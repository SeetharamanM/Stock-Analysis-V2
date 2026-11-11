"""
Local backend for the Equity Analysis web app.

Pipeline:  stock name -> Screener.in scrape -> Yahoo quote -> Claude ->
           standalone HTML report written to outputs/

Run:  python app/server.py       (see README.md)
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyst  # noqa: E402
import market  # noqa: E402
import report as report_mod  # noqa: E402
import screener  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
STATIC = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_dotenv() -> None:
    """Minimal .env loader so the key can live in a file instead of the shell."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    # utf-8-sig strips a BOM if PowerShell wrote one, which would otherwise
    # turn the first key into "﻿ANTHROPIC_API_KEY" and silently not match.
    for line in env_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# --------------------------------------------------------------------------
# static
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC, filename)


@app.route("/reports/<path:filename>")
def serve_report(filename):
    # send_from_directory rejects traversal outside OUTPUTS.
    return send_from_directory(OUTPUTS, filename)


# --------------------------------------------------------------------------
# api
# --------------------------------------------------------------------------

@app.route("/api/health")
def health():
    try:
        cfg = analyst.resolve_config()
        provider, model, key_ok = cfg.provider, cfg.model, bool(cfg.api_key)
        key_env, model_fixed = cfg.key_env, cfg.model_fixed
        cfg_error = None
    except analyst.AnalystError as exc:
        provider = analyst.env_provider()
        model, key_ok, key_env, model_fixed = analyst.DEFAULT_MODEL, False, "", False
        cfg_error = str(exc)
    return jsonify({
        "ok": True,
        "provider": provider,
        "model": model,
        "model_fixed": model_fixed,
        "key_env": key_env,
        "api_key_configured": key_ok,
        "config_error": cfg_error,
        "outputs": str(OUTPUTS),
        "default_model": model,
        "providers": analyst.list_providers(),
    })


@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify([])
    try:
        return jsonify(screener.search(q)[:8])
    except screener.ScreenerError as exc:
        return jsonify({"error": str(exc)}), 502


STAMP_RE = re.compile(r"-equity-analysis-(\d{8})-(\d{6})\.html$")


def _stamp_from_name(name: str):
    """
    Read the generation time out of the filename rather than the filesystem —
    st_mtime is unreliable on the network share this project lives on.
    """
    m = STAMP_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


@app.route("/api/reports")
def api_reports():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    found = []
    for p in OUTPUTS.glob("*-equity-analysis-*.html"):
        found.append((_stamp_from_name(p.name) or datetime.min, p))
    found.sort(key=lambda pair: pair[0], reverse=True)

    items = []
    for stamp, p in found[:25]:
        items.append({
            "filename": p.name,
            "url": f"/reports/{p.name}",
            "modified": stamp.strftime("%d %b %Y, %H:%M") if stamp != datetime.min else "—",
            "kb": max(1, round(p.stat().st_size / 1024)),
        })
    return jsonify(items)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    stock = (payload.get("stock") or "").strip()
    model = (payload.get("model") or "").strip() or None
    provider = (payload.get("provider") or "").strip().lower() or None
    if not stock:
        return jsonify({"error": "No stock supplied."}), 400
    try:
        cfg = analyst.resolve_config(model, provider)
    except analyst.AnalystError as exc:
        return jsonify({"error": str(exc)}), 400
    provider_label = analyst.PROVIDER_DEFAULTS[cfg.provider]["label"]

    def stream():
        try:
            # 1 — resolve + scrape ------------------------------------------
            yield sse("progress", {"step": "screener", "message": f"Looking up {stock} on Screener.in…"})
            company = screener.fetch_company(stock)
            yield sse("progress", {
                "step": "screener",
                "done": True,
                "message": f"Fetched fundamentals for {company['name']}",
                "company": {"name": company["name"], "ticker": company["ticker"],
                            "url": company["screener_url"]},
            })

            # 2 — live quote -------------------------------------------------
            yield sse("progress", {"step": "quote", "message": "Cross-referencing live price…"})
            live = market.quote(company.get("ticker", ""))
            yield sse("progress", {
                "step": "quote", "done": True,
                "message": (f"Live: {live.get('currency','')} {live.get('price')} "
                            f"({live.get('symbol')})") if live else
                           "Yahoo quote unavailable — using Screener price.",
            })

            # 3 — analyse ----------------------------------------------------
            yield sse("progress", {"step": "analyse",
                             "message": f"{cfg.model} ({provider_label}) is writing the report…"})

            q: queue.Queue = queue.Queue()
            result: dict = {}

            def work():
                try:
                    result["report"] = analyst.analyse(
                        company, live, model=cfg.model, provider=cfg.provider,
                        on_progress=lambda n, phase="writing": q.put(("progress", (n, phase))),
                    )
                except Exception as exc:  # surfaced to the client below
                    result["error"] = exc
                finally:
                    q.put(("done", None))

            thread = threading.Thread(target=work, daemon=True)
            thread.start()

            while True:
                kind, value = q.get()
                if kind == "done":
                    break
                count, phase = value
                if phase == "thinking":
                    yield sse("progress", {
                        "step": "analyse",
                        "message": f"Model is thinking… ({count:,} reasoning characters so far)",
                    })
                else:
                    yield sse("progress", {"step": "analyse", "chars": count})
            thread.join()

            if "error" in result:
                raise result["error"]
            rep = result["report"]
            usage = rep.get("_usage", {})
            yield sse("progress", {
                "step": "analyse", "done": True,
                "message": f"Report written ({usage.get('output_tokens','?')} output tokens)",
            })

            # 4 — save -------------------------------------------------------
            # A failure here must not discard a paid model call, so the report
            # is still returned to the browser if writing the file fails.
            yield sse("progress", {"step": "save", "message": "Saving HTML report…"})
            saved, save_error = None, None
            try:
                saved = report_mod.save(rep, company, live)
                yield sse("progress", {"step": "save", "done": True,
                                       "message": f"Saved {saved['filename']}"})
            except Exception as exc:
                traceback.print_exc()
                save_error = f"Report generated but could not be saved to outputs/: {exc}"
                yield sse("progress", {"step": "save", "done": True, "message": save_error})

            yield sse("result", {"report": rep, "file": saved, "save_error": save_error,
                                 "charts": report_mod.build_charts(company),
                                 "company": {"name": company["name"],
                                             "ticker": company["ticker"],
                                             "screener_url": company["screener_url"]},
                                 "live": live})

        except (screener.ScreenerError, analyst.AnalystError) as exc:
            yield sse("error", {"message": str(exc)})
        except Exception as exc:  # unexpected — log fully, report briefly
            traceback.print_exc()
            yield sse("error", {"message": f"Unexpected error: {exc}"})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "5057"))
    try:
        cfg = analyst.resolve_config()
        provider, model, configured, key_env = (
            cfg.provider, cfg.model, bool(cfg.api_key), cfg.key_env)
    except analyst.AnalystError as exc:
        provider, model, configured, key_env = "?", str(exc), False, "API key"
    print("=" * 66)
    print("  Equity Analysis backend")
    print(f"  URL:      http://127.0.0.1:{port}")
    print(f"  Outputs:  {OUTPUTS}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  API key:  {'configured' if configured else f'NOT SET ({key_env}) — see README.md'}")
    print("=" * 66)
    # load_dotenv=False: our own loader already read .env, and leaving it on
    # makes Flask print a misleading "install python-dotenv" tip.
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False, load_dotenv=False)
