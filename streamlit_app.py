"""
Streamlit front end for the Equity Analysis Report Generator.

Reuses the same pipeline as the Flask app (app/server.py):
    Screener.in scrape -> Yahoo quote -> LLM -> standalone HTML report

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud — see README.md ("Deploying to Streamlit").

API keys come from Streamlit secrets (.streamlit/secrets.toml locally, or the
app's Secrets box on Streamlit Cloud) and fall back to the .env file.
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

import analyst  # noqa: E402
import market  # noqa: E402
import report as report_mod  # noqa: E402
import screener  # noqa: E402

st.set_page_config(page_title="Equity Analysis Report", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
# config: secrets -> environment (analyst.py reads os.environ)
# --------------------------------------------------------------------------

CONFIG_KEYS = (
    "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY",
    "LLM_PROVIDER", "LLM_MODEL", "LLM_MAX_TOKENS", "LLM_BASE_URL",
)


def load_config() -> None:
    try:
        secrets = dict(st.secrets)
    except Exception:  # no secrets file — fine when running off .env
        secrets = {}
    for key in CONFIG_KEYS:
        value = secrets.get(key)
        if value not in (None, ""):
            os.environ[key] = str(value)

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_config()


def app_password() -> str:
    try:
        return str(st.secrets.get("APP_PASSWORD", "") or "")
    except Exception:
        return ""


def check_password() -> bool:
    """
    Optional gate. A public Streamlit app would otherwise let anyone spend
    your API credit, so set APP_PASSWORD in secrets for a deployed app.
    """
    expected = app_password()
    if not expected or st.session_state.get("authed"):
        return True
    st.title("Equity Analysis Report")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw, expected):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong password.")
    return False


if not check_password():
    st.stop()


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("Equity Analysis Report")
st.caption(
    "Enter an Indian listed company name or NSE ticker. Live fundamentals come from "
    "Screener.in, the price is cross-checked against Yahoo Finance, and an LLM writes a "
    "seven-section report."
)

providers = analyst.list_providers()
by_id = {p["id"]: p for p in providers}
ids = [p["id"] for p in providers]
default_pid = analyst.env_provider() if analyst.env_provider() in by_id else ids[0]
CUSTOM = "Custom model id…"

col_stock, col_prov, col_model = st.columns([3, 2, 2])
with col_stock:
    stock = st.text_input("Stock name or ticker", placeholder="e.g. HDFC Bank, INFY, ITC")
with col_prov:
    pid = st.selectbox(
        "Provider", ids, index=ids.index(default_pid),
        format_func=lambda i: by_id[i]["label"] + ("" if by_id[i]["api_key_configured"] else " — no key"),
    )
prov = by_id[pid]
with col_model:
    if prov["model_fixed"]:
        model = prov["default_model"]
        st.selectbox("Model", [model], disabled=True, help="Pinned by LLM_MODEL.")
    else:
        options = [m["id"] for m in prov["models"]]
        if prov["default_model"] not in options:
            options.insert(0, prov["default_model"])
        labels = {m["id"]: m["label"] for m in prov["models"]}
        choice = st.selectbox(
            "Model", options + [CUSTOM], index=options.index(prov["default_model"]),
            format_func=lambda m: labels.get(m, m), key=f"model_{pid}",
        )
        model = st.text_input("Custom model id", key=f"custom_{pid}") if choice == CUSTOM else choice

if not prov["api_key_configured"]:
    st.warning(
        f"No `{prov['key_env']}` configured for {prov['label']}. Add it to the app's "
        "secrets (Streamlit Cloud → app → Settings → Secrets). Analysis will fail."
    )

go = st.button("Generate report", type="primary", disabled=not stock.strip())

if go:
    if not (model or "").strip():
        st.error("Enter a model id.")
        st.stop()
    try:
        with st.status(f"Analysing {stock}…", expanded=True) as status:
            st.write("Looking up on Screener.in…")
            company = screener.fetch_company(stock.strip())
            st.write(f"Fetched fundamentals for **{company['name']}** ({company['ticker']})")

            live = market.quote(company.get("ticker", ""))
            st.write(
                f"Live price: {live.get('currency', '')} {live.get('price')} ({live.get('symbol')})"
                if live else "Yahoo quote unavailable — using Screener price."
            )

            st.write(f"{model} ({prov['label']}) is writing the report…")
            ticker = st.empty()

            def on_progress(n, phase="writing"):
                ticker.caption(f"{'Thinking' if phase == 'thinking' else 'Writing'}… {n:,} characters")

            rep = analyst.analyse(company, live, model=model, provider=pid, on_progress=on_progress)
            usage = rep.get("_usage", {})
            ticker.caption(f"Done — {usage.get('output_tokens', '?')} output tokens")
            status.update(label=f"Report ready: {company['name']}", state="complete")

        html = report_mod.render_html(rep, company, live)
        name = rep.get("companyName") or company.get("name") or "report"
        st.session_state["last"] = {
            "html": html,
            "filename": f"{report_mod.slugify(name)}-equity-analysis.html",
        }
    except (screener.ScreenerError, analyst.AnalystError) as exc:
        st.error(str(exc))
    except Exception as exc:  # unexpected — show briefly, log fully
        st.exception(exc)

last = st.session_state.get("last")
if last:
    st.download_button("Download HTML report", last["html"], file_name=last["filename"],
                       mime="text/html")
    components.html(last["html"], height=1400, scrolling=True)

st.divider()
st.caption(
    "Educational and informational use only — not investment advice. Figures are scraped and "
    "model-processed and may be wrong. Consult a SEBI-registered investment adviser before acting."
)
