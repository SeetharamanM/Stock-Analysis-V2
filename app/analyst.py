"""
Turns a scraped data pack into a structured equity research report via an LLM.

The model is given the real figures and is explicitly forbidden from using
remembered numbers, which is what separates this from a pure-LLM report.

Provider is chosen at run time from the environment (usually the .env file):

    LLM_PROVIDER = anthropic | openrouter | deepseek     (default: anthropic;
                   the UI's provider dropdown overrides it per request)
    LLM_MODEL    = <model id>        overrides the provider's default model
    LLM_MAX_TOKENS = <int>          overrides the provider's output-token cap
    LLM_BASE_URL = <url>            overrides the provider's API base URL

    ANTHROPIC_API_KEY / OPENROUTER_API_KEY / DEEPSEEK_API_KEY   per-provider keys
    LLM_API_KEY                     generic fallback used by any provider

anthropic talks the native Messages API (tool use + prompt caching); openrouter
and deepseek are OpenAI-compatible and go through the `openai` SDK.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-sonnet-5"
# A full seven-section report runs ~10-14k output tokens; detailed smallcaps run
# longer. Truncation mid-tool-use silently loses the sections, so leave headroom.
MAX_TOKENS = 32000

# Per-provider defaults. `max_tokens` reflects a comfortable ceiling for a full
# seven-section report (~10-14k output tokens). DeepSeek's v4 models allow far
# more (v4-flash up to 65,536, v4-pro more), so 32000 leaves ample headroom;
# LLM_MAX_TOKENS overrides it per environment.
PROVIDER_DEFAULTS = {
    "anthropic": {
        "label": "Claude (Anthropic)",
        "model": DEFAULT_MODEL,
        "base_url": None,
        "key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 32000,
    },
    "deepseek": {
        "label": "DeepSeek",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "max_tokens": 32000,
    },
    "openrouter": {
        "label": "OpenRouter",
        "model": "anthropic/claude-sonnet-4",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "max_tokens": 32000,
    },
}

# Models offered in the UI dropdown, per provider: (model id, label). Any other
# id can still be typed in via the UI's "Custom model…" option — useful for
# OpenRouter, which exposes hundreds of models.
PROVIDER_MODELS = {
    "anthropic": [
        ("claude-sonnet-5", "Claude Sonnet 5 (balanced)"),
        ("claude-opus-4-8", "Claude Opus 4.8 (deepest analysis)"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 (fastest)"),
        ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
        ("claude-opus-5-5", "Claude Opus 5.5"),
        ("claude-haiku-5-5", "Claude Haiku 5.5"),
    ],
    "deepseek": [
        ("deepseek-v4-pro", "DeepSeek V4 Pro (deeper)"),
        ("deepseek-v4-flash", "DeepSeek V4 Flash (faster)"),
    ],
    "openrouter": [
        ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
        ("deepseek/deepseek-chat", "DeepSeek Chat"),
        ("openai/gpt-4o", "GPT-4o"),
        ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
    ],
}

# The user's own brief. Loaded at call time so edits to it take effect
# without touching the code.
PROMPT_FILE = Path(__file__).resolve().parent.parent / "Stock Analysis Prompt.txt"


class AnalystError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str
    key_env: str
    max_tokens: int
    model_fixed: bool  # True when LLM_MODEL pins the model regardless of the UI


def env_provider() -> str:
    """The provider named by LLM_PROVIDER in .env (the UI's default choice)."""
    return (os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower()


def resolve_config(requested_model: str | None = None,
                   requested_provider: str | None = None) -> LLMConfig:
    """
    Work out which provider/model/key to use.

    Provider: the UI's selection if given, else LLM_PROVIDER from .env, else
    anthropic.

    Model: LLM_MODEL in .env pins the model, but only for the .env provider
    (it is a model id for that provider and means nothing to the others).
    Otherwise the UI's selection is used, falling back to the provider default.
    LLM_BASE_URL likewise only applies to the .env provider.
    """
    default_provider = env_provider()
    provider = (requested_provider or "").strip().lower() or default_provider
    if provider not in PROVIDER_DEFAULTS:
        raise AnalystError(
            f"Unknown LLM provider '{provider}'. Use one of: "
            + ", ".join(PROVIDER_DEFAULTS)
        )
    defaults = PROVIDER_DEFAULTS[provider]
    is_env_provider = provider == default_provider

    env_model = (os.environ.get("LLM_MODEL") or "").strip() if is_env_provider else ""
    model_fixed = bool(env_model)
    if env_model:
        model = env_model
    elif requested_model and requested_model.strip():
        model = requested_model.strip()
    else:
        model = defaults["model"]

    env_base = (os.environ.get("LLM_BASE_URL") or "").strip() if is_env_provider else ""
    base_url = env_base or defaults["base_url"]

    key_env = defaults["key_env"]
    api_key = (os.environ.get(key_env) or os.environ.get("LLM_API_KEY") or "").strip()

    max_tokens = defaults["max_tokens"]
    env_max = (os.environ.get("LLM_MAX_TOKENS") or "").strip()
    if env_max:
        try:
            max_tokens = int(env_max)
        except ValueError:
            pass

    return LLMConfig(
        provider=provider, model=model, base_url=base_url, api_key=api_key,
        key_env=key_env, max_tokens=max_tokens, model_fixed=model_fixed,
    )


def list_providers() -> list[dict]:
    """Everything the UI needs to render the provider + model pickers."""
    out = []
    for pid, d in PROVIDER_DEFAULTS.items():
        cfg = resolve_config(requested_provider=pid)
        out.append({
            "id": pid,
            "label": d["label"],
            "key_env": cfg.key_env,
            "api_key_configured": bool(cfg.api_key),
            "default_model": cfg.model,
            "model_fixed": cfg.model_fixed,
            "models": [{"id": m, "label": lbl} for m, lbl in PROVIDER_MODELS.get(pid, [])],
        })
    return out


# --------------------------------------------------------------------------
# formatting the scraped data for the model
# --------------------------------------------------------------------------

def _md_table(table: dict, keep: int = 6) -> str:
    """Render a scraped table as markdown, keeping the last `keep` periods."""
    cols = table.get("columns") or []
    rows = table.get("rows") or {}
    if not cols or not rows:
        return "_not available_"
    idx = list(range(max(0, len(cols) - keep), len(cols)))
    header = "| Metric | " + " | ".join(cols[i] for i in idx) + " |"
    sep = "|---" * (len(idx) + 1) + "|"
    lines = [header, sep]
    for label, values in rows.items():
        cells = [values[i] if i < len(values) else "" for i in idx]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _peers_md(peers: dict) -> str:
    if not peers.get("available"):
        return "_No peer group available from Screener for this company. Compare qualitatively._"
    cols = peers.get("columns") or []
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for row in peers.get("rows", []):
        lines.append("| " + " | ".join(row + [""] * (len(cols) - len(row))) + " |")
    if peers.get("median"):
        med = peers["median"]
        lines.append("| " + " | ".join(med + [""] * (len(cols) - len(med))) + " |")
    return "\n".join(lines)


def build_data_pack(company: dict, live: dict) -> str:
    """Assemble every scraped figure into one readable block for the model."""
    derived = company.get("derived", {})
    per_year = derived.get("per_year", [])

    computed_rows = []
    for y in per_year[-6:]:
        computed_rows.append(
            f"| {y['period']} | {y.get('sales')} | {y.get('net_profit')} | "
            f"{y.get('net_profit_margin_pct')} | {y.get('operating_margin_pct')} | "
            f"{y.get('debt_to_equity')} | {y.get('asset_turnover')} | "
            f"{y.get('eps')} | {y.get('cfo')} | {y.get('dividend_payout_pct')} |"
        )

    parts = [
        f"# LIVE DATA PACK — {company.get('name')}",
        "",
        f"Source: {company.get('screener_url')}",
        f"NSE ticker: {company.get('ticker')}",
        "",
        "## Headline ratios (Screener.in, live)",
        "\n".join(f"- **{k}**: {v}" for k, v in (company.get("top_ratios") or {}).items())
        or "_not available_",
        "",
        "## Live quote cross-reference (Yahoo Finance)",
        (
            "\n".join(f"- **{k}**: {v}" for k, v in live.items() if v is not None)
            if live
            else "_Yahoo Finance quote unavailable; use the Screener price._"
        ),
        "",
        "## Company description (Screener.in)",
        company.get("about") or "_not available_",
        "",
        "## Key points",
        company.get("key_points") or "_not available_",
        "",
        "## Profit & Loss (Rs Cr)",
        _md_table(company.get("profit_loss", {})),
        "",
        "## Balance Sheet (Rs Cr)",
        _md_table(company.get("balance_sheet", {})),
        "",
        "## Cash Flow (Rs Cr)",
        _md_table(company.get("cash_flow", {})),
        "",
        "## Efficiency / other ratios",
        _md_table(company.get("ratios", {})),
        "",
        "## Shareholding pattern (%)",
        _md_table(company.get("shareholding", {}), keep=7),
        "",
        "## Quarterly results (Rs Cr)",
        _md_table(company.get("quarters", {}), keep=6),
        "",
        "## Growth / CAGR (Screener)",
        json.dumps(company.get("growth", {}), indent=2),
        "",
        "## Peer comparison (Screener; final row is the industry median)",
        _peers_md(company.get("peers", {})),
        "",
        "## PRE-COMPUTED RATIOS — use these verbatim, do not recalculate",
        "| Period | Sales | Net Profit | NPM % | OPM % | Debt/Equity | Asset Turnover | EPS | CFO | Payout % |",
        "|---|---|---|---|---|---|---|---|---|---|",
        "\n".join(computed_rows) or "_not available_",
        "",
        "### Pre-computed CAGR (%)",
        json.dumps(derived.get("cagr", {}), indent=2),
        f"\nLatest reported period in the data above: **{derived.get('latest_period')}**",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

DATA_RULES = """
=== ABSOLUTE DATA RULES ===
1. Every number you state MUST come from the LIVE DATA PACK in the user message.
   You are strictly forbidden from using figures recalled from training data.
2. If a metric is not in the data pack, say so plainly (e.g. "not disclosed by the
   company") and give a qualitative assessment instead. NEVER invent a figure.
3. Use the pre-computed ratio and CAGR tables verbatim. Do not recompute them.
4. Screener columns are calendar month-ends: "Mar 2026" is FY26 (1 Apr 2025 -
   31 Mar 2026). Always label years in FY form. The most recent full fiscal year
   in the pack is the latest annual period - use it as "current".
5. Amounts are in Rs crore unless stated. Preserve Indian digit grouping.
6. The target price must be arithmetically consistent with the current price:
   if current price is P and target is T, the upside is (T-P)/P as a percentage.
   Compute this correctly and state the direction (upside/downside) accordingly.
7. Report only the last 5 fiscal years plus TTM where shown.
"""

HTML_CONTRACT = """
=== HTML FORMATTING CONTRACT ===
Each section is an HTML fragment (no <html>, <head>, <body>, <style>, <script>).
Use only these elements and classes:
  <h3> major heading, <h4> sub-heading, <p>, <ul>, <ol>, <li>, <strong>, <em>
  <span class="figure">..</span>            highlight an important number / the recommendation
  <div class="table-wrap"><table><thead><tr><th>..</th></tr></thead>
       <tbody><tr><td>..</td></tr></tbody></table></div>
  <div class="callout"><span class="callout-label">Trend Analysis</span><p>..</p></div>
  <div class="callout takeaway"><span class="callout-label">Key Takeaways</span><p>..</p></div>
  <div class="callout risk"><span class="callout-label">Red Flags</span><p>..</p></div>
Trend words in the final column of every table must be wrapped:
  <span class="t-up">Improving</span> / <span class="t-down">Declining</span> /
  <span class="t-flat">Stable</span>
Do NOT hand-write bar charts. A five-year visual summary is generated automatically
from the source figures and appended to section 2 for you.
Every sub-section a) to h) must contain a data table.

=== REQUIRED STRUCTURE ===
1 overview      Company intro (name, sector, activities, promoters), core segments as a
                bulleted list with revenue share where known, recent significant events.
2 quantitative  Sub-sections a) to h), each = data table + "Trend Analysis" callout +
                "Key Takeaways" callout, over the last 5 fiscal years:
                a) Market Valuation and Price Metrics (Market Cap, Price, P/E)
                b) Profitability and Returns (ROE, ROCE, Net Profit Margin, Operating Margin)
                c) Growth Metrics (Revenue CAGR, EPS CAGR - 5yr and 3yr)
                d) Balance Sheet Strength (Debt-to-Equity)
                e) Cash Flow Analysis (Cash Flow from Operations)
                f) Dividend Analysis (Dividend Yield, Payout Ratio)
                g) Efficiency Ratios (Asset Turnover; add cash conversion cycle /
                   working capital days when present in the data)
                h) Valuation vs peers - use the peer table if supplied, otherwise a
                   qualitative premium/in-line/discount assessment. Treat the peer P/E
                   comparison qualitatively rather than as precise fact.
                Highlight year-on-year changes, compare to industry benchmarks,
                flag red flags, and explain implications for investors.
3 qualitative   a) Business Model (products/services, revenue streams, competitive
                advantages) b) Management Quality (track record, governance)
                c) Growth Strategy (expansion, R&D)
4 shareholding  Table of the reported quarters in the pack (Promoters, FIIs, DIIs,
                Government, Public) with trend spans, then analysis of promoter holding,
                the FII/DII shift, public shareholding, and a Key Takeaways callout.
5 thesis        "Key Drivers for Future Growth", "Potential Catalysts for Stock Price
                Movement", "Positioning for Industry Trends and Challenges" - bulleted,
                each bullet led by a bolded phrase. Focus on this industry's specific
                trends, challenges and innovations.
6 valuation     "Fair Value Estimate" with a visible build-up (SOTP / DCF / P-E multiple;
                use P/B and NIM-driven methods for banks and NBFCs), the resulting fair
                value, and implied upside vs the current price. Then "Investment
                Recommendation" and "Rationale for Recommendation" as bullets.
                Wrap the recommendation and target price in <span class="figure">.
7 conclusion    Three to four paragraphs restating the analysis and the recommendation.

=== SECTOR ADAPTATION ===
For banks and NBFCs substitute NIM, GNPA/NNPA, CASA, provision coverage, capital
adequacy and Price-to-Book for metrics that do not apply (ROCE, asset turnover,
operating margin). Say why the substitution was made.

Maintain an objective tone: include genuine risks and red flags, not only positives.
Do not mention the report generation date.
"""

TOOL = {
    "name": "render_report",
    "description": "Render the completed seven-section equity analysis report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "companyName": {"type": "string"},
            "ticker": {"type": "string"},
            "exchange": {"type": "string"},
            "sector": {"type": "string", "description": "Sector / industry"},
            "recommendation": {
                "type": "object",
                "properties": {
                    "rating": {
                        "type": "string",
                        "description": "One of: Strong Buy, Buy, Hold, Sell, Strong Sell",
                    },
                    "targetPrice": {"type": "string", "description": "e.g. 'Rs 4,250'"},
                    "currentPrice": {"type": "string", "description": "from the data pack"},
                    "changePct": {"type": "string", "description": "e.g. '34% upside'"},
                    "direction": {"type": "string", "description": "'upside' or 'downside'"},
                    "horizon": {"type": "string", "description": "e.g. '18-24 months'"},
                },
                "required": ["rating", "targetPrice", "currentPrice", "changePct", "horizon"],
            },
            # Sections are FLAT top-level fields, deliberately not nested under a
            # "sections" object. Nesting made the model occasionally emit the whole
            # group as one hand-escaped JSON string, and at ~27KB of HTML that
            # escaping sometimes broke, losing the entire report body.
            "overview": {"type": "string", "description": "Section 1 HTML fragment"},
            "quantitative": {"type": "string", "description": "Section 2 HTML fragment, a) to h)"},
            "qualitative": {"type": "string", "description": "Section 3 HTML fragment"},
            "shareholding": {"type": "string", "description": "Section 4 HTML fragment"},
            "thesis": {"type": "string", "description": "Section 5 HTML fragment"},
            "valuation": {"type": "string", "description": "Section 6 HTML fragment"},
            "conclusion": {"type": "string", "description": "Section 7 HTML fragment"},
        },
        "required": ["companyName", "sector", "recommendation",
                     "overview", "quantitative", "qualitative",
                     "shareholding", "thesis", "valuation", "conclusion"],
    },
}


def _system_blocks() -> list[dict]:
    """System prompt, cached so repeated runs are cheap."""
    brief = ""
    try:
        brief = PROMPT_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass

    intro = (
        "You are a senior equity research analyst covering Indian listed equities, "
        "writing objective, evidence-led fundamental research in the house style of "
        "SEETHARAMAN. You will be given a LIVE DATA PACK scraped from Screener.in and "
        "Yahoo Finance. Call the render_report tool exactly once. Never reply in plain text."
    )

    blocks = [{"type": "text", "text": intro + "\n" + DATA_RULES + "\n" + HTML_CONTRACT}]
    if brief.strip():
        blocks.append({
            "type": "text",
            "text": "=== ANALYST'S OWN BRIEF (follow its structure and style) ===\n" + brief,
        })
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def _system_text() -> str:
    """Flatten the system prompt to plain text for OpenAI-compatible providers,
    which do not support Anthropic's structured/cached system blocks."""
    return "\n".join(b["text"] for b in _system_blocks())


SECTION_KEYS = ["overview", "quantitative", "qualitative", "shareholding",
                "thesis", "valuation", "conclusion"]


def _as_dict(value) -> dict:
    """
    Tool-use sometimes returns a nested object as a JSON *string* rather than a
    real object. Accept both so a whole report is never lost to that variance.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _salvage_sections(blob: str) -> dict:
    """
    Last resort: pull each section out of a malformed JSON string by hand.

    The model occasionally hand-escapes a large nested object and gets it
    slightly wrong, so json.loads fails on ~27KB of otherwise-good HTML.
    Rather than discard the whole report, recover what is recoverable.
    """
    out = {}
    for key in SECTION_KEYS:
        # "key": "....." up to the next `", "<known key>":` boundary or the end
        pattern = (
            r'"' + key + r'"\s*:\s*"(.*?)"\s*(?:,\s*"(?:' + "|".join(SECTION_KEYS) + r')"\s*:|\s*\}?\s*$)'
        )
        m = re.search(pattern, blob, re.S)
        if not m:
            continue
        text = m.group(1)
        # undo the JSON escaping that would have been handled by json.loads
        text = (text.replace('\\"', '"').replace("\\n", "\n")
                    .replace("\\t", "\t").replace("\\/", "/").replace("\\\\", "\\"))
        if text.strip():
            out[key] = text
    return out


def _extract_json_object(text: str) -> dict:
    """
    Parse a JSON *object* out of free text.

    Reasoning models called with tool_choice="auto" sometimes answer in prose
    with the JSON inline, or wrap it in a ```json code fence, instead of putting
    it in clean tool-call arguments. Try the whole string, then a fenced block,
    then the first `{`…last `}` span. Return {} if nothing parses to an object.
    """
    if not text or not text.strip():
        return {}
    s = text.strip()
    candidates = [s]
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last > first:
        candidates.append(s[first:last + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_report_input(tool_args: str | None, text: str | None) -> dict:
    """
    Turn whatever the model returned into a report dict, regardless of model.

    With tool_choice="auto" (so thinking models keep their reasoning), the
    payload can arrive three ways: as tool-call arguments, as JSON in the text
    reply, or as a slightly malformed blob. Try them in that order of fidelity,
    salvaging sections by hand as a last resort. Returns {} when nothing usable
    is found; the caller turns that into an AnalystError.
    """
    # 1. Preferred: a tool call carrying JSON arguments.
    if tool_args and tool_args.strip():
        obj = _extract_json_object(tool_args)
        if obj:
            return obj
    # 2. The model replied in text instead of (or as well as) calling the tool.
    if text and text.strip():
        obj = _extract_json_object(text)
        if obj:
            return obj
    # 3. Last resort: hand-salvage sections from whichever blob we have.
    for blob in (tool_args, text):
        if blob and blob.strip():
            salvaged = _salvage_sections(blob)
            if salvaged:
                return salvaged
    return {}


def _normalise(report: dict) -> dict:
    """Coerce the report into the shape the renderer expects."""
    report["recommendation"] = _as_dict(report.get("recommendation"))

    # Preferred path: flat top-level section fields (see TOOL schema).
    sections = {k: report[k] for k in SECTION_KEYS
                if isinstance(report.get(k), str) and report[k].strip()}

    # Fall back to a nested "sections" object or string if one turns up anyway.
    if len(sections) < len(SECTION_KEYS):
        raw = report.get("sections")
        nested = _as_dict(raw)
        if not nested and isinstance(raw, str) and raw.strip():
            nested = _salvage_sections(raw)
        for key, value in nested.items():
            if key not in sections:
                sections[key] = value if isinstance(value, str) else json.dumps(value)

    report["sections"] = {
        key: (value if isinstance(value, str) else json.dumps(value))
        for key, value in sections.items()
    }
    for key in SECTION_KEYS:  # keep the payload tidy
        report.pop(key, None)
    return report


def get_client(api_key: str | None = None) -> anthropic.Anthropic:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise AnalystError(
            "No Anthropic API key configured. Set the ANTHROPIC_API_KEY environment "
            "variable (see README) and restart the server."
        )
    return anthropic.Anthropic(api_key=key)


def _finish_report(report_input: dict, usage: dict, stop_reason: str) -> dict:
    """Normalise, validate completeness, and stamp usage — shared by all
    providers so the completeness guarantee is identical everywhere."""
    report = _normalise(report_input)

    filled = [k for k, v in report["sections"].items() if v and v.strip()]
    if len(filled) < len(SECTION_KEYS):
        missing = [k for k in SECTION_KEYS if k not in filled]
        raise AnalystError(
            "The model returned an incomplete report — missing section(s): "
            f"{', '.join(missing)}. This usually means the response was cut short. "
            f"(stop_reason={stop_reason}, "
            f"output_tokens={usage.get('output_tokens')}). Try again."
        )

    report["_usage"] = usage
    return report


def analyse(company: dict, live: dict, model: str | None = None,
            api_key: str | None = None, on_progress=None,
            provider: str | None = None) -> dict:
    """Run the report through the chosen provider and return the report dict."""
    cfg = resolve_config(model, provider)
    if not (api_key or cfg.api_key):
        raise AnalystError(
            f"No API key configured for provider '{cfg.provider}'. Set {cfg.key_env} "
            "in your .env (see README) and restart the server."
        )

    pack = build_data_pack(company, live)
    user_msg = (
        f"Produce the complete equity analysis report for **{company.get('name')}** "
        f"({company.get('ticker')}).\n\n"
        "Use ONLY the figures in the data pack below. Call render_report once with every "
        "section fully written.\n\n" + pack
    )

    if cfg.provider == "anthropic":
        return _analyse_anthropic(cfg, user_msg, api_key, on_progress)
    return _analyse_openai(cfg, user_msg, api_key, on_progress)


def _analyse_anthropic(cfg: LLMConfig, user_msg: str, api_key: str | None,
                       on_progress=None) -> dict:
    """Native Anthropic path: tool use + prompt caching over the Messages API."""
    client = get_client(api_key or cfg.api_key)

    try:
        # tool_choice="auto" rather than forcing render_report, so the path is
        # model-agnostic: models that reason before answering (which restrict a
        # forced tool_choice) keep their reasoning. The prompt still directs the
        # model to call the tool; if it answers in text, we recover it below.
        with client.messages.stream(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=_system_blocks(),
            tools=[TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            chars = 0
            for event in stream:
                if event.type == "content_block_delta":
                    dtype = getattr(event.delta, "type", "")
                    if dtype == "input_json_delta":
                        chars += len(event.delta.partial_json or "")
                    elif dtype == "text_delta":
                        chars += len(event.delta.text or "")
                    if on_progress:
                        on_progress(chars, "writing")
            final = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise AnalystError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from exc
    except anthropic.RateLimitError as exc:
        raise AnalystError("Rate limited or out of credit on the Anthropic API.") from exc
    except anthropic.APIStatusError as exc:
        raise AnalystError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIError as exc:
        raise AnalystError(f"Anthropic API error: {exc}") from exc

    # A tool_use block truncated at max_tokens yields incomplete JSON, which the
    # SDK surfaces as a partial object. Catch that here rather than saving a
    # report whose sections silently came back empty.
    if final.stop_reason == "max_tokens":
        raise AnalystError(
            f"The report exceeded the {cfg.max_tokens:,}-token output limit and was cut off "
            "before it finished. Raise LLM_MAX_TOKENS, or try a different model."
        )

    # Prefer a render_report tool_use block; fall back to any text the model
    # produced instead (thinking models may answer in prose under auto).
    tool_input = None
    text_out = ""
    for block in final.content:
        if block.type == "tool_use" and block.name == "render_report":
            tool_input = dict(block.input)
        elif block.type == "text":
            text_out += block.text or ""

    report_input = tool_input if tool_input else _extract_report_input(None, text_out)
    if not report_input:
        raise AnalystError(
            "The model did not return a parseable report (stop_reason="
            f"{final.stop_reason}) — try again, or switch model."
        )

    return _finish_report(
        report_input,
        {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
            "stop_reason": final.stop_reason,
            "model": cfg.model,
            "provider": cfg.provider,
        },
        final.stop_reason,
    )


def _analyse_openai(cfg: LLMConfig, user_msg: str, api_key: str | None,
                    on_progress=None) -> dict:
    """OpenAI-compatible path (OpenRouter, DeepSeek) via the `openai` SDK.

    The render_report tool is offered with tool_choice="auto" (not forced), so
    reasoning models keep their reasoning. The report normally arrives as JSON in
    the function-call arguments, but a model that answers in text is handled too:
    both go through the same _extract_report_input/normalise machinery as the
    native path.
    """
    try:
        import openai
        from openai import OpenAI
    except ImportError as exc:
        raise AnalystError(
            f"The 'openai' package is required for the '{cfg.provider}' provider. "
            "Install it with:  pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key or cfg.api_key, base_url=cfg.base_url)

    openai_tool = {
        "type": "function",
        "function": {
            "name": TOOL["name"],
            "description": TOOL["description"],
            "parameters": TOOL["input_schema"],
        },
    }

    # OpenRouter uses these headers for attribution/leaderboards; harmless
    # elsewhere but only sent when talking to OpenRouter.
    extra_headers = None
    extra_body = None
    if cfg.provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "http://127.0.0.1:5057",
            "X-Title": "Equity Analysis Report Generator",
        }
        # Ask OpenRouter to STREAM reasoning tokens rather than let the upstream
        # compute them silently. A reasoning model that thinks without emitting
        # anything makes the connection go idle, which is what trips OpenRouter's
        # "Upstream idle timeout". Streaming the reasoning keeps tokens flowing
        # (and feeds the "thinking" progress indicator). Harmless for models that
        # don't reason. Reasoning is kept ON — only its streaming is requested.
        extra_body = {"reasoning": {"enabled": True}}

    try:
        # tool_choice="auto" (not a forced function) so reasoning/"thinking"
        # models keep their reasoning on: a forced tool_choice is what they
        # reject ("Thinking mode does not support this tool_choice"). The strong
        # system prompt still steers them to call render_report; if a model
        # answers in text instead, _extract_report_input recovers it below.
        stream = client.chat.completions.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": _system_text()},
                {"role": "user", "content": user_msg},
            ],
            tools=[openai_tool],
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

        buf = ""            # accumulated tool-call arguments (preferred payload)
        text_buf = ""       # accumulated assistant text (fallback payload)
        reasoning_chars = 0  # size of the model's thinking (shown, never stored)
        finish_reason = None
        usage_obj = None
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if not delta:
                continue
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        buf += tc.function.arguments
            if getattr(delta, "content", None):
                text_buf += delta.content
            # Reasoning streams separately (delta.reasoning / .reasoning_content).
            # It is the model's private thinking, so it must never enter the
            # report — but its size is reported as a "thinking" phase so a long
            # think doesn't look like a hang at 0 characters.
            rdelta = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if isinstance(rdelta, str):
                reasoning_chars += len(rdelta)
            if on_progress:
                written = len(buf) + len(text_buf)
                if written:
                    on_progress(written, "writing")
                elif reasoning_chars:
                    on_progress(reasoning_chars, "thinking")
    except openai.AuthenticationError as exc:
        raise AnalystError(
            f"{cfg.provider} rejected the API key. Check {cfg.key_env}."
        ) from exc
    except openai.RateLimitError as exc:
        raise AnalystError(f"Rate limited or out of credit on the {cfg.provider} API.") from exc
    except openai.APIStatusError as exc:
        raise AnalystError(
            f"{cfg.provider} API error {exc.status_code}: {getattr(exc, 'message', exc)}"
        ) from exc
    except openai.APIError as exc:
        if "idle timeout" in str(exc).lower():
            raise AnalystError(
                f"{cfg.provider} dropped the connection with an upstream idle timeout. "
                "A reasoning model thought for too long without streaming any output. "
                "Retry (routing may land on a faster provider), pick a model whose "
                "reasoning streams, or switch to a lighter/non-reasoning model."
            ) from exc
        raise AnalystError(f"{cfg.provider} API error: {exc}") from exc

    if finish_reason == "length":
        raise AnalystError(
            f"The report hit the {cfg.max_tokens:,}-token output limit for model "
            f"'{cfg.model}' and was cut off. Raise LLM_MAX_TOKENS if the model allows "
            "it, or use a model/provider with a larger output limit."
        )

    if not buf.strip() and not text_buf.strip():
        raise AnalystError(
            f"{cfg.provider} returned an empty response (finish_reason={finish_reason}). "
            "Try again, or switch model/provider."
        )

    report_input = _extract_report_input(buf, text_buf)
    if not report_input:
        raise AnalystError(
            f"{cfg.provider} did not return a parseable report (finish_reason="
            f"{finish_reason}). The model replied without usable structured output — "
            "try again, or switch model/provider."
        )

    usage = {
        "input_tokens": getattr(usage_obj, "prompt_tokens", None),
        "output_tokens": getattr(usage_obj, "completion_tokens", None),
        "stop_reason": finish_reason,
        "model": cfg.model,
        "provider": cfg.provider,
    }
    return _finish_report(report_input, usage, finish_reason)
