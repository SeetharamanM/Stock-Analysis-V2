# Equity Analysis Report Generator

A local web app that takes a stock name, pulls **live fundamentals** from Screener.in,
cross-checks the price against Yahoo Finance, and has an LLM write a seven-section equity
research report — saved as **its own standalone HTML file** in `outputs/`.

The LLM provider is switchable in `.env` — **Anthropic (Claude)**, **OpenRouter**, or
**DeepSeek** — with no code changes. See [Choosing a provider](#choosing-a-provider).

Report structure follows `Stock Analysis Prompt.txt`, which the backend reads at run time.
Edit that file and the reports change — no code changes needed.

---

## Quick start

**Double-click `START.bat`.**

It checks for Python, installs any missing packages on first run, verifies your API key,
starts the server and opens the browser. Keep the console window open while you use the
app; closing it (or Ctrl+C) stops the server.

The only thing it can't do for you is create the API key file — see step 2 below.

---

## Manual setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

The key lives **on the server**, never in the browser. Create a file named `.env` next to
this README (copy `.env.example` and fill it in). For the default Anthropic provider:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Get an Anthropic key from <https://console.anthropic.com/settings/keys>.

> You can also set it as an environment variable with `setx ANTHROPIC_API_KEY "…"`. If you
> use `setx`, open a **new** terminal afterwards — it does not affect the current one.

### 3. Run

```bash
python app/server.py
```

Then open <http://127.0.0.1:5057>.

---

## Choosing a provider

The provider is selected entirely in `.env` — **no code changes**. Set `LLM_PROVIDER` and
the matching API key, then restart the server.

| `LLM_PROVIDER` | Key variable | Default model | Base URL |
|---|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | Anthropic |
| `openrouter` | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4` | `https://openrouter.ai/api/v1` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` | `https://api.deepseek.com` |

**Setting the model (your original question):** there is no persistent "default model" stored
on OpenRouter's side — the model is chosen per request. In this app you set it once via
`LLM_MODEL` in `.env`, and every request uses it:

```
# Use DeepSeek's chat model through OpenRouter
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-your-key-here
LLM_MODEL=deepseek/deepseek-chat
```

```
# Or hit DeepSeek's own API directly
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
LLM_MODEL=deepseek-v4-pro   # or deepseek-v4-flash
```

When `LLM_MODEL` is set it **overrides the model dropdown** in the UI (the dropdown is then
shown disabled, reflecting the fixed model). Leave `LLM_MODEL` blank to use the provider's
default. Optional `LLM_MAX_TOKENS` and `LLM_BASE_URL` override the token cap and endpoint.

**How the providers differ under the hood:** Anthropic uses the native Messages API with
tool use and prompt caching. OpenRouter and DeepSeek are OpenAI-compatible and are called
through the `openai` SDK; the same report structure is requested as a forced function call.

> **DeepSeek + `LLM_MAX_TOKENS`:** DeepSeek's v4 models (`deepseek-v4-pro` / `deepseek-v4-flash`)
> support large outputs (flash up to 65,536 tokens, pro more), but a full seven-section report
> runs ~10–14K tokens — so set `LLM_MAX_TOKENS` (e.g. `32000`) or the response can be truncated.
> Note the app disables DeepSeek's default *thinking mode* automatically, because thinking mode
> rejects the forced tool call this app uses to return structured sections.

---

## How it works

```
stock name
    |
    v
Screener.in  ──  company search API resolves name/ticker
    |            company page scraped for P&L, balance sheet, cash flow,
    |            ratios, shareholding, quarterly results, CAGR, peers
    v
Yahoo Finance ── live price, 52-week range, 1-year change
    |
    v
derived metrics ── net margin, operating margin, debt/equity, asset
    |              turnover and CAGRs computed in Python
    v
Claude ────────── writes the report from the supplied figures only
    |
    v
outputs/<company>-equity-analysis-<timestamp>.html
```

**Why the metrics are computed in Python:** the model is given a pre-computed ratio table
and told to use it verbatim. It is explicitly forbidden from using figures recalled from
training data. That removes both stale numbers and arithmetic mistakes — the two things
a pure-LLM report gets wrong most often.

**Charts are also generated in Python**, not by the model. The five-year bar charts
(revenue, net profit, operating margin, cash flow) are rendered from the scraped figures
and appended to section 2. Asking the model to hand-write bar widths was unreliable — it
routinely skipped them — and the exact numbers are already known, so there is no reason
to delegate it.

---

## Layout

| Path | Purpose |
|---|---|
| `app/server.py` | Flask backend, SSE progress streaming |
| `app/screener.py` | Screener.in scraper + derived metrics |
| `app/market.py` | Yahoo Finance live quote |
| `app/analyst.py` | Prompt construction + provider dispatch (Anthropic / OpenRouter / DeepSeek) |
| `app/report.py` | Renders and saves the standalone HTML report |
| `app/static/` | Web UI (`index.html`, `styles.css`) |
| `app/legacy/` | Earlier browser-only version (no backend, no live data) |
| `outputs/` | Generated reports — one file per analysis |
| `Stock Analysis Prompt.txt` | The analyst brief, loaded at run time |

---

## Notes and limitations

- **Screener.in has no public API.** This scrapes the public company page, so a layout
  change on their side can break parsing. Responses are cached for 15 minutes and each
  analysis makes a single page fetch. Keep usage modest and personal.
- **Not every company has a peer group** on Screener (the endpoint 404s, e.g. TCS). The
  report then falls back to a qualitative peer comparison, which is what the brief asks
  for anyway.
- **Banks and NBFCs** report different line items (Revenue not Sales, Deposits not
  Borrowings). The scraper handles this, and the prompt tells the model to substitute
  NIM, GNPA, CASA and Price-to-Book for metrics that do not apply.
- **Fiscal years are Indian.** A Screener column of `Mar 2026` is FY26 (1 Apr 2025 –
  31 Mar 2026).

## Disclaimer

Generated reports are for educational and informational purposes only and are **not
investment advice**. Figures are scraped and model-processed and may be wrong. Verify
everything against company filings and exchange disclosures before making any investment
decision. Consult a SEBI-registered investment adviser before acting.
