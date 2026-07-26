"""
Integration test for the seam between extraction and report finalisation.

A reasoning model called with tool_choice="auto" may answer with the report as
JSON in its text reply (in a code fence) instead of via a tool call. This checks
that such a reply flows through _extract_report_input -> _finish_report into a
complete, validated report — the exact path the model-agnostic change relies on.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analyst import (  # noqa: E402
    SECTION_KEYS,
    _extract_report_input,
    _finish_report,
)


def _complete_report_dict():
    report = {
        "companyName": "Acme Ltd",
        "ticker": "ACME",
        "sector": "Widgets",
        "recommendation": {
            "rating": "Buy",
            "targetPrice": "Rs 100",
            "currentPrice": "Rs 80",
            "changePct": "25% upside",
            "horizon": "12 months",
        },
    }
    for key in SECTION_KEYS:
        report[key] = f"<p>{key} body</p>"
    return report


def test_text_mode_reply_produces_complete_report():
    payload = _complete_report_dict()
    # Model answered in prose with the JSON wrapped in a ```json fence.
    text = "Here is the analysis:\n```json\n" + json.dumps(payload) + "\n```"

    report_input = _extract_report_input(None, text)
    report = _finish_report(report_input, {"output_tokens": 1234}, "stop")

    assert set(report["sections"]) == set(SECTION_KEYS)
    assert all(report["sections"][k].strip() for k in SECTION_KEYS)
    assert report["recommendation"]["rating"] == "Buy"
    assert report["_usage"]["output_tokens"] == 1234


def test_tool_args_mode_reply_produces_complete_report():
    payload = _complete_report_dict()
    # Model called the tool: JSON arrives as function-call arguments.
    report_input = _extract_report_input(json.dumps(payload), "")
    report = _finish_report(report_input, {"output_tokens": 42}, "tool_calls")

    assert set(report["sections"]) == set(SECTION_KEYS)
    assert report["recommendation"]["targetPrice"] == "Rs 100"
