"""
Tests for the model-agnostic report extraction helpers in app.analyst.

These cover the variance that thinking/reasoning models introduce once we stop
forcing the tool call: the model may return clean tool arguments, JSON wrapped
in a Markdown code fence, JSON surrounded by prose, or a slightly malformed blob
that only the section salvager can recover. All are pure functions — no network.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analyst import _extract_report_input, _extract_json_object  # noqa: E402


def test_tool_args_clean_json():
    args = '{"overview": "o", "conclusion": "c"}'
    assert _extract_report_input(args, None) == {"overview": "o", "conclusion": "c"}


def test_text_fenced_json_block():
    text = 'Sure, here it is:\n```json\n{"overview": "o", "conclusion": "c"}\n```\n'
    assert _extract_report_input(None, text) == {"overview": "o", "conclusion": "c"}


def test_text_json_with_surrounding_prose():
    text = 'Here is the report: {"overview": "o", "conclusion": "c"} Hope it helps.'
    assert _extract_report_input(None, text) == {"overview": "o", "conclusion": "c"}


def test_malformed_json_is_salvaged():
    # Unescaped inner quotes break json.loads; the section salvager recovers it.
    blob = '{"overview": "a "bad" quote", "conclusion": "the end"}'
    out = _extract_report_input(blob, None)
    assert out.get("overview") == 'a "bad" quote'
    assert out.get("conclusion") == "the end"


def test_empty_inputs_return_empty_dict():
    assert _extract_report_input(None, None) == {}
    assert _extract_report_input("", "   ") == {}


def test_tool_args_preferred_over_text():
    args = '{"overview": "from_tool"}'
    text = '{"overview": "from_text"}'
    assert _extract_report_input(args, text)["overview"] == "from_tool"


def test_text_used_when_no_tool_args():
    text = '{"overview": "from_text"}'
    assert _extract_report_input(None, text)["overview"] == "from_text"


def test_extract_json_object_ignores_non_object_json():
    # A bare JSON array is valid JSON but not a report object.
    assert _extract_json_object('["a", "b"]') == {}


def test_extract_json_object_empty():
    assert _extract_json_object("") == {}
    assert _extract_json_object("   ") == {}
