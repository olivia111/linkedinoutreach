# tests/test_connect_note.py
"""LLM connect-note generation (openoutreach.linkedin.pipeline.connect_note)."""
from unittest.mock import MagicMock, patch

from openoutreach.linkedin.pipeline import connect_note as cn


def _profile(first="Hill", headline="Founder & CEO at StealthCo"):
    return {"first_name": first, "profile": {"headline": headline}}


def _agent_returning(text):
    """Patch Agent + run_agent_with_backoff so the agent 'returns' `text`."""
    result = MagicMock()
    result.output = cn.ConnectNote(note=text)
    return result


def test_first_name_from_top_level_and_nested():
    assert cn._first_name_from({"first_name": "Amelie"}) == "Amelie"
    assert cn._first_name_from({"profile": {"first_name": "Rami"}}) == "Rami"
    assert cn._first_name_from({}) == ""


def test_generate_returns_llm_note():
    note = "Hi Hill, love your stealth venture. I'm building an AI assistant for founders. 15-min call?"
    with patch("openoutreach.core.llm.get_llm_model", return_value=MagicMock()), \
         patch("pydantic_ai.Agent"), \
         patch("openoutreach.core.llm.run_agent_with_backoff", return_value=_agent_returning(note)):
        out = cn.generate_connect_note(_profile(), "product", "objective")
    assert out == note


def test_generate_truncates_to_max_chars():
    long = "Hi Hill, " + "x" * 400
    with patch("openoutreach.core.llm.get_llm_model", return_value=MagicMock()), \
         patch("pydantic_ai.Agent"), \
         patch("openoutreach.core.llm.run_agent_with_backoff", return_value=_agent_returning(long)):
        out = cn.generate_connect_note(_profile(), "p", "o", max_chars=200)
    assert len(out) <= 200


def test_generate_falls_back_when_empty():
    with patch("openoutreach.core.llm.get_llm_model", return_value=MagicMock()), \
         patch("pydantic_ai.Agent"), \
         patch("openoutreach.core.llm.run_agent_with_backoff", return_value=_agent_returning("   ")):
        out = cn.generate_connect_note(_profile(first="Gina"), "p", "o")
    assert out.startswith("Hi Gina,")
    assert len(out) <= cn.DEFAULT_NOTE_MAX_CHARS
