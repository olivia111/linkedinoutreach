# tests/test_llm_backoff.py
"""429 rate-limit backoff in openoutreach.core.llm."""
from unittest.mock import patch

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from openoutreach.core import llm


def _429(retry_delay_s=None, message=""):
    body = {"error": {"message": message, "details": []}}
    if retry_delay_s is not None:
        body["error"]["details"].append({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": f"{retry_delay_s}s",
        })
    return ModelHTTPError(status_code=429, model_name="gemini-2.5-flash", body=body)


# ── _retry_delay_from_429 ────────────────────────────────────────────


def test_parses_structured_retry_info():
    assert llm._retry_delay_from_429(_429(retry_delay_s=46)) == 46.0


def test_parses_message_fallback():
    exc = _429(message="You exceeded quota. Please retry in 30.5s.")
    assert llm._retry_delay_from_429(exc) == 30.5


def test_returns_none_when_absent():
    assert llm._retry_delay_from_429(_429()) is None


# ── run_agent_with_backoff ───────────────────────────────────────────


def test_retries_then_succeeds():
    calls = {"n": 0}

    def fake_run_sync(coro):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _429(retry_delay_s=0)
        return "ok"

    with patch.object(llm, "run_agent_sync", side_effect=fake_run_sync), \
         patch.object(llm.time, "sleep") as sleep:
        result = llm.run_agent_with_backoff(lambda: object())

    assert result == "ok"
    assert calls["n"] == 2
    sleep.assert_called_once()  # waited once before the retry


def test_gives_up_after_max_attempts():
    with patch.object(llm, "run_agent_sync", side_effect=lambda c: (_ for _ in ()).throw(_429(retry_delay_s=0))), \
         patch.object(llm.time, "sleep"):
        with pytest.raises(ModelHTTPError):
            llm.run_agent_with_backoff(lambda: object(), max_attempts=3)


def test_non_429_propagates_immediately():
    err = ModelHTTPError(status_code=500, model_name="m", body={})
    calls = {"n": 0}

    def fake(coro):
        calls["n"] += 1
        raise err

    with patch.object(llm, "run_agent_sync", side_effect=fake), \
         patch.object(llm.time, "sleep") as sleep:
        with pytest.raises(ModelHTTPError):
            llm.run_agent_with_backoff(lambda: object())

    assert calls["n"] == 1        # no retry
    sleep.assert_not_called()
