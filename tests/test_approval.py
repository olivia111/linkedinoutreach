# tests/test_approval.py
"""The human-in-the-loop approval gate (openoutreach.core.approval)."""
import pytest

from openoutreach.core import approval, conf

# These tests drive conf.REQUIRE_APPROVAL themselves, so opt out of the
# autouse fixture that disables the gate for the rest of the suite.
pytestmark = pytest.mark.require_approval_gate


@pytest.fixture
def _restore_flag():
    original = conf.REQUIRE_APPROVAL
    yield
    conf.REQUIRE_APPROVAL = original


def test_disabled_always_approves(_restore_flag, monkeypatch):
    conf.REQUIRE_APPROVAL = False
    # Even with no TTY, a disabled gate approves and never prompts.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("should not prompt"))
    assert approval.require_approval("anything", "detail") is True


def test_enabled_no_tty_denies(_restore_flag, monkeypatch):
    conf.REQUIRE_APPROVAL = True
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert approval.require_approval("send", "to alice") is False


@pytest.mark.parametrize("answer,expected", [
    ("y", True), ("Y", True), ("yes", True), ("YES", True), (" y ", True),
    ("n", False), ("no", False), ("", False), ("nope", False), ("x", False),
])
def test_enabled_tty_prompts(_restore_flag, monkeypatch, answer, expected):
    conf.REQUIRE_APPROVAL = True
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    assert approval.require_approval("send", "to alice") is expected


def test_eof_denies(_restore_flag, monkeypatch):
    conf.REQUIRE_APPROVAL = True
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _raise(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    assert approval.require_approval("send") is False
