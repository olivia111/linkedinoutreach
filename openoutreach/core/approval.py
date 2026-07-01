# openoutreach/core/approval.py
"""Human-in-the-loop approval gate for outbound actions.

Every action that leaves the local machine — a LinkedIn connection request, a
follow-up message, a cold email, a newsletter signup, a contacts-store
contribution, a paid email lookup — is routed through ``require_approval`` first.

Behavior is governed by ``conf.REQUIRE_APPROVAL`` (default ``True``):

- ``REQUIRE_APPROVAL`` False  → always approve (original autonomous mode).
- no interactive terminal     → always deny (never act unattended).
- otherwise                   → prompt y/N on the terminal; only an explicit
  ``y``/``yes`` approves.

Denial is non-destructive: the caller skips that single action and the daemon
moves on to the next slot. Nothing leaves the machine without an explicit yes.
"""
from __future__ import annotations

import logging
import sys

from termcolor import colored

from openoutreach.core import conf

logger = logging.getLogger(__name__)


def require_approval(action: str, detail: str = "") -> bool:
    """Return ``True`` if the operator approves *action*, else ``False``.

    *action* is a short label (e.g. ``"LinkedIn connection request"``); *detail*
    is the specific target/content shown to the operator before they decide.

    Reads ``conf.REQUIRE_APPROVAL`` at call time so the toggle can be flipped
    (env, settings, tests) without re-importing this module.
    """
    if not conf.REQUIRE_APPROVAL:
        return True

    if not sys.stdin or not sys.stdin.isatty():
        logger.warning(
            colored("BLOCKED", "red", attrs=["bold"])
            + " — approval required but no interactive terminal; skipping %s%s",
            action, f" ({detail})" if detail else "",
        )
        return False

    print("\n" + colored("APPROVAL NEEDED", "yellow", attrs=["bold"]) + f"  {action}")
    if detail:
        print("  " + detail)
    try:
        answer = input("  Proceed? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = ""

    approved = answer in ("y", "yes")
    logger.info(
        "%s — %s",
        colored("✓ approved", "green") if approved else colored("✗ skipped", "magenta"),
        action,
    )
    return approved
