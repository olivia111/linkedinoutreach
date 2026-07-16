#!/usr/bin/env python
"""Check/open a LinkedIn session, search People for a query, and send a
note-bearing connection request to each result, using a provided note template.

Three explicit phases:

  1. CHECK / OPEN SESSION  — resolve the first active ``LinkedInProfile``, reuse
     the process-wide cached ``AccountSession`` (``registry.get_or_create_session``),
     and ``ensure_browser()`` to resume saved cookies (or log in if expired).
     Prints who we're logged in as.
  2. SEARCH PEOPLE          — ``search_people(session, query)``.
  3. CONNECT WITH NOTE      — hand off to the ``search_connect`` management
     command, which (per result) scrapes the profile, fills the template's
     ``{first_name}`` / ``XXX`` placeholder, skips already connected/pending,
     honours the daily connect limit + weekly-invite cap, paces between sends,
     and gates every send behind the human-in-the-loop approval prompt.

The session opened in phase 1 is the *same* cached object the command reuses in
phase 3 (the registry keys by profile id), so there is only ever one browser.

Note template (phase 3 fill): plain text with ``{first_name}`` or ``XXX`` as the
recipient's first-name placeholder. Resolution order: ``--template-file`` >
``--template`` > the bundled ``scripts/connect_note.template.txt``. Use
``--personalize`` to have the LLM write a per-person note instead of the template.

Run with the project venv so Django + Playwright are wired:

    # Template-driven, interactive approval on each send (default):
    .venv/Scripts/python scripts/search_connect.py --query "CEO Healthcare Startup" --max 8

    # Provide your own template file:
    .venv/Scripts/python scripts/search_connect.py --query "Head of Growth" \
        --template-file path/to/note.txt --max 5

    # Inline template:
    .venv/Scripts/python scripts/search_connect.py --query "Founder fintech" \
        --template "Hi {first_name}, loved your work — open to a quick chat?"

    # LLM-personalized notes instead of a template:
    .venv/Scripts/python scripts/search_connect.py --query "VP Sales" --personalize

    # Unattended (skip the y/N approval gate) — use with care, real invites:
    .venv/Scripts/python scripts/search_connect.py --query "..." --yes

On POSIX use ``.venv/bin/python`` instead of ``.venv/Scripts/python``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("search_connect_script")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_FILE = PROJECT_ROOT / "scripts" / "connect_note.template.txt"


def setup_django() -> None:
    # scripts/ (not the project root) lands on sys.path when running this file,
    # so ``openoutreach`` isn't importable without adding the root explicitly.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "openoutreach.settings")
    import django

    django.setup()


def resolve_template(opts) -> str:
    """Provided-template resolution: --template-file > --template > bundled file."""
    if opts.template_file:
        return Path(opts.template_file).read_text(encoding="utf-8").strip()
    if opts.template:
        return opts.template
    return DEFAULT_TEMPLATE_FILE.read_text(encoding="utf-8").strip()


def open_session():
    """Phase 1: check/open the session and report the logged-in identity."""
    from openoutreach.linkedin.browser.registry import (
        get_first_active_profile,
        get_or_create_session,
    )

    profile = get_first_active_profile()
    if profile is None:
        logger.error("No active LinkedInProfile (need active=True). Onboard first.")
        raise SystemExit(1)

    session = get_or_create_session(profile)
    logger.info("[1/3] Opening session for %s ...", profile.linkedin_username)
    session.ensure_browser()
    logger.info("[1/3] Landed on: %s", session.page.url)
    try:
        me = session.self_profile
        logger.info(
            "[1/3] Logged in as: %s %s (%s)",
            me.get("first_name"), me.get("last_name"), me.get("public_identifier"),
        )
    except Exception as e:  # best-effort; session may still be usable
        logger.info("[1/3] Session open; self-profile lookup skipped: %s", e)
    return session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open session, search People, send note-bearing connection requests.",
    )
    parser.add_argument("--query", required=True, help="LinkedIn People search query.")
    parser.add_argument("--max", type=int, default=10, help="Max results to contact (default 10).")
    parser.add_argument("--campaign", default="", help="Campaign name (default: first non-freemium).")
    parser.add_argument("--note-max", type=int, default=300,
                        help="Note char cap (200 Basic / 300 Premium; default 300).")

    tmpl = parser.add_mutually_exclusive_group()
    tmpl.add_argument("--template-file", default="",
                      help=f"Path to a note template file (default: {DEFAULT_TEMPLATE_FILE.name}). "
                           "Use {first_name} or XXX as the name placeholder.")
    tmpl.add_argument("--template", default="",
                      help="Inline note template. Use {first_name} or XXX as the name placeholder.")

    parser.add_argument("--personalize", action="store_true",
                        help="Generate a per-person note with the LLM instead of the template.")
    parser.add_argument("--min-pause", type=float, default=45.0, help="Min seconds between sends.")
    parser.add_argument("--max-pause", type=float, default=90.0, help="Max seconds between sends.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the per-send approval prompt (REQUIRE_APPROVAL=0). Real invites — use with care.")
    opts = parser.parse_args()

    setup_django()
    from openoutreach.core.logging import configure_logging
    configure_logging(level=logging.INFO)

    if opts.yes:
        os.environ["REQUIRE_APPROVAL"] = "0"

    template = resolve_template(opts)
    if not opts.personalize:
        logger.info("Using note template: %s", template)

    # Phase 1: check/open session (warms the registry-cached session that the
    # search_connect command reuses in phase 3 — one browser total).
    open_session()

    # Phases 2 + 3: search + note-bearing connect, reusing the vetted command
    # (approval gate, daily/weekly limits, pacing, CRM state) with our template.
    from django.core.management import call_command
    logger.info("[2/3] Searching People for %r; [3/3] sending note-bearing invites ...", opts.query)
    call_command(
        "search_connect",
        query=opts.query,
        max=opts.max,
        campaign=opts.campaign,
        note_max=opts.note_max,
        note_template=template,
        personalize=opts.personalize,
        min_pause=opts.min_pause,
        max_pause=opts.max_pause,
    )


if __name__ == "__main__":
    main()
