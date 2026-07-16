#!/usr/bin/env python
"""Open a LinkedIn browser session using the OpenOutreach stack, and hold it open.

Reuses the daemon's own launch path (``AccountSession.ensure_browser()`` ->
``openoutreach.linkedin.browser.launch.start_browser_session``): it loads the
stored ``LinkedInProfile`` cookies, launches the stealthed browser, resumes the
saved session, and only falls back to a full username/password login when the
cookies are expired. Handy for manual poking, debugging, or warming cookies
without running the full daemon.

Run with the project venv so Django + Playwright are wired:

    .venv/Scripts/python scripts/open_session.py                 # default profile (id=1)
    .venv/Scripts/python scripts/open_session.py --profile 2     # a specific profile
    .venv/Scripts/python scripts/open_session.py --once          # open, report, then close

On POSIX use ``.venv/bin/python`` instead of ``.venv/Scripts/python``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def setup_django() -> None:
    # Running ``scripts/foo.py`` puts scripts/ on sys.path, not the project root,
    # so the ``openoutreach`` package isn't importable without this.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "openoutreach.settings")
    import django

    django.setup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=int,
        default=1,
        help="LinkedInProfile id to open (default: 1).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Open, report status, then close immediately (don't hold the browser open).",
    )
    args = parser.parse_args()

    setup_django()

    from openoutreach.linkedin.browser.session import AccountSession
    from openoutreach.linkedin.models import LinkedInProfile

    profile = LinkedInProfile.objects.get(id=args.profile)
    print(f"[open_session] Opening session for {profile.linkedin_username} ...", flush=True)

    session = AccountSession(profile)
    session.ensure_browser()

    print(f"[open_session] Landed on: {session.page.url}", flush=True)
    try:
        me = session.self_profile
        print(
            f"[open_session] Logged in as: {me.get('first_name')} {me.get('last_name')} "
            f"({me.get('public_identifier')})",
            flush=True,
        )
    except Exception as e:  # self-profile scrape is best-effort; the session may still be fine
        print(f"[open_session] Session open; self-profile lookup skipped: {e}", flush=True)

    if args.once:
        session.close()
        print("[open_session] Closed (--once).", flush=True)
        return

    print("[open_session] Session is OPEN and being held alive. Ctrl-C to close.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        session.close()
        print("[open_session] Closed.", flush=True)


if __name__ == "__main__":
    main()
