# openoutreach/linkedin/browser/launch.py
"""Persist + orchestrate the daemon's LinkedIn browser session.

Cookie persistence (to the Django DB) and the launch/login orchestration are
OpenOutreach concerns, so they live here. The reusable *mechanics* — launching a
stealthed browser, driving the login form, clearing checkpoints — stay in the
Django-free ``linkedin_cli.browser`` library and are called from here.
"""
from __future__ import annotations

import logging

from termcolor import colored

from linkedin_cli.auth import authenticate
from linkedin_cli.browser.login import dismiss_comply_gate, launch_browser
from linkedin_cli.exceptions import CheckpointChallengeError

logger = logging.getLogger(__name__)

LINKEDIN_FEED_URL = "https://www.linkedin.com/feed/"

# URL fragments that mean the saved cookies are no longer authenticated. A valid
# logged-in session can legitimately land on /feed, /messaging, /mynetwork, a
# notification, etc. — so we validate by the ABSENCE of these logged-out markers
# rather than requiring an exact /feed match (LinkedIn often redirects a restored
# session to the last-viewed surface, e.g. a messaging thread).
_LOGGED_OUT_MARKERS = ("/login", "/authwall", "/uas/login", "/signup")


def _save_cookies(session):
    """Persist Playwright storage state (cookies) to the DB."""
    state = session.context.storage_state()
    session.linkedin_profile.cookie_data = state
    session.linkedin_profile.save(update_fields=["cookie_data"])


def _resume_saved_session(session) -> bool:
    """Try to resume a cookie-restored session. Return True if logged in.

    Navigates to the feed and accepts any authenticated landing page. Raises
    ``CheckpointChallengeError`` if LinkedIn interrupts with a challenge; returns
    False (so the caller re-authenticates from scratch) only when the cookies are
    genuinely expired — i.e. we get bounced to a login/authwall page.
    """
    page = session.page
    page.goto(LINKEDIN_FEED_URL)
    dismiss_comply_gate(page)
    session.wait()

    url = page.url
    if "/checkpoint/" in url:
        raise CheckpointChallengeError(url)
    if any(marker in url for marker in _LOGGED_OUT_MARKERS):
        logger.info("Saved session expired for %s — re-authenticating", session)
        return False

    logger.info(colored("Saved session valid", "green", attrs=["bold"]))
    return True


def start_browser_session(session):
    logger.debug("Configuring browser for %s", session)

    session.linkedin_profile.refresh_from_db(fields=["cookie_data"])
    cookie_data = session.linkedin_profile.cookie_data

    storage_state = cookie_data if cookie_data else None
    if storage_state:
        logger.info("Loading saved session for %s", session)

    session.page, session.context, session.browser, session.playwright = launch_browser(storage_state=storage_state)

    if not storage_state or not _resume_saved_session(session):
        lp = session.linkedin_profile
        authenticate(session, username=lp.linkedin_username, password=lp.linkedin_password)
        _save_cookies(session)
        logger.info(colored("Login successful – session saved", "green", attrs=["bold"]))

    # "domcontentloaded" — "load" waits for every subresource (analytics
    # beacons, lazy media) and on LinkedIn that event may never fire,
    # hanging the daemon for the duration of the browser timeout.
    session.page.wait_for_load_state("domcontentloaded")
    logger.info(colored("Browser ready", "green", attrs=["bold"]))
