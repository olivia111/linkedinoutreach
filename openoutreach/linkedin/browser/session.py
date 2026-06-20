# openoutreach/linkedin/browser/session.py
from __future__ import annotations

import logging
import random
import time
from functools import cached_property

from openoutreach.core.conf import MIN_DELAY, MAX_DELAY

logger = logging.getLogger(__name__)

# The main LinkedIn auth cookie
_AUTH_COOKIE_NAME = "li_at"


def random_sleep(min_val, max_val):
    delay = random.uniform(min_val, max_val)
    logger.debug(f"Pause: {delay:.2f}s")
    time.sleep(delay)


class AccountSession:
    def __init__(self, linkedin_profile):
        self.linkedin_profile = linkedin_profile
        self.django_user = linkedin_profile.user

        # Active campaign — set by the daemon before each lane execution
        self.campaign = None

        # Playwright objects – created on first access or after crash
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None

    @cached_property
    def campaigns(self):
        """All campaigns this user belongs to (cached)."""
        from openoutreach.core.models import Campaign
        return list(Campaign.objects.filter(users=self.django_user))

    def ensure_browser(self):
        """Launch or recover browser + login if needed. Call before using .page"""
        from openoutreach.linkedin.browser.launch import start_browser_session

        if not self.page or self.page.is_closed():
            logger.debug("Launching/recovering browser for %s", self)
            start_browser_session(session=self)
        else:
            self._maybe_refresh_cookies()

    @cached_property
    def self_profile(self) -> dict:
        """Authenticated user's profile dict, fetched once per session.

        The dict isn't persisted to DB (we dropped ``Lead.profile_data``),
        so the first access per session triggers a Voyager call via the
        ``linkedin_cli`` self-discovery primitive; the ``cached_property``
        keeps it warm for the rest of the session. CRM-side persistence
        (the disqualified ``self_lead``) is layered on in ``register_self_lead``.
        """
        from linkedin_cli.setup.self_profile import discover_self_profile
        from openoutreach.linkedin.db.leads import register_self_lead

        profile = discover_self_profile(self)
        register_self_lead(self, profile)
        return profile

    @cached_property
    def active_timezone(self) -> str | None:
        """IANA zone for the active-hours window, resolved once per session.

        An explicit ``ACTIVE_TIMEZONE`` in conf wins (operator override);
        otherwise the zone is inferred from the LinkedIn profile country.
        None when neither yields a zone — the scheduler/daemon treat None as
        "no active-hours gating" rather than guessing UTC. Resolving via
        ``self_profile`` means this fires only after login.
        """
        from openoutreach.core.conf import ACTIVE_TIMEZONE
        from openoutreach.core.tz_country import timezone_for_country

        if ACTIVE_TIMEZONE:
            return ACTIVE_TIMEZONE
        return timezone_for_country(self.self_profile.get("country_code"))

    def active_timezone_provenance(self) -> str:
        """Human-readable note on where ``active_timezone`` came from — used in
        the daemon's active-hours log so an inferred (and possibly wrong) zone
        is visible and overridable."""
        from openoutreach.core.conf import ACTIVE_TIMEZONE

        if ACTIVE_TIMEZONE:
            return f"{ACTIVE_TIMEZONE} (configured via ACTIVE_TIMEZONE)"
        tz = self.active_timezone
        country = (self.self_profile.get("country_code") or "?").upper()
        if tz:
            return (
                f"{tz} (inferred from LinkedIn profile country {country}; "
                "override with ACTIVE_TIMEZONE)"
            )
        return "unknown (no profile country and no ACTIVE_TIMEZONE) — not gating"

    def wait(self, min_delay=MIN_DELAY, max_delay=MAX_DELAY):
        random_sleep(min_delay, max_delay)
        self.page.wait_for_load_state("domcontentloaded")

    def reauthenticate(self):
        """Force a fresh login: close browser, clear saved cookies, re-launch."""
        from openoutreach.linkedin.browser.launch import start_browser_session

        logger.warning("Re-authenticating %s — clearing saved session", self)
        self.close()
        self.linkedin_profile.cookie_data = None
        self.linkedin_profile.save(update_fields=["cookie_data"])
        start_browser_session(session=self)

    def _maybe_refresh_cookies(self):
        """Re-login if the li_at auth cookie in the saved DB state is expired."""
        from openoutreach.linkedin.browser.launch import start_browser_session

        self.linkedin_profile.refresh_from_db(fields=["cookie_data"])
        cookie_data = self.linkedin_profile.cookie_data
        if not cookie_data:
            return
        for cookie in cookie_data.get("cookies", []):
            if cookie.get("name") == _AUTH_COOKIE_NAME:
                expires = cookie.get("expires", -1)
                if expires > 0 and expires < time.time():
                    logger.warning("Auth cookie expired for %s — re-authenticating", self)
                    self.close()
                    start_browser_session(session=self)
                return

    def close(self):
        if self.context:
            try:
                self.context.close()
                if self.browser:
                    self.browser.close()
                if self.playwright:
                    self.playwright.stop()
                logger.info("Browser closed gracefully (%s)", self)
            except Exception as e:
                logger.debug("Error closing browser: %s", e)
            finally:
                self.page = self.context = self.browser = self.playwright = None

        logger.info("Account session closed → %s", self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return self.linkedin_profile.linkedin_username
