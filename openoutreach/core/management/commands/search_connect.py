# openoutreach/core/management/commands/search_connect.py
"""Search LinkedIn People for a query, then send a personalized, note-bearing
connection request to each result.

Reuses the existing machinery: ``search_people`` (discovery), ``Lead.get_profile``
(scrape), ``pipeline.connect_note.generate_connect_note`` (LLM note, like the
qualify/keyword agents), ``browser.connect.send_connection_request_with_note``
(the note-bearing send), the ``ActionLog`` daily-limit guard, and the
``core.approval`` human-in-the-loop gate.

  python manage.py search_connect --query "CEO Healthcare Startup" --max 8

Every send is gated (y/N unless REQUIRE_APPROVAL=0), skips already
connected/pending, respects the account's daily connect limit, paces between
sends, and stops on a weekly-invite cap or a checkpoint.
"""
from __future__ import annotations

import logging
import random
import sys
import time

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)

# Fixed note used by default; {first_name} (or XXX) is filled from the profile.
DEFAULT_NOTE_TEMPLATE = (
    "Hi {first_name}, your background and new venture really caught my eye. "
    "I'm building a personal assistant for CEOs and would love to learn more "
    "about your daily pain points. Would you be open to a quick 15-min call?"
)


class Command(BaseCommand):
    help = "Search People for a query and send note-bearing connection requests."

    def add_arguments(self, parser):
        parser.add_argument("--query", required=True, help="LinkedIn People search query.")
        parser.add_argument("--max", type=int, default=10, help="Max results to contact (default 10).")
        parser.add_argument("--campaign", default="", help="Campaign name (default: first non-freemium).")
        parser.add_argument("--note-max", type=int, default=300,
                            help="Note char cap (200 Basic / 300 Premium; default 300).")
        parser.add_argument("--note-template", default=DEFAULT_NOTE_TEMPLATE,
                            help="Fixed note; use {first_name} or XXX as the name placeholder. "
                                 "Ignored when --personalize is set.")
        parser.add_argument("--personalize", action="store_true",
                            help="Generate a per-person note with the LLM instead of the fixed template.")
        parser.add_argument("--min-pause", type=float, default=45.0)
        parser.add_argument("--max-pause", type=float, default=90.0)

    def handle(self, *args, **opts):
        from openoutreach.core.logging import configure_logging
        configure_logging(level=logging.INFO)

        campaign = self._resolve_campaign(opts["campaign"])
        session = self._build_session(campaign)
        try:
            self._run(session, campaign, opts)
        finally:
            session.close()

    # -- setup (mirrors discover.py) ------------------------------------

    def _resolve_campaign(self, name):
        from openoutreach.core.models import Campaign
        qs = Campaign.objects.all()
        if name:
            c = qs.filter(name=name).first()
            if c is None:
                self.stderr.write(f"Campaign {name!r} not found."); sys.exit(1)
            return c
        c = qs.filter(is_freemium=False).first() or qs.first()
        if c is None:
            self.stderr.write("No campaigns exist. Onboard a campaign first."); sys.exit(1)
        return c

    def _build_session(self, campaign):
        from openoutreach.core.models import SiteConfig
        from openoutreach.linkedin.browser.registry import (
            get_first_active_profile, get_or_create_session,
        )
        if not SiteConfig.load().llm_api_key:
            self.stderr.write("LLM_API_KEY is not set — wire the LLM first."); sys.exit(1)
        profile = get_first_active_profile()
        if profile is None:
            self.stderr.write("No active LinkedIn profile (need active=True)."); sys.exit(1)
        session = get_or_create_session(profile)
        session.campaign = campaign
        return session

    # -- main loop ------------------------------------------------------

    def _run(self, session, campaign, opts):
        from linkedin_cli.actions.search import search_people
        from linkedin_cli.actions.status import get_connection_status
        from linkedin_cli.exceptions import (
            AuthenticationError, CheckpointChallengeError, ProfileInaccessibleError,
            ReachedConnectionLimit, SkipProfile,
        )
        from linkedin_cli.url_utils import url_to_public_id
        from openoutreach.core.approval import require_approval
        from openoutreach.crm.models import DealState, Lead
        from openoutreach.linkedin.browser.connect import send_connection_request_with_note
        from openoutreach.linkedin.models import ActionLog

        query, max_n = opts["query"], opts["max"]
        results = search_people(session, query)["profiles"][:max_n]
        logger.info("Search %r → %d result(s) to process", query, len(results))

        sent = skipped = failed = 0
        for i, prof in enumerate(results):
            url = prof["url"]
            public_id = prof.get("public_identifier") or url_to_public_id(url)
            if not public_id:
                continue

            if not session.linkedin_profile.can_execute(ActionLog.ActionType.CONNECT):
                logger.info("Daily connect limit reached — stopping."); break

            lead, _ = Lead.objects.get_or_create(
                public_identifier=public_id, defaults={"linkedin_url": url},
            )
            if lead.disqualified:
                logger.info("[%d/%d] %s: skip (disqualified)", i + 1, len(results), public_id)
                skipped += 1
                continue

            try:
                profile = lead.get_profile(session)  # live scrape (caches urn/country)
                note = self._build_note(profile, campaign, opts)

                status = DealState(get_connection_status(session, {"public_identifier": public_id, "url": url}).value)
                if status in (DealState.CONNECTED, DealState.PENDING):
                    logger.info("[%d/%d] %s: skip (%s)", i + 1, len(results), public_id, status.value)
                    skipped += 1
                    continue

                if not require_approval("connection request WITH note", f"{public_id}: {note}"):
                    logger.info("[%d/%d] %s: skip (not approved)", i + 1, len(results), public_id)
                    skipped += 1
                    continue

                if send_connection_request_with_note(session, {"public_identifier": public_id, "url": url}, note):
                    session.linkedin_profile.record_action(ActionLog.ActionType.CONNECT, campaign)
                    self._mark_pending(session, lead, campaign, note)
                    logger.info("[%d/%d] %s: SENT ✅ — %s", i + 1, len(results), public_id, note)
                    sent += 1
                else:
                    logger.warning("[%d/%d] %s: FAILED (couldn't attach note)", i + 1, len(results), public_id)
                    failed += 1
            except ReachedConnectionLimit:
                logger.warning("Weekly invitation limit reached — stopping.")
                session.linkedin_profile.mark_exhausted(ActionLog.ActionType.CONNECT)
                break
            except CheckpointChallengeError as e:
                logger.error("Checkpoint challenge — stopping. Clear it in a browser: %s", e.url)
                break
            except AuthenticationError:
                logger.error("Session expired — stopping."); break
            except (ProfileInaccessibleError, SkipProfile) as e:
                logger.warning("[%d/%d] %s: skip (%s)", i + 1, len(results), public_id, e)
                skipped += 1
            except Exception:
                logger.exception("[%d/%d] %s: error", i + 1, len(results), public_id)
                failed += 1

            if i < len(results) - 1:
                time.sleep(random.uniform(opts["min_pause"], opts["max_pause"]))

        logger.info("Done. Sent %d, skipped %d, failed %d.", sent, skipped, failed)
        self.stdout.write(self.style.SUCCESS(f"search_connect: sent {sent}, skipped {skipped}, failed {failed}"))

    def _build_note(self, profile, campaign, opts):
        """Fixed template (default) or LLM-personalized (--personalize)."""
        from openoutreach.linkedin.pipeline.connect_note import _first_name_from
        if opts["personalize"]:
            from openoutreach.linkedin.pipeline.connect_note import generate_connect_note
            return generate_connect_note(
                profile, campaign.product_docs, campaign.campaign_objective,
                max_chars=opts["note_max"],
            )
        first = _first_name_from(profile) or "there"
        note = opts["note_template"].replace("{first_name}", first).replace("XXX", first)
        return note[:opts["note_max"]]

    def _mark_pending(self, session, lead, campaign, note):
        """Record the outbound invite in the CRM: ensure a Deal, set it PENDING."""
        from openoutreach.crm.models import Deal, DealState
        from openoutreach.core.db.deals import set_profile_state

        Deal.objects.get_or_create(
            lead=lead, campaign=campaign,
            defaults={"state": DealState.QUALIFIED, "reason": note},
        )
        set_profile_state(session, lead.public_identifier, DealState.PENDING.value)
