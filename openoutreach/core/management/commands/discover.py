# openoutreach/core/management/commands/discover.py
"""Read-only discovery: search + qualify profiles into a vetted lead list.

Runs the exact search -> enrich -> qualify -> rank pipeline the daemon uses, but
stops before any outreach: it never sends a connection request, message, or
email, and never calls the paid email finder. The result is a campaign-scoped
list of QUALIFIED leads (LinkedIn profile + the LLM's fit reason), written to CSV.

Emails are not available at discovery time — LinkedIn only exposes a 1st-degree
address once a connection is accepted (harvested by the daemon's approval-gated
connect flow). Re-run with ``--export-only`` after some connections land to
refresh the CSV with whatever emails have been captured.

  python manage.py discover --count 50 --out leads.csv
  python manage.py discover --export-only --out leads.csv

Note: discovery still logs in to LinkedIn (the People search uses the
authenticated Voyager API) and views profiles, which may be visible to those
users. No connection requests, messages, or emails are sent.
"""
from __future__ import annotations

import csv
import logging
import sys

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Read-only discovery -> qualified lead list (CSV). "
        "Sends no connections, messages, or emails."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--count", type=int, default=25,
            help="Target number of qualified leads to accumulate (default 25).",
        )
        parser.add_argument(
            "--out", default="leads.csv",
            help="CSV output path (default leads.csv).",
        )
        parser.add_argument(
            "--campaign", default="",
            help="Campaign name (default: first non-freemium campaign).",
        )
        parser.add_argument(
            "--queries", nargs="+", default=None, metavar="QUERY",
            help="LinkedIn People search queries to use directly. Supplying "
                 "queries turns OFF LLM keyword generation for the campaign "
                 "(use --auto-keywords to keep it on).",
        )
        parser.add_argument(
            "--queries-file", default="",
            help="Path to a file of People search queries, one per line "
                 "(blank lines and lines starting with # are ignored). "
                 "Combined with --queries; also turns off LLM generation.",
        )
        parser.add_argument(
            "--auto-keywords", action="store_true",
            help="Keep LLM keyword generation on even when --queries are given "
                 "(generated keywords run after yours are exhausted).",
        )
        parser.add_argument(
            "--export-only", action="store_true",
            help="Skip discovery; just export the current qualified leads.",
        )
        parser.add_argument(
            "--include-rejected", action="store_true",
            help="Also write disqualified/rejected profiles to the CSV, with a "
                 "`qualified` column and the rejection reason.",
        )

    def handle(self, *args, **opts):
        from openoutreach.core.logging import configure_logging

        configure_logging(level=logging.INFO)
        campaign = self._resolve_campaign(opts["campaign"])

        if not opts["export_only"]:
            self._seed_queries(campaign, opts)
            session = self._build_session(campaign)
            try:
                self._discover(session, campaign, opts["count"])
            finally:
                session.close()

        n = self._export(campaign, opts["out"], include_rejected=opts["include_rejected"])
        label = "leads (incl. rejected)" if opts["include_rejected"] else "qualified leads"
        self.stdout.write(self.style.SUCCESS(f"Exported {n} {label} to {opts['out']}"))

    # -- helpers ---------------------------------------------------------

    def _resolve_campaign(self, name):
        from openoutreach.core.models import Campaign

        qs = Campaign.objects.all()
        if name:
            campaign = qs.filter(name=name).first()
            if campaign is None:
                self.stderr.write(f"Campaign {name!r} not found.")
                sys.exit(1)
            return campaign
        campaign = qs.filter(is_freemium=False).first() or qs.first()
        if campaign is None:
            self.stderr.write("No campaigns exist. Onboard a campaign first.")
            sys.exit(1)
        return campaign

    def _seed_queries(self, campaign, opts):
        """Insert operator-supplied People search queries and, unless
        --auto-keywords is set, turn off LLM keyword generation for the campaign.

        No queries supplied → leave the campaign's keyword settings untouched
        (default: LLM generation from product_docs + campaign_objective)."""
        from openoutreach.linkedin.models import SearchKeyword

        queries = list(opts["queries"] or [])
        if opts["queries_file"]:
            try:
                with open(opts["queries_file"], encoding="utf-8") as f:
                    queries += [
                        line.strip() for line in f
                        if line.strip() and not line.lstrip().startswith("#")
                    ]
            except OSError as exc:
                self.stderr.write(f"Could not read --queries-file: {exc}")
                sys.exit(1)

        # De-dupe while preserving order.
        seen, deduped = set(), []
        for q in queries:
            if q not in seen:
                seen.add(q)
                deduped.append(q)

        if not deduped:
            return

        existing = set(
            SearchKeyword.objects.filter(campaign=campaign, keyword__in=deduped)
            .values_list("keyword", flat=True)
        )
        new = [SearchKeyword(campaign=campaign, keyword=q) for q in deduped if q not in existing]
        SearchKeyword.objects.bulk_create(new, ignore_conflicts=True)

        campaign.auto_generate_keywords = bool(opts["auto_keywords"])
        campaign.save(update_fields=["auto_generate_keywords"])

        logger.info(
            "Seeded %d new search query(ies) (%d already present); LLM keyword "
            "generation is now %s for %s.",
            len(new), len(existing),
            "ON" if campaign.auto_generate_keywords else "OFF", campaign,
        )

    def _build_session(self, campaign):
        from openoutreach.core.models import SiteConfig
        from openoutreach.linkedin.browser.registry import (
            get_first_active_profile, get_or_create_session,
        )

        if not SiteConfig.load().llm_api_key:
            self.stderr.write("LLM_API_KEY is not set — wire the LLM first.")
            sys.exit(1)
        profile = get_first_active_profile()
        if profile is None:
            self.stderr.write(
                "No active LinkedIn profile. Add credentials with active=True first."
            )
            sys.exit(1)
        session = get_or_create_session(profile)
        session.campaign = campaign
        return session

    def _qualified_qs(self, campaign):
        """Deals that passed qualification (everything that isn't a FAILED/rejected
        deal or a permanently disqualified lead)."""
        from openoutreach.crm.models import Deal, DealState

        return (
            Deal.objects.filter(campaign=campaign, lead__disqualified=False)
            .exclude(state=DealState.FAILED)
        )

    def _discover(self, session, campaign, target):
        from openoutreach.core.conf import CAMPAIGN_CONFIG
        from openoutreach.crm.models import Lead
        from openoutreach.linkedin.ml.qualifier import BayesianQualifier
        from openoutreach.linkedin.pipeline.pools import qualify_source

        qualifier = BayesianQualifier(
            seed=42,
            n_mc_samples=CAMPAIGN_CONFIG["qualification_n_mc_samples"],
            campaign=campaign,
        )
        X, y = Lead.get_labeled_arrays(campaign)
        if len(X) > 0:
            qualifier.warm_start(X, y)
            logger.info("Warm-started on %d labelled samples", len(y))

        have = self._qualified_qs(campaign).count()
        logger.info("Discovery for %s — %d qualified so far, target %d", campaign, have, target)

        gen = qualify_source(session, qualifier)
        while self._qualified_qs(campaign).count() < target:
            if next(gen, None) is None:
                logger.info("Pipeline exhausted before reaching target.")
                break
            logger.info("Qualified %d / %d", self._qualified_qs(campaign).count(), target)

    def _export(self, campaign, out, *, include_rejected=False):
        from openoutreach.crm.models import Deal, DealState

        if include_rejected:
            deals = (
                Deal.objects.filter(campaign=campaign)
                .select_related("lead").order_by("-update_date")
            )
        else:
            deals = self._qualified_qs(campaign).select_related("lead").order_by("-update_date")

        rows = 0
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "qualified", "public_identifier", "linkedin_url", "country_code",
                "state", "outcome", "email", "reason",
            ])
            for deal in deals:
                lead = deal.lead
                email = lead.api_email or _overlay_email(lead.contact_info) or ""
                qualified = deal.state != DealState.FAILED and not lead.disqualified
                writer.writerow([
                    qualified, lead.public_identifier, lead.linkedin_url,
                    lead.country_code, deal.state, deal.outcome, email, deal.reason,
                ])
                rows += 1
        return rows


def _overlay_email(contact_info) -> str:
    """Best-effort email from the raw LinkedIn contact-info overlay JSON."""
    if not contact_info or not isinstance(contact_info, dict):
        return ""
    value = contact_info.get("emails") or contact_info.get("email") or ""
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""
