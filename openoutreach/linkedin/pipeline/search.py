# openoutreach/linkedin/pipeline/search.py
"""Search keyword management and LinkedIn People search."""
from __future__ import annotations

import logging

from django.utils import timezone
from termcolor import colored

logger = logging.getLogger(__name__)


def run_search(session) -> str | None:
    """Use the next search keyword to discover new profiles. Returns keyword or None."""
    from linkedin_cli.actions.search import search_people
    from openoutreach.linkedin.db.leads import discover_and_enrich
    from openoutreach.linkedin.pipeline.search_keywords import generate_search_keywords
    from openoutreach.linkedin.models import SearchKeyword

    campaign = session.campaign

    if not SearchKeyword.objects.filter(campaign=campaign, used=False).exists():
        if not campaign.auto_generate_keywords:
            logger.info(
                colored("▶ search", "magenta", attrs=["bold"])
                + " no unused queries and keyword generation is off for %s — "
                "stopping (supply more via `discover --queries`).", campaign,
            )
            return None
        used = list(
            SearchKeyword.objects.filter(campaign=campaign, used=True)
            .values_list("keyword", flat=True)
        )
        fresh = generate_search_keywords(
            product_docs=campaign.product_docs,
            campaign_objective=campaign.campaign_objective,
            exclude_keywords=used if used else None,
        )

        if not fresh:
            return None

        objs = [SearchKeyword(campaign=campaign, keyword=k) for k in fresh]
        SearchKeyword.objects.bulk_create(objs, ignore_conflicts=True)

    kw = (
        SearchKeyword.objects.filter(campaign=campaign, used=False)
        .order_by("pk")
        .first()
    )
    if not kw:
        return None

    kw.used = True
    kw.used_at = timezone.now()
    kw.save()

    logger.info(colored("\u25b6 search", "magenta", attrs=["bold"]) + " keyword=%r", kw.keyword)
    urls = [p["url"] for p in search_people(session, kw.keyword)["profiles"]]
    discover_and_enrich(session, urls)
    return kw.keyword
