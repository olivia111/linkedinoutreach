# tests/test_search.py
"""Operator-supplied search queries + the auto_generate_keywords switch."""
from unittest.mock import MagicMock, patch

import pytest

from openoutreach.core.management.commands.discover import Command
from openoutreach.core.models import Campaign
from openoutreach.linkedin.models import SearchKeyword
from openoutreach.linkedin.pipeline.search import run_search


@pytest.fixture
def campaign(db):
    return Campaign.objects.create(name="c1", product_docs="p", campaign_objective="o")


def _session(campaign):
    session = MagicMock()
    session.campaign = campaign
    return session


# ── run_search respects auto_generate_keywords ───────────────────────


def test_skips_generation_when_off_and_no_keywords(campaign):
    campaign.auto_generate_keywords = False
    campaign.save()
    with patch(
        "openoutreach.linkedin.pipeline.search_keywords.generate_search_keywords"
    ) as gen, patch("linkedin_cli.actions.search.search_people") as search_people:
        result = run_search(_session(campaign))
    assert result is None
    gen.assert_not_called()
    search_people.assert_not_called()


def test_uses_provided_keyword_without_generating(campaign):
    campaign.auto_generate_keywords = False
    campaign.save()
    SearchKeyword.objects.create(campaign=campaign, keyword="cto fintech")
    with patch(
        "openoutreach.linkedin.pipeline.search_keywords.generate_search_keywords"
    ) as gen, patch(
        "linkedin_cli.actions.search.search_people",
        return_value={"profiles": [{"url": "u1"}]},
    ) as search_people, patch(
        "openoutreach.linkedin.db.leads.discover_and_enrich"
    ) as enrich:
        result = run_search(_session(campaign))
    assert result == "cto fintech"
    gen.assert_not_called()
    search_people.assert_called_once()
    enrich.assert_called_once()
    assert SearchKeyword.objects.get(keyword="cto fintech").used is True


def test_generates_when_on_and_no_keywords(campaign):
    # default auto_generate_keywords == True
    with patch(
        "openoutreach.linkedin.pipeline.search_keywords.generate_search_keywords",
        return_value=["k1"],
    ) as gen, patch(
        "linkedin_cli.actions.search.search_people",
        return_value={"profiles": []},
    ), patch("openoutreach.linkedin.db.leads.discover_and_enrich"):
        result = run_search(_session(campaign))
    assert result == "k1"
    gen.assert_called_once()


# ── discover._seed_queries ───────────────────────────────────────────


def test_seed_queries_inserts_and_disables_generation(campaign):
    Command()._seed_queries(
        campaign, {"queries": ["q1", "q2", "q1"], "queries_file": "", "auto_keywords": False},
    )
    campaign.refresh_from_db()
    assert campaign.auto_generate_keywords is False
    kws = set(SearchKeyword.objects.filter(campaign=campaign).values_list("keyword", flat=True))
    assert kws == {"q1", "q2"}


def test_seed_queries_file_and_auto_keywords(campaign, tmp_path):
    qfile = tmp_path / "q.txt"
    qfile.write_text("# comment\nq3\n\n  q4  \n", encoding="utf-8")
    Command()._seed_queries(
        campaign, {"queries": ["q5"], "queries_file": str(qfile), "auto_keywords": True},
    )
    campaign.refresh_from_db()
    assert campaign.auto_generate_keywords is True
    kws = set(SearchKeyword.objects.filter(campaign=campaign).values_list("keyword", flat=True))
    assert kws == {"q5", "q3", "q4"}


def test_seed_queries_noop_when_none(campaign):
    Command()._seed_queries(
        campaign, {"queries": None, "queries_file": "", "auto_keywords": False},
    )
    campaign.refresh_from_db()
    assert campaign.auto_generate_keywords is True  # untouched
    assert SearchKeyword.objects.filter(campaign=campaign).count() == 0
