# tests/test_qualify.py
"""Tests for the qualification logic in qualify module."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from openoutreach.linkedin.pipeline.qualify import run_qualification
from openoutreach.linkedin.ml.qualifier import BayesianQualifier


def _make_trained_qualifier(seed=42):
    qualifier = BayesianQualifier(seed=seed)
    rng = np.random.RandomState(seed)
    for _ in range(5):
        qualifier.update(rng.randn(384).astype(np.float32) + 1.0, 1)
        qualifier.update(rng.randn(384).astype(np.float32) - 1.0, 0)
    return qualifier


def _create_lead_with_embedding(lead_id, public_id):
    from openoutreach.crm.models import Lead
    emb = np.ones(384, dtype=np.float32)
    return Lead.objects.create(
        pk=lead_id,
        public_identifier=public_id,
        linkedin_url=f"https://linkedin.com/in/{public_id}/",
        embedding=emb.tobytes(),
    )


def _fake_leads(lead_id=1, public_id="alice"):
    """Return a list matching get_leads_for_qualification output."""
    return [{"lead_id": lead_id, "public_identifier": public_id, "url": "", "profile": {}}]


def _enable_email_channel():
    """Make email a viable channel: a sendable mailbox + a BetterContact key.

    `_resolve_email` is gated on `has_mailbox()`, and the paid leg on
    `bettercontact.is_configured()` — without both, enrichment is skipped.
    """
    from openoutreach.core.models import SiteConfig
    from openoutreach.emails.models import Mailbox

    cfg = SiteConfig.load()
    cfg.bettercontact_api_key = "k"
    cfg.save()
    Mailbox.objects.create(username="me@acme.com", password="p", from_address="me@acme.com")


class TestQualifyAutoDecisions:
    def test_always_calls_llm(self, db):
        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")) as mock_llm,
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal"),
        ):
            run_qualification(session, qualifier)
            mock_llm.assert_called_once()

    def test_llm_on_cold_start(self, db):
        qualifier = BayesianQualifier(seed=42)
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(0, "Bad fit")) as mock_llm,
            patch.object(qualifier, "update"),
            patch("openoutreach.core.db.deals.create_disqualified_deal"),
        ):
            run_qualification(session, qualifier)
            mock_llm.assert_called_once()

    def test_disqualify_on_promote_failure(self, db):
        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")),
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal",
                  side_effect=ValueError("no company_name")),
            patch("openoutreach.core.db.deals.create_disqualified_deal") as mock_disqualify,
        ):
            run_qualification(session, qualifier)
            mock_disqualify.assert_called_once()

    def test_qualified_lead_is_enriched(self, db):
        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")
        _enable_email_channel()

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")),
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal") as mock_promote,
        ):
            run_qualification(session, qualifier)
            mock_promote.return_value.lead.resolve_api_email.assert_called_once_with()

    def test_finder_hit_routes_to_ready_to_email(self, db):
        """A BetterContact hit (True) routes the Deal QUALIFIED → READY_TO_EMAIL."""
        from openoutreach.crm.models import DealState

        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")
        _enable_email_channel()

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")),
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal") as mock_promote,
            patch("openoutreach.core.db.deals.set_profile_state") as mock_set_state,
            patch("openoutreach.contacts.service.contribute") as mock_contribute,
        ):
            # Model the api_email null→non-null transition the give-back gate keys on.
            lead = mock_promote.return_value.lead
            lead.api_email = None

            def _resolve():
                lead.api_email = "alice@acme.com"
                return True

            lead.resolve_api_email.side_effect = _resolve
            run_qualification(session, qualifier)
            mock_set_state.assert_called_once()
            assert mock_set_state.call_args.args[2] == DealState.READY_TO_EMAIL
            mock_contribute.assert_called_once()  # fresh BetterContact hit → moment-1 give-back
            assert mock_contribute.call_args.args[3] == "bettercontact"  # tagged with the BetterContact origin

    def test_cached_api_email_routes_but_does_not_recontribute(self, db):
        """An already-resolved api_email still routes to READY_TO_EMAIL but is not
        re-sent to the hub — the give-back already happened on the first resolve."""
        from openoutreach.crm.models import DealState

        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")
        _enable_email_channel()

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")),
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal") as mock_promote,
            patch("openoutreach.core.db.deals.set_profile_state") as mock_set_state,
            patch("openoutreach.contacts.service.contribute") as mock_contribute,
        ):
            lead = mock_promote.return_value.lead
            lead.api_email = "alice@acme.com"  # already resolved before this run
            lead.resolve_api_email.return_value = True  # cached hit
            run_qualification(session, qualifier)
            mock_set_state.assert_called_once()
            assert mock_set_state.call_args.args[2] == DealState.READY_TO_EMAIL
            mock_contribute.assert_not_called()

    def test_genuine_email_miss_stays_qualified(self, db):
        """A genuine BetterContact miss (False) leaves the Deal QUALIFIED → the connect funnel."""
        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")
        _enable_email_channel()

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")),
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal") as mock_promote,
            patch("openoutreach.core.db.deals.set_profile_state") as mock_set_state,
        ):
            mock_promote.return_value.lead.resolve_api_email.return_value = False
            run_qualification(session, qualifier)
            mock_set_state.assert_not_called()

    def test_bettercontact_unavailable_leaves_qualified(self, db):
        """BetterContact that couldn't run (None) leaves the Deal QUALIFIED — no parking."""
        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")
        _enable_email_channel()

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")),
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal") as mock_promote,
            patch("openoutreach.core.db.deals.set_profile_state") as mock_set_state,
        ):
            mock_promote.return_value.lead.resolve_api_email.return_value = None
            run_qualification(session, qualifier)
            mock_set_state.assert_not_called()

    def test_no_mailbox_skips_enrichment(self, db):
        """With no mailbox to send from, neither finder runs — the Deal takes the connect leg."""
        qualifier = _make_trained_qualifier()
        session = MagicMock()
        _create_lead_with_embedding(1, "alice")
        # No mailbox created → has_mailbox() is False, even with a BetterContact key set.
        from openoutreach.core.models import SiteConfig
        cfg = SiteConfig.load()
        cfg.bettercontact_api_key = "k"
        cfg.save()

        with (
            patch("openoutreach.linkedin.db.leads.get_leads_for_qualification", return_value=_fake_leads()),
            patch("openoutreach.linkedin.pipeline.qualify._fetch_profile_text", return_value="engineer at acme"),
            patch("openoutreach.linkedin.ml.qualifier.qualify_with_llm", return_value=(1, "Good fit")),
            patch.object(qualifier, "update"),
            patch("openoutreach.linkedin.db.leads.promote_lead_to_deal") as mock_promote,
            patch("openoutreach.core.db.deals.set_profile_state") as mock_set_state,
            patch("openoutreach.contacts.service.resolve") as mock_resolve,
        ):
            run_qualification(session, qualifier)
            mock_resolve.assert_not_called()  # hub lookup skipped — nothing to send
            mock_promote.return_value.lead.resolve_api_email.assert_not_called()  # paid finder skipped
            mock_set_state.assert_not_called()  # stays QUALIFIED for the connect funnel
