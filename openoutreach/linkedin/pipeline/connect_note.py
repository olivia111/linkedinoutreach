# openoutreach/linkedin/pipeline/connect_note.py
"""LLM generation of personalized LinkedIn connection-request notes.

Mirrors ``pipeline/search_keywords.py``: render a Jinja prompt, run a pydantic-ai
agent through the shared rate-limit backoff, return validated output. The note is
hard-capped to ``max_chars`` (LinkedIn's note limit — 200 for Basic, 300 Premium).
"""
from __future__ import annotations

import logging

import jinja2
from pydantic import BaseModel, Field

from openoutreach.core.conf import PROMPTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_NOTE_MAX_CHARS = 200


class ConnectNote(BaseModel):
    """Structured LLM output for a connection-request note."""

    note: str = Field(description="The connection-request note, plain text")


def _first_name_from(profile: dict) -> str:
    """Best-effort first name from a Voyager profile dict (top-level or nested)."""
    return (
        (profile.get("first_name") or "")
        or ((profile.get("profile") or {}).get("first_name") or "")
    ).strip()


def generate_connect_note(
    profile: dict,
    product_docs: str,
    campaign_objective: str,
    max_chars: int = DEFAULT_NOTE_MAX_CHARS,
) -> str:
    """Return a personalized connection note (<= ``max_chars``) for *profile*.

    Falls back to a safe template if the LLM output is empty. Never raises for a
    missing name/headline — those just make the note more generic.
    """
    from pydantic_ai import Agent

    from openoutreach.core.llm import get_llm_model, run_agent_with_backoff
    from openoutreach.linkedin.ml.profile_text import build_profile_text

    first_name = _first_name_from(profile) or "there"
    profile_text = build_profile_text(profile).strip()

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    template = env.get_template("connect_note.j2")
    prompt = template.render(
        product_docs=product_docs,
        campaign_objective=campaign_objective,
        first_name=first_name,
        profile_text=profile_text or "(no profile details available)",
        max_chars=max_chars,
    )

    agent = Agent(get_llm_model(), output_type=ConnectNote, model_settings={"temperature": 0.7})
    note = run_agent_with_backoff(lambda: agent.run(prompt)).output.note.strip()

    if not note:
        note = (
            f"Hi {first_name}, your work caught my eye. I'm building a personal AI "
            "assistant for founders and would love to hear your daily pain points. "
            "Open to a quick 15-min call?"
        )
    if len(note) > max_chars:
        logger.debug("Connect note over %d chars — truncating", max_chars)
        note = note[:max_chars].rstrip()
    return note
