# openoutreach/linkedin/browser/connect.py
"""Send a connection request WITH a personalized note.

``linkedin_cli.actions.connect.send_connection_request`` deliberately sends
note-less invites (fastest/safest). This repo-side wrapper adds the "Add a note"
modal step, reusing the library's Connect-button primitives (``_connect_direct`` /
``_connect_via_more``) so the click-flow stays in one place. Same spirit as
``browser/launch.py`` wrapping the library's login mechanics.

Assumes the profile page is already loaded (caller navigates first, e.g. via
``get_connection_status``).
"""
from __future__ import annotations

import logging

from linkedin_cli.actions.connect import (
    SELECTORS,
    _connect_direct,
    _connect_via_more,
    _check_weekly_invitation_limit,
)

logger = logging.getLogger(__name__)

_ADD_NOTE = 'button[aria-label*="Add a note"], button:has-text("Add a note")'
_NOTE_BOX = (
    'textarea[name="message"], textarea#custom-message, '
    'textarea[id*="custom-message"], textarea'
)
_SEND_WITH_NOTE = (
    'button[aria-label="Send invitation"], button[aria-label*="Send invitation"], '
    'button:has-text("Send"):not(:has-text("without"))'
)


def send_connection_request_with_note(session, profile: dict, note: str) -> bool:
    """Click Connect, attach *note*, and send. Returns True if the note was sent.

    Raises ``ReachedConnectionLimit`` (via ``_check_weekly_invitation_limit``) if
    LinkedIn shows the weekly-invite cap. Returns False when the flow can't be
    completed (no Connect button, no note field, no Send button) — the caller
    decides whether to fall back to a note-less invite.
    """
    public_id = profile.get("public_identifier")

    if not (_connect_direct(session) or _connect_via_more(session)):
        logger.debug("No Connect button for %s", public_id)
        return False

    session.wait()
    page = session.page

    add = page.locator(_ADD_NOTE).first
    if add.count() == 0:
        logger.debug("No 'Add a note' control for %s", public_id)
        return False
    add.click()
    session.wait()

    box = page.locator(_NOTE_BOX).first
    if box.count() == 0:
        logger.debug("No note text box for %s", public_id)
        return False
    box.click()
    box.fill(note)
    session.wait()

    send = page.locator(_SEND_WITH_NOTE).first
    if send.count() == 0:
        logger.debug("No Send button for %s", public_id)
        return False
    send.click(force=True)
    session.wait()

    _check_weekly_invitation_limit(session)
    logger.debug("Connection request WITH note sent to %s", public_id)
    return True
