"""Credential check for the single demo user of the web UI (values from .env)."""

from __future__ import annotations

import hmac
import logging

from research_assistant.config import Settings

logger = logging.getLogger(__name__)


def demo_login_configured(settings: Settings) -> bool:
    return bool(
        settings.demo_username
        and settings.demo_password
        and settings.demo_password.get_secret_value()
    )


def check_demo_credentials(username: str, password: str, settings: Settings) -> bool:
    """True if ``username``/``password`` match DEMO_USERNAME/DEMO_PASSWORD.

    Comparisons run in constant time. Without configured credentials every login fails.
    """
    expected_user = settings.demo_username or ""
    expected_pw = settings.demo_password.get_secret_value() if settings.demo_password else ""
    if not (expected_user and expected_pw):
        logger.warning("Login attempt, but DEMO_USERNAME/DEMO_PASSWORD are not set")
        return False
    user_ok = hmac.compare_digest(username.encode(), expected_user.encode())
    pw_ok = hmac.compare_digest(password.encode(), expected_pw.encode())
    return user_ok and pw_ok
