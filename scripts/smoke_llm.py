"""Smoke test: send 'Reply with OK' to the main model and print the answer."""

from __future__ import annotations

import logging
import sys

from langchain_core.messages import BaseMessage

from research_assistant.config import ConfigError, get_settings
from research_assistant.llm import LLMConfigError, get_chat_model

logger = logging.getLogger("smoke_llm")


def main() -> int:
    try:
        settings = get_settings()
        model = get_chat_model("main", settings)
    except (ConfigError, LLMConfigError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        reply = model.invoke("Reply with OK")
    except Exception as exc:  # friendly message, no traceback / secrets
        logger.debug("LLM call failed", exc_info=True)
        print(f"LLM call failed ({type(exc).__name__}). Check key, model name and network.")
        return 1
    text = reply.content if isinstance(reply, BaseMessage) else reply
    print(f"[{settings.models.main}] {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
