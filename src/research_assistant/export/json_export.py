"""JSON export: the full conversation (citations, tool results, traces, strategy, times)."""

from __future__ import annotations

from research_assistant.export.models import Conversation


def export_json(conversation: Conversation) -> bytes:
    """UTF-8 JSON; :func:`load_json` reads it back into an equal :class:`Conversation`."""
    return conversation.model_dump_json(indent=2).encode("utf-8")


def load_json(data: bytes | str) -> Conversation:
    return Conversation.model_validate_json(data)
