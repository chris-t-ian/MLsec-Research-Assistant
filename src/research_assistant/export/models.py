"""Conversation model for exports, and its (de)serialization from stored chat messages.

The UI stores each :class:`TurnResult` in the answer message's metadata under
:data:`METADATA_KEY` (see :func:`turn_metadata`). :func:`conversation_from_thread` rebuilds a
:class:`Conversation` from a stored thread (plain dicts as returned by the chat data layer:
``{"id", "name", "createdAt", "userIdentifier", "steps": [...]}``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_assistant.agent.state import TurnResult
from research_assistant.models import Citation

logger = logging.getLogger(__name__)

METADATA_KEY = "research_assistant"
METADATA_VERSION = 1

Role = Literal["user", "assistant"]

_STEP_ROLES: dict[str, Role] = {"user_message": "user", "assistant_message": "assistant"}


class ConversationMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    role: Role
    content: str = ""
    created_at: datetime | None = None
    strategy: str | None = Field(default=None, description="Retrieval strategy of this turn.")
    top_k: int | None = None
    turn: TurnResult | None = Field(
        default=None, description="Full turn (citations, tool results, trace) for answers."
    )

    @property
    def citations(self) -> list[Citation]:
        return list(self.turn.citations) if self.turn else []

    def cited_arxiv_ids(self) -> list[str]:
        """arXiv IDs cited in this message, in citation order, without duplicates."""
        return list(dict.fromkeys(c.arxiv_id for c in self.citations))


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    title: str | None = None
    user: str | None = None
    created_at: datetime | None = None
    exported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    messages: list[ConversationMessage] = Field(default_factory=list)


# --------------------------------------------------------------------------- metadata


def turn_metadata(result: TurnResult, strategy: str | None, top_k: int | None = None) -> dict:
    """JSON-serializable metadata to store with the answer message of one turn."""
    return {
        METADATA_KEY: {
            "version": METADATA_VERSION,
            "strategy": strategy,
            "top_k": top_k,
            "turn_result": result.model_dump(mode="json"),
        }
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_turn_metadata(metadata: Any) -> tuple[TurnResult | None, str | None, int | None]:
    """Inverse of :func:`turn_metadata`; tolerates missing or malformed metadata."""
    entry = _as_dict(_as_dict(metadata).get(METADATA_KEY))
    if not entry:
        return None, None, None
    strategy = entry.get("strategy") if isinstance(entry.get("strategy"), str) else None
    top_k = entry.get("top_k") if isinstance(entry.get("top_k"), int) else None
    turn: TurnResult | None = None
    if entry.get("turn_result") is not None:
        try:
            turn = TurnResult.model_validate(entry["turn_result"])
        except ValidationError:
            logger.warning("Stored turn result does not match the current schema; skipped")
    return turn, strategy, top_k


def _parse_time(value: Any) -> datetime | None:
    """Parse an ISO timestamp; naive values are taken as UTC."""
    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def conversation_from_thread(thread: Mapping[str, Any]) -> Conversation:
    """Build a :class:`Conversation` from a stored thread (user and assistant messages only)."""
    steps: Iterable[Mapping[str, Any]] = thread.get("steps") or []
    messages: list[ConversationMessage] = []
    for step in steps:
        role = _STEP_ROLES.get(str(step.get("type")))
        if role is None or not step.get("id"):
            continue
        content = str(step.get("output") or "")
        turn, strategy, top_k = parse_turn_metadata(step.get("metadata"))
        if role == "assistant" and not content and turn is None:
            continue  # placeholder of a message that was never filled
        messages.append(
            ConversationMessage(
                id=str(step["id"]),
                role=role,
                content=content,
                created_at=_parse_time(step.get("createdAt")),
                strategy=strategy,
                top_k=top_k,
                turn=turn,
            )
        )
    # Stable sort; messages without a timestamp keep their place relative to each other, last.
    messages.sort(key=lambda m: m.created_at or datetime.max.replace(tzinfo=UTC))
    return Conversation(
        id=str(thread.get("id") or "unknown"),
        title=thread.get("name") or None,
        user=thread.get("userIdentifier") or None,
        created_at=_parse_time(thread.get("createdAt")),
        messages=messages,
    )
