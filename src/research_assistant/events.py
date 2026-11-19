"""Progress reporting from core to any UI. Core code talks only to an EventSink."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class EventSink(Protocol):
    """Receives progress events from core logic."""

    def step_started(self, step_id: str, name: str, input: Any) -> None: ...

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None: ...

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None: ...

    def token(self, text: str) -> None: ...


class NullSink:
    """Discards all events."""

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        pass

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        pass

    def token(self, text: str) -> None:
        pass


EventKind = Literal["step_started", "step_finished", "tool_result", "token"]


class RecordedEvent(BaseModel):
    kind: EventKind
    data: dict[str, Any] = Field(default_factory=dict)


class RecordingSink:
    """Stores every event in `events`, in order. Useful for tests and A/B logs."""

    def __init__(self) -> None:
        self.events: list[RecordedEvent] = []

    def _add(self, kind: EventKind, **data: Any) -> None:
        self.events.append(RecordedEvent(kind=kind, data=data))

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        self._add("step_started", step_id=step_id, name=name, input=input)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        self._add("step_finished", step_id=step_id, output=output, metadata=metadata or {})

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        self._add("tool_result", name=name, args=args, result=result)

    def token(self, text: str) -> None:
        self._add("token", text=text)

    def kinds(self) -> list[EventKind]:
        return [e.kind for e in self.events]

    def text(self) -> str:
        """Concatenate all streamed tokens."""
        return "".join(e.data["text"] for e in self.events if e.kind == "token")
