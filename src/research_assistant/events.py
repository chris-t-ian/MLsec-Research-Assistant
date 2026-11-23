"""Progress reporting from core to any UI. Core code talks only to an EventSink."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from research_assistant.models import Answer


@runtime_checkable
class EventSink(Protocol):
    """Receives progress events from core logic."""

    def step_started(self, step_id: str, name: str, input: Any) -> None: ...

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None: ...

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None: ...

    def token(self, text: str) -> None: ...

    def answer_ready(self, answer: Answer, verifying: bool) -> None:
        """The generated answer passed the output checks. ``verifying``: its claims are
        checked next, so the final answer may still change (lets a UI show the sources
        while the check runs)."""
        ...


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

    def answer_ready(self, answer: Answer, verifying: bool) -> None:
        pass


class NoTokenSink:
    """Forwards steps and tool results to ``inner`` but drops streamed tokens (used when an
    answer is regenerated and the UI should only show the final text)."""

    def __init__(self, inner: EventSink) -> None:
        self.inner = inner

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        self.inner.step_started(step_id, name, input)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        self.inner.step_finished(step_id, output, metadata)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        self.inner.tool_result(name, args, result)

    def token(self, text: str) -> None:
        pass

    def answer_ready(self, answer: Answer, verifying: bool) -> None:
        self.inner.answer_ready(answer, verifying)


EventKind = Literal["step_started", "step_finished", "tool_result", "token", "answer_ready"]


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

    def answer_ready(self, answer: Answer, verifying: bool) -> None:
        self._add("answer_ready", answer=answer.model_dump(mode="json"), verifying=verifying)

    def kinds(self) -> list[EventKind]:
        return [e.kind for e in self.events]

    def text(self) -> str:
        """Concatenate all streamed tokens."""
        return "".join(e.data["text"] for e in self.events if e.kind == "token")
