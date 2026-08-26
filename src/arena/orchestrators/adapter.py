"""A LangChain chat model backed by the arena's own client.

This is the load-bearing piece of the orchestration axis. LangChain and
LangGraph want a `BaseChatModel`; the benchmark needs every token to land in a
phase-tagged `Ledger` and every response to be content-addressed on disk. Point
`create_agent` at `ChatAnthropic` instead and both properties are lost -- the
cache stops replaying, and the framework's spend never shows up in the cost
column that the whole report is built around.

So the framework runs on top of the harness rather than beside it: LangChain
messages are translated into the provider-agnostic transcript that
`providers.chat()` speaks, and the answer is translated back.

The phase decision is made *after* the model responds, not before. A step that
comes back with tool calls decided what to do next and is billed `orchestrate`;
a step that comes back with prose is the answer and is billed `answer`. That
rule is applied identically in the graph orchestrator, which is what makes the
two orchestration styles comparable on cost at all.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from ..providers import Completion
from ..types import Phase


def phase_of(completion: Completion) -> Phase:
    """Control flow or answer? Decided by what the model actually did."""
    return "orchestrate" if completion.wants_tools else "answer"


def to_transcript(messages: Sequence[BaseMessage]) -> tuple[str | None, list[dict]]:
    """LangChain messages -> (system prompt, provider-agnostic transcript)."""
    system_parts: list[str] = []
    transcript: list[dict] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(str(message.content))
        elif isinstance(message, HumanMessage):
            transcript.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, ToolMessage):
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": str(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            transcript.append(
                {
                    "role": "assistant",
                    "content": str(message.content or ""),
                    "tool_calls": [
                        {
                            "id": call.get("id") or f"call_{i}",
                            "name": call["name"],
                            "args": call.get("args", {}),
                        }
                        for i, call in enumerate(message.tool_calls or [])
                    ],
                }
            )
        else:  # pragma: no cover - LangChain has no other concrete message type here
            transcript.append({"role": "user", "content": str(message.content)})

    return ("\n\n".join(system_parts) or None), transcript


def to_tool_specs(bound: Any) -> list[dict] | None:
    """OpenAI-shaped tool schemas (what `bind_tools` stores) -> arena shape."""
    if not bound:
        return None
    specs: list[dict] = []
    for entry in bound:
        fn = entry.get("function", entry) if isinstance(entry, dict) else entry
        specs.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return specs or None


class ArenaChatModel(BaseChatModel):
    """`BaseChatModel` over `arena.llm.LLM`: cached, ledgered, provider-agnostic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: An `arena.llm.LLM`, or anything with the same `.chat/.model/.effort`
    #: surface. Left untyped because pydantic would otherwise reject the stub
    #: client the offline test suite runs the whole matrix on.
    client: Any
    max_tokens: int = 400

    @property
    def _llm_type(self) -> str:
        return "arena"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.client.model, "effort": self.client.effort}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return super().bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, transcript = to_transcript(messages)
        got = self.client.chat(
            transcript,
            phase=phase_of,
            system=system,
            tools=to_tool_specs(kwargs.get("tools")),
            max_tokens=self.max_tokens,
        )
        reply = AIMessage(
            content=got.text,
            tool_calls=[
                {"name": c.name, "args": c.args, "id": c.id} for c in got.tool_calls
            ],
            usage_metadata={
                "input_tokens": got.input_tokens,
                "output_tokens": got.output_tokens,
                "total_tokens": got.input_tokens + got.output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])
