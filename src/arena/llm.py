"""Phase-tagged accounting + a disk cache, over a pluggable provider.

Three things this buys the benchmark:

1. Reproducibility. Claude models removed the sampling parameters, so decoding
   cannot be pinned there; local models can pin temperature but still drift
   across server versions. Either way every call is content-addressed and cached
   on disk, so re-running a report replays identical responses for free. Delete
   results/cache/ to resample.

2. Honest cost attribution. A graph backend spends tokens at *write* time
   extracting entities; a vector backend spends almost nothing at write time and
   more at read time. If you only measure the answer call you will conclude the
   wrong thing. Every call is tagged with the phase that caused it.

3. Per-phase model routing. The judge can run on a different (stronger) model
   than the agent under test. On a local setup this is the single most important
   knob: a 3B judge grading a 3B agent measures mostly the judge.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel

from .providers import Completion, Provider, ToolCall, build
from .types import Phase, Usage

# USD per 1M tokens (input, output). Source: Anthropic pricing, cached 2026-06.
# A model absent from this table -- every local model -- is billed at zero.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_LOCAL_MODEL = "qwen2.5:7b-instruct"

T = TypeVar("T", bound=BaseModel)


def cost_usd(u: Usage) -> float:
    """Cost of one call. Local models are free; cache reads bill at 10%."""
    price = PRICES.get(u.model)
    if price is None:
        return 0.0
    inp, out = price
    return (
        u.input_tokens * inp
        + u.cache_read_tokens * inp * 0.10
        + u.cache_write_tokens * inp * 1.25
        + u.output_tokens * out
    ) / 1_000_000


class Ledger:
    """Collects Usage records for one run."""

    def __init__(self) -> None:
        self.records: list[Usage] = []

    def add(self, u: Usage) -> None:
        self.records.append(u)

    def by_phase(self, phase: Phase) -> list[Usage]:
        return [u for u in self.records if u.phase == phase]

    def tokens(self, phase: Phase | None = None) -> tuple[int, int]:
        rs = self.records if phase is None else self.by_phase(phase)
        return sum(r.input_tokens for r in rs), sum(r.output_tokens for r in rs)

    def cost(self, phase: Phase | None = None) -> float:
        rs = self.records if phase is None else self.by_phase(phase)
        return sum(cost_usd(r) for r in rs)

    def live_latency(self, phase: Phase | None = None) -> float:
        """Wall time of calls that actually hit the model (cache hits excluded)."""
        rs = self.records if phase is None else self.by_phase(phase)
        return sum(r.latency_s for r in rs if not r.cached_locally)

    def repairs(self) -> int:
        """Structured-output round trips beyond the first. Local models only."""
        return sum(r.calls - 1 for r in self.records)


class LLM:
    """Caching model client. One per run; the Ledger attributes the spend."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        cache_dir: Path | None = None,
        ledger: Ledger | None = None,
        provider: Provider | str = "anthropic",
        base_url: str | None = None,
        judge_model: str | None = None,
        judge_provider: Provider | str | None = None,
        judge_base_url: str | None = None,
    ) -> None:
        self.model = model
        self.judge_model = judge_model or model
        self.effort = effort
        self.provider = build(provider, base_url) if isinstance(provider, str) else provider
        if judge_provider is None:
            self.judge_provider = self.provider
        elif isinstance(judge_provider, str):
            self.judge_provider = build(judge_provider, judge_base_url or base_url)
        else:
            self.judge_provider = judge_provider
        self.cache_dir = cache_dir or Path("results/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger or Ledger()

    def model_for(self, phase: Phase) -> str:
        return self.judge_model if phase == "judge" else self.model

    def provider_for(self, phase: Phase) -> Provider:
        """Grading may run on an entirely different transport than the agent --
        the point being that you can drive the expensive write path on local
        hardware while still grading with a model you trust."""
        return self.judge_provider if phase == "judge" else self.provider

    # -- cache ---------------------------------------------------------------

    def _key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:32]

    def _load(self, key: str) -> dict | None:
        path = self.cache_dir / (key + ".json")
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _store(self, key: str, value: dict) -> None:
        (self.cache_dir / (key + ".json")).write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )

    def _replay(self, hit: dict, phase: Phase, model: str) -> None:
        self.ledger.add(
            Usage(
                phase=phase,
                model=model,
                input_tokens=hit["usage"]["input_tokens"],
                output_tokens=hit["usage"]["output_tokens"],
                latency_s=hit["usage"].get("latency_s", 0.0),
                calls=hit["usage"].get("calls", 1),
                cached_locally=True,
            )
        )

    # -- calls ---------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        phase: Phase,
        system: str | None = None,
        max_tokens: int = 2000,
    ) -> str:
        model = self.model_for(phase)
        provider = self.provider_for(phase)
        key = self._key(
            {
                "provider": provider.name,
                "model": model,
                "effort": self.effort,
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
            }
        )
        hit = self._load(key)
        if hit is not None:
            self._replay(hit, phase, model)
            return hit["text"]

        started = time.perf_counter()
        got = provider.complete(
            model=model,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            effort=self.effort,
        )
        elapsed = time.perf_counter() - started

        self.ledger.add(
            Usage(
                phase=phase,
                model=model,
                input_tokens=got.input_tokens,
                output_tokens=got.output_tokens,
                latency_s=elapsed,
                calls=got.calls,
            )
        )
        self._store(
            key,
            {
                "text": got.text,
                "usage": {
                    "input_tokens": got.input_tokens,
                    "output_tokens": got.output_tokens,
                    "latency_s": elapsed,
                    "calls": got.calls,
                },
            },
        )
        return got.text

    def chat(
        self,
        messages: list[dict],
        *,
        phase: Phase | Callable[[Completion], Phase],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 1000,
    ) -> Completion:
        """A multi-turn call, optionally with tools. The unit an agent loop runs on.

        `phase` may be a callable, which is how the orchestration axis stays
        honestly accounted: the caller cannot know before the model responds
        whether a step will emit tool calls (control flow, billed
        `orchestrate`) or prose (the answer, billed `answer`). Deciding after
        the fact means a loop and a graph are measured by the same rule.
        """
        model = self.model_for(phase if isinstance(phase, str) else "answer")
        provider = self.provider_for(phase if isinstance(phase, str) else "answer")
        key = self._key(
            {
                "kind": "chat",
                "provider": provider.name,
                "model": model,
                "effort": self.effort,
                "system": system,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
            }
        )

        hit = self._load(key)
        if hit is not None:
            got = Completion(
                text=hit["text"],
                input_tokens=hit["usage"]["input_tokens"],
                output_tokens=hit["usage"]["output_tokens"],
                calls=hit["usage"].get("calls", 1),
                tool_calls=tuple(
                    ToolCall(id=c["id"], name=c["name"], args=c["args"])
                    for c in hit.get("tool_calls", [])
                ),
            )
            resolved = phase if isinstance(phase, str) else phase(got)
            self.ledger.add(
                Usage(
                    phase=resolved,
                    model=model,
                    input_tokens=got.input_tokens,
                    output_tokens=got.output_tokens,
                    latency_s=hit["usage"].get("latency_s", 0.0),
                    calls=got.calls,
                    cached_locally=True,
                )
            )
            return got

        started = time.perf_counter()
        got = provider.chat(
            model=model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            effort=self.effort,
        )
        elapsed = time.perf_counter() - started

        resolved = phase if isinstance(phase, str) else phase(got)
        self.ledger.add(
            Usage(
                phase=resolved,
                model=model,
                input_tokens=got.input_tokens,
                output_tokens=got.output_tokens,
                latency_s=elapsed,
                calls=got.calls,
            )
        )
        self._store(
            key,
            {
                "text": got.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "args": c.args} for c in got.tool_calls
                ],
                "usage": {
                    "input_tokens": got.input_tokens,
                    "output_tokens": got.output_tokens,
                    "latency_s": elapsed,
                    "calls": got.calls,
                },
            },
        )
        return got

    def parse(
        self,
        prompt: str,
        schema: type[T],
        *,
        phase: Phase,
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> T:
        model = self.model_for(phase)
        provider = self.provider_for(phase)
        key = self._key(
            {
                "provider": provider.name,
                "model": model,
                "system": system,
                "prompt": prompt,
                "schema": schema.model_json_schema(),
                "max_tokens": max_tokens,
            }
        )
        hit = self._load(key)
        if hit is not None:
            self._replay(hit, phase, model)
            return schema.model_validate(hit["parsed"])

        started = time.perf_counter()
        parsed, got = provider.parse(
            model=model,
            prompt=prompt,
            system=system,
            schema=schema,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - started

        self.ledger.add(
            Usage(
                phase=phase,
                model=model,
                input_tokens=got.input_tokens,
                output_tokens=got.output_tokens,
                latency_s=elapsed,
                calls=got.calls,
            )
        )
        self._store(
            key,
            {
                "parsed": parsed.model_dump(),
                "usage": {
                    "input_tokens": got.input_tokens,
                    "output_tokens": got.output_tokens,
                    "latency_s": elapsed,
                    "calls": got.calls,
                },
            },
        )
        return parsed  # type: ignore[return-value]


def credentials_present() -> bool:
    """The SDK also resolves an `ant auth login` profile, so a missing env var
    is not proof that there are no credentials."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to build an LLM for one run.

    Kept as one immutable object so the runner signature does not grow a
    parameter every time a provider gains a knob.
    """

    provider: str = "anthropic"
    model: str = DEFAULT_MODEL
    judge_model: str | None = None
    judge_provider: str | None = None
    judge_base_url: str | None = None
    effort: str = "low"
    base_url: str | None = None

    @classmethod
    def local(cls, model: str = DEFAULT_LOCAL_MODEL, **kw) -> "ModelConfig":
        return cls(provider="local", model=model, **kw)

    @property
    def priced(self) -> bool:
        """Whether any spend on this config translates into money."""
        return PRICES.get(self.model) is not None

    def build(self, ledger: Ledger, cache_dir: Path) -> LLM:
        return LLM(
            model=self.model,
            judge_model=self.judge_model,
            judge_provider=self.judge_provider,
            judge_base_url=self.judge_base_url,
            effort=self.effort,
            provider=self.provider,
            base_url=self.base_url,
            cache_dir=cache_dir,
            ledger=ledger,
        )
