"""Thin wrapper over the Anthropic SDK: phase-tagged accounting + a disk cache.

Two things this buys the benchmark:

1. Reproducibility. Current Claude models removed the sampling parameters
   (temperature / top_p / top_k), so you cannot pin decoding to be
   deterministic. Instead every call is content-addressed and cached on disk, so
   re-running a report replays the exact same responses at zero cost. Delete
   results/cache/ to force a fresh sample.

2. Honest cost attribution. A graph backend spends tokens at *write* time
   extracting entities; a vector backend spends almost nothing at write time and
   more at read time. If you only measure the answer call you will conclude the
   wrong thing. Every call is tagged with the phase that caused it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from .types import Phase, Usage

# USD per 1M tokens (input, output). Source: Anthropic pricing, cached 2026-06.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

DEFAULT_MODEL = "claude-opus-5"

T = TypeVar("T", bound=BaseModel)


def cost_usd(u: Usage) -> float:
    """Cost of one call. Cache reads bill at 10% of the input rate."""
    inp, out = PRICES.get(u.model, PRICES[DEFAULT_MODEL])
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
        """Wall time of calls that actually hit the API (cache hits excluded)."""
        rs = self.records if phase is None else self.by_phase(phase)
        return sum(r.latency_s for r in rs if not r.cached_locally)


class LLM:
    """Caching Anthropic client. One per run; pass a Ledger to attribute spend."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        cache_dir: Path | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.client = anthropic.Anthropic()
        self.cache_dir = cache_dir or Path("results/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger or Ledger()

    # -- cache ---------------------------------------------------------------

    def _key(self, payload: dict[str, Any]) -> str:
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
        path = self.cache_dir / (key + ".json")
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    # -- calls ---------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        phase: Phase,
        system: str | None = None,
        max_tokens: int = 2000,
    ) -> str:
        """Single-shot text completion."""
        payload = {
            "model": self.model,
            "effort": self.effort,
            "system": system,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        key = self._key(payload)
        hit = self._load(key)
        if hit is not None:
            self.ledger.add(
                Usage(
                    phase=phase,
                    model=self.model,
                    input_tokens=hit["usage"]["input_tokens"],
                    output_tokens=hit["usage"]["output_tokens"],
                    latency_s=hit["usage"].get("latency_s", 0.0),
                    cached_locally=True,
                )
            )
            return hit["text"]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system

        started = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        elapsed = time.perf_counter() - started

        if response.stop_reason == "refusal":
            text = "[refused]"
        else:
            text = "".join(b.text for b in response.content if b.type == "text").strip()

        self.ledger.add(
            Usage(
                phase=phase,
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                latency_s=elapsed,
            )
        )
        self._store(
            key,
            {
                "text": text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "latency_s": elapsed,
                },
            },
        )
        return text

    def parse(
        self,
        prompt: str,
        schema: type[T],
        *,
        phase: Phase,
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> T:
        """Structured completion validated against a pydantic model."""
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "schema": schema.model_json_schema(),
            "max_tokens": max_tokens,
        }
        key = self._key(payload)
        hit = self._load(key)
        if hit is not None:
            self.ledger.add(
                Usage(
                    phase=phase,
                    model=self.model,
                    input_tokens=hit["usage"]["input_tokens"],
                    output_tokens=hit["usage"]["output_tokens"],
                    cached_locally=True,
                )
            )
            return schema.model_validate(hit["parsed"])

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_format": schema,
        }
        if system:
            kwargs["system"] = system

        started = time.perf_counter()
        response = self.client.messages.parse(**kwargs)
        elapsed = time.perf_counter() - started
        parsed = response.parsed_output

        self.ledger.add(
            Usage(
                phase=phase,
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=elapsed,
            )
        )
        self._store(
            key,
            {
                "parsed": parsed.model_dump(),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "latency_s": elapsed,
                },
            },
        )
        return parsed


def credentials_present() -> bool:
    """The SDK also resolves an `ant auth login` profile, so a missing env var
    is not proof that there are no credentials."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()
