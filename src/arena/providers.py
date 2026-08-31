"""Model transports. Two of them, one interface.

`AnthropicProvider` talks to the Claude API. `OpenAICompatProvider` talks to
anything that speaks the OpenAI chat-completions shape -- Ollama, LM Studio,
llama.cpp's server, vLLM, text-generation-webui -- which is every practical way
to run a model locally.

The interesting difference between them is structured output. The Claude API
validates against a schema server-side and retries internally; a local 7B model
hands you a JSON object wrapped in a markdown fence, or preceded by a paragraph
of reasoning, or with a trailing comma. Extraction and the judge both depend on
structured output, so the local path carries its own defenses: strip reasoning
blocks, scan for the first balanced JSON object, validate, and on failure make
exactly one repair call before giving up.

That repair call is counted in the ledger like any other. A local model that
needs two calls to produce one valid verdict costs twice as much wall time as a
model that gets it right first try, and the report should say so.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

# Reasoning models (deepseek-r1 and friends) emit a visible scratchpad. It is
# never part of the answer and always breaks JSON parsing.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool, normalized across transports."""

    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class Completion:
    """One provider call, normalized."""

    text: str
    input_tokens: int
    output_tokens: int
    calls: int = 1  # >1 when a repair round-trip was needed
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        """True when the model asked to act rather than to answer.

        The orchestration ledger keys off this: a response carrying tool calls
        is billed as control flow, a response without them as the answer.
        """
        return bool(self.tool_calls)


class Provider(Protocol):
    name: str

    def complete(
        self, *, model: str, prompt: str, system: str | None, max_tokens: int, effort: str
    ) -> Completion: ...

    def parse(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> tuple[BaseModel, Completion]: ...

    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
        max_tokens: int,
        effort: str,
    ) -> Completion: ...


# -- the normalized message shape --------------------------------------------
#
# A tool loop is a conversation, not a prompt, so `complete()` cannot express
# it. `chat()` takes a provider-agnostic transcript instead:
#
#   {"role": "user",      "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall-shaped dicts]}
#   {"role": "tool",      "tool_call_id": str, "content": str}
#
# and a tool list shaped {"name", "description", "parameters" (JSON Schema)}.
# Keeping this layer in plain dicts is deliberate: the LangChain adapter
# translates into it, so nothing above this file has to import LangChain and
# the base install stays framework-free.


# -- shared JSON salvage ------------------------------------------------------


def strip_reasoning(text: str) -> str:
    return THINK_BLOCK.sub("", text).strip()


def extract_json(text: str) -> str | None:
    """First balanced JSON object in the text, ignoring braces inside strings."""
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def coerce(text: str, schema: type[BaseModel]) -> BaseModel | None:
    blob = extract_json(strip_reasoning(text))
    if blob is None:
        return None
    try:
        return schema.model_validate(json.loads(blob))
    except (json.JSONDecodeError, ValidationError):
        return None


def example_for(node: dict, defs: dict | None = None) -> Any:
    """A filled-in instance of a JSON Schema node, for showing rather than telling."""
    defs = defs if defs is not None else {}
    if "$ref" in node:
        return example_for(defs.get(node["$ref"].rsplit("/", 1)[-1], {}), defs)
    for key in ("anyOf", "oneOf", "allOf"):
        if node.get(key):
            return example_for(node[key][0], defs)
    kind = node.get("type")
    if kind == "object":
        return {
            name: example_for(sub, defs)
            for name, sub in (node.get("properties") or {}).items()
        }
    if kind == "array":
        return [example_for(node.get("items") or {}, defs), "..."]
    if kind == "boolean":
        return True
    if kind in ("integer", "number"):
        return 0
    return "..."


def schema_instruction(schema: type[BaseModel]) -> str:
    """Show the shape; do not paste the schema.

    Pasting the JSON Schema puts a JSON object in the prompt and then asks for a
    JSON object, and smaller models resolve that ambiguity by handing the schema
    straight back: llama3.2:3b returns it verbatim, qwen3.5 buries the real
    answer one level down under `properties`. Measured on qwen3.5 9B, this
    instruction style takes `Rewrites` from 0/3 to 3/3 valid.

    Constrained decoding fixes the same failure at the sampler and is strictly
    better where it applies -- but Ollama does not honour it for every model, so
    this is what defends the models it drops.

    Field descriptions still have to reach the model, since they carry which
    fact belongs in which slot. They come as a plain list rather than as the
    nested blob that caused the problem.
    """
    spec = schema.model_json_schema()
    defs = spec.get("$defs", {})
    lines = [
        "Respond with a single JSON object and nothing else -- no prose, no "
        "markdown fence, no explanation.",
        "",
        "Shape:",
        json.dumps(example_for(spec, defs), indent=2),
    ]
    described = [
        (name, sub.get("description", ""))
        for name, sub in (spec.get("properties") or {}).items()
    ]
    if any(desc for _, desc in described):
        lines += ["", "Fields:"]
        lines += [f"  {name} -- {desc}" for name, desc in described if desc]
    if spec.get("required"):
        lines += ["", f"Required: {', '.join(spec['required'])}"]
    return "\n".join(lines)


# -- Anthropic ----------------------------------------------------------------


class AnthropicProvider:
    """The Claude API. Schema validation happens server-side."""

    name = "anthropic"

    def __init__(self) -> None:
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(
        self, *, model: str, prompt: str, system: str | None, max_tokens: int, effort: str
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": effort},
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            text = "[refused]"
        else:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def parse(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> tuple[BaseModel, Completion]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_format": schema,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.parse(**kwargs)
        return response.parsed_output, Completion(
            text="",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    @staticmethod
    def _blocks(messages: list[dict]) -> list[dict]:
        """Normalized transcript -> Anthropic content blocks.

        Anthropic carries tool results as *user* messages, and consecutive
        results from one parallel round have to share a single message, so this
        cannot be a straight per-message map.
        """
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m.get("content", ""),
                }
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list)                         and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif role == "assistant":
                content: list[dict] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", ()):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc.get("args", {}),
                        }
                    )
                # An assistant turn with neither text nor tool calls is not a
                # legal message; skip it rather than have the API reject the run.
                if content:
                    out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": "user", "content": m.get("content", "")})
        return out

    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
        max_tokens: int,
        effort: str,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._blocks(messages),
            "output_config": {"effort": effort},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

        response = self.client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            return Completion(
                text="[refused]",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        calls = tuple(
            ToolCall(id=b.id, name=b.name, args=dict(b.input or {}))
            for b in response.content
            if b.type == "tool_use"
        )
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=calls,
        )


# -- OpenAI-compatible (Ollama, LM Studio, llama.cpp, vLLM) -------------------

#: Substrings that mark a server *rejecting the concept* of
#: `response_format: json_schema`, rather than rejecting this particular
#: request. Only these downgrade the session to prompt-and-validate. Anything
#: else -- a context overflow, an auth failure, a rate limit -- is re-raised,
#: because quietly answering under a weaker decoding mode is the same class of
#: bug this change exists to remove.
_NO_JSON_SCHEMA = ("json_schema", "response_format", "structured output")


#: The dialect differences `_create` knows how to recover from. Named here so
#: the retry bound and the give-up message cannot drift apart.
_ADAPTATIONS = ("json_schema", "max_completion_tokens", "temperature")


def rejects_max_tokens(exc: Exception) -> bool:
    """Azure's GPT-5 family renamed the field and says so in the error."""
    text = str(exc).lower()
    return "max_tokens" in text and "max_completion_tokens" in text


def rejects_temperature(exc: Exception) -> bool:
    """Reasoning models that allow only the default sampling temperature."""
    text = str(exc).lower()
    return "temperature" in text and (
        "unsupported" in text or "does not support" in text
    )


def rejects_json_schema(exc: Exception) -> bool:
    """Did the server refuse constrained decoding itself, or refuse the call?"""
    text = str(exc).lower()
    return any(marker in text for marker in _NO_JSON_SCHEMA)


#: Env vars checked, in order, for a key to send to an OpenAI-compatible
#: endpoint. A local Ollama needs none, which is why the fallback is a
#: placeholder rather than an error: the server ignores whatever is sent.
API_KEY_VARS = ("ARENA_JUDGE_API_KEY", "ARENA_API_KEY", "OPENAI_API_KEY")


def dotenv_values(path: Path = Path(".env")) -> dict[str, str]:
    """`KEY=value` lines from a .env file. No dependency, and no cleverness.

    Deliberately not python-dotenv: no interpolation, no `export` prefix, no
    multi-line values. `.env.example` has always implied this file is where
    keys live, and until now nothing read it -- which mattered because the
    environment cannot always be relied on to carry them in.
    """
    found: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return found
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        found[key.strip()] = value.strip().strip("'\"")
    return found


def load_dotenv(path: Path = Path(".env")) -> None:
    """Copy .env into the process environment without overriding what is set.

    A real exported variable always wins: the file is the fallback, so a run
    can be steered from the shell without editing it.
    """
    for key, value in dotenv_values(path).items():
        os.environ.setdefault(key, value)


def resolve_api_key() -> str:
    """The key for an OpenAI-compatible endpoint, environment or .env."""
    for source in (os.environ, dotenv_values()):
        for var in API_KEY_VARS:
            value = source.get(var)
            if value:
                return value
    return "not-needed"


REPAIR = """\
Your previous response was not valid JSON for the required schema.

Previous response:
{previous}

{instruction}

Output only the corrected JSON object."""


#: The arena's effort scale mapped onto the OpenAI `reasoning_effort` field.
#:
#: Local reasoning models (qwen3.5, deepseek-r1, ...) think on every call unless
#: told not to. That is not merely slow -- thinking tokens are drawn from the
#: same `max_tokens` budget as the answer, so a 300-token answer cap can be
#: consumed entirely by reasoning and return an EMPTY `content`. The judge
#: scores that as a wrong answer, and a truncation artifact gets reported as a
#: memory failure. Measured on qwen3.5:9b: "reply with exactly: ok" costs 133
#: completion tokens at the default and 2 with reasoning off.
#:
#: So the benchmark default (`--effort low`) turns reasoning off. Raise it only
#: when you are deliberately measuring a thinking model, and raise `max_tokens`
#: with it.
REASONING_EFFORT = {
    "low": "none",
    "medium": "low",
    "high": "medium",
    "xhigh": "high",
    "max": "high",
}


class OpenAICompatProvider:
    """Any OpenAI chat-completions endpoint. Defends its own structured output."""

    name = "local"

    #: Fraction of the server's context window at which a prompt is worth
    #: complaining about. Ollama truncates silently rather than erroring, so
    #: without this the first symptom of overflow is an unexplained score drop.
    WARN_AT = 0.8

    # Class-level defaults so the accounting is safe even on instances built
    # without __init__ (tests do this to stub the HTTP layer).
    context_limit: int = 4096
    peak_prompt_tokens: int = 0
    context_warnings: int = 0
    #: Tri-state: None until a schema call has been tried, then latched. Probing
    #: once per session and remembering it keeps a server that cannot do
    #: constrained decoding from paying a failed round trip on every parse.
    supports_json_schema: bool | None = None
    #: Two more dialect differences this class discovers rather than being
    #: configured with. Azure's GPT-5 family renamed `max_tokens` and refuses
    #: any temperature but its default; Ollama accepts both. Each is latched on
    #: the first rejection, so a session pays one failed round trip rather than
    #: one per call, and no flag has to be threaded through the CLI for it.
    token_param: str = "max_tokens"
    supports_temperature: bool = True

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "not-needed",
        timeout: float = 300.0,
        context_limit: int = 4096,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Local providers need the 'local' extra. Run: uv sync --extra local"
            ) from exc
        # Azure API Management authenticates on an `api-key` header and 401s a
        # bearer token; OpenAI and Ollama want the bearer and ignore unknown
        # headers. Sending both satisfies all three without asking the caller
        # which flavour of endpoint they pointed us at.
        headers = {"api-key": api_key} if api_key and api_key != "not-needed" else None
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            default_headers=headers,
        )
        self.base_url = base_url
        self.context_limit = context_limit
        #: Largest prompt seen, so a run can report its actual headroom.
        self.peak_prompt_tokens = 0
        self.context_warnings = 0

    def _note_prompt_size(self, prompt_tokens: int, model: str) -> None:
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, prompt_tokens)
        if prompt_tokens > self.context_limit * self.WARN_AT:
            self.context_warnings += 1
            if self.context_warnings <= 3:  # say it, do not spam it
                print(
                    f"  [context] {model}: prompt was {prompt_tokens} tokens against "
                    f"a {self.context_limit}-token window. Ollama truncates silently "
                    f"-- raise num_ctx via a Modelfile or results will degrade.",
                    flush=True,
                )

    def _create(self, kwargs: dict[str, Any], *, constrained: bool, json_mode: bool):
        """One request, retried against the dialects servers actually differ on.

        Every branch here is a difference that is *reported by the server* --
        never guessed from a URL or a model name, because those are exactly the
        heuristics that rot. A rejection this does not recognise is re-raised
        untouched: silently answering a weaker question is the failure mode the
        constrained-decoding work exists to remove, and it would be perverse to
        reintroduce it here.
        """
        for _ in range(len(_ADAPTATIONS) + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                if constrained:
                    self.supports_json_schema = True
                return response
            except Exception as exc:
                if constrained and rejects_json_schema(exc):
                    # Older Ollama, LM Studio, llama.cpp: no grammar support.
                    self.supports_json_schema = False
                    constrained = False
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    else:
                        kwargs.pop("response_format", None)
                    continue
                if "max_tokens" in kwargs and rejects_max_tokens(exc):
                    self.token_param = "max_completion_tokens"
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "temperature" in kwargs and rejects_temperature(exc):
                    self.supports_temperature = False
                    kwargs.pop("temperature")
                    continue
                raise
        raise RuntimeError(
            "no request dialect this server accepts: tried "
            f"{', '.join(_ADAPTATIONS)}"
        )

    def _chat(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        effort: str = "low",
        json_mode: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            self.token_param: max_tokens,
            "reasoning_effort": REASONING_EFFORT.get(effort, "none"),
        }
        if self.supports_temperature:
            kwargs["temperature"] = 0  # local models expose this; pin it
        # Constrained decoding, when the server can do it: the schema is
        # compiled to a grammar and the sampler masked to it, so a
        # non-conforming object is unreachable rather than merely discouraged.
        # This is the whole difference between a 3B model being unusable here
        # and being fine. Under plain `json_object` the prompt contains a JSON
        # object (the schema) and asks for a JSON object, and small models
        # resolve that by echoing the schema back -- which used to validate.
        constrained = schema is not None and self.supports_json_schema is not False
        if constrained:
            kwargs["response_format"] = {
                "type": "json_schema",
                # No `strict`: it would require every property in `required` and
                # `additionalProperties: false`, which several of these schemas
                # deliberately violate by giving fields defaults. Ollama
                # constrains on the grammar either way.
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            }
        elif json_mode:
            # Widely supported; servers that do not know it ignore it, which is
            # why the balanced-brace salvage below is not optional.
            kwargs["response_format"] = {"type": "json_object"}

        response = self._create(kwargs, constrained=constrained, json_mode=json_mode)
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        self._note_prompt_size(prompt_tokens, model)
        return Completion(
            text=(response.choices[0].message.content or "").strip(),
            input_tokens=prompt_tokens,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def complete(
        self, *, model: str, prompt: str, system: str | None, max_tokens: int, effort: str
    ) -> Completion:
        got = self._chat(
            model=model,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
        )
        return Completion(
            text=strip_reasoning(got.text),
            input_tokens=got.input_tokens,
            output_tokens=got.output_tokens,
        )

    def parse(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> tuple[BaseModel, Completion]:
        instruction = schema_instruction(schema)
        # The instruction stays even when decoding is constrained: the grammar
        # fixes the shape, but the field descriptions are what tell the model
        # which fact belongs in which slot. Structure and intent are separate.
        first = self._chat(
            model=model,
            prompt=f"{prompt}\n\n{instruction}",
            system=system,
            max_tokens=max_tokens,
            json_mode=True,
            schema=schema,
        )
        parsed = coerce(first.text, schema)
        if parsed is not None:
            return parsed, first

        second = self._chat(
            model=model,
            prompt=REPAIR.format(previous=first.text[:2000], instruction=instruction),
            system=system,
            max_tokens=max_tokens,
            json_mode=True,
            schema=schema,
        )
        billed = Completion(
            text=second.text,
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            calls=2,
        )
        parsed = coerce(second.text, schema)
        if parsed is None:
            raise ValueError(
                f"{model} could not produce valid {schema.__name__} JSON after a "
                f"repair attempt. Last output: {second.text[:300]!r}"
            )
        return parsed, billed


    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
        max_tokens: int,
        effort: str,
    ) -> Completion:
        payload: list[dict[str, Any]] = []
        if system:
            payload.append({"role": "system", "content": system})
        for m in messages:
            role = m["role"]
            if role == "tool":
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m.get("content", ""),
                    }
                )
            elif role == "assistant" and m.get("tool_calls"):
                payload.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc.get("args", {})),
                                },
                            }
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            else:
                payload.append({"role": role, "content": m.get("content", "")})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": payload,
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning_effort": REASONING_EFFORT.get(effort, "none"),
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
                for t in tools
            ]

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = response.usage

        calls: list[ToolCall] = []
        for i, raw in enumerate(getattr(message, "tool_calls", None) or []):
            # Small models routinely emit arguments that are almost-JSON. The
            # same salvage the structured-output path uses applies here; a call
            # whose arguments cannot be recovered is dropped rather than crashed
            # on, which shows up as a wasted hop in the report.
            blob = raw.function.arguments or "{}"
            try:
                args = json.loads(blob)
            except json.JSONDecodeError:
                recovered = extract_json(strip_reasoning(blob))
                try:
                    args = json.loads(recovered) if recovered else None
                except json.JSONDecodeError:
                    args = None
            if args is None:
                continue
            calls.append(
                ToolCall(
                    id=raw.id or f"call_{i}",
                    name=raw.function.name,
                    args=args if isinstance(args, dict) else {"input": args},
                )
            )

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        # The loop/graph orchestrators accumulate tool results across hops, so
        # this is the path most likely to run into the context window.
        self._note_prompt_size(prompt_tokens, model)
        return Completion(
            text=strip_reasoning(message.content or "").strip(),
            input_tokens=prompt_tokens,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            tool_calls=tuple(calls),
        )


def build(
    provider: str, base_url: str | None = None, api_key: str | None = None
) -> Provider:
    if provider == "anthropic":
        return AnthropicProvider()
    if provider in ("local", "ollama", "openai-compat"):
        # Until now this passed no key at all, which made every remote
        # OpenAI-compatible endpoint unreachable no matter what was configured.
        return OpenAICompatProvider(
            base_url=base_url or "http://localhost:11434/v1",
            api_key=api_key or resolve_api_key(),
        )
    raise ValueError(f"unknown provider {provider!r}")
