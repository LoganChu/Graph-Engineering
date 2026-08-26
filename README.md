# Memory Arena

**Seven memory architectures x three orchestration styles. One task suite.
Measured against each other.**

Every memory library claims to beat "naive RAG." Every agent framework claims
its control flow is the one that matters. Almost none publish what either costs
to find out. This harness holds the prompts, the model, and the tasks fixed,
varies exactly one thing at a time, and reports accuracy *and* the token spend
required to get it — split by whether the tokens were burned writing to the
store, reading from it, or deciding what to read next.

Two axes, deliberately orthogonal:

| Axis | Question | Values |
|---|---|---|
| **Memory** | What is worth remembering, and in what form? | 7 backends |
| **Orchestration** | Who decides when the agent has looked hard enough? | `single_shot`, `loop`, `graph` |

The interesting result is not which backend wins. It is that they win on
different probe types, that the expensive ones don't always win at all, and that
the orchestration you wrap around them can matter more than the store itself —
or cost 3x and buy nothing.

---

## The interface

Every architecture in the arena implements two methods. The whole design space
lives inside them:

```python
class Memory(ABC):
    def observe(self, event: Event) -> None:
        """WRITE policy: what to store, in what form, at what cost."""

    def recall(self, query: str, as_of: date | None = None) -> Recall:
        """READ policy: what earns a place in the context window."""
```

| Backend | What it tests | Write cost |
|---|---|---|
| `full_transcript` | Control — everything in context, no retrieval | free |
| `window_summary` | The default most agents ship | 1 LLM call / 4 turns |
| `bm25` | Lexical baseline. Stubbornly hard to beat | free |
| `vector` | Dense embeddings, the standard RAG answer | free (local model) |
| `entity_graph` | Triples + k-hop walks: does structure beat similarity? | 1 LLM call / turn |
| `temporal_graph` | Bi-temporal facts — memory as *belief revision* | 1 LLM call / turn |
| `agent_notes` | Model authors its own markdown, Claude-Code style | 1 LLM call / 4 turns |

## The second interface

`Memory` decides what gets stored. It says nothing about how hard the agent
works to find it — and that turns out to be a separate, measurable choice.

An orchestrator turns a question plus a store into an answer. There are three,
and they differ in exactly one respect: who owns the decision to search again.

```python
class Orchestrator(ABC):
    def run(self, store: Memory, question: str, as_of: date | None) -> Attempt:
        """CONTROL FLOW: how many times to go back, and with what query."""
```

| Orchestrator | Who owns control flow | Implementation |
|---|---|---|
| `single_shot` | Nobody. One recall, one answer. | ~10 lines, no framework |
| `loop` | **The model.** It calls a `recall` tool until satisfied. | LangChain `create_agent` |
| `graph` | **You.** An explicit retrieve/assess cycle with a declared exit. | LangGraph `StateGraph` |

### Loop engineering vs. graph engineering

These are the two positions people mean by those phrases, and the comparison is
usually framed as LangChain versus LangGraph. That framing is out of date. Since
LangChain v1.0 (October 2025) `AgentExecutor` is deprecated and `create_agent`
compiles to a LangGraph graph — so both orchestrators here run on the *same*
runtime. One library did not win.

What actually differs is where the iteration policy is written down:

```
loop    the cycle exists, but only in the model's judgement
        prompt -> [model decides: search again? answer?] -> ...
        cheap to write, adapts to queries you did not anticipate,
        terminates when the model feels like it

graph   the cycle is the topology
        START -> retrieve -> assess --(enough | out of hops)--> answer -> END
                     ^                        |
                     +------(not enough)------+
        exit condition is a validated field, hop cap is structural,
        every intermediate query is inspectable state
```

Both get the thing single-shot cannot do: a second attempt with better wording
when the store misses on the question's own phrasing. Both pay a model call per
decision to get it. The table the report prints asks whether that trade landed.

To keep the comparison fair rather than flattering:

- **The answer prompt is identical across all three.** `graph` and `single_shot`
  literally call the same `agent.answer()`; `loop` shares the same `RULES` block.
  Anything an orchestrator wins, it wins by assembling better evidence.
- **`graph` assesses and reformulates in one model call**, not two. Splitting
  them reads better on a diagram but would spend 2x per cycle against the loop's
  1x, and this is a cost comparison.
- **Both are capped at `--max-hops` (default 4).** An unbounded loop is not a
  baseline, it is a bill.

### LangChain runs on the harness, not beside it

Pointing `create_agent` at `ChatAnthropic` would have been three lines and would
have broken the benchmark: the framework's tokens would never reach the ledger,
and the response cache would stop replaying. So `ArenaChatModel` is a
`BaseChatModel` over the arena's own client — every LangChain and LangGraph call
is content-addressed, cached, and phase-tagged like any other.

That is also what makes the phase split honest. A model call is billed
`orchestrate` if it came back with tool calls (it decided what to do next) and
`answer` if it came back with prose. The rule is applied identically to both
orchestrators, so their costs are actually comparable.

## The probes

Recall accuracy on questions you can answer by grepping the transcript is not
where memory systems differ. These six categories are:

- **`simple_recall`** — stated once, asked back. The floor.
- **`multi_hop`** — the answer is a join across two turns sharing no vocabulary.
- **`contradiction`** — a later turn overrides an earlier one. Append-only stores
  hand the model both and hope recency wins.
- **`temporal`** — "where was I living in March?" Requires facts to have a
  *lifetime*, not just a timestamp.
- **`negation`** — never stated. The correct answer is "I don't know," and
  confident fabrication is the failure mode being measured.
- **`aggregation`** — count or summarize across many turns. Top-k retrieval
  structurally cannot do this: the answer is not in any k documents.

Probes fire at a declared turn position, so a question can be asked *before* the
contradicting turn arrives — testing that a store handled a change rather than
just that it happens to hold the latest value.

### The corpus is LongMemEval

Earlier versions ran on two hand-authored conversations. They were a harness
demonstration, not a benchmark: a 14-turn haystack with nine probes hung off it
is a needle density no real session history has, and at that size
`full_transcript` fits everything in context while top-k returns half the store.
The memory axis barely separates.

The corpus is now [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(Wu et al., ICLR 2025) — 500 human-curated questions over multi-session chat
histories with per-session timestamps. Its categories map onto ours almost
one-to-one, which is why it is the right fit:

| LongMemEval `question_type` | probe type here |
|---|---|
| `single-session-user` / `-assistant` / `-preference` | `simple_recall` |
| `multi-session` | `multi_hop` |
| `knowledge-update` | `contradiction` |
| `temporal-reasoning` | `temporal` |
| any `question_id` ending `_abs` | `negation` |

Build it (raw JSON goes to the HuggingFace cache, only YAML lands in the tree):

```bash
python scripts/build_longmemeval.py                    # 60 oracle questions
python scripts/build_longmemeval.py --variant s -n 20  # 20 full haystacks
```

Two variants, and the choice is a budget decision. `oracle` ships only the
evidence sessions — a median 23 turns per question. `s` ships ~50 sessions per
question, a median of 492 turns. The graph backends spend one `llm.parse` per
turn ingested, so the full `s` split is **246,930 extraction calls per backend**.
Subset it; `--limit` stratifies across probe types so the by-type table stays
readable.

Three honest caveats:

- **`aggregation` goes unpopulated.** LongMemEval has no equivalent category, and
  reclassifying `multi-session` questions by heuristic would be inventing labels.
  It remains covered only by the hand-authored tasks.
- **`must_not_contain` is lost.** The hard-fail guard needs a distractor string
  and the dataset ships no such annotation, so imported `contradiction` probes
  are graded by the judge alone.
- **`as_of` is lost.** LongMemEval puts the temporal reference in the question
  text rather than a separate field, so there is no date to query the store at.
  Every probe fires at the final turn, which also leaves the `after_turn`
  machinery unexercised.

Those three are exactly what the hand-authored pair still exercises, so it is
kept as an offline smoke test rather than deleted:

```bash
uv run arena run --tasks tasks/handwritten
```

---

## Running it

```bash
uv sync                          # base install
uv sync --extra local            # OpenAI-compatible client (Ollama, LM Studio, llama.cpp)
uv sync --extra embeddings       # adds the vector backend (local sentence-transformers)
uv sync --extra orchestration    # adds the loop and graph orchestrators (LangChain + LangGraph)

uv run arena list                # backends, orchestrators, and tasks
uv run arena doctor              # is the configured provider reachable?
uv run arena run                 # the full matrix
uv run arena inspect             # every probe that wasn't graded correct
```

The orchestration axis is opt-in, because it multiplies the run. Without `-o`
nothing changes: the matrix is backends x tasks under `single_shot`, exactly as
before.

```bash
# One backend, all three orchestration styles -- the loop-vs-graph comparison
uv run arena run -b bm25 -o single_shot -o loop -o graph

# Cross both axes, and give the iterative ones a longer leash
uv run arena run -b bm25 -b temporal_graph -o single_shot -o graph --max-hops 6

# Why did it answer that? `inspect` prints the queries it actually searched for
uv run arena run -b bm25 -o loop
uv run arena inspect -o loop
```

Two providers, same harness:

```bash
# Claude API
export ANTHROPIC_API_KEY=sk-...          # or: ant auth login
uv run arena run -b bm25 -b temporal_graph

# Fully local — no API key, no network
uv run arena run --provider local --model qwen2.5:7b-instruct
```

Outputs land in `results/`: `summary.json`, `runs.json` (every answer with its
grade and reason), `table.md`, and two charts — score by probe type, and the
accuracy-vs-cost scatter that shows which backends are actually on the frontier.

Start with `-b bm25 -b temporal_graph` on a small corpus (`-n 20`) to see the
shape of the result before committing to the full matrix.

---

## Running on local models

Ollama is the default target, but `--provider local` speaks plain OpenAI
chat-completions, so LM Studio, `llama-server`, vLLM, and anything else with
that endpoint work via `--base-url`.

```bash
ollama pull qwen2.5:7b-instruct
uv run arena doctor --provider local --model qwen2.5:7b-instruct
uv run arena run --provider local --model qwen2.5:7b-instruct
```

**Pick for instruction-following, not for benchmark scores.** Two of the seven
backends (`entity_graph`, `temporal_graph`) and the judge all depend on the model
emitting valid JSON against a schema. On ~7B models that is the binding
constraint, not reasoning ability. Qwen2.5-instruct and Llama 3.1 8B hold up;
`deepseek-r1` works but spends most of its output on a visible scratchpad, which
is slow and fragile even though the harness strips `<think>` blocks. Under ~4B,
expect the extraction backends to degrade into a graph full of nothing —
which the report will show as a genuine result, not an error.

The local path defends its own structured output: strip reasoning blocks, scan
for the first balanced JSON object, validate, and on failure make exactly one
repair call. Repairs are counted and reported. A high repair count means your
model is too small for the extraction backends and the numbers below it are
measuring JSON compliance rather than memory.

### The judge problem (read this before quoting any local number)

By default the judge runs on the same model as the agent. With a 3B model on
both sides, **you are mostly measuring the judge**, and the harness prints a
warning saying so. Three ways out, best first:

```bash
# 1. Local agent, API judge -- the write path (the expensive part) stays free
uv run arena run --provider local --model llama3.2:3b \
                 --judge-provider anthropic --judge-model claude-opus-5

# 2. Bigger local judge than the agent under test
uv run arena run --provider local --model llama3.2:3b --judge-model qwen2.5:7b-instruct

# 3. Same model both sides -- fine for smoke-testing the pipeline, not for a result
uv run arena run --provider local --model llama3.2:3b
```

Option 1 is the sweet spot on a laptop. The write path is where token spend
actually accumulates — `entity_graph` makes one extraction call *per turn* — so
running it locally removes almost all the cost while grading stays trustworthy.

### Cost columns go to zero

Local models are unpriced, so the money columns read `$0.00` and the frontier
chart switches its x-axis to **tokens per probe**. That is the right
substitution: on local hardware the scarce resources are context and wall time,
and tokens track both.

---

## Design decisions worth arguing with

**Reproducibility without temperature.** Current Claude models removed
`temperature`/`top_p`/`top_k`, so decoding cannot be pinned. Instead every API
call is content-addressed and cached to `results/cache/`, so regenerating a
report replays identical responses at zero cost. Delete the cache to resample.
Run-to-run variance is real and unmeasured — see limitations.

**Write cost is counted.** A graph backend spends an LLM call per turn extracting
triples; BM25 spends nothing. Measuring only the answer call makes the graph look
free and produces the wrong conclusion. Every call is tagged `write` / `read` /
`answer` / `judge`, and the judge's spend is excluded from the backend's cost.

**The agent is forced closed-book.** Its system prompt permits only the retrieved
excerpt as a source. Without that, the model answers plausible personal questions
from its priors, and a backend that retrieved *nothing* scores as if it had
retrieved correctly — silently inflating every number in the report.

**Control-flow spend is counted, and counted as reading.** A loop that searches
four times makes three extra model calls before it answers a word. Those land in
the ledger as `orchestrate` and are folded into the read-side cost, because that
is what they are: what the agent paid to decide what to look at. Report them
separately from the answer call and every orchestrator looks equally cheap.

**Two grading gates.** Probes declare strings that must not appear (usually the
superseded value). That check is deterministic and runs first, so answering
"Durham" after the move fails regardless of how confident it sounded, and no
judge call is spent. Everything else goes to a rubric judge.

---

## Limitations (read before believing any number)

1. **The judge is the largest error source.** `judge.calibrate()` reports
   agreement against hand-labeled examples. It has not been run yet. Until it
   has, treat differences under ~0.1 as noise. This matters roughly ten times
   more when the judge is a small local model — see the judge problem above.
2. **No repeated trials.** Every cell runs once. No variance estimate, no
   confidence intervals, so small gaps are not meaningful.
3. **Entity linking in the graph backends is string-normalized, not semantic.**
   "my sister" and "Rachel" stay separate nodes unless the extractor resolves
   them. This is a real weakness of the implementation and shows up in the
   multi-hop scores — it is not a property of graph memory in general.
4. **Extraction quality is a confound on small models.** Below ~7B the
   extraction prompt drives the result more than the architecture does. A graph
   backend scoring badly on a 3B model has not shown that graph memory is worse;
   it has shown that the model cannot populate a graph. Compare architectures
   only at a fixed model, and say which model in the caption.

   Concretely, on the first local run (`llama3.2:3b`, `oncall` task) `bm25`
   scored 0.56 while `entity_graph` scored 0.00. Inspecting the store showed the
   graph was the problem, not the idea: relations came out inverted
   (`checkout service --[owns]--> sam`), entities fragmented into separate nodes
   (`checkout` vs `checkout service`, which stops supersession from ever
   firing), and the same fact landed three times with different phrasings.
   **That number is a measurement of llama3.2:3b, not of graph memory.** The
   comparison needs a model that can hold a schema — start at 7B.

   The same run did surface a genuine bug, which is what the harness is for:
   `temporal_graph` matched query terms against fact *subjects* only, so asking
   "who owns checkout?" retrieved nothing because "checkout" appears as an
   object. Fixed by routing retrieval through the k-hop walk; locked in by
   `tests/test_graph_retrieval.py`.
5. **Two hand-written tasks.** Enough to show the method, not enough to rank
   architectures. LoCoMo and LongMemEval are the real suites to port next.
6. **Prices are hardcoded** in `llm.py` from a June 2026 snapshot and drift.
7. **The hop cap is a free parameter, and it moves the result.** Four is a
   guess. `loop` and `graph` are both bounded by it, so the comparison between
   them is fair at any setting, but neither number means much in isolation —
   quote `--max-hops` alongside any orchestration figure.
8. **`loop` needs a model that can actually call tools.** The orchestrator sends
   real tool schemas over whichever transport is configured. Claude handles this;
   so do the larger Ollama models. A small local model that ignores the schema
   will answer without ever searching, which the report shows as `hops` near
   zero. Check that column before concluding the loop was bad at reasoning — it
   may never have run.
9. **The graph's `assess` node is a second judge inside the measurement.** It is
   the same model deciding whether its own evidence is sufficient, and it can be
   wrong in both directions: stopping early on thin evidence, or burning hops
   chasing a fact the store never held. That is a property of this graph, not of
   graph orchestration, and a better exit condition is the obvious next
   experiment.

## Roadmap

- [ ] Run the judge calibration set and publish the agreement number
- [ ] N=5 trials per cell with error bars
- [ ] Port LoCoMo / LongMemEval as additional task suites
- [ ] Semantic entity resolution in `consolidate()`, measured as an ablation
- [ ] A retrieval-quality metric separate from answer accuracy (did the right
      turn make it into context at all?) to separate retrieval failures from
      reasoning failures
- [ ] Sweep `--max-hops` to find where the orchestration lift saturates
- [ ] A non-LLM exit condition for `graph` (retrieval score threshold) as an
      ablation against the model-judged one

## Layout

```
src/arena/
  memory.py       the two-method interface every backend implements
  providers.py    Anthropic + OpenAI-compatible transports, JSON salvage
  llm.py          caching client, phase-tagged ledger, per-phase model routing
  agent.py        the agent under test — one prompt, shared by all three
  judge.py        deterministic guard + rubric judge + calibration
  orchestration.py  the control-flow interface, registry, and hop accounting
  orchestrators/  single_shot (no deps), loop (LangChain), graph (LangGraph),
                  adapter.py — BaseChatModel over the arena's cached client
  runner.py       the (backend x orchestrator x task) matrix
  report.py       aggregation, markdown table, charts
  backends/       one file per architecture
  longmemeval.py  LongMemEval -> task YAML, and the stratified subsetter
tasks/
  longmemeval/    the corpus — generated, gitignored (see below)
  handwritten/    two hand-authored conversations, kept as an offline smoke test
scripts/
  build_longmemeval.py   fetch + convert + stratify
tests/            offline suite — stub LLM, no network
```

```bash
uv run pytest -q                                    # 75 tests, no model needed
uv run pytest --cov=src/arena --cov-report=term-missing
```

The test suite runs against a stub LLM, so it needs neither an API key nor a
local server. It covers the JSON-salvage paths (markdown fences, reasoning
blocks, braces inside strings, the repair round trip) and the graph-retrieval
regressions found by the first real run.

The stub speaks tool calls too, so the loop and the graph run offline against a
scripted model: `tool_hops` sets how many times it asks to search, `assess_hops`
how many times the graph's exit condition says "not yet". That is what lets the
tests assert that the cycle actually cycles, that the hop cap holds against a
model that will not stop, and that control-flow tokens land in the right column
— none of which you can pin down against a real model that might simply choose
not to loop. Tests for `loop` and `graph` skip cleanly without the extra.
