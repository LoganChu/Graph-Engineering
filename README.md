# Memory Arena

**Six agent memory architectures. One agent loop. One task suite. Measured against each other.**

Every memory library claims to beat "naive RAG." Almost none publish what it costs
to find out. This harness holds the agent, the prompts, the model, and the tasks
fixed, swaps only the memory backend, and reports accuracy *and* the token spend
required to get it — split by whether the tokens were burned writing to the store
or reading from it.

The interesting result is not which backend wins. It is that they win on
different probe types, and the expensive ones don't always win at all.

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

---

## Running it

```bash
uv sync                          # base install
uv sync --extra embeddings       # adds the vector backend (local sentence-transformers)

export ANTHROPIC_API_KEY=sk-...  # or: ant auth login

uv run arena list                # backends and tasks
uv run arena run                 # the full matrix
uv run arena run -b temporal_graph -b bm25 -t relocation
uv run arena inspect             # every probe that wasn't graded correct
```

Outputs land in `results/`: `summary.json`, `runs.json` (every answer with its
grade and reason), `table.md`, and two charts — score by probe type, and the
accuracy-vs-cost scatter that shows which backends are actually on the frontier.

Start with `-b bm25 -b temporal_graph -t relocation` to see the shape of the
result for a few cents before committing to the full matrix.

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

**Two grading gates.** Probes declare strings that must not appear (usually the
superseded value). That check is deterministic and runs first, so answering
"Durham" after the move fails regardless of how confident it sounded, and no
judge call is spent. Everything else goes to a rubric judge.

---

## Limitations (read before believing any number)

1. **The judge is the largest error source.** `judge.calibrate()` reports
   agreement against hand-labeled examples. It has not been run yet. Until it
   has, treat differences under ~0.1 as noise.
2. **No repeated trials.** Every cell runs once. No variance estimate, no
   confidence intervals, so small gaps are not meaningful.
3. **Entity linking in the graph backends is string-normalized, not semantic.**
   "my sister" and "Rachel" stay separate nodes unless the extractor resolves
   them. This is a real weakness of the implementation and shows up in the
   multi-hop scores — it is not a property of graph memory in general.
4. **Two hand-written tasks.** Enough to show the method, not enough to rank
   architectures. LoCoMo and LongMemEval are the real suites to port next.
5. **Prices are hardcoded** in `llm.py` from a June 2026 snapshot and drift.

## Roadmap

- [ ] Run the judge calibration set and publish the agreement number
- [ ] N=5 trials per cell with error bars
- [ ] Port LoCoMo / LongMemEval as additional task suites
- [ ] Semantic entity resolution in `consolidate()`, measured as an ablation
- [ ] A retrieval-quality metric separate from answer accuracy (did the right
      turn make it into context at all?) to separate retrieval failures from
      reasoning failures

## Layout

```
src/arena/
  memory.py       the two-method interface every backend implements
  llm.py          caching Anthropic client + phase-tagged cost ledger
  agent.py        the agent under test — identical across backends
  judge.py        deterministic guard + rubric judge + calibration
  runner.py       the (backend x task) matrix
  report.py       aggregation, markdown table, charts
  backends/       one file per architecture
tasks/            YAML conversations with positioned probes
tests/            offline suite — stub LLM, no API calls
```

```bash
uv run pytest -q                                    # 19 tests, no API needed
uv run pytest --cov=src/arena --cov-report=term-missing
```
