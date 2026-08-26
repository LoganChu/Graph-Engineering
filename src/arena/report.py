"""Aggregation and output: summary.json, runs.json, a markdown table, charts.

The scatter is the one that matters. Accuracy alone always ranks the full
transcript first; plotting accuracy against what it *cost* is what shows which
architectures are on the efficient frontier and which are merely expensive.

With a local model the money axis collapses to zero, so the same chart is drawn
against tokens per probe instead. That is the right substitution: on local
hardware the scarce resource is context and time, not dollars, and tokens track
both.

When a run covers more than one orchestrator the report grows a third table and
a third chart, because the question has changed. It is no longer "which memory
architecture wins" but "does letting the agent search again earn back what the
extra model calls cost" -- and that is a per-backend answer, not a global one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .judge import score
from .llm import cost_usd
from .memory import get_backend
from .types import RunResult

PROBE_ORDER = [
    "simple_recall",
    "multi_hop",
    "contradiction",
    "temporal",
    "negation",
    "aggregation",
]

#: The control. Cells running it are keyed by bare backend name, so a default
#: run produces exactly the table it always did.
BASELINE_ORCHESTRATOR = "single_shot"


def cell_key(backend: str, orchestrator: str) -> str:
    if orchestrator == BASELINE_ORCHESTRATOR:
        return backend
    return f"{backend}/{orchestrator}"


def summarize(results: list[RunResult]) -> dict:
    """Collapse the raw cells into per-(backend, orchestrator) metrics."""
    grouped: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for r in results:
        grouped[(r.backend, r.orchestrator)].append(r)

    summary: dict[str, dict] = {}
    for (backend, orchestrator), cells in grouped.items():
        probes = [p for c in cells for p in c.probes]
        grades = [p.grade for p in probes]

        per_type: dict[str, float] = {}
        counts: dict[str, int] = {}
        for ptype in PROBE_ORDER:
            subset = [p.grade for p in probes if p.probe.type == ptype]
            if subset:
                per_type[ptype] = score(subset)
                counts[ptype] = len(subset)

        usage = [u for c in cells for u in c.usage]
        n_probes = len(probes) or 1

        def spend(*phases: str) -> tuple[float, int]:
            picked = [u for u in usage if u.phase in phases]
            return (
                sum(cost_usd(u) for u in picked),
                sum(u.input_tokens + u.output_tokens for u in picked),
            )

        write_cost, write_tokens = spend("write")
        # Control-flow spend belongs on the read side of the ledger: it is what
        # the agent paid to decide what to look at. Folding it in is the only
        # way an orchestrator that searches four times reads as more expensive
        # than one that searches once.
        read_cost, read_tokens = spend("read", "answer", "orchestrate")
        orch_cost, orch_tokens = spend("orchestrate")
        judge_cost, judge_tokens = spend("judge")

        summary[cell_key(backend, orchestrator)] = {
            "backend": backend,
            "orchestrator": orchestrator,
            "blurb": getattr(get_backend(backend), "blurb", ""),
            "score": score(grades),
            "by_type": per_type,
            "type_counts": counts,
            "n_probes": len(probes),
            "hard_fails": sum(1 for p in probes if p.hard_fail),
            "write_cost_usd": write_cost,
            "read_cost_usd": read_cost,
            "orchestrate_cost_usd": orch_cost,
            "judge_cost_usd": judge_cost,
            "cost_per_probe_usd": (write_cost + read_cost) / n_probes,
            "write_tokens": write_tokens,
            "read_tokens": read_tokens,
            "orchestrate_tokens": orch_tokens,
            "judge_tokens": judge_tokens,
            "tokens_per_probe": (write_tokens + read_tokens) / n_probes,
            # Store round trips per probe. 1.0 means the orchestrator never went
            # back for more; the gap above 1.0 is what the extra spend bought.
            "avg_hops": (sum(p.hops for p in probes) / n_probes if probes else 0),
            # Structured-output round trips beyond the first. Zero on the API,
            # a real reliability signal for small local models.
            "json_repairs": sum(u.calls - 1 for u in usage),
            "avg_context_chars": (
                sum(p.context_chars for p in probes) / n_probes if probes else 0
            ),
            "avg_read_latency_s": (
                sum(p.read_latency_s for p in probes) / n_probes if probes else 0
            ),
            "wall_s": sum(c.wall_s for c in cells),
            "errors": [c.error for c in cells if c.error],
            "store_stats": {c.task_id: c.store_stats for c in cells},
        }
    return summary


def orchestrators_in(summary: dict) -> list[str]:
    """Distinct orchestrators present, control first."""
    found = {m.get("orchestrator", BASELINE_ORCHESTRATOR) for m in summary.values()}
    head = [BASELINE_ORCHESTRATOR] if BASELINE_ORCHESTRATOR in found else []
    return head + sorted(found - {BASELINE_ORCHESTRATOR})


def to_markdown(summary: dict, priced: bool = True) -> str:
    rows = sorted(summary.items(), key=lambda kv: kv[1]["score"], reverse=True)

    if priced:
        head = "| Backend | Score | Write $ | Read $ | $/probe | Avg ctx | Hard fails |"
        fmt = lambda m: (  # noqa: E731
            f"${m['write_cost_usd']:.4f} | ${m['read_cost_usd']:.4f} | "
            f"${m['cost_per_probe_usd']:.5f}"
        )
    else:
        head = "| Backend | Score | Write tok | Read tok | Tok/probe | Avg ctx | Hard fails |"
        fmt = lambda m: (  # noqa: E731
            f"{m['write_tokens']:,} | {m['read_tokens']:,} | "
            f"{m['tokens_per_probe']:,.0f}"
        )

    lines = [head, "|---" * 7 + "|"]
    for name, m in rows:
        lines.append(
            f"| `{name}` | {m['score']:.2f} | {fmt(m)} | "
            f"{m['avg_context_chars']:.0f} | {m['hard_fails']} |"
        )

    lines += ["", "### Score by probe type", ""]
    types = [t for t in PROBE_ORDER if any(t in m["by_type"] for _, m in rows)]
    lines.append("| Backend | " + " | ".join(types) + " |")
    lines.append("|---" * (len(types) + 1) + "|")
    for name, m in rows:
        cells = [f"{m['by_type'][t]:.2f}" if t in m["by_type"] else "--" for t in types]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += orchestration_table(summary, priced=priced)

    repairs = sum(m["json_repairs"] for _, m in rows)
    if repairs:
        lines += [
            "",
            f"_{repairs} structured-output repair round trips were needed. A high "
            "count means the model struggles to emit valid JSON, which degrades "
            "both the extraction backends and the judge._",
        ]
    return "\n".join(lines)


def orchestration_table(summary: dict, priced: bool = True) -> list[str]:
    """Per-backend lift from handing control flow to the model, or to a graph.

    Rendered only when there is something to compare. Deltas are against
    `single_shot` because that is the honest question -- not "is the loop good"
    but "was the loop worth more than one retrieval and one answer call".
    """
    orders = orchestrators_in(summary)
    if len(orders) < 2:
        return []

    by_backend: dict[str, dict[str, dict]] = defaultdict(dict)
    for m in summary.values():
        by_backend[m["backend"]][m.get("orchestrator", BASELINE_ORCHESTRATOR)] = m

    if priced:
        unit = "$/probe"
        cost_of = lambda m: m["cost_per_probe_usd"]  # noqa: E731
        render = lambda v: f"${v:.5f}"  # noqa: E731
        delta = lambda v: f"{v:+.5f}"  # noqa: E731
    else:
        unit = "tok/probe"
        cost_of = lambda m: m["tokens_per_probe"]  # noqa: E731
        render = lambda v: f"{v:,.0f}"  # noqa: E731
        delta = lambda v: f"{v:+,.0f}"  # noqa: E731

    lines = [
        "",
        "### Orchestration: does searching again pay for itself?",
        "",
        f"| Backend | Orchestrator | Score | Score lift | Hops | {unit} | {unit} lift |",
        "|---" * 7 + "|",
    ]
    for backend in sorted(by_backend):
        control = by_backend[backend].get(BASELINE_ORCHESTRATOR)
        for orchestrator in orders:
            m = by_backend[backend].get(orchestrator)
            if m is None:
                continue
            if control is None or orchestrator == BASELINE_ORCHESTRATOR:
                d_score, d_cost = "--", "--"
            else:
                d_score = f"{m['score'] - control['score']:+.2f}"
                d_cost = delta(cost_of(m) - cost_of(control))
            lines.append(
                f"| `{backend}` | `{orchestrator}` | {m['score']:.2f} | {d_score} | "
                f"{m['avg_hops']:.1f} | {render(cost_of(m))} | {d_cost} |"
            )

    lines += [
        "",
        "_Hops are store round trips per probe. A `+0.00` score delta beside a "
        "positive cost delta is the result worth reporting: the extra "
        "orchestration bought nothing._",
    ]
    return lines


def write_charts(summary: dict, out_dir: Path, priced: bool = True) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    backends = sorted(summary, key=lambda b: summary[b]["score"], reverse=True)
    types = [t for t in PROBE_ORDER if any(t in summary[b]["by_type"] for b in backends)]

    # 1. Grouped bars: score by probe type.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.8 / max(len(backends), 1)
    for i, backend in enumerate(backends):
        vals = [summary[backend]["by_type"].get(t, 0.0) for t in types]
        ax.bar([x + i * width for x in range(len(types))], vals, width, label=backend)
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(types))])
    ax.set_xticklabels([t.replace("_", "\n") for t in types])
    ax.set_ylabel("score (correct=1, partial=0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Memory architecture vs. probe type")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "by_probe_type.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2. The frontier: accuracy against what it cost to get there.
    key = "cost_per_probe_usd" if priced else "tokens_per_probe"
    xlabel = (
        "cost per probe (USD, write + read amortized)"
        if priced
        else "tokens per probe (write + read amortized)"
    )
    orders = orchestrators_in(summary)
    markers = dict(zip(orders, ["o", "s", "^", "D", "v"]))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for backend in backends:
        m = summary[backend]
        ax.scatter(
            m[key],
            m["score"],
            s=90,
            marker=markers.get(m.get("orchestrator", BASELINE_ORCHESTRATOR), "o"),
        )
        ax.annotate(
            backend,
            (m[key], m["score"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=9,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_title("Accuracy vs. cost -- upper left is the frontier")
    ax.grid(alpha=0.3)
    if len(orders) > 1:
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D([], [], marker=markers[o], ls="", color="gray", label=o)
                for o in orders
            ],
            fontsize=8,
        )
    fig.tight_layout()
    path = out_dir / "cost_vs_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    if len(orders) > 1:
        written.append(_orchestration_chart(summary, out_dir, priced, orders))
    return written


def _orchestration_chart(
    summary: dict, out_dir: Path, priced: bool, orders: list[str]
) -> Path:
    """Score and spend side by side, grouped by backend. The 30-second read."""
    import matplotlib.pyplot as plt

    by_backend: dict[str, dict[str, dict]] = defaultdict(dict)
    for m in summary.values():
        by_backend[m["backend"]][m.get("orchestrator", BASELINE_ORCHESTRATOR)] = m
    names = sorted(by_backend)

    key = "cost_per_probe_usd" if priced else "tokens_per_probe"
    cost_label = "cost per probe (USD)" if priced else "tokens per probe"

    fig, (ax_score, ax_cost) = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.8 / max(len(orders), 1)
    for i, orchestrator in enumerate(orders):
        offs = [x + i * width for x in range(len(names))]
        ax_score.bar(
            offs,
            [by_backend[n].get(orchestrator, {}).get("score", 0.0) for n in names],
            width,
            label=orchestrator,
        )
        ax_cost.bar(
            offs,
            [by_backend[n].get(orchestrator, {}).get(key, 0.0) for n in names],
            width,
            label=orchestrator,
        )

    for ax, ylabel, title in (
        (ax_score, "score", "Accuracy by orchestration style"),
        (ax_cost, cost_label, "What that accuracy cost"),
    ):
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(names))])
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    ax_score.set_ylim(0, 1.05)

    fig.suptitle("Loop vs. graph vs. neither, holding memory fixed", fontsize=12)
    fig.tight_layout()
    path = out_dir / "orchestration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save(results: list[RunResult], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    raw = [
        {
            "backend": r.backend,
            "orchestrator": r.orchestrator,
            "task": r.task_id,
            "wall_s": r.wall_s,
            "error": r.error,
            "store_stats": r.store_stats,
            "probes": [
                {
                    "probe_id": p.probe.probe_id,
                    "type": p.probe.type,
                    "question": p.probe.question,
                    "expected": p.probe.expected,
                    "answer": p.answer,
                    "grade": p.grade,
                    "reason": p.reason,
                    "hard_fail": p.hard_fail,
                    "context_chars": p.context_chars,
                    "hops": p.hops,
                    # What the orchestrator actually searched for. This is the
                    # error-analysis payload: a wrong answer after four hops of
                    # bad queries is a different bug from one after a good query.
                    "queries": list(p.queries),
                }
                for p in r.probes
            ],
        }
        for r in results
    ]
    (out_dir / "runs.json").write_text(
        json.dumps(raw, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "table.md").write_text(to_markdown(summary), encoding="utf-8")
