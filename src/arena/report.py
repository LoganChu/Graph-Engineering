"""Aggregation and output: results.json, a markdown table, and two charts.

The scatter is the one that matters. Accuracy alone always ranks the full
transcript first; plotting accuracy against cost per probe is what shows which
architectures are actually on the efficient frontier and which are just
expensive.
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


def summarize(results: list[RunResult]) -> dict:
    """Collapse the raw cells into per-backend metrics."""
    by_backend: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_backend[r.backend].append(r)

    summary: dict[str, dict] = {}
    for backend, cells in by_backend.items():
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
        write_cost = sum(cost_usd(u) for u in usage if u.phase == "write")
        read_cost = sum(cost_usd(u) for u in usage if u.phase in ("read", "answer"))
        judge_cost = sum(cost_usd(u) for u in usage if u.phase == "judge")
        n_probes = len(probes) or 1

        summary[backend] = {
            "blurb": getattr(get_backend(backend), "blurb", ""),
            "score": score(grades),
            "by_type": per_type,
            "type_counts": counts,
            "n_probes": len(probes),
            "hard_fails": sum(1 for p in probes if p.hard_fail),
            "write_cost_usd": write_cost,
            "read_cost_usd": read_cost,
            "judge_cost_usd": judge_cost,
            "cost_per_probe_usd": (write_cost + read_cost) / n_probes,
            "write_tokens": sum(
                u.input_tokens + u.output_tokens for u in usage if u.phase == "write"
            ),
            "answer_tokens": sum(
                u.input_tokens + u.output_tokens for u in usage if u.phase == "answer"
            ),
            "avg_context_chars": (
                sum(p.context_chars for p in probes) / n_probes if probes else 0
            ),
            "avg_read_latency_s": (
                sum(p.read_latency_s for p in probes) / n_probes if probes else 0
            ),
            "errors": [c.error for c in cells if c.error],
            "store_stats": {c.task_id: c.store_stats for c in cells},
        }
    return summary


def to_markdown(summary: dict) -> str:
    rows = sorted(summary.items(), key=lambda kv: kv[1]["score"], reverse=True)
    lines = [
        "| Backend | Score | Write $ | Read $ | $/probe | Avg ctx (chars) | Hard fails |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| `{name}` | {m['score']:.2f} | ${m['write_cost_usd']:.4f} | "
            f"${m['read_cost_usd']:.4f} | ${m['cost_per_probe_usd']:.5f} | "
            f"{m['avg_context_chars']:.0f} | {m['hard_fails']} |"
        )

    lines += ["", "### Score by probe type", ""]
    types = [t for t in PROBE_ORDER if any(t in m["by_type"] for _, m in rows)]
    lines.append("| Backend | " + " | ".join(types) + " |")
    lines.append("|---" * (len(types) + 1) + "|")
    for name, m in rows:
        cells = [
            f"{m['by_type'][t]:.2f}" if t in m["by_type"] else "--" for t in types
        ]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_charts(summary: dict, out_dir: Path) -> list[Path]:
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

    # 2. The frontier: accuracy against cost.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for backend in backends:
        m = summary[backend]
        ax.scatter(m["cost_per_probe_usd"], m["score"], s=90)
        ax.annotate(
            backend,
            (m["cost_per_probe_usd"], m["score"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=9,
        )
    ax.set_xlabel("cost per probe (USD, write + read amortized)")
    ax.set_ylabel("score")
    ax.set_title("Accuracy vs. cost -- upper left is the frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cost_vs_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def save(results: list[RunResult], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    raw = [
        {
            "backend": r.backend,
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
