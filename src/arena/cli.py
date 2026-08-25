"""Command line: `arena run`, `arena list`, `arena inspect`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import backends as _backends  # noqa: F401  (registers every backend)
from . import report, runner, tasks
from .llm import DEFAULT_MODEL, credentials_present
from .memory import available, get_backend

console = Console()


def cmd_list(args: argparse.Namespace) -> int:
    console.print("[bold]Registered backends[/bold]")
    for name in available():
        console.print(f"  [cyan]{name:<16}[/cyan] {get_backend(name).blurb}")
    task_dir = Path(args.tasks)
    if task_dir.exists():
        console.print("\n[bold]Tasks[/bold]")
        for task in tasks.load_all(task_dir):
            console.print(
                f"  [cyan]{task.task_id:<16}[/cyan] "
                f"{len(task.events)} turns, {len(task.probes)} probes"
            )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not credentials_present():
        console.print(
            "[red]No Anthropic credentials found.[/red] Set ANTHROPIC_API_KEY "
            "or run `ant auth login`."
        )
        return 1

    task_list = tasks.load_all(Path(args.tasks))
    if args.task:
        task_list = [t for t in task_list if t.task_id in args.task]
    if not task_list:
        console.print("[red]No tasks matched.[/red]")
        return 1

    backend_names = args.backend or available()
    for name in backend_names:
        get_backend(name)  # fail fast on a typo before spending anything

    total_probes = len(backend_names) * sum(len(t.probes) for t in task_list)
    console.print(
        f"[bold]{len(backend_names)} backends x {len(task_list)} tasks[/bold] "
        f"= {total_probes} probes, model={args.model}, effort={args.effort}"
    )

    def on_cell(cell) -> None:
        if cell.error:
            console.print(f"  [red]{cell.backend} / {cell.task_id}: failed[/red]")
            console.print(cell.error)
            return
        graded = [p.grade for p in cell.probes]
        from .judge import score as _score

        console.print(
            f"  [green]{cell.backend:<16}[/green] {cell.task_id:<14} "
            f"score={_score(graded):.2f}  ({cell.wall_s:.1f}s)"
        )

    results = runner.run_matrix(
        backend_names,
        task_list,
        model=args.model,
        effort=args.effort,
        token_budget=args.token_budget,
        cache_dir=Path(args.cache),
        on_cell=on_cell,
    )

    summary = report.summarize(results)
    out_dir = Path(args.out)
    report.save(results, summary, out_dir)
    console.print()
    console.print(report.to_markdown(summary))

    if not args.no_charts:
        try:
            for path in report.write_charts(summary, out_dir):
                console.print(f"\n[dim]chart: {path}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]chart generation skipped: {exc}[/yellow]")

    console.print(f"[dim]results: {out_dir}/summary.json, {out_dir}/runs.json[/dim]")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show every probe a backend got wrong -- the error analysis loop."""
    import json

    raw = json.loads((Path(args.out) / "runs.json").read_text(encoding="utf-8"))
    for cell in raw:
        if args.backend and cell["backend"] not in args.backend:
            continue
        for probe in cell["probes"]:
            if probe["grade"] == "correct" and not args.all:
                continue
            colour = {"correct": "green", "partial": "yellow"}.get(probe["grade"], "red")
            console.print(
                f"[{colour}]{probe['grade']:<9}[/{colour}] "
                f"[cyan]{cell['backend']}[/cyan] {probe['type']}"
            )
            console.print(f"   Q: {probe['question']}")
            console.print(f"   expected: {probe['expected']}")
            console.print(f"   got:      {probe['answer']}")
            console.print(f"   [dim]{probe['reason']}[/dim]\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arena", description="Benchmark agent memory architectures."
    )
    parser.add_argument("--tasks", default="tasks", help="task directory")
    parser.add_argument("--out", default="results", help="output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show backends and tasks")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run the matrix")
    p_run.add_argument("-b", "--backend", action="append", help="repeatable")
    p_run.add_argument("-t", "--task", action="append", help="repeatable")
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p_run.add_argument("--token-budget", type=int, default=2000)
    p_run.add_argument("--cache", default="results/cache")
    p_run.add_argument("--no-charts", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_inspect = sub.add_parser("inspect", help="review graded answers")
    p_inspect.add_argument("-b", "--backend", action="append")
    p_inspect.add_argument("--all", action="store_true", help="include correct ones")
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
