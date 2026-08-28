"""Command line: `arena run`, `arena list`, `arena inspect`, `arena doctor`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import backends as _backends  # noqa: F401  (registers every backend)
from . import orchestrators as _orchestrators  # noqa: F401  (registers each one)
from . import report, runner, tasks
from .llm import DEFAULT_LOCAL_MODEL, DEFAULT_MODEL, ModelConfig, credentials_present
from .memory import available, get_backend
from .orchestration import DEFAULT_MAX_HOPS
from .orchestration import available as available_orchestrators
from .orchestration import get_orchestrator

console = Console()
OLLAMA_URL = "http://localhost:11434/v1"

#: The corpus is LongMemEval, generated into this directory rather than checked
#: in -- the `s` variant is 278 MB of raw JSON upstream. The two hand-authored
#: YAML tasks now live in tasks/handwritten/ and are no longer on the default
#: path; point --tasks at them if you want the old offline smoke run.
DEFAULT_TASKS = "tasks/longmemeval"
BUILD_HINT = "python scripts/build_longmemeval.py"


def _default_model(provider: str) -> str:
    return DEFAULT_LOCAL_MODEL if provider == "local" else DEFAULT_MODEL


def _default_url(provider: str, given: str | None) -> str | None:
    return given or (OLLAMA_URL if provider == "local" else None)


def build_config(args: argparse.Namespace) -> ModelConfig:
    """Resolve provider defaults so `--provider local` needs no other flags."""
    judge_provider = getattr(args, "judge_provider", None)
    judge_model = args.judge_model
    if judge_provider and not judge_model:
        # Switching the judge's transport without naming a model is almost
        # always a mistake -- fall back to that provider's default rather than
        # asking the new endpoint for a model it does not serve.
        judge_model = _default_model(judge_provider)
    return ModelConfig(
        provider=args.provider,
        model=args.model or _default_model(args.provider),
        judge_model=judge_model,
        judge_provider=judge_provider,
        judge_base_url=_default_url(judge_provider or args.provider,
                                    getattr(args, "judge_base_url", None)),
        effort=args.effort,
        base_url=_default_url(args.provider, args.base_url),
    )


def check_endpoint(provider: str, base_url: str | None, models: set[str]) -> str | None:
    """Return an error string if this transport cannot serve these models."""
    if provider == "anthropic":
        if not credentials_present():
            return (
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY, run "
                "`ant auth login`, or switch to a local model with "
                "`--provider local`."
            )
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return "Local providers need the extra: uv sync --extra local"

    try:
        client = OpenAI(base_url=base_url, api_key="not-needed", timeout=10.0)
        served = {m.id for m in client.models.list().data}
    except Exception as exc:
        return (
            f"Could not reach a local model server at {base_url} ({exc}). "
            "Start one with `ollama serve`."
        )

    missing = sorted(m for m in models if m not in served)
    if missing:
        return (
            f"Model(s) not served at {base_url}: {', '.join(missing)}.\n"
            f"Available: {', '.join(sorted(served)) or '(none)'}\n"
            f"Pull one with: ollama pull {missing[0]}"
        )
    return None


def preflight(config: ModelConfig) -> str | None:
    """Check every transport the run will actually use.

    The judge may sit on a different endpoint than the agent, and discovering
    that only after the write path has already run is expensive.
    """
    judge_provider = config.judge_provider or config.provider
    judge_model = config.judge_model or config.model

    if judge_provider == config.provider:
        return check_endpoint(
            config.provider, config.base_url, {config.model, judge_model}
        )
    return check_endpoint(config.provider, config.base_url, {config.model}) or (
        check_endpoint(judge_provider, config.judge_base_url, {judge_model})
    )


def cmd_list(args: argparse.Namespace) -> int:
    console.print("[bold]Registered backends[/bold] [dim](what to remember)[/dim]")
    for name in available():
        console.print(f"  [cyan]{name:<16}[/cyan] {get_backend(name).blurb}")

    console.print()
    console.print(
        "[bold]Registered orchestrators[/bold] [dim](who owns control flow)[/dim]"
    )
    for name in available_orchestrators():
        console.print(f"  [cyan]{name:<16}[/cyan] {get_orchestrator(name).blurb}")
    if not {"loop", "graph"} <= set(available_orchestrators()):
        console.print(
            "  [dim]loop/graph unavailable -- uv sync --extra orchestration[/dim]"
        )

    task_dir = Path(args.tasks)
    task_list = tasks.load_all(task_dir) if task_dir.exists() else []
    console.print("\n[bold]Tasks[/bold]")
    if not task_list:
        console.print(f"  [dim]none in {task_dir}/ -- build them: {BUILD_HINT}[/dim]")
        return 0
    for task in task_list[:20]:
        console.print(
            f"  [cyan]{task.task_id:<22}[/cyan] "
            f"{len(task.events):>4} turns, {len(task.probes)} probes  "
            f"[dim]{task.probes[0].type}[/dim]"
        )
    if len(task_list) > 20:
        console.print(f"  [dim]... and {len(task_list) - 20} more[/dim]")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that the configured provider can actually serve a run."""
    config = build_config(args)
    console.print(
        f"provider=[cyan]{config.provider}[/cyan] model=[cyan]{config.model}[/cyan]"
    )
    console.print(
        f"judge=[cyan]{config.judge_provider or config.provider}[/cyan]:"
        f"[cyan]{config.judge_model or config.model}[/cyan]"
    )
    if config.base_url:
        console.print(f"base_url=[cyan]{config.base_url}[/cyan]")
    problem = preflight(config)
    if problem:
        console.print(f"[red]{problem}[/red]")
        return 1
    console.print("[green]ready[/green]")
    if not config.priced:
        console.print("[dim]unpriced model -- cost columns will read $0.00[/dim]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = build_config(args)
    problem = preflight(config)
    if problem:
        console.print(f"[red]{problem}[/red]")
        return 1

    task_list = tasks.load_all(Path(args.tasks))
    if args.task:
        task_list = [t for t in task_list if t.task_id in args.task]
    if not task_list:
        console.print(
            f"[red]No tasks in {args.tasks}/.[/red] Build the corpus first: {BUILD_HINT}"
        )
        return 1

    backend_names = args.backend or available()
    for name in backend_names:
        get_backend(name)  # fail fast on a typo before spending anything

    orchestrator_names = args.orchestrator or ["single_shot"]
    for name in orchestrator_names:
        # A missing extra is an install problem, not a typo. Say so plainly
        # rather than raising a traceback at someone mid-benchmark.
        try:
            get_orchestrator(name)
        except KeyError as exc:
            console.print(f"[red]{exc.args[0]}[/red]")
            return 1

    trials = max(args.trials, 1)
    total_probes = (
        len(backend_names)
        * len(orchestrator_names)
        * trials
        * sum(len(t.probes) for t in task_list)
    )
    console.print(
        f"[bold]{len(backend_names)} backends x {len(orchestrator_names)} "
        f"orchestrators x {len(task_list)} tasks"
        + (f" x {trials} trials" if trials > 1 else "")
        + f"[/bold] = {total_probes} probes | {config.provider}:{config.model}"
        + (f" | judge {config.judge_model}" if config.judge_model else "")
    )
    if trials == 1:
        console.print(
            "[dim]Single trial: scores will print without a usable interval. "
            "Use --trials 3 to get error bars and pass^k.[/dim]"
        )
    if config.judge_model in (None, config.model) and not config.priced:
        console.print(
            "[yellow]The judge is the same local model as the agent. Scores will "
            "partly measure the judge -- see --judge-model.[/yellow]"
        )

    def on_cell(cell) -> None:
        from .judge import score as _score

        label = f"{cell.backend} / {cell.orchestrator} / {cell.task_id}"
        if cell.failure is not None:
            f = cell.failure
            turn = f" at turn {f.turn_id}" if f.turn_id else ""
            console.print(
                f"  [red]{label}: {cell.outcome} -- {f.where} phase{turn}, "
                f"{f.probes_done}/{f.probes_expected} probes graded. "
                f"Excluded from the report.[/red]"
            )
            console.print(f.detail)
            return
        hops = sum(p.hops for p in cell.probes) / max(len(cell.probes), 1)
        trial = f" t{cell.trial}" if cell.trial else ""
        console.print(
            f"  [green]{cell.backend:<16}[/green] {cell.orchestrator:<12} "
            f"{cell.task_id:<12}{trial} "
            f"score={_score([p.grade for p in cell.probes]):.2f}  "
            f"hops={hops:.1f}  ({cell.wall_s:.1f}s)"
        )

    results = runner.run_matrix(
        backend_names,
        task_list,
        config=config,
        orchestrators=orchestrator_names,
        trials=trials,
        token_budget=args.token_budget,
        max_hops=args.max_hops,
        cache_dir=Path(args.cache),
        on_cell=on_cell,
    )

    summary = report.summarize(results)
    out_dir = Path(args.out)
    report.save(results, summary, out_dir)
    console.print()
    console.print(report.to_markdown(summary, priced=config.priced))

    if not args.no_charts:
        try:
            for path in report.write_charts(summary, out_dir, priced=config.priced):
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
        orchestrator = cell.get("orchestrator", "single_shot")
        if args.orchestrator and orchestrator not in args.orchestrator:
            continue
        for probe in cell["probes"]:
            if probe["grade"] == "correct" and not args.all:
                continue
            colour = {"correct": "green", "partial": "yellow"}.get(probe["grade"], "red")
            console.print(
                f"[{colour}]{probe['grade']:<9}[/{colour}] "
                f"[cyan]{cell['backend']}[/cyan]/[magenta]{orchestrator}[/magenta] "
                f"{probe['type']}"
            )
            console.print(f"   Q: {probe['question']}")
            console.print(f"   expected: {probe['expected']}")
            console.print(f"   got:      {probe['answer']}")
            # Which queries were tried separates a retrieval failure from a
            # reasoning one -- four hops of bad phrasings is a different bug
            # from one good query the store could not serve.
            queries = probe.get("queries") or []
            if len(queries) > 1 or (queries and queries[0] != probe["question"]):
                console.print(f"   searched: {' -> '.join(repr(q) for q in queries)}")
            # Retrieval recall separates the two failures the answer cannot:
            # recall 1.0 means the evidence was in front of the model and it
            # still got this wrong.
            ev = probe.get("evidence")
            if ev:
                console.print(
                    f"   evidence: recall={ev['recall']:.2f} "
                    f"({ev['n_hit']}/{ev['n_gold']} gold sessions), "
                    f"stopped={probe.get('stop', '?')}"
                )
            console.print(f"   [dim]{probe['reason']}[/dim]")
            console.print()
    return 0


def add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "local"],
        help="'local' targets any OpenAI-compatible server (Ollama by default)",
    )
    parser.add_argument("--model", default=None, help="agent + write-path model")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="grade with a different (stronger) model than the one under test",
    )
    parser.add_argument("--base-url", default=None, help=f"default {OLLAMA_URL}")
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=["anthropic", "local"],
        help="grade on a different transport than the agent under test",
    )
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument(
        "--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arena", description="Benchmark agent memory architectures."
    )
    # Shared paths live on a parent parser so they work in either position --
    # `arena --out x run` and `arena run --out x` both parse.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tasks", default=DEFAULT_TASKS, help="task directory")
    common.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help=argparse.SUPPRESS)
    parser.add_argument("--out", default="results", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", parents=[common], help="show backends and tasks")
    p_list.set_defaults(func=cmd_list)

    p_doctor = sub.add_parser(
        "doctor", parents=[common], help="check the provider is reachable"
    )
    add_model_flags(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", parents=[common], help="run the matrix")
    p_run.add_argument("-b", "--backend", action="append", help="repeatable")
    p_run.add_argument("-t", "--task", action="append", help="repeatable")
    p_run.add_argument(
        "-o",
        "--orchestrator",
        action="append",
        help="repeatable; default single_shot. 'loop' and 'graph' need "
        "`uv sync --extra orchestration`",
    )
    add_model_flags(p_run)
    p_run.add_argument(
        "--trials",
        type=int,
        default=1,
        help="repeat the whole matrix N times. Anything above 1 buys error "
        "bars and pass^k; it also multiplies the bill, since trials past the "
        "first deliberately miss the response cache",
    )
    p_run.add_argument("--token-budget", type=int, default=2000)
    p_run.add_argument(
        "--max-hops",
        type=int,
        default=DEFAULT_MAX_HOPS,
        help="ceiling on store round trips per probe for loop/graph "
        f"(default {DEFAULT_MAX_HOPS})",
    )
    p_run.add_argument("--cache", default="results/cache")
    p_run.add_argument("--no-charts", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_inspect = sub.add_parser(
        "inspect", parents=[common], help="review graded answers"
    )
    p_inspect.add_argument("-b", "--backend", action="append")
    p_inspect.add_argument("-o", "--orchestrator", action="append")
    p_inspect.add_argument("--all", action="store_true", help="include correct ones")
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
