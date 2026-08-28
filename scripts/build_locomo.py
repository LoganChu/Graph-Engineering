"""Fetch LoCoMo and write it out as arena task files.

    python scripts/build_locomo.py                     # 3 conversations, 40 probes each
    python scripts/build_locomo.py -c 10 -p 60         # more of both
    python scripts/build_locomo.py --all               # everything (read the warning)

LoCoMo is a single 2.8 MB JSON in the authors' repository -- small enough to
fetch at build time and small enough to cache locally, unlike LongMemEval's
278 MB `s` split. Only the converted YAML lands in the tree.

Conversation count is the expensive axis: each one is ~590 turns, and the graph
backends spend one `llm.parse` per turn ingested. Probe count is nearly free by
comparison, since a conversation is ingested once and then interrogated many
times -- which is the whole reason this corpus is cheaper per probe than
LongMemEval.

Conversion lives in `arena.locomo` so it can be tested without a network.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arena.locomo import subset, to_task  # noqa: E402

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

#: Per-turn write cost is what bites: one extraction call per turn ingested,
#: per graph backend.
COST_WARNING = 5_000


def load(cache: Path) -> list[dict]:
    if not cache.exists():
        print(f"fetching {URL} ...", file=sys.stderr)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as response:
            cache.write_bytes(response.read())
    else:
        print(f"using cached {cache}", file=sys.stderr)
    return json.loads(cache.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--conversations", type=int, default=3)
    ap.add_argument(
        "-p",
        "--probes",
        type=int,
        default=40,
        help="max probes per conversation, round-robined across probe types",
    )
    ap.add_argument("--all", action="store_true", help="every conversation and probe")
    ap.add_argument("--out", default="tasks/locomo", help="output directory")
    ap.add_argument(
        "--cache",
        default="results/cache/locomo10.json",
        help="where to keep the downloaded corpus",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="skip the cost prompt")
    args = ap.parse_args()

    instances = load(Path(args.cache))
    print(f"  {len(instances)} conversations upstream", file=sys.stderr)

    picked = subset(
        instances,
        conversations=None if args.all else args.conversations,
        probes_per_conversation=None if args.all else args.probes,
        seed=args.seed,
    )
    tasks = [to_task(i) for i in picked]
    tasks = [t for t in tasks if t["probes"]]

    turns = sum(len(t["turns"]) for t in tasks)
    probes = sum(len(t["probes"]) for t in tasks)
    chars = sum(len(x["text"]) for t in tasks for x in t["turns"])
    print(
        f"  keeping {len(tasks)} conversations -- {turns:,} turns, {probes:,} probes, "
        f"{chars / 1e6:.1f} MB of text",
        file=sys.stderr,
    )
    print(
        f"  graph backends will make ~{turns:,} extraction calls per backend "
        f"({turns / max(probes, 1):.1f} per probe)",
        file=sys.stderr,
    )
    if turns > COST_WARNING and not args.force:
        reply = input("  that is a lot of write-path spend. continue? [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            print("aborted", file=sys.stderr)
            return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.yaml"):
        stale.unlink()

    for task in tasks:
        (out / f"{task['id']}.yaml").write_text(
            yaml.safe_dump(task, allow_unicode=True, sort_keys=False, width=100),
            encoding="utf-8",
        )

    counts: dict[str, int] = {}
    for task in tasks:
        for probe in task["probes"]:
            counts[probe["type"]] = counts.get(probe["type"], 0) + 1
    guarded = sum(
        1 for t in tasks for p in t["probes"] if p.get("must_not_contain")
    )
    print(f"\nwrote {len(tasks)} task files to {out}/", file=sys.stderr)
    for kind in sorted(counts):
        print(f"  {kind:<16} {counts[kind]}", file=sys.stderr)
    print(f"  {guarded} probes carry a must_not_contain distractor", file=sys.stderr)
    print(f"\n  uv run arena run --tasks {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
