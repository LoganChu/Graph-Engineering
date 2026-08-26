"""Fetch LongMemEval and write it out as arena task files.

    python scripts/build_longmemeval.py                    # 60 oracle questions
    python scripts/build_longmemeval.py --variant s -n 20  # 20 full haystacks
    python scripts/build_longmemeval.py --all              # all 500 (read the warning)

The raw JSON lands in the HuggingFace cache, not in the repo -- `s` is 278 MB
and `m` is 2.7 GB, neither of which belongs in a synced project directory. Only
the converted YAML is written into the tree.

Conversion lives in `arena.longmemeval` so it can be tested without a network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arena.longmemeval import stratified, to_task  # noqa: E402

REPO = "xiaowu0162/longmemeval"
VARIANTS = {"oracle": "longmemeval_oracle", "s": "longmemeval_s", "m": "longmemeval_m"}

#: Per-turn write cost is the thing that bites: the graph backends spend one
#: `llm.parse` on every turn ingested, evidence-bearing or not.
COST_WARNING = 20_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="oracle")
    ap.add_argument("-n", "--limit", type=int, default=60, help="questions to keep")
    ap.add_argument("--all", action="store_true", help="keep all 500")
    ap.add_argument("--out", default="tasks/longmemeval", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="skip the cost prompt")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("needs huggingface_hub:  uv pip install huggingface_hub", file=sys.stderr)
        return 1

    filename = VARIANTS[args.variant]
    print(f"fetching {REPO}/{filename} ...", file=sys.stderr)
    path = hf_hub_download(REPO, filename, repo_type="dataset")
    instances = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"  {len(instances)} instances", file=sys.stderr)

    picked = stratified(instances, None if args.all else args.limit, seed=args.seed)
    tasks = [to_task(i) for i in picked]

    turns = sum(len(t["turns"]) for t in tasks)
    chars = sum(len(x["text"]) for t in tasks for x in t["turns"])
    print(
        f"  keeping {len(tasks)} questions -- {turns:,} turns, {chars / 1e6:.1f} MB of text",
        file=sys.stderr,
    )
    print(
        f"  graph backends will make ~{turns:,} extraction calls per backend",
        file=sys.stderr,
    )
    if turns > COST_WARNING and not args.force:
        reply = input(f"  that is a lot of write-path spend. continue? [y/N] ")
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
        kind = task["probes"][0]["type"]
        counts[kind] = counts.get(kind, 0) + 1
    print(f"\nwrote {len(tasks)} task files to {out}/", file=sys.stderr)
    for kind in sorted(counts):
        print(f"  {kind:<16} {counts[kind]}", file=sys.stderr)
    print(f"\n  uv run arena run --tasks {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
