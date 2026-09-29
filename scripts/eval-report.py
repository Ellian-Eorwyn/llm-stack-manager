#!/usr/bin/env python3
"""One comparison table from every eval result under benchmarks/.

    scripts/eval-report.py                    # print markdown
    scripts/eval-report.py --out benchmarks/report.md
    scripts/eval-report.py --only qwen3.8-27b-mtplx-speed,flash-next-3bit-g64-mtplx

Reads what `bench-offload.py`, `eval-quality.py` and `eval-workload.py` wrote,
keeps the newest result per model, suite and task, and puts the models side
by side: decode speed at each depth, peak memory, the deepest context that
answered, and every quality and workload score. docs/model-evals.md says how
to read it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _newest(files):
    return sorted(files, key=lambda p: p.stat().st_mtime)


def collect(root: pathlib.Path) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for path in _newest(root.rglob("*.json")):
        if "eval-data" in path.parts or "invalid" in path.parts:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        name = data.get("variant") or data.get("model")
        if not name:
            continue
        entry = models.setdefault(name, {"speed": {}, "scores": {}})
        if "rows" in data:  # bench-offload
            for row in data["rows"]:
                key = (row["depth"], row["task"])
                entry["speed"][key] = row
        elif "results" in data:  # eval-quality / eval-workload
            for task, result in data["results"].items():
                entry["scores"][task] = result
    return models


def _depth_label(depth: int) -> str:
    return "0" if depth == 0 else f"{round(depth / 1024)}k"


def render(models: dict[str, dict], only: list[str]) -> str:
    names = [n for n in (only or sorted(models)) if n in models]
    if not names:
        return "No results yet. Run scripts/eval-run.py first.\n"
    lines = ["# Model evaluation report", ""]

    depths = sorted({d for n in names for (d, _t) in models[n]["speed"]})
    if depths:
        lines += ["## Speed (decode tok/s, chat · code)", "",
                  "| Model | " + " | ".join(_depth_label(d) for d in depths)
                  + " | Peak GiB | Deepest answered |",
                  "|---|" + "---|" * len(depths) + "---|---|"]
        for name in names:
            speed = models[name]["speed"]
            if not speed:
                continue
            cells, peak, deepest = [], 0.0, 0
            for depth in depths:
                parts = []
                for task in ("chat", "code"):
                    row = speed.get((depth, task))
                    if not row or "error" in row:
                        parts.append("✗" if row else "–")
                        continue
                    rate = row.get("server_decode_tps") or row.get("decode_tps")
                    parts.append(f"{rate:.0f}" if rate else "–")
                    peak = max(peak, (row.get("peak_mib") or 0) / 1024)
                    deepest = max(deepest, row.get("prompt_tokens") or 0)
                cells.append(" · ".join(parts))
            lines.append(f"| {name} | " + " | ".join(cells)
                         + f" | {peak:.1f} | {_depth_label(deepest)} |")
        lines += ["", "✗ = refused or failed at that depth (usually memory); – = not run.", ""]

    tasks = []
    for name in names:
        for task in models[name]["scores"]:
            if task not in tasks:
                tasks.append(task)
    if tasks:
        lines += ["## Scores (%)", "",
                  "| Model | " + " | ".join(tasks) + " |",
                  "|---|" + "---|" * len(tasks)]
        for name in names:
            scores = models[name]["scores"]
            if not scores:
                continue
            cells = []
            for task in tasks:
                r = scores.get(task)
                if not r:
                    cells.append("–")
                    continue
                errors = sum(1 for f in r.get("failures", []) if str(f.get("reply", "")).startswith("ERROR"))
                note = f" ⚠{errors} err" if errors else ""
                if r.get("truncated"):
                    note += f" ✂{r['truncated']}"
                cells.append(f"{r['score']:.1f} ({r['total']}){note}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "(n) = questions asked. ⚠ = requests that errored rather than being answered; "
                  "a score with errors understates the model. ✂ = replies cut off at the token budget.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(ROOT / "benchmarks"))
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    text = render(collect(pathlib.Path(args.root)), [n for n in args.only.split(",") if n])
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
