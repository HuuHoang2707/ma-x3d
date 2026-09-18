"""Collect finished runs into result tables (Markdown + LaTeX) and training curves."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# (column title, getter) for the summary table; values are fractions unless noted.
COLUMNS = [
    ("test acc", lambda r: r["test"]["accuracy"]),
    ("test F1", lambda r: r["test"]["f1"]),
    ("test AUC", lambda r: r["test"].get("auc")),
    ("val F1", lambda r: r["val"]["f1"]),
    ("ring off acc", lambda r: r.get("test_ring_off", {}).get("accuracy")),
    ("ring energy", lambda r: r.get("ring_energy_mean")),
]


def load_runs(root: str | Path) -> list[dict]:
    runs = []
    for res in sorted(Path(root).glob("**/results.json")):
        r = json.loads(res.read_text())
        cfg = yaml.safe_load((res.parent / "config.yaml").read_text())
        r["dir"] = str(res.parent)
        r["description"] = cfg.get("description", "")
        runs.append(r)
    return runs


def _stats(values: list) -> tuple[float, float, int] | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return float(np.mean(v)), std, len(v)


def summarize(runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        groups[r["name"]].append(r)
    rows = []
    for name in sorted(groups):
        rs = groups[name]
        rows.append({
            "name": name,
            "description": rs[0]["description"],
            "protocol": rs[0]["protocol"],
            "seeds": sorted(r["seed"] for r in rs),
            "stats": {title: _stats([get(r) for r in rs]) for title, get in COLUMNS},
        })
    return rows


def _pct(s, latex=False) -> str:
    if s is None:
        return "-"
    mean, std, n = s
    pm = r"$\pm$" if latex else "±"
    if mean > 1.0:  # not a fraction
        return f"{mean:.3f}" if n == 1 else f"{mean:.3f} {pm} {std:.3f}"
    return f"{100 * mean:.2f}" if n == 1 else f"{100 * mean:.2f} {pm} {100 * std:.2f}"


def markdown_table(rows: list[dict]) -> str:
    head = ["experiment", "description", "seeds"] + [c for c, _ in COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["name"], r["description"], str(len(r["seeds"]))]
        for title, _ in COLUMNS:
            s = r["stats"][title]
            cells.append(f"{s[0]:.4f}" if title == "ring energy" and s else _pct(s))
        if r["protocol"] != "holdout":
            cells[0] += " (test-selected)"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_rows(rows: list[dict]) -> str:
    """Rows for a booktabs table: description & accuracy & F1 (mean +- std in %)."""
    out = []
    for r in rows:
        acc, f1 = r["stats"]["test acc"], r["stats"]["test F1"]
        out.append(f"{r['description'] or r['name']} & {_pct(acc, True)} & {_pct(f1, True)} \\\\")
    return "\n".join(out)


def plot_curves(run: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = run / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (title, tr, va) in zip(axes, [("loss", "train_loss", "val_loss"),
                                           ("accuracy", "train_acc", "val_acc"),
                                           ("val F1", None, "val_f1")], strict=True):
        if tr:
            ax.plot(ep, [r[tr] for r in rows], "o-", label="train")
        ax.plot(ep, [r[va] for r in rows], "s--", label="val")
        switch = next((r["epoch"] for r in rows if r["phase"] == "finetune"), None)
        if switch:
            ax.axvline(switch - 0.5, color="gray", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(str(run), fontsize=9)
    fig.tight_layout()
    out = run / "curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def write_report(runs_root: str | Path, out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_root)
    for r in runs:
        plot_curves(Path(r["dir"]))
    rows = summarize(runs)
    md = markdown_table(rows)
    (out_dir / "summary.md").write_text(
        "# Results\n\nTest numbers are mean ± std over seeds, in %. The test split is the "
        "official 400-clip RWF-2000 partition, evaluated once with the checkpoint picked on "
        "validation F1.\n\n" + md + "\n")
    (out_dir / "ablation_rows.tex").write_text(latex_rows(rows) + "\n")
    (out_dir / "runs.json").write_text(json.dumps(rows, indent=2))
    return md
