"""Binary classification metrics with Fight (label 1) as the positive class."""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ("NonFight", "Fight")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic (ties get average rank)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):  # average ranks of ties
        tie = scores == s
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(labels, preds, fight_scores=None) -> dict:
    labels, preds = np.asarray(labels).astype(int), np.asarray(preds).astype(int)
    cm = np.zeros((2, 2), dtype=int)  # rows: true class, cols: predicted class
    for t, p in zip(labels, preds, strict=True):
        cm[t, p] += 1
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c, c]
        precision = _safe_div(tp, cm[:, c].sum())
        recall = _safe_div(tp, cm[c, :].sum())
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall),
            "support": int(cm[c, :].sum()),
        }
    out = {
        "n": int(len(labels)),
        "accuracy": _safe_div(np.trace(cm), cm.sum()),
        "f1": per_class["Fight"]["f1"],
        "precision": per_class["Fight"]["precision"],
        "recall": per_class["Fight"]["recall"],
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class,
        "confusion": cm.tolist(),
    }
    if fight_scores is not None:
        out["auc"] = roc_auc(labels, fight_scores)
    return out


def format_report(m: dict) -> str:
    lines = [f"{'':10s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'support':>8s}"]
    for name in CLASS_NAMES:
        c = m["per_class"][name]
        lines.append(f"{name:10s} {c['precision']:9.3f} {c['recall']:7.3f} {c['f1']:6.3f} "
                     f"{c['support']:8d}")
    lines.append(f"accuracy {m['accuracy']:.4f}   fight-F1 {m['f1']:.4f}   macro-F1 "
                 f"{m['macro_f1']:.4f}" + (f"   AUC {m['auc']:.4f}" if "auc" in m else ""))
    (tn, fp), (fn, tp) = m["confusion"]
    lines.append(f"confusion [[TN {tn}, FP {fp}], [FN {fn}, TP {tp}]]")
    return "\n".join(lines)
