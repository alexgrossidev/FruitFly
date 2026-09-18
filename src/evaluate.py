"""Metrics: per-label accuracy, F1, and balanced accuracy, plus the trivial
"always predict 1" baseline from src.task.baseline_positive_rate() for
comparison. Used by src/train.py after each milestone and written into
MILESTONES.md by hand (not auto-synced -- keep MILESTONES.md as the
human-readable source of truth).
"""

from __future__ import annotations

import math

import torch

LABEL_NAMES = ("has_six", "has_seven")
METRIC_NAMES = ("accuracy", "f1", "balanced_accuracy", "precision", "recall")


def _metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = (recall + specificity) / 2
    return {
        "accuracy": accuracy,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "n": tp + tn + fp + fn,
        "positives": tp + fn,
    }


def per_label_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, dict[str, float]]:
    """logits, labels: [n, 2] (has_six, has_seven). Returns per-label accuracy/F1/balanced accuracy."""
    preds = (logits.sigmoid() > 0.5).float()
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(LABEL_NAMES):
        p, y = preds[:, i], labels[:, i]
        tp = int(((p == 1) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        out[name] = _metrics_from_counts(tp, tn, fp, fn)
    return out


def constant_positive_baseline(labels: torch.Tensor) -> dict[str, dict[str, float]]:
    """Metrics of a predictor that always says 1, on this exact label set.

    Its accuracy is the empirical positive rate (~0.6513 in expectation), its
    balanced accuracy is exactly 0.5, and its F1 is 2p / (1 + p) ~ 0.789.
    """
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(LABEL_NAMES):
        y = labels[:, i]
        tp, fn = int((y == 1).sum()), 0
        fp, tn = int((y == 0).sum()), 0
        out[name] = _metrics_from_counts(tp, tn, fp, fn)
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; 0.0 for a single value)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def aggregate(runs: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    """Per-label, per-metric mean/std/min/max across seeds."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for label in LABEL_NAMES:
        out[label] = {}
        for metric in METRIC_NAMES:
            values = [run[label][metric] for run in runs]
            mean, std = mean_std(values)
            out[label][metric] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "n_seeds": len(values),
            }
    return out


def format_metrics(metrics: dict[str, dict[str, float]]) -> str:
    """One line per label: acc / F1 / balanced acc."""
    return "  ".join(
        f"{label}: acc {m['accuracy']:.4f} f1 {m['f1']:.4f} bal {m['balanced_accuracy']:.4f}"
        for label, m in metrics.items()
    )
