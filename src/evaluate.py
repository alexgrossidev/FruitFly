"""Metrics: per-label accuracy, F1, and balanced accuracy, plus the baseline
from src.task.baseline_positive_rate() for comparison. Used by src/train.py
after each milestone and written into MILESTONES.md by hand (not auto-synced
-- keep MILESTONES.md as the human-readable source of truth).
"""

from __future__ import annotations

import torch


def per_label_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, dict[str, float]]:
    """logits, labels: [n, 2] (has_six, has_seven). Returns per-label accuracy/F1/balanced accuracy."""
    preds = (logits.sigmoid() > 0.5).float()
    names = ["has_six", "has_seven"]
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        p, y = preds[:, i], labels[:, i]
        tp = ((p == 1) & (y == 1)).sum().item()
        tn = ((p == 0) & (y == 0)).sum().item()
        fp = ((p == 1) & (y == 0)).sum().item()
        fn = ((p == 0) & (y == 1)).sum().item()
        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        specificity = tn / max(1, tn + fp)
        balanced_accuracy = (recall + specificity) / 2
        out[name] = {
            "accuracy": accuracy,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
        }
    return out
