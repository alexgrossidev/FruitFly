"""Hyperparameter sweep on the real sample, seed 0, 8 short epochs. Diagnostic only, not a result.

    python scripts/sweep_milestone1.py 8 > results/sweep_milestone1.txt

Run from the repo root with the venv active. Output for the run that chose the
defaults in src/model.py and src/train.py is checked in as results/sweep_milestone1.txt.
"""

import itertools
import json
import sys
import time

import torch

from src.model import ModelConfig, build_malecns_sample_model
from src.train import TrainConfig, train_one

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
grid = {
    "init_row_sum": [None, 4.0, 2.0, 1.0, 0.5],
    "lr": [2e-3, 1e-2],
    "leak": [0.4, 1.0],
    "bias": [False, True],
}
rows = []
for values in itertools.product(*grid.values()):
    cfg = dict(zip(grid.keys(), values))
    mc = ModelConfig(init_row_sum=cfg["init_row_sum"], leak=cfg["leak"], bias=cfg["bias"])
    tc = TrainConfig(epochs=EPOCHS, lr=cfg["lr"], train_size=8192, val_size=2048, test_size=2048)
    torch.manual_seed(0)
    model = build_malecns_sample_model("real", 0, mc)
    t0 = time.perf_counter()
    r = train_one(model, seed=0, cfg=tc, log=lambda *a, **k: None)
    bal = {k: v["balanced_accuracy"] for k, v in r["test"].items()}
    acc = {k: v["accuracy"] for k, v in r["test"].items()}
    row = {
        **cfg,
        "best_epoch": r["best_epoch"],
        "loss_last": r["train_loss_last_epoch"],
        "bal6": bal["has_six"],
        "bal7": bal["has_seven"],
        "acc6": acc["has_six"],
        "acc7": acc["has_seven"],
        "row_sum_after": r["diagnostics_after"]["max_abs_row_sum"],
        "s": time.perf_counter() - t0,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
rows.sort(key=lambda r: -(r["bal6"] + r["bal7"]))
print("\n=== top 10 by mean balanced accuracy ===")
for r in rows[:10]:
    print(json.dumps(r))
