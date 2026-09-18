"""Training loop and control-condition harness. Run via `python -m src.train --milestone N`.

Milestone 2 must run the real connectome plus both controls
(connectorch.transforms.degree_preserving_rewire, .random_topology) at a
matched trainable-parameter budget (ct.nn.count_parameters), across >=5 seeds
each. Do not report Milestone 2 as done with fewer seeds or missing controls
-- see README.md's Milestone 2 and the "single seed is not a result" gotcha.

Seeding convention
------------------
For a run with seed ``s``:
- ``torch.manual_seed(s)`` is set before the model is built, so encoder/decoder
  initialisation depends on ``s``;
- the control graphs are rewired with ``seed=s``;
- the train/val/test data come from a generator seeded with ``1000 + s``, so
  every condition sees the *same* data for a given seed and the only thing
  that differs between conditions is the wiring in the middle.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import connectorch as ct
import torch
from torch import nn

from .evaluate import aggregate, constant_positive_baseline, format_metrics, per_label_metrics
from .model import (
    CONDITIONS,
    ModelConfig,
    build_malecns_sample_model,
    build_model,
    build_toy_model,
    checkpoint_payload,
    malecns_sample_brain,
)
from .task import LIST_LENGTH, baseline_positive_rate, make_batch, one_hot_sequence

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    # README.md suggests starting around 2e-3; the sweep in MILESTONES.md
    # (Milestone 1) found 1e-2 converges in 1-4 epochs where 2e-3 needs 8+.
    lr: float = 1e-2
    train_size: int = 16384
    val_size: int = 2048
    test_size: int = 4096
    clip_grad_norm: float = 0.0  # 0 disables clipping
    data_seed_offset: int = 1000

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_datasets(
    seed: int, cfg: TrainConfig
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fixed train/val/test splits. Returns {split: (digits, x_onehot, labels)}."""
    generator = torch.Generator().manual_seed(cfg.data_seed_offset + seed)
    out = {}
    for split, n in (("train", cfg.train_size), ("val", cfg.val_size), ("test", cfg.test_size)):
        digits, labels = make_batch(n, generator=generator)
        out[split] = (digits, one_hot_sequence(digits), labels)
    return out


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i : i + batch_size]) for i in range(0, x.shape[0], batch_size)])


@torch.no_grad()
def evaluate_split(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, dict]:
    logits = predict(model, x)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
    return loss, per_label_metrics(logits, y)


def _mean_balanced_accuracy(metrics: dict) -> float:
    return sum(m["balanced_accuracy"] for m in metrics.values()) / len(metrics)


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------


def train_one(
    model: ct.nn.ConnectomeModel,
    *,
    seed: int,
    cfg: TrainConfig,
    log=print,
    tag: str = "",
) -> dict:
    """Train ``model`` on the 10-digit task and evaluate on held-out data.

    Model selection uses the validation split (best mean balanced accuracy
    across the two labels); the reported numbers are on the test split, which
    is never used for anything else. The connectome's ``edge_index`` is
    asserted bit-identical before and after training.
    """
    data = make_datasets(seed, cfg)
    _, x_train, y_train = data["train"]
    _, x_val, y_val = data["val"]
    _, x_test, y_test = data["test"]
    assert x_train.shape[1] == LIST_LENGTH

    wiring_before = model.core.edge_index.clone()
    diagnostics_before = model.core.diagnostics()
    parameters = ct.nn.count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()
    shuffle = torch.Generator().manual_seed(seed)

    history: list[dict] = []
    best_score, best_epoch, best_state = -1.0, -1, copy.deepcopy(model.state_dict())
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0], generator=shuffle)
        total, count = 0.0, 0
        for i in range(0, perm.numel(), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[idx]), y_train[idx])
            loss.backward()
            if cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            total += loss.item() * idx.numel()
            count += idx.numel()
        train_loss = total / count
        val_loss, val_metrics = evaluate_split(model, x_val, y_val)
        score = _mean_balanced_accuracy(val_metrics)
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_balanced_accuracy_mean": score,
                "val_accuracy_has_six": val_metrics["has_six"]["accuracy"],
                "val_accuracy_has_seven": val_metrics["has_seven"]["accuracy"],
            }
        )
        log(
            f"{tag}epoch {epoch:>3}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
            f"val {format_metrics(val_metrics)}"
        )
    seconds = time.perf_counter() - start
    epochs_to_95 = next(
        (h["epoch"] + 1 for h in history if h["val_balanced_accuracy_mean"] >= 0.95), None
    )

    model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate_split(model, x_test, y_test)
    baseline = constant_positive_baseline(y_test)
    topology_unchanged = bool(torch.equal(wiring_before, model.core.edge_index))
    if not topology_unchanged:
        raise RuntimeError("connectome topology changed during training -- this must never happen")

    return {
        "seed": seed,
        "train_config": cfg.to_dict(),
        "parameters": parameters,
        "topology_unchanged": topology_unchanged,
        "diagnostics_before": diagnostics_before,
        "diagnostics_after": model.core.diagnostics(),
        "train_loss_first_epoch": history[0]["train_loss"],
        "train_loss_last_epoch": history[-1]["train_loss"],
        "best_epoch": best_epoch,
        "epochs_to_val_balanced_accuracy_95": epochs_to_95,
        "history": history,
        "test_loss": test_loss,
        "test": test_metrics,
        "baseline_always_positive_on_test": baseline,
        "seconds": seconds,
    }


def save_checkpoint(
    model, *, condition: str, seed: int, config: ModelConfig, result: dict, subdir: str = ""
) -> Path:
    directory = CHECKPOINT_DIR / subdir if subdir else CHECKPOINT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{condition}_seed{seed}.pt"
    payload = checkpoint_payload(
        model,
        condition=condition,
        seed=seed,
        config=config,
        extra={"test": result["test"], "best_epoch": result["best_epoch"]},
    )
    torch.save(payload, path)
    return path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Milestones
# ---------------------------------------------------------------------------


def run_milestone_0(args, model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    """Toy graph. Proves the data pipeline and training loop, not biology."""
    torch.manual_seed(args.seed)
    model = build_toy_model(args.seed, model_cfg)
    print(f"{model.core}\n")
    result = train_one(model, seed=args.seed, cfg=train_cfg, tag="[toy] ")
    decreased = result["train_loss_last_epoch"] < result["train_loss_first_epoch"]
    print(
        f"\nloss decreased: {decreased} "
        f"({result['train_loss_first_epoch']:.4f} -> {result['train_loss_last_epoch']:.4f})"
    )
    print(f"topology unchanged after training: {result['topology_unchanged']}")
    print(f"parameters: {result['parameters']}")
    print(f"test: {format_metrics(result['test'])}")
    print(f"baseline (always 1): {format_metrics(result['baseline_always_positive_on_test'])}")
    if not decreased:
        raise RuntimeError("training loss did not decrease on the toy graph")
    out = {"milestone": 0, "condition": "toy", "model_config": model_cfg.to_dict(), "run": result}
    write_json(RESULTS_DIR / "milestone0.json", out)
    save_checkpoint(model, condition="toy", seed=args.seed, config=model_cfg, result=result)
    return out


def run_milestone_1(args, model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    """Real connectome sample, one seed, no controls. Not yet a reportable result."""
    torch.manual_seed(args.seed)
    model = build_malecns_sample_model("real", args.seed, model_cfg)
    print(f"{model.core}\n")
    result = train_one(model, seed=args.seed, cfg=train_cfg, tag="[real] ")
    print(f"\ntopology unchanged after training: {result['topology_unchanged']}")
    print(f"parameters: {result['parameters']}")
    print(f"best epoch (by val balanced accuracy): {result['best_epoch']}")
    print(f"test:     {format_metrics(result['test'])}")
    print(f"baseline: {format_metrics(result['baseline_always_positive_on_test'])}")
    print(f"expected always-1 accuracy: {baseline_positive_rate():.4f}")
    out = {"milestone": 1, "condition": "real", "model_config": model_cfg.to_dict(), "run": result}
    write_json(RESULTS_DIR / "milestone1.json", out)
    path = save_checkpoint(model, condition="real", seed=args.seed, config=model_cfg, result=result)
    print(f"checkpoint: {path}")
    return out


def run_milestone_2(args, model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    """Real vs degree-preserving rewiring vs random topology, >=5 seeds each."""
    if args.seeds < 5:
        raise SystemExit("Milestone 2 needs at least 5 seeds per condition (README.md Gotchas).")
    seeds = list(range(args.seed, args.seed + args.seeds))
    real_edges = set(zip(*malecns_sample_brain("real").edge_index.tolist()))
    conditions: dict[str, dict] = {}
    for condition in CONDITIONS:
        runs, overlaps = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            brain = malecns_sample_brain(condition, seed)
            overlaps.append(
                len(real_edges & set(zip(*brain.edge_index.tolist()))) / len(real_edges)
            )
            model = build_model(brain, model_cfg)
            quiet = args.quiet
            result = train_one(
                model,
                seed=seed,
                cfg=train_cfg,
                tag=f"[{condition} s{seed}] ",
                log=(lambda *a, **k: None) if quiet else print,
            )
            save_checkpoint(
                model,
                condition=condition,
                seed=seed,
                config=model_cfg,
                result=result,
                subdir=Path(args.out_name).stem if args.out_name else "",
            )
            print(
                f"[{condition:>7} seed {seed}] edges {model.core.num_edges}  "
                f"params {result['parameters']}  best epoch {result['best_epoch']:>2}  "
                f"{result['seconds']:.0f}s  test {format_metrics(result['test'])}"
            )
            runs.append(result)
        reached = [r["epochs_to_val_balanced_accuracy_95"] for r in runs]
        conditions[condition] = {
            "parameters": runs[0]["parameters"],
            "num_edges": model.core.num_edges,
            "edge_overlap_with_real": {"per_seed": overlaps, "mean": sum(overlaps) / len(overlaps)},
            "epochs_to_val_balanced_accuracy_95": {
                "per_seed": reached,
                "n_reached": sum(r is not None for r in reached),
                "mean_when_reached": (
                    sum(r for r in reached if r is not None)
                    / max(1, sum(r is not None for r in reached))
                ),
            },
            "aggregate": aggregate([r["test"] for r in runs]),
            "baseline_always_positive": aggregate(
                [r["baseline_always_positive_on_test"] for r in runs]
            ),
            "per_seed": runs,
        }

    print("\n=== Milestone 2: test-set mean ± std across seeds ===")
    print(f"seeds: {seeds}")
    for condition, block in conditions.items():
        print(
            f"\n{condition}  (params {block['parameters']}, edges {block['num_edges']}, "
            f"edge overlap with real {block['edge_overlap_with_real']['mean']:.3f}, "
            f"epochs to 0.95 val bal-acc: {block['epochs_to_val_balanced_accuracy_95']['per_seed']})"
        )
        for label, metrics in block["aggregate"].items():
            print(
                f"  {label:>9}: "
                + "  ".join(
                    f"{name} {m['mean']:.4f}±{m['std']:.4f}"
                    for name, m in metrics.items()
                    if name in ("accuracy", "f1", "balanced_accuracy")
                )
            )
    out = {
        "milestone": 2,
        "seeds": seeds,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "expected_always_positive_accuracy": baseline_positive_rate(),
        "conditions": conditions,
    }
    write_json(RESULTS_DIR / (args.out_name or "metrics.json"), out)
    return out


def run_milestone_4(args, model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    from .full import run_full  # imported lazily: touching the full graph is slow

    return run_full(args, model_cfg, train_cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--milestone", type=int, required=True, choices=[0, 1, 2, 4])
    parser.add_argument("--seed", type=int, default=0, help="first (or only) seed")
    parser.add_argument("--seeds", type=int, default=5, help="seeds per condition for milestone 2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--clip", type=float, default=None, help="grad-norm clip, 0 = off")
    parser.add_argument("--leak", type=float, default=None)
    parser.add_argument("--num-input-nodes", type=int, default=None)
    parser.add_argument("--num-output-nodes", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument(
        "--init-row-sum", type=float, default=None, help="-1 = initializer's raw scale"
    )
    parser.add_argument("--quiet", action="store_true", help="milestone 2: no per-epoch logs")
    parser.add_argument(
        "--out-name", default=None, help="milestone 2: results file name (default metrics.json)"
    )
    # milestone 4 only
    parser.add_argument(
        "--download", action="store_true", help="milestone 4: allow the ~520 MB download"
    )
    parser.add_argument("--device", default=None, help="milestone 4: cpu or mps")
    parser.add_argument("--max-steps", type=int, default=None, help="milestone 4: optimizer steps")
    parser.add_argument("--time-budget", type=float, default=None, help="milestone 4: seconds")
    parser.add_argument("--conditions", default=None, help="milestone 4: comma list of real,random")
    parser.add_argument(
        "--accumulate", type=int, default=None, help="milestone 4: grad accumulation passes"
    )
    parser.add_argument(
        "--test-size", type=int, default=None, help="milestone 4: held-out examples"
    )
    parser.add_argument("--eval-batch", type=int, default=None, help="milestone 4: eval batch size")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        default=None,
        help="milestone 4: cap on activation memory",
    )
    return parser


def configs_from_args(args) -> tuple[ModelConfig, TrainConfig]:
    model_kwargs = {
        k: v
        for k, v in {
            "leak": args.leak,
            "num_input_nodes": args.num_input_nodes,
            "num_output_nodes": args.num_output_nodes,
            "settle_steps": args.settle_steps,
            "init_row_sum": (
                None
                if args.init_row_sum is not None and args.init_row_sum < 0
                else args.init_row_sum
            ),
        }.items()
        if v is not None or (k == "init_row_sum" and args.init_row_sum is not None)
    }
    train_kwargs = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_size": args.train_size,
            "clip_grad_norm": args.clip,
        }.items()
        if v is not None
    }
    if args.milestone == 0 and "epochs" not in train_kwargs:
        train_kwargs["epochs"] = 10
    return ModelConfig(**model_kwargs), TrainConfig(**train_kwargs)


def main() -> None:
    args = build_parser().parse_args()
    model_cfg, train_cfg = configs_from_args(args)
    print(f"model config: {model_cfg}\ntrain config: {train_cfg}\n")
    runner = {0: run_milestone_0, 1: run_milestone_1, 2: run_milestone_2, 4: run_milestone_4}[
        args.milestone
    ]
    runner(args, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
