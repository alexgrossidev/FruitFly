"""Export a trained model's activity trajectory into the replay JSON schema
fly-connectome-template's validator enforces. See README.md's "Visualization
bridge" section for the exact schema and constraints (verified against
viewer/src/lib/replay.ts, not assumed from prose) before changing this file.

    python -m src.export_replay                       # results/checkpoints/real_seed0.pt
    python -m src.export_replay --checkpoint results/checkpoints/real_seed3.pt
    node scripts/validate_replay.mjs results/replays/*.json   # the viewer's own validator

What gets exported, per example list:
- the core is run directly (not the wrapped ConnectomeModel) on the encoded
  10-digit sequence, with gradients off, through a second ConnectomeRNN that
  reads out *every* neuron and carries the trained weights, so the replay
  shows all 100 sample neurons rather than the 16 the decoder sees. The
  16-neuron readout the decoder actually used is listed in the file's
  ``experiment`` block, and the full trajectory restricted to those 16
  neurons is asserted equal to what the trained core itself produces.
- one frame per digit, ``SECONDS_PER_DIGIT`` apart, starting at time 0; the
  value for a neuron at frame t is the normalised state after digit t has
  been presented (see ``NORMALIZATION`` -- it is declared in the JSON and the
  viewer requires the declaration to be honest).
- every neuron is listed in every frame explicitly, because the viewer treats
  an omitted id as zero for that frame, not as "hold".

Body ids: the sample's node ids are MaleCNS body ids and all 100 of them are in
the viewer's visible selection (groups.bin < 3); ``validate_replay`` checks
that against the atlas files, and scripts/validate_replay.mjs re-checks it
with the viewer's own TypeScript.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import connectorch as ct
import numpy as np
import torch

from .evaluate import LABEL_NAMES
from .model import ModelConfig, build_core, load_checkpoint, malecns_sample_brain
from .task import LIST_LENGTH, one_hot_sequence
from .train import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, make_datasets, predict

REPLAY_VERSION = 1
DATASET = "male-cns:v1.0"
ATLAS_DIR = Path(__file__).resolve().parent.parent / "viewer" / "public" / "data" / "brain-atlas"
REPLAY_DIR = RESULTS_DIR / "replays"
SECONDS_PER_DIGIT = 1.0

#: Declared in every exported file. Keep this string and ``normalize`` in sync.
NORMALIZATION = (
    "|h_t|: absolute value of each neuron's leaky-tanh state after recurrent step t. "
    "h_t is in [-1, 1] by construction (leak-weighted average of tanh outputs from a "
    "zero initial state), so this is a fixed rescale to [0, 1] with no per-run "
    "min-max; sign is discarded; 0 = rest. One frame per recurrent step, "
    f"{SECONDS_PER_DIGIT:g} s apart: the first 10 steps each present one digit of the "
    "list, any further frames are silent zero-drive settle steps (count in "
    "experiment.settle_steps)."
)

#: The four representative cases README.md's Milestone 3 asks for.
CASES = {
    "six_only": (1.0, 0.0),
    "seven_only": (0.0, 1.0),
    "both": (1.0, 1.0),
    "neither": (0.0, 0.0),
}


def normalize(states: torch.Tensor) -> torch.Tensor:
    """[..., nodes] leaky-tanh states in [-1, 1] -> [0, 1]. Must match NORMALIZATION."""
    return states.abs().clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Replay file construction and validation
# ---------------------------------------------------------------------------


def build_replay(
    *,
    body_ids: list[int],
    activity_per_step: list[list[float]],  # [step][node], already normalized to [0, 1]
    model_name: str,
    normalization: str = NORMALIZATION,
    seconds_per_step: float = SECONDS_PER_DIGIT,
    extra: dict | None = None,
) -> dict:
    """activity_per_step[t] must have the same length and order as body_ids."""
    frames = [
        {
            "time": t * seconds_per_step,
            "values": [[int(bid), float(val)] for bid, val in zip(body_ids, row, strict=True)],
        }
        for t, row in enumerate(activity_per_step)
    ]
    replay = {
        "version": REPLAY_VERSION,
        "dataset": DATASET,
        "source": {
            "kind": "predicted",
            "name": model_name,
            "normalization": normalization,
        },
        "frames": frames,
    }
    if extra:
        # The viewer ignores keys it does not know (verified in replay.ts).
        replay["experiment"] = extra
    return replay


def load_visible_ids(atlas_dir: Path = ATLAS_DIR) -> set[int]:
    """The viewer's visible brain selection, decoded exactly as viewer/src/lib/atlas.ts does."""
    manifest = json.loads((atlas_dir / "manifest.json").read_text())
    ids = np.fromfile(atlas_dir / "ids.bin", dtype="<u4")
    groups = np.fromfile(atlas_dir / "groups.bin", dtype="u1")
    if ids.size != manifest["count"] or groups.size != manifest["count"]:
        raise ValueError("atlas file lengths do not match the manifest")
    visible = set(ids[groups < 3].tolist())
    if len(visible) != manifest["brainCount"]:
        raise ValueError("brain selection does not match the manifest")
    return visible


def validate_replay(replay: dict, visible_ids: set[int]) -> None:
    """Python mirror of parseReplay() in viewer/src/lib/replay.ts. Raises ValueError.

    The authoritative check is the TypeScript itself (scripts/validate_replay.mjs);
    this exists so the Python side fails fast and so pytest can cover it.
    """
    if (
        not isinstance(replay, dict)
        or replay.get("version") != 1
        or replay.get("dataset") != DATASET
    ):
        raise ValueError("Expected replay version 1 for male-cns:v1.0.")
    source = replay.get("source")
    if (
        not isinstance(source, dict)
        or source.get("kind") not in ("synthetic", "predicted", "measured")
        or not isinstance(source.get("name"), str)
        or not source["name"].strip()
        or not isinstance(source.get("normalization"), str)
        or not source["normalization"].strip()
    ):
        raise ValueError("Declare source kind, name and normalization.")
    frames = replay.get("frames")
    if not isinstance(frames, list) or not 2 <= len(frames) <= 10_000:
        raise ValueError("Supply 2 to 10,000 frames.")
    previous = -1.0
    for frame in frames:
        time = frame.get("time") if isinstance(frame, dict) else None
        if (
            not isinstance(time, (int, float))
            or isinstance(time, bool)
            or not math.isfinite(time)
            or time < 0
            or time <= previous
        ):
            raise ValueError("Frame times must be finite, nonnegative and strictly increasing.")
        previous = time
        values = frame.get("values")
        if not isinstance(values, list) or len(values) > len(visible_ids):
            raise ValueError("Invalid frame values.")
        seen: set[int] = set()
        for pair in values:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], int)
                or isinstance(pair[0], bool)
                or pair[0] not in visible_ids
            ):
                raise ValueError("Every value must use a visible MaleCNS brain body ID.")
            if pair[0] in seen:
                raise ValueError("Duplicate body ID in a frame.")
            value = pair[1]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or value > 1
            ):
                raise ValueError("Activity values must be finite and normalized to [0, 1].")
            seen.add(pair[0])
    if frames[0]["time"] != 0:
        raise ValueError("The first frame must start at time 0.")


def write_replay(path: Path, replay: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replay, indent=None, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Running the trained core and reading out every neuron
# ---------------------------------------------------------------------------


def full_readout_core(
    model: ct.nn.ConnectomeModel, brain: ct.Connectome, config: ModelConfig
) -> ct.nn.ConnectomeRNN:
    """A copy of the trained core that reads out all nodes instead of the decoder's slice."""
    core = build_core(brain, config, output_nodes=brain.node_ids)
    with torch.no_grad():
        core.weights.weight.copy_(model.core.weights.weight)
        if model.core.bias is not None:
            core.bias.copy_(model.core.bias)
    if not torch.equal(core.edge_index, model.core.edge_index):
        raise RuntimeError("full-readout core has a different topology from the trained core")
    if not torch.equal(core.edge_weight, model.core.edge_weight):
        raise RuntimeError("full-readout core did not receive the trained weights")
    if not torch.equal(core.input_index, model.core.input_index):
        raise RuntimeError("full-readout core has different input nodes from the trained core")
    return core


@torch.no_grad()
def trajectory(
    model: ct.nn.ConnectomeModel, core_all: ct.nn.ConnectomeRNN, digits: torch.Tensor
) -> torch.Tensor:
    """[LIST_LENGTH] digits -> [LIST_LENGTH, num_nodes] raw states (before normalisation)."""
    model.eval()
    core_all.eval()
    drive = model.encoder(one_hot_sequence(digits.view(1, LIST_LENGTH)))  # [1, 10, num_input]
    states = core_all(drive)  # [1, 10, num_nodes]
    # Consistency: the trained core's own readout must equal the same columns here.
    own = model.core(drive)  # [1, 10, num_output]
    if not torch.allclose(states[:, :, model.core.output_index], own, atol=1e-6):
        raise RuntimeError("full-readout trajectory disagrees with the trained core's readout")
    return states[0]


@torch.no_grad()
def pick_examples(
    model: ct.nn.ConnectomeModel, seed: int, train_cfg: TrainConfig
) -> dict[str, dict]:
    """First *correctly classified* held-out example of each case in CASES.

    Uses the same seeded test split the checkpoint was evaluated on, so the
    picked lists are held-out data, and deterministic for a given checkpoint.
    """
    digits, x, labels = make_datasets(seed, train_cfg)["test"]
    logits = predict(model, x)
    preds = (logits.sigmoid() > 0.5).float()
    correct = (preds == labels).all(dim=1)
    picked = {}
    for case, target in CASES.items():
        target_t = torch.tensor(target)
        mask = correct & (labels == target_t).all(dim=1)
        index = int(mask.nonzero()[0]) if mask.any() else None
        if index is None:
            # Fall back to the first example of the case even if misclassified,
            # and say so in the export rather than silently skipping the case.
            mask = (labels == target_t).all(dim=1)
            index = int(mask.nonzero()[0])
        picked[case] = {
            "index": index,
            "digits": digits[index].tolist(),
            "labels": {n: float(labels[index, i]) for i, n in enumerate(LABEL_NAMES)},
            "logits": {n: float(logits[index, i]) for i, n in enumerate(LABEL_NAMES)},
            "probabilities": {
                n: float(logits[index, i].sigmoid()) for i, n in enumerate(LABEL_NAMES)
            },
            "predicted": {n: float(preds[index, i]) for i, n in enumerate(LABEL_NAMES)},
            "correct": bool(correct[index]),
        }
    return picked


def export_checkpoint(
    checkpoint: Path,
    out_dir: Path = REPLAY_DIR,
    *,
    seconds_per_digit: float = SECONDS_PER_DIGIT,
    train_cfg: TrainConfig | None = None,
) -> list[Path]:
    model, payload = load_checkpoint(checkpoint)
    condition, seed = payload["condition"], int(payload["seed"])
    config = ModelConfig.from_dict(payload["config"])
    if condition == "toy":
        raise SystemExit("toy graphs have no MaleCNS body ids; export a malecns_sample checkpoint")
    brain = malecns_sample_brain(condition, seed)
    core_all = full_readout_core(model, brain, config)
    visible = load_visible_ids()
    body_ids = [int(i) for i in brain.node_ids[core_all.output_index.numpy()]]
    missing = [i for i in body_ids if i not in visible]
    if missing:
        raise RuntimeError(
            f"{len(missing)} body ids are not in the viewer's visible set: {missing[:10]}"
        )
    input_ids = [int(i) for i in brain.node_ids[model.core.input_index.numpy()]]
    readout_ids = [int(i) for i in brain.node_ids[model.core.output_index.numpy()]]

    examples = pick_examples(model, seed, train_cfg or TrainConfig())
    written = []
    index = {
        "checkpoint": str(checkpoint),
        "condition": condition,
        "seed": seed,
        "normalization": NORMALIZATION,
        "seconds_per_digit": seconds_per_digit,
        "input_body_ids": input_ids,
        "readout_body_ids": readout_ids,
        "num_neurons_exported": len(body_ids),
        "files": {},
    }
    for case, info in examples.items():
        states = trajectory(model, core_all, torch.tensor(info["digits"]))
        activity = normalize(states)
        rows = [[round(float(v), 5) for v in row] for row in activity.tolist()]
        name = (
            f"fly-number-recognition: {condition} MaleCNS sample (100 neurons, 3,630 edges), "
            f"seed {seed} checkpoint; case '{case}', digits {info['digits']}, "
            f"predicted has_six={info['predicted']['has_six']:.0f} "
            f"has_seven={info['predicted']['has_seven']:.0f} "
            f"({'correct' if info['correct'] else 'INCORRECT'})"
        )
        replay = build_replay(
            body_ids=body_ids,
            activity_per_step=rows,
            model_name=name,
            seconds_per_step=seconds_per_digit,
            extra={
                "case": case,
                **info,
                "checkpoint": str(checkpoint.name),
                "condition": condition,
                "seed": seed,
                "input_body_ids": input_ids,
                "readout_body_ids": readout_ids,
                "settle_steps": config.settle_steps,
                "frame_t_means": (
                    f"frames 0-{LIST_LENGTH - 1}: state after digit t of the list is presented; "
                    f"frames {LIST_LENGTH}-{LIST_LENGTH + config.settle_steps - 1}: silent settle steps"
                    if config.settle_steps
                    else "frame t: state after digit t of the list is presented"
                ),
                "raw_state_range": [float(states.min()), float(states.max())],
            },
        )
        validate_replay(replay, visible)
        path = out_dir / f"{condition}_seed{seed}_{case}.json"
        write_replay(path, replay)
        written.append(path)
        index["files"][case] = {"path": str(path.relative_to(RESULTS_DIR.parent)), **info}
        print(
            f"wrote {path}  digits={info['digits']}  "
            f"pred=({info['predicted']['has_six']:.0f},{info['predicted']['has_seven']:.0f}) "
            f"{'correct' if info['correct'] else 'INCORRECT'}  frames={len(rows)}  "
            f"activity range [{activity.min():.3f}, {activity.max():.3f}]"
        )
    index_path = (
        out_dir.parent / f"{out_dir.name}_index.json"
    )  # not a replay: keep it out of the glob
    index_path.write_text(json.dumps(index, indent=2))
    print(f"wrote {index_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "real_seed0.pt")
    parser.add_argument("--out", type=Path, default=REPLAY_DIR)
    parser.add_argument("--seconds-per-digit", type=float, default=SECONDS_PER_DIGIT)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(
            f"no checkpoint at {args.checkpoint}; run `python -m src.train --milestone 1` first"
        )
    export_checkpoint(args.checkpoint, args.out, seconds_per_digit=args.seconds_per_digit)


if __name__ == "__main__":
    main()
