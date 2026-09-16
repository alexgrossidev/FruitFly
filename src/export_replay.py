"""Export a trained model's activity trajectory into the replay JSON schema
fly-connectome-template's validator enforces. See README.md's "Visualization
bridge" section for the exact schema and constraints (verified against
viewer/src/lib/replay.ts, not assumed from prose) before changing this file.

Key points, restated so they aren't rediscovered the hard way:
- call the ConnectomeRNN core directly (not the wrapped ConnectomeModel), so
  you get raw per-step activity for every output node instead of the
  decoder's collapsed logits.
- map output node indices back to MaleCNS body IDs via core.output_index /
  brain.node_ids.
- normalize activity to [0, 1] and record exactly how in source.normalization
  -- the validator requires a non-empty, honest string there.
- first frame time must be 0, every later timestamp strictly increasing.
- every body ID used must be in the viewer's visible set (groups.bin < 3) --
  cross-check a handful by hand before trusting a full export, since the
  Python-side dataset loader and the viewer's bundled atlas load the ID space
  from different files.
"""

from __future__ import annotations

import json
from pathlib import Path

REPLAY_VERSION = 1
DATASET = "male-cns:v1.0"


def build_replay(
    *,
    body_ids: list[int],
    activity_per_step: list[list[float]],  # [step][node], already normalized to [0, 1]
    model_name: str,
    normalization: str,
) -> dict:
    """activity_per_step[t] must have the same length and order as body_ids."""
    frames = [
        {"time": t, "values": [[bid, val] for bid, val in zip(body_ids, row, strict=True)]}
        for t, row in enumerate(activity_per_step)
    ]
    return {
        "version": REPLAY_VERSION,
        "dataset": DATASET,
        "source": {
            "kind": "predicted",
            "name": model_name,
            "normalization": normalization,
        },
        "frames": frames,
    }


def write_replay(path: Path, replay: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replay, indent=2))


if __name__ == "__main__":
    raise SystemExit(
        "wire this up to a trained checkpoint once Milestone 1/2 is done -- "
        "see README.md Milestone 3."
    )
