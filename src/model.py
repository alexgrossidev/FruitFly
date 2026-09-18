"""Encoder -> connectome core -> decoder, per README.md's Architecture section.

One builder, ``build_model``, takes a Connectome and a ``ModelConfig`` and
returns the three-part ``ct.nn.ConnectomeModel``. The milestone-specific
helpers below only decide *which* connectome goes in the middle:

- ``build_toy_model``             Milestone 0, a non-biological random graph
- ``build_malecns_sample_model``  Milestones 1-3, the bundled 100-neuron sample
                                  and its two topology-matched controls
- ``build_full_malecns_model``    Milestone 4, the full 164,587-neuron graph

Every condition in Milestone 2 goes through the same ``build_model`` with the
same ``ModelConfig``, so the encoder/decoder sizes and the number of trainable
connectome weights are matched by construction: the controls keep the exact
edge count of the graph they are a control for (see
``connectorch.transforms.rewire``).

Hyperparameters here are starting points from README.md, not requirements.
If you change one, record what and why in MILESTONES.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import connectorch as ct
import numpy as np
import torch
from connectorch.transforms import degree_preserving_rewire, random_topology
from torch import nn

from .task import NUM_DIGITS

CONDITIONS = ("real", "rewired", "random")


@dataclass(frozen=True)
class ModelConfig:
    """Everything about the model that is not the connectome itself."""

    num_input_nodes: int = 16
    num_output_nodes: int = 16
    leak: float = 0.4
    activation: str = "tanh"
    initializer: str = "normalized_synapse_count"
    bias: bool = False
    # After the named initializer runs, rescale every starting weight by one
    # scalar so the largest absolute row sum of A (the biggest total recurrent
    # drive any neuron can receive) equals this value. None keeps the
    # initializer's raw scale. On the dense 100-neuron sample the raw scale
    # gives row sums of ~36, which saturates tanh on the first step and makes
    # the sub-unit input drive invisible -- see MILESTONES.md, Milestone 0.
    # This is a numerical choice about the starting point only; the weights
    # are trainable and the topology is untouched.
    init_row_sum: float | None = 1.0
    # Silent recurrent steps with zero drive appended after the 10 digits, so
    # the decoder ("last" readout) sees a state that the final digit has had
    # time to reach. Input and output neurons are disjoint slices, so a digit
    # presented at step t can only influence an output neuron at step t+1 or
    # later; with 0 settle steps the tenth digit is invisible to the decoder
    # and accuracy is capped at ~96.1% (1 - 0.1 * 0.9**9 of lists have their
    # only 6, or only 7, in the last slot). See MILESTONES.md, Milestone 1.
    settle_steps: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def build_encoder(num_input_nodes: int) -> nn.Module:
    """One-hot digit [*, NUM_DIGITS] -> connectome input drive [*, num_input_nodes].

    Applied to a [batch, steps, NUM_DIGITS] tensor it acts on the last axis, so
    each of the 10 digits gets the same linear map and the sequence axis is
    preserved for the core to consume one slice per recurrent step.
    """
    return nn.Sequential(nn.Linear(NUM_DIGITS, num_input_nodes), nn.Tanh())


class SettlingEncoder(nn.Module):
    """Encode a [batch, 10, NUM_DIGITS] sequence, then append zero-drive steps.

    The core reads ``x.shape[1]`` as the number of recurrent steps, so padding
    the *encoded* drive with ``settle_steps`` all-zero slices runs the
    recurrence that many extra times with no external input. Every digit is
    still presented one per step, in order, exactly once.
    """

    def __init__(self, encoder: nn.Module, settle_steps: int) -> None:
        super().__init__()
        if settle_steps < 0:
            raise ValueError("settle_steps must be >= 0")
        self.encoder = encoder
        self.settle_steps = int(settle_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [batch, steps, features], got {tuple(x.shape)}")
        drive = self.encoder(x)
        if self.settle_steps == 0:
            return drive
        silence = drive.new_zeros(drive.shape[0], self.settle_steps, drive.shape[2])
        return torch.cat([drive, silence], dim=1)

    def extra_repr(self) -> str:
        return f"settle_steps={self.settle_steps}"


def build_decoder(num_output_nodes: int) -> nn.Module:
    """Final-step readout [*, num_output_nodes] -> [has_six_logit, has_seven_logit]."""
    return nn.Linear(num_output_nodes, 2)


def input_output_nodes(brain: ct.Connectome, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Arbitrary, disjoint index slices of the node id list, as README.md suggests.

    The first ``num_input_nodes`` ids receive the drive; the last
    ``num_output_nodes`` are read out. Node ids are the biological body ids, so
    the same neurons play the same roles in every control condition.
    """
    if config.num_input_nodes + config.num_output_nodes > brain.num_nodes:
        raise ValueError(
            f"{config.num_input_nodes} input + {config.num_output_nodes} output nodes "
            f"do not fit in a {brain.num_nodes}-node connectome without overlapping."
        )
    return (
        brain.node_ids[: config.num_input_nodes],
        brain.node_ids[-config.num_output_nodes :],
    )


def build_core(
    brain: ct.Connectome,
    config: ModelConfig,
    *,
    output_nodes: np.ndarray | None = None,
) -> ct.nn.ConnectomeRNN:
    """The connectome runtime. ``output_nodes=None`` keeps the config's slice.

    ``src/export_replay.py`` passes an explicit ``output_nodes`` to read out
    every neuron for visualisation; the trained weights are then copied across.
    """
    inputs, default_outputs = input_output_nodes(brain, config)
    core = ct.nn.ConnectomeRNN(
        brain,
        input_nodes=inputs,
        output_nodes=default_outputs if output_nodes is None else output_nodes,
        weights="trainable",
        initializer=config.initializer,
        activation=config.activation,
        leak=config.leak,
        bias=config.bias,
        backend="auto",  # never "dense" -- see README.md Gotchas
    )
    if config.init_row_sum is not None:
        rescale_initial_weights(core, config.init_row_sum)
    return core


@torch.no_grad()
def rescale_initial_weights(core: ct.nn.ConnectomeRNN, target_row_sum: float) -> float:
    """Multiply the starting weights by one scalar so max |row sum| == target.

    Returns the gain applied. Only the values change; ``edge_index`` and the
    parameter count are untouched, so the control conditions stay matched.
    """
    current = float(core.diagnostics()["max_abs_row_sum"])
    if current <= 0:
        return 1.0
    gain = target_row_sum / current
    core.weights.weight.mul_(gain)
    return gain


def build_model(brain: ct.Connectome, config: ModelConfig | None = None) -> ct.nn.ConnectomeModel:
    config = config or ModelConfig()
    core = build_core(brain, config)
    return ct.nn.ConnectomeModel(
        core,
        encoder=SettlingEncoder(build_encoder(core.num_input_nodes), config.settle_steps),
        decoder=build_decoder(core.num_output_nodes),
        readout="last",  # decode only after all 10 digits (and the settle steps)
    )


# ---------------------------------------------------------------------------
# Connectome selection, one helper per milestone family
# ---------------------------------------------------------------------------


def toy_brain(seed: int = 0, *, num_nodes: int = 100, num_edges: int = 4000) -> ct.Connectome:
    """Milestone 0: a random sparse graph. Not biology; wiring-correctness only."""
    return ct.datasets.random_sparse(num_nodes=num_nodes, num_edges=num_edges, seed=seed)


def malecns_sample_brain(condition: str = "real", seed: int = 0) -> ct.Connectome:
    """The bundled 100-neuron MaleCNS sample, or one of its two controls.

    ``"real"``     the measured wiring
    ``"rewired"``  degree-preserving double-edge-swap rewiring (same in/out
                   degree for every neuron, same edge count, different wiring)
    ``"random"``   random graph with the same node and edge count only

    The control graphs keep the original node ids, so the input/output slices
    chosen by ``input_output_nodes`` refer to the same body ids in every
    condition. Every control's provenance records that it is not biological.
    """
    brain = ct.datasets.malecns_sample()
    if condition == "real":
        return brain
    if condition == "rewired":
        return degree_preserving_rewire(brain, seed=seed)
    if condition == "random":
        return random_topology(brain, seed=seed)
    raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")


def build_toy_model(seed: int = 0, config: ModelConfig | None = None) -> ct.nn.ConnectomeModel:
    """Milestone 0: sanity pipeline on a non-biological graph."""
    return build_model(toy_brain(seed), config)


def build_malecns_sample_model(
    condition: str = "real", seed: int = 0, config: ModelConfig | None = None
) -> ct.nn.ConnectomeModel:
    """Milestones 1-3: ct.datasets.malecns_sample() (100 real neurons, offline)."""
    return build_model(malecns_sample_brain(condition, seed), config)


def full_malecns_brain(*, download: bool | None = None) -> ct.Connectome:
    """Milestone 4 (stretch): the full graph, ~520 MB on first use. Not before M4."""
    return ct.datasets.malecns(download=download)


def build_full_malecns_model(
    config: ModelConfig | None = None, *, download: bool | None = None
) -> ct.nn.ConnectomeModel:
    """Milestone 4 (stretch): 164,587 neurons, 25,563,197 connections.

    Check ``model.core.activation_bytes(batch, steps)`` and
    ``model.core.diagnostics()['max_abs_row_sum']`` before training; never pass
    ``backend="dense"``. Input/output slices are the same arbitrary id-order
    slices as the sample; choosing populations by biological role is a further
    option discussed in README.md.
    """
    return build_model(full_malecns_brain(download=download), config)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def checkpoint_payload(
    model: ct.nn.ConnectomeModel, *, condition: str, seed: int, config: ModelConfig, extra: dict
) -> dict:
    """What ``torch.save`` writes: enough to rebuild the identical model."""
    return {
        "state_dict": model.state_dict(),
        "condition": condition,
        "seed": seed,
        "config": config.to_dict(),
        "fingerprint": model.core.fingerprint,
        **extra,
    }


def load_checkpoint(path) -> tuple[ct.nn.ConnectomeModel, dict]:
    """Rebuild the model a checkpoint came from and load its weights into it."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload["config"])
    condition, seed = payload["condition"], int(payload["seed"])
    if condition == "toy":
        model = build_toy_model(seed, config)
    else:
        model = build_malecns_sample_model(condition, seed, config)
    if model.core.fingerprint != payload["fingerprint"]:
        raise RuntimeError(
            f"checkpoint {path} was trained on connectome {payload['fingerprint']} "
            f"but the rebuilt model has {model.core.fingerprint}."
        )
    model.load_state_dict(payload["state_dict"])
    return model, payload
