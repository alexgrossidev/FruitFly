"""Encoder -> connectome core -> decoder, per README.md's Architecture section.

Three model-building functions, one per milestone family. Fill these in
following README.md exactly — node counts, weight strategy, activation and
leak are starting points from that document, not requirements; tune them if
training doesn't converge, but record what you changed and why in
MILESTONES.md.
"""

from __future__ import annotations

from torch import nn

import connectorch as ct

from .task import NUM_DIGITS


def build_encoder(num_input_nodes: int) -> nn.Module:
    """One-hot digit [*, NUM_DIGITS] -> connectome input drive [*, num_input_nodes]."""
    return nn.Sequential(nn.Linear(NUM_DIGITS, num_input_nodes), nn.Tanh())


def build_toy_model() -> ct.nn.ConnectomeModel:
    """Milestone 0: sanity pipeline on a non-biological graph.

    Use ct.datasets.random_sparse(...) or ct.datasets.chain(...). The point of
    this milestone is proving the data/model/training loop is correct before
    biology enters the picture — see README.md Milestone 0.
    """
    raise NotImplementedError


def build_malecns_sample_model() -> ct.nn.ConnectomeModel:
    """Milestone 1/2: ct.datasets.malecns_sample(), 100 real neurons, offline.

    input_nodes / output_nodes are arbitrary index slices to start (see the
    connectorch MNIST example for precedent); weights="trainable",
    initializer="normalized_synapse_count" is the documented safe default.
    decoder must output 2 logits (has_six, has_seven) — BCEWithLogitsLoss,
    not CrossEntropyLoss.
    """
    raise NotImplementedError


def build_full_malecns_model() -> ct.nn.ConnectomeModel:
    """Milestone 4 (stretch): ct.datasets.malecns(download=True), 164,587 neurons.

    Read README.md's Gotchas section before calling this — check
    core.activation_bytes(batch, steps) and core.diagnostics() before training,
    never pass backend="dense".
    """
    raise NotImplementedError
