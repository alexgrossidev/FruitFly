"""Model construction, control matching, and the topology invariant."""

from __future__ import annotations

import connectorch as ct
import numpy as np
import torch

from src.model import (
    CONDITIONS,
    ModelConfig,
    build_malecns_sample_model,
    build_toy_model,
    malecns_sample_brain,
)
from src.task import LIST_LENGTH, make_batch, one_hot_sequence
from src.train import TrainConfig, train_one

TINY = TrainConfig(epochs=2, batch_size=64, train_size=256, val_size=64, test_size=64)


def test_model_forward_shapes():
    torch.manual_seed(0)
    model = build_toy_model(0)
    digits, _ = make_batch(4, generator=torch.Generator().manual_seed(0))
    x = one_hot_sequence(digits)
    assert model(x).shape == (4, 2)
    trajectory = model.core(model.encoder(x))
    assert trajectory.shape == (
        4,
        LIST_LENGTH + ModelConfig().settle_steps,
        model.core.num_output_nodes,
    )


def test_settle_steps_append_silent_zero_drive():
    torch.manual_seed(0)
    model = build_toy_model(0, ModelConfig(settle_steps=3))
    digits, _ = make_batch(2, generator=torch.Generator().manual_seed(0))
    drive = model.encoder(one_hot_sequence(digits))
    assert drive.shape == (2, LIST_LENGTH + 3, model.core.num_input_nodes)
    assert torch.equal(drive[:, LIST_LENGTH:], torch.zeros_like(drive[:, LIST_LENGTH:]))
    assert torch.equal(drive[:, :LIST_LENGTH], model.encoder.encoder(one_hot_sequence(digits)))


def test_sequence_axis_is_the_step_axis():
    """A wrong stacking axis must raise, not silently run (README Gotchas)."""
    torch.manual_seed(0)
    model = build_toy_model(0)
    bad = torch.zeros(4, model.core.num_input_nodes, LIST_LENGTH)  # transposed
    try:
        model.core(bad)
    except ValueError:
        return
    raise AssertionError("core accepted a [batch, num_input_nodes, steps] tensor")


def test_topology_unchanged_after_training():
    torch.manual_seed(0)
    model = build_toy_model(0)
    before = model.core.edge_index.clone()
    weights_before = model.core.edge_weight.detach().clone()
    result = train_one(model, seed=0, cfg=TINY, log=lambda *a, **k: None)
    assert result["topology_unchanged"]
    assert torch.equal(before, model.core.edge_index)
    assert not torch.equal(weights_before, model.core.edge_weight.detach()), "weights never moved"


def test_controls_match_node_edge_and_parameter_budget():
    real = malecns_sample_brain("real")
    counts = {}
    for condition in CONDITIONS:
        brain = malecns_sample_brain(condition, seed=0)
        assert brain.num_nodes == real.num_nodes
        assert brain.num_edges == real.num_edges
        assert np.array_equal(brain.node_ids, real.node_ids)
        torch.manual_seed(0)
        model = build_malecns_sample_model(condition, 0)
        counts[condition] = ct.nn.count_parameters(model)
    assert len({tuple(sorted(c.items())) for c in counts.values()}) == 1, counts
    assert counts["real"]["connectome"] == real.num_edges


def test_degree_preserving_rewire_preserves_degrees_and_changes_wiring():
    real = malecns_sample_brain("real")
    rewired = malecns_sample_brain("rewired", seed=0)
    for axis in (0, 1):
        assert np.array_equal(
            np.bincount(real.edge_index[axis], minlength=real.num_nodes),
            np.bincount(rewired.edge_index[axis], minlength=real.num_nodes),
        )
    real_edges = set(zip(*real.edge_index.tolist()))
    rewired_edges = set(zip(*rewired.edge_index.tolist()))
    # This sample is dense (36% of all possible directed pairs), so most swaps
    # are rejected as duplicates and about half the edges survive. The
    # milestone-2 report records the exact overlap; here only assert it moved.
    assert len(rewired_edges) == len(real_edges)
    assert len(real_edges & rewired_edges) < len(real_edges)


def test_random_topology_changes_degrees():
    real = malecns_sample_brain("real")
    random = malecns_sample_brain("random", seed=0)
    assert not np.array_equal(
        np.bincount(real.edge_index[1], minlength=100),
        np.bincount(random.edge_index[1], minlength=100),
    )


def test_init_row_sum_rescales_only_values():
    torch.manual_seed(0)
    raw = build_malecns_sample_model("real", 0, ModelConfig(init_row_sum=None))
    torch.manual_seed(0)
    scaled = build_malecns_sample_model("real", 0, ModelConfig(init_row_sum=1.0))
    assert torch.equal(raw.core.edge_index, scaled.core.edge_index)
    assert abs(scaled.core.diagnostics()["max_abs_row_sum"] - 1.0) < 1e-5
    ratio = scaled.core.edge_weight / raw.core.edge_weight
    assert torch.allclose(ratio, ratio[0].expand_as(ratio))
