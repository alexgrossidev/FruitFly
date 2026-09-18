"""Replay export: schema, the viewer's constraints, and the id bridge."""

from __future__ import annotations

import copy

import pytest
import torch

from src.export_replay import (
    NORMALIZATION,
    build_replay,
    full_readout_core,
    load_visible_ids,
    normalize,
    validate_replay,
)
from src.model import ModelConfig, build_malecns_sample_model, malecns_sample_brain
from src.task import make_batch, one_hot_sequence


@pytest.fixture(scope="module")
def visible():
    return load_visible_ids()


def test_all_sample_body_ids_are_visible_in_the_viewer(visible):
    brain = malecns_sample_brain("real")
    ids = [int(i) for i in brain.node_ids]
    assert all(i in visible for i in ids)


def _replay(body_ids, steps=10):
    activity = [[(t + i) % 7 / 6 for i in range(len(body_ids))] for t in range(steps)]
    return build_replay(body_ids=body_ids, activity_per_step=activity, model_name="test model")


def test_built_replay_passes_validator(visible):
    brain = malecns_sample_brain("real")
    replay = _replay([int(i) for i in brain.node_ids])
    validate_replay(replay, visible)
    assert replay["frames"][0]["time"] == 0
    assert replay["source"]["kind"] == "predicted"
    assert replay["source"]["normalization"] == NORMALIZATION


def test_validator_rejects_the_same_things_the_viewer_does(visible):
    brain = malecns_sample_brain("real")
    good = _replay([int(i) for i in brain.node_ids])

    bad = copy.deepcopy(good)
    bad["dataset"] = "flywire:v783"
    with pytest.raises(ValueError):
        validate_replay(bad, visible)
    bad = copy.deepcopy(good)
    bad["frames"][0]["values"][0][0] = 1  # not a visible id
    with pytest.raises(ValueError):
        validate_replay(bad, visible)
    bad = copy.deepcopy(good)
    bad["frames"][0]["values"][1][0] = bad["frames"][0]["values"][0][0]
    with pytest.raises(ValueError):
        validate_replay(bad, visible)
    bad = copy.deepcopy(good)
    bad["frames"][0]["values"][0][1] = 1.01
    with pytest.raises(ValueError):
        validate_replay(bad, visible)
    bad = copy.deepcopy(good)
    bad["frames"][1]["time"] = 0
    with pytest.raises(ValueError):
        validate_replay(bad, visible)
    bad = copy.deepcopy(good)
    bad["frames"] = bad["frames"][:1]
    with pytest.raises(ValueError):
        validate_replay(bad, visible)
    bad = copy.deepcopy(good)
    bad["source"]["normalization"] = "  "
    with pytest.raises(ValueError):
        validate_replay(bad, visible)


def test_normalization_maps_state_range_into_unit_interval():
    states = torch.tensor([[-1.0, -0.25, 0.0, 0.5, 1.0]])
    out = normalize(states)
    assert out.min() >= 0 and out.max() <= 1
    assert torch.equal(out, torch.tensor([[1.0, 0.25, 0.0, 0.5, 1.0]]))


def test_full_readout_core_matches_trained_core():
    torch.manual_seed(0)
    config = ModelConfig()
    model = build_malecns_sample_model("real", 0, config)
    brain = malecns_sample_brain("real")
    core_all = full_readout_core(model, brain, config)
    assert core_all.num_output_nodes == brain.num_nodes
    digits, _ = make_batch(2, generator=torch.Generator().manual_seed(1))
    drive = model.encoder(one_hot_sequence(digits))
    with torch.no_grad():
        full = core_all(drive)
        own = model.core(drive)
    assert torch.allclose(full[:, :, model.core.output_index], own, atol=1e-6)
    assert full.abs().max() <= 1.0
