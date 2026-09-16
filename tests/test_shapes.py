"""Shape and invariant tests. Keep these passing at every milestone -- they
are cheap and catch the two mistakes README.md calls out explicitly: getting
the one-hot sequence axis wrong, and accidentally letting the connectome's
topology change during training.
"""

from __future__ import annotations

import torch

from src.task import LIST_LENGTH, NUM_DIGITS, baseline_positive_rate, make_batch, one_hot_sequence


def test_make_batch_shapes():
    digits, labels = make_batch(16)
    assert digits.shape == (16, LIST_LENGTH)
    assert labels.shape == (16, 2)
    assert set(torch.unique(digits).tolist()) <= set(range(NUM_DIGITS))
    assert set(torch.unique(labels).tolist()) <= {0.0, 1.0}


def test_labels_match_digits():
    digits, labels = make_batch(200)
    expected_six = (digits == 6).any(dim=1).float()
    expected_seven = (digits == 7).any(dim=1).float()
    assert torch.equal(labels[:, 0], expected_six)
    assert torch.equal(labels[:, 1], expected_seven)


def test_one_hot_sequence_shape():
    digits, _ = make_batch(4)
    encoded = one_hot_sequence(digits)
    assert encoded.shape == (4, LIST_LENGTH, NUM_DIGITS)
    assert torch.equal(encoded.argmax(dim=-1), digits)


def test_baseline_positive_rate_matches_closed_form():
    assert abs(baseline_positive_rate() - (1 - 0.9**10)) < 1e-9


# Once src/model.py and src/train.py have real implementations, add:
#   def test_topology_unchanged_after_training(): ...
# asserting torch.equal(core.edge_index, wiring_before) after a few optimizer
# steps, exactly as README.md's Milestone 0 acceptance criterion requires.
