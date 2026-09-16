"""Synthetic data generator for the locked task in README.md.

Input: a list of 10 integers, each drawn independently and uniformly from
{0, ..., 9}, WITH REPLACEMENT. Do not switch to a permutation of 0-9 without
re-reading README.md's "Task definition (locked)" section first — a
permutation always contains every digit, which makes the task trivial.

Labels: two independent binary targets, has_six and has_seven. This is a
multi-label problem, not a multi-class one (both, either, or neither can be
true) — train with BCEWithLogitsLoss, not CrossEntropyLoss.
"""

from __future__ import annotations

import torch

NUM_DIGITS = 10  # digits 0-9
LIST_LENGTH = 10  # numbers per example


def baseline_positive_rate(list_length: int = LIST_LENGTH, num_digits: int = NUM_DIGITS) -> float:
    """P(a given digit appears at least once), i.e. the "always predict 1" baseline.

    Report this next to every accuracy number in MILESTONES.md — an accuracy
    below this is worse than a constant predictor.
    """
    return 1.0 - ((num_digits - 1) / num_digits) ** list_length


def make_batch(
    n: int, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate n examples.

    Returns
    -------
    digits: LongTensor [n, LIST_LENGTH], values in [0, NUM_DIGITS)
    labels: FloatTensor [n, 2], columns are (has_six, has_seven) as 0.0/1.0
    """
    digits = torch.randint(0, NUM_DIGITS, (n, LIST_LENGTH), generator=generator)
    has_six = (digits == 6).any(dim=1).float()
    has_seven = (digits == 7).any(dim=1).float()
    labels = torch.stack([has_six, has_seven], dim=1)
    return digits, labels


def one_hot_sequence(digits: torch.Tensor) -> torch.Tensor:
    """[n, LIST_LENGTH] int digits -> [n, LIST_LENGTH, NUM_DIGITS] one-hot float.

    This is the raw feature sequence to hand to the encoder in src/model.py,
    one slice per recurrent step — see README.md's "Feeding the list one
    number at a time" section for why this must stay a genuine sequence
    rather than a single flattened vector.
    """
    return torch.nn.functional.one_hot(digits, num_classes=NUM_DIGITS).float()


if __name__ == "__main__":
    print(f"baseline positive rate per label: {baseline_positive_rate():.4f}")
    x, y = make_batch(8)
    print("digits:\n", x)
    print("labels (has_six, has_seven):\n", y)
