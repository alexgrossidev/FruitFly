"""Training loop and control-condition harness. Run via `python -m src.train --milestone N`.

Milestone 2 must run the real connectome plus both controls
(connectorch.transforms.degree_preserving_rewire, .random_topology) at a
matched trainable-parameter budget (ct.nn.count_parameters), across >=5 seeds
each. Do not report Milestone 2 as done with fewer seeds or missing controls
-- see README.md's Milestone 2 and the "single seed is not a result" gotcha.
"""

from __future__ import annotations

import argparse


def run_milestone(milestone: int) -> None:
    raise NotImplementedError(f"milestone {milestone} not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", type=int, required=True, choices=[0, 1, 2, 4])
    args = parser.parse_args()
    run_milestone(args.milestone)


if __name__ == "__main__":
    main()
