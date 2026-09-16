# Milestones

Mirrors the build plan in `README.md`. Check a box only when its acceptance criterion (also in `README.md`) is genuinely met — not when the code merely runs without error. Fill in the results line under each item as you go; a fresh session should be able to read this file alone and know exactly where things stand.

- [ ] **0 — Sanity pipeline on a toy graph.** `ct.datasets.random_sparse(...)` or `ct.datasets.chain(...)`, full encoder/core/decoder/training loop, loss decreases, `torch.equal(wiring_before, wiring_after)` holds.
  - Result:

- [ ] **1 — Real connectome, small scale.** `ct.datasets.malecns_sample()`, trained to convergence.
  - Baseline (always predict 1): ~65.13% per label (computed from `1 - 0.9**10`).
  - has_six — accuracy / F1 / balanced accuracy:
  - has_seven — accuracy / F1 / balanced accuracy:

- [ ] **2 — Controls.** Real connectome vs. `degree_preserving_rewire` vs. `random_topology`, matched trainable-parameter budget, ≥5 seeds each.
  - Parameter budget (connectome / other, from `ct.nn.count_parameters`):
  - Real connectome — mean ± spread across seeds:
  - Degree-preserving rewiring — mean ± spread:
  - Random topology — mean ± spread:
  - Honest verdict (did real wiring help at this scale? say so either way):

- [ ] **3 — Export a replay for the viewer.** At least one true positive and one true negative, loads in the running viewer with zero validator errors.
  - Files: `results/replays/`
  - Normalization used (must match what's declared in the JSON's `source.normalization`):

- [ ] **4 — Stretch: full connectome.** `ct.datasets.malecns(download=True)`, repeat 1–2.
  - Skipped / attempted / done:

- [ ] **5 — Write-up.** Final `README.md` (this project's own, not the upstream ones) states real Milestone 2 numbers, keeps the "what this actually is" framing near the top, and `ATTRIBUTION.md` is filled in and linked.
