# Fly Number Recognition

Train a slice of a real fruit fly's nervous system — actual synaptic wiring, reconstructed from electron microscopy — to detect whether the digits 6 or 7 appear in a list of 10 random numbers, then watch the trained network's activity replay on a 3D map of the real neurons doing it.

This document is the spec and build plan for a fresh Claude Code instance starting from an empty repository. It assumes no prior context: everything the agent needs to know about the two upstream projects, the exact task definition, the architecture, and the order of work is below.

## Read this first: what this actually is

This is **not** a fly learning arithmetic. It is a real, measured connectome — the wiring diagram of ~166,000 real neurons from a dissected fly, reconstructed by electron microscopy — compiled into an ordinary recurrent neural network and trained with gradient descent on a synthetic classification task. The fly never evolved to read digit lists; nothing here claims otherwise.

What makes it interesting and legitimate rather than a gimmick: the *topology* of that network — which of the ~25 million connections exist — is fixed to what was actually measured in the animal. Gradient descent can change what a connection is worth. It cannot invent a connection the fly does not have. This is "reservoir computing" / "connectome-constrained learning," an active, published research direction (see [References](#references)), not a novel technique invented for this project. Say this plainly in the project's own README when it's published — don't let it read as "a fly brain learned to count."

Two upstream projects do the two halves of the work. Neither does both:

| Project | Role | What it actually contains |
|---|---|---|
| [`connectorch`](https://github.com/us/connectorch) | Training | Compiles the real MaleCNS connectome into a trainable `torch.nn.Module`. This is where the actual computation and learning happens. |
| [`fly-connectome-template`](https://github.com/cobanov/fly-connectome-template) | Visualization | A React/Three.js viewer that draws 124,289 real neuron cell-body positions and replays a JSON file of activity values over time. It contains **no computation** — no connectivity graph, no simulator, no model. It only draws dots and plays back numbers you hand it. Confirmed by reading its source: `src/lib/atlas.ts` just loads static position/ID arrays, `src/lib/replay.ts` is pure JSON validation plus a binary search over timestamps. If you skip the training half, this half has nothing real to show. |

The bridge between them is a JSON file in the exact schema `fly-connectome-template` expects (specified below). `connectorch` trains and produces activity; a small export script reshapes that activity into the replay format; the viewer plays it back.

## Task definition (locked — read before writing any data code)

**Input:** a list of 10 integers, each drawn independently and uniformly at random from `{0, 1, ..., 9}`, **with replacement**.

**Labels (multi-label, 2 independent binary targets):**
- `has_six` = 1 if `6` appears anywhere in the list, else 0
- `has_seven` = 1 if `7` appears anywhere in the list, else 0

**Why "with replacement" and not a shuffled permutation of 0–9:** a permutation of the digits 0–9 always contains every digit exactly once, which means 6 and 7 would be present in 100% of examples — there would be nothing to discriminate and no task at all. Sampling with replacement gives real, learnable class balance:

```
P(at least one 6 in 10 draws) = 1 - (9/10)^10 ≈ 0.6513
P(at least one 7 in 10 draws) = 1 - (9/10)^10 ≈ 0.6513
```

So a trivial "always predict 1" baseline gets ~65% accuracy per label — not useless, but not a wall either. Report accuracy *and* F1/balanced accuracy per label so a lazy baseline doesn't look like success.

Do not change this task definition without updating this section. If a later milestone wants a harder variant (exact count of 6s and 7s, position of first occurrence, or distinguishing 6-only vs 7-only vs both vs neither as a 4-way single-label task), add it as a new milestone, don't silently swap the base task.

## Repository layout

```
fly-number-recognition/
├── README.md                  # this file
├── pyproject.toml             # or requirements.txt — connectorch, torch
├── src/
│   ├── task.py                 # synthetic data generator + labels
│   ├── model.py                 # encoder / connectome core / decoder construction
│   ├── train.py                  # training loop, controls, multi-seed harness
│   ├── evaluate.py                # metrics, comparison table/plots
│   └── export_replay.py            # activity trajectory -> viewer JSON
├── results/
│   ├── metrics.json                # per-seed, per-condition results
│   └── replays/                     # exported JSON files, one per example case
├── viewer/                       # fly-connectome-template lives here (own git history if you want it as a submodule, otherwise a plain copy — see Licensing)
└── ATTRIBUTION.md                # required credits, see Licensing section
```

## Setup

Python side (training):

```bash
python -m venv .venv && source .venv/bin/activate
pip install connectorch torch
```

`connectorch` needs no GPU for this task — the bundled 100-neuron sample (`ct.datasets.malecns_sample()`) runs on a laptop CPU in seconds per epoch. Only reach for the full 164,587-neuron graph (`ct.datasets.malecns(download=True)`, ~520 MB, resumable download) in the stretch milestone, and expect to need real RAM/VRAM budgeting — see [Gotchas](#gotchas-read-before-training-on-the-full-graph).

Viewer side (visualization), Node.js 22.18+:

```bash
# generate the template into viewer/ (https://github.com/new?template_name=fly-connectome-template&template_owner=cobanov),
# or clone it directly if you don't need your own copy's history:
git clone https://github.com/cobanov/fly-connectome-template.git viewer
cd viewer && npm ci && npm run dev
```

## Architecture

### Encoder → connectome core → decoder

`connectorch` provides exactly this three-part shape via `connectorch.nn.ConnectomeModel`:

```python
import connectorch as ct
import torch
from torch import nn

brain = ct.datasets.malecns_sample()   # 100 real neurons, 3,630 real edges, offline

core = ct.nn.ConnectomeRNN(
    brain,
    input_nodes=brain.node_ids[:16],     # arbitrary subset receiving external drive
    output_nodes=brain.node_ids[-16:],   # arbitrary subset read out
    weights="trainable",
    initializer="normalized_synapse_count",
    activation="tanh",
    leak=0.4,
)

model = ct.nn.ConnectomeModel(
    core,
    encoder=nn.Sequential(nn.Linear(10, core.num_input_nodes), nn.Tanh()),  # one-hot digit -> input nodes
    decoder=nn.Linear(core.num_output_nodes, 2),                            # -> [has_six_logit, has_seven_logit]
    readout="last",   # decode only after the network has seen all 10 numbers
)
```

### Feeding the list one number at a time (this is the part worth doing carefully)

`ConnectomeRNN.forward` accepts `x` as either a constant drive `[batch, num_input_nodes]` held over `steps`, **or** a genuinely time-varying drive `[batch, steps, num_input_nodes]`, one slice consumed per recurrent step, with `steps` inferred from the sequence length. Use the second form — it's what makes this a real "reading the list one number at a time" task rather than "look at the whole list at once," and it's a first-class, already-supported input shape, not a workaround.

Encode each of the 10 numbers as a one-hot 10-vector, run each through the same `nn.Linear(10, core.num_input_nodes)` encoder, stack along the step dimension into `[batch, 10, num_input_nodes]`, and hand that whole tensor to `model(x)` — `steps` doesn't need to be passed; it's read off `x.shape[1]`. With `readout="last"`, the decoder only sees the state after all 10 digits have been presented, matching "read the whole list, then answer."

`connectorch/examples/02_train_connectome.py` in the connectorch repo is the closest existing precedent for a sequence-position task built exactly this way (a pulse in the first vs. second half of a sequence) — read it before writing `model.py`.

### Loss and optimizer

Two independent binary targets → `nn.BCEWithLogitsLoss()` against a `[batch, 2]` float tensor of `{0, 1}` labels, not `CrossEntropyLoss` (that's for mutually-exclusive classes, and "has a 6" and "has a 7" are not mutually exclusive — both, either, or neither can be true). `torch.optim.Adam`, start around `lr=2e-3`, same as the connectorch MNIST example.

## Build plan

Work through these in order. Each milestone should run and produce a saved artifact (metrics file, plot, or JSON) before moving to the next — don't jump to the real connectome before the pipeline is proven, and don't jump to visualization before training is proven.

**Milestone 0 — Sanity pipeline on a toy graph.**
Build the full encoder/core/decoder/training loop against `ct.datasets.random_sparse(...)` or `ct.datasets.chain(...)` (not biological, just for wiring correctness) with the real 10-number task. Confirm shapes, confirm loss decreases, confirm `core.edge_index` is bit-identical before and after training (the library's own invariant — copy the assertion pattern from `examples/02_train_connectome.py`). This step is about proving your data pipeline and training loop are correct before biology enters the picture at all.

**Milestone 1 — Real connectome, small scale.**
Swap in `ct.datasets.malecns_sample()` (100 real neurons, no download needed). Train to convergence. Report per-label accuracy, F1, and balanced accuracy on a held-out set, against the ~65%-always-predict-1 baseline computed above.

**Milestone 2 — Controls (do not skip this).**
A result on the real connectome alone proves nothing by itself. `connectorch.transforms` ships exactly the comparisons its own documentation says are required before claiming biological wiring helped:

```python
from connectorch.transforms import degree_preserving_rewire, random_topology
```

Train the identical model (same parameter budget, same encoder/decoder sizes) on: (a) the real MaleCNS sample, (b) a degree-preserving rewiring of it, (c) a random graph of the same node/edge count. Run each across at least 5 random seeds. Report all three, honestly, even if the real connectome doesn't win — a 100-neuron/3,630-edge sample is small, and the connectome's own maintainers are explicit that "a single-seed difference between a connectome and one control is not a result." If the real wiring doesn't clearly outperform the controls at this scale, say so in the results — that's a genuine, reportable finding about how much structure this size of task actually needs, not a failure to hide.

**Milestone 3 — Export a replay for the viewer.**
Pick 3–4 representative example lists (a true positive for 6, a true positive for 7, a case with both, a clean negative). Run each through the trained model with gradients off, capture the **core's own trajectory** (call `core(...)` directly, not the wrapped `model(...)`, so you get raw per-step activity for every output node rather than the decoder's collapsed logits — shape `[1, 10, num_output_nodes]`). Normalize activity to `[0, 1]` (document your normalization choice, e.g. min-max per run or a fixed tanh-based rescale — `docs/MODEL-INTEGRATION.md` in the viewer repo requires you to state it, it doesn't accept an undeclared normalization). Map each output node's index back to its MaleCNS body ID via `core.output_index` / `brain.node_ids`, and write the replay JSON (schema below). Load it in the running viewer with "Load model JSON" and confirm it plays without validator errors.

**Milestone 4 — Stretch: full connectome.**
Repeat Milestone 1–2 on `ct.datasets.malecns(download=True)` (164,587 nodes, ~520 MB download, cached and resumable). Read [Gotchas](#gotchas-read-before-training-on-the-full-graph) first — this is not a drop-in scale-up, it changes memory behavior by orders of magnitude. Optionally choose input/output node populations by biological role instead of arbitrary index slices, e.g. `fly.where(cell_class="visual")` for input, a central-complex or descending population for output — this is closer to how the fly's own sensory-to-motor pipeline is organized, though note it's still an arbitrary choice for a task with no biological equivalent.

**Milestone 5 — Write it up.**
Finalize this project's own README with real numbers from Milestone 2's controls, keep the "what this actually is" honesty framing near the top, and satisfy both projects' attribution requirements (below) before publishing.

## Visualization bridge: the exact replay format

From `fly-connectome-template/docs/MODEL-INTEGRATION.md`, verified by reading `src/lib/replay.ts`'s validator directly — it will reject anything that doesn't match this exactly:

```json
{
  "version": 1,
  "dataset": "male-cns:v1.0",
  "source": {
    "kind": "predicted",
    "name": "fly-number-recognition, milestone-3 checkpoint",
    "normalization": "state after tanh, min-max rescaled to [0,1] per run"
  },
  "frames": [
    {"time": 0, "values": [[1234567, 0.12], [7654321, 0.83]]},
    {"time": 1, "values": [[1234567, 0.40]]}
  ]
}
```

Hard constraints the validator enforces (all confirmed by reading the code, not assumed from prose): `version` must be `1`, `dataset` must be exactly `"male-cns:v1.0"`, `source.kind` must be `"synthetic"`, `"predicted"`, or `"measured"` (use `"predicted"` — this is model output, not a biological recording) with non-empty `name` and `normalization` strings, 2–10,000 frames, the first frame's `time` must be `0`, every subsequent `time` must be strictly increasing, every body ID in `values` must belong to the viewer's *visible* brain selection (not every ID in the atlas — only classified optic/central/descending somata, `groups.bin` value `< 3`), no duplicate IDs within a frame, and every value must be a finite number in `[0, 1]`. An omitted ID in a given frame is treated as zero for that frame, not as "unchanged" — if you want a neuron to hold its value, repeat it explicitly.

Note the ID space: `connectorch`'s `malecns`/`malecns_sample` node IDs and the viewer's atlas body IDs both come from the same MaleCNS v1.0 dataset, so they should refer to the same neurons — but verify a handful by hand before trusting a full export, since one loads via the Python-side dataset loader and the other via bundled binary files, and a mismatch would fail loudly (the validator rejects unknown IDs) rather than silently.

## Acceptance criteria

- Milestone 0 passes with `torch.equal(wiring_before, wiring_after)` still true (topology never changes, only weights).
- Milestone 1 reports per-label accuracy, F1, and balanced accuracy against the computed ~65% baseline, on held-out data.
- Milestone 2 reports the same metrics for the real connectome, a degree-preserving rewiring, and a random graph, each across ≥5 seeds, at a matched trainable-parameter count (`ct.nn.count_parameters(model)` splits connectome vs. encoder/decoder parameters — report both).
- Milestone 3 produces a replay JSON that loads in the running viewer with zero validator errors, for at least one true-positive and one true-negative example.
- The project's own README states plainly, near the top, that this is connectome-constrained reservoir learning on a synthetic task, not the fly "understanding numbers."

## Gotchas (read before training on the full graph)

- **The module is stateless between calls.** Nothing carries over from one `model(x)` call to the next automatically — if you ever need to continue a run (e.g., feeding digits in separate calls instead of one stacked tensor), pass `state=` explicitly using `return_state=True`'s output. For this task, feeding the whole `[batch, 10, num_input_nodes]` tensor in one call sidesteps this entirely — prefer that.
- **`x` must be exactly 2D or 3D.** `[batch, num_input_nodes]` (constant drive, requires explicit `steps=`) or `[batch, steps, num_input_nodes]` (time-varying, `steps` inferred). Get the one-hot stacking axis wrong and you'll get a shape error, not silently wrong behavior — that's by design, but write a shape-assertion test anyway.
- **Full-graph memory scales fast.** The full MaleCNS graph has 25,563,197 edges. Training (not inference) holds `2 * num_edges * batch * steps * itemsize` bytes for backprop — at batch 4 over 8 steps that's already ~6.5 GiB, and it grows linearly with both batch and steps. `model.core.activation_bytes(batch, steps)` computes this before you run anything; call it first. `model.core.diagnostics()` reports `max_abs_row_sum`, the number to watch for exploding recurrent state — if it's large, use `initializer="normalized_synapse_count"`, a smaller `leak`, or a saturating activation (`tanh` is the default and a reasonable choice here).
- **Never request the `"dense"` backend on the full graph.** A dense adjacency for 164,587 nodes needs ~100.9 GiB and the library will refuse rather than try. Leave `backend="auto"` — it already picks `"scatter"` for trainable weights.
- **Synapse count is not a sign.** Nothing in the raw connectivity says excitatory vs. inhibitory by default; `weights="trainable"` with `initializer="normalized_synapse_count"` starts from magnitude only. If you want biologically-signed weights (a stretch beyond this project's scope), that requires the transmitter-prediction columns and `connectorch.transforms.infer_signs` — read `docs/biology.md` in connectorch before touching this, it's a prediction-on-a-prediction and the library is explicit about not inferring it for you silently.
- **A single seed is not a result.** Said above, worth repeating here as a gate: don't report Milestone 2 with fewer than 5 seeds per condition.

## Licensing and attribution — do not skip this before publishing

Both upstream projects have real, binding attribution requirements. Check both before making the finished repo public:

- **`fly-connectome-template`** is source-available under a custom *Cobanov Template Attribution License 1.0* — not OSI-approved. It requires a visible, linked credit ("Built with fly-connectome-template by Mert Cobanov") in **both** the running web UI (the `Attribution.tsx` component — you may restyle or relocate it to an About view, but not hide or remove it) and this project's own README. Your trained model and weights do not have to be published; the credit does.
- **MaleCNS v1.0 data** (used by both projects) is **CC BY 4.0**, credited to FlyEM/HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research. Cite the dataset's authors (Berg, S. et al., *Cell*, 2026) directly, not just the two GitHub repos.
- **`connectorch`** itself is MIT-licensed for the library code; the connectome data it loads keeps the license above, not MIT.

Put all of this in an `ATTRIBUTION.md` at the repo root and link it from the main README, alongside the template's required on-screen credit.

## References

- [`fly-connectome-template`](https://github.com/cobanov/fly-connectome-template) — the viewer used in this project.
- [`connectorch`](https://github.com/us/connectorch) — the library used to train on the real connectome; see its `docs/` for the full API reference, and `examples/02_train_connectome.py` / `examples/04_mnist_connectome.py` for the two closest precedents to this task.
- Berg, S. et al. *Sexual dimorphism in the complete Drosophila male central nervous system connectome.* Cell 189, 5504–5526.e15 (2026). — the MaleCNS v1.0 dataset itself.
- Eckstein, N. et al. *Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila.* Cell 187, 2574–2594 (2024). — source of the transmitter-prediction columns, relevant only if you pursue signed weights.
- Lappalainen, J. K. et al. *Connectome-constrained deep mechanistic networks predict neural responses across the fly visual system.* Nature 634, 1132–1140 (2024). ("FlyVis") — closest published proof that this general approach (real connectome + gradient descent on a task) works.
- "The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction" — [PMC12109256](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/) — direct precedent for treating the fly connectome as a fixed reservoir for a classification/prediction task, the same framing this project uses.