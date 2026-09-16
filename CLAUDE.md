# Working notes for Claude Code in this repo

Read `README.md` in full before writing any code. It is the spec — task definition, architecture, milestone order, acceptance criteria, licensing obligations. This file is only the operational supplement: environment facts and hard constraints that would otherwise cost a turn to rediscover.

## Do this before anything else

1. Run `scripts/bootstrap.sh`. It vendors both upstream dependencies as pinned git submodules and sets up the Python and Node environments. Do not `pip install connectorch` directly — see below.
2. Open `MILESTONES.md` and work top to bottom. Check off a milestone only once its acceptance criterion in `README.md` is actually met, not when the code merely runs.
3. Treat `README.md`'s "Task definition (locked)" section as fixed. If a change to it seems necessary, say so and ask rather than silently redefining the task.

## Hard facts about the two dependencies (verified, not assumed)

- **`connectorch` is not on PyPI.** `pip install connectorch` will fail with "No matching distribution found." It must be installed from source. `scripts/bootstrap.sh` adds it as a git submodule at `vendor/connectorch`, pinned to commit `4bbfb645099aeb85bdbf850e1a87cc094769af87`, and installs it editable (`pip install -e vendor/connectorch`). If you ever re-run the bootstrap or update the pin, record the new commit hash here and in the script.
- **`fly-connectome-template`** (the viewer) is vendored the same way, as a submodule at `viewer/`, pinned to commit `38f55332055328d38c29e72474c4ad5b6876101f`. It needs Node.js `>=22.18.0` (see its own `.nvmrc`) and is a separate `npm` project — `npm ci` inside `viewer/`, not at the repo root.
- **Both submodules require network access to `github.com` at `git submodule update --init` time.** If this environment has no outbound network access at all, stop and say so rather than trying to work around it — there is no offline fallback for fetching the initial connectome data or the viewer's JS dependencies.
- `connectorch` requires Python `>=3.10`, `torch>=2.1`, `numpy>=1.24`, `pyarrow>=14`. `.python-version` in this repo pins `3.11`.
- The bundled `ct.datasets.malecns_sample()` (100 real neurons, 3,630 real edges) ships inside the `connectorch` submodule itself and needs no download. The full graph (`ct.datasets.malecns(download=True)`, 164,587 neurons, ~520 MB) is a separate, resumable download on first use — don't trigger it before Milestone 4, it's slow and unnecessary for earlier milestones.

## Constraints that are easy to accidentally violate

- **Do not skip the controls in Milestone 2.** A trained result on the real connectome alone, without a degree-preserving-rewiring and a random-topology control at a matched parameter budget and ≥5 seeds, is not a reportable result — this is the upstream library's own stated bar, not an invented one. If you're tempted to report Milestone 2 with one seed "to see if it's working," that's fine as a smoke test, but say explicitly that it isn't the real result yet.
- **`ConnectomeRNN` is stateless between calls.** Don't reach for a per-timestep Python loop calling `model()` repeatedly and expect state to persist — it won't, unless you thread `state=`/`return_state=True` through explicitly. Feed the whole `[batch, 10, num_input_nodes]` sequence in one call instead; this is a first-class supported input shape, not a workaround.
- **Never pass `backend="dense"` on the full connectome.** It needs ~100.9 GiB and the library will refuse. Leave `backend="auto"`.
- **Check `model.core.activation_bytes(batch, steps)` before scaling up batch or steps on the full graph**, and `model.core.diagnostics()['max_abs_row_sum']` before trusting that training won't explode.
- **Multi-label, not multi-class.** "Contains a 6" and "contains a 7" are independent and can both be true — `nn.BCEWithLogitsLoss` against a `[batch, 2]` float target, not `CrossEntropyLoss`.

## Commands

```bash
bash scripts/bootstrap.sh          # one-time setup: submodules, venv, pip, npm
source .venv/bin/activate
pytest tests/                       # shape and invariant tests — keep these passing at every milestone
python -m src.train --milestone 0   # toy graph sanity pipeline
python -m src.train --milestone 1   # malecns_sample, no controls
python -m src.train --milestone 2   # malecns_sample + rewired + random controls, multi-seed
python -m src.export_replay          # writes results/replays/*.json for the viewer
cd viewer && npm run dev             # then use "Load model JSON" in the browser UI
```

## Reporting results

When you finish a milestone, update `MILESTONES.md` with the actual numbers (accuracy/F1/balanced accuracy per label, per condition, per seed or aggregated with a spread) rather than just checking the box. If the real connectome doesn't beat the controls at the small (100-neuron) scale, write that down plainly in `MILESTONES.md` and in the final README write-up — it's a real finding about how much structure this task needs, not a bug to hide.
