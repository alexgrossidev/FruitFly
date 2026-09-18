# Fly Number Recognition

I took a piece of a real fruit fly's brain, the actual wiring diagram reconstructed from electron microscopy, turned it into a recurrent neural network, and taught it to answer one question: *does this list of ten random digits contain a 6 or a 7?* Then I played the trained network's activity back on a 3D map of the real neurons so you can watch it "think".

It works. It also turns out that a random pile of neurons with the same number of wires works exactly as well, which is the more interesting result. More on that below.

## Wait, did a fly learn to count?

No. Let's get this out of the way, because it's the first thing anyone asks.

The fly connectome (MaleCNS v1.0, roughly 166,000 neurons and 25 million synapses) is a *map*. It says which neuron connects to which, and roughly how strongly. What this project does is compile that map into a plain PyTorch recurrent network, where every real connection becomes one trainable weight, and then run ordinary gradient descent on a synthetic task the fly has never seen and would have no reason to care about.

The one thing gradient descent is *not* allowed to do is invent a connection. If two neurons aren't wired together in the animal, they stay unwired. The topology is the fly's; the weight values are learned. This is "connectome-constrained learning" or, if you squint, reservoir computing. It is an active research direction (see the [references](#references) at the bottom), not something I made up.

So: a network *shaped like* a fly brain learned to spot sixes and sevens. The fly itself remains blissfully unaware.

## The task

Ten integers, each drawn uniformly from 0 to 9, with replacement. Two yes/no questions:

- `has_six`: is there a 6 anywhere in the list?
- `has_seven`: is there a 7 anywhere in the list?

Both can be true at once, so this is multi-label, not a four-way choice. The digits are fed in one at a time, ten recurrent steps, and the network only gets to answer after it has seen the whole list. That's the point: it has to *remember* whether it saw a 6 five steps ago.

Why "with replacement"? Because a shuffled permutation of 0–9 always contains every digit, and then there's nothing to learn. With replacement, a 6 shows up in about 65% of lists (`1 - 0.9^10`), so a lazy model that always says "yes" gets 65% accuracy. Every number below is reported against that baseline, with F1 and balanced accuracy alongside plain accuracy so the lazy model can't hide.

## The result

Everything was run on a laptop (Apple M4, 16 GiB). Three networks were trained on identical data, ten random seeds each:

1. **The real thing.** 100 real neurons and 3,630 real synaptic connections, a sample bundled with `connectorch`.
2. **Degree-preserving rewiring.** Same neurons, same number of inputs and outputs per neuron, connections shuffled.
3. **Random topology.** Same node and edge count, no other resemblance.

Every network has the same trainable budget: 3,630 connectome weights plus 210 encoder/decoder parameters.

| Condition | has_six acc / F1 / bal. acc | has_seven acc / F1 / bal. acc | epochs to 95% |
|---|---|---|---|
| Real connectome | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1 |
| Degree-preserving rewiring | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1 |
| Random topology | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1 |
| Always say "yes" | 0.645 / 0.784 / 0.500 | 0.658 / 0.794 / 0.500 | n/a |

Mean over 10 seeds, standard deviation 0.000 across the board, 4,096 held-out lists each. Full per-seed numbers live in `results/metrics.json` and are narrated in [MILESTONES.md](MILESTONES.md).

**So the fly's wiring doesn't help.** It doesn't hurt either. At 100 neurons, this task is just too easy for any recurrent network of that size to reveal a difference. I tried starving the models of data to open up a gap: with only 512 training lists all three sit around 99.5–99.9%, and with 128 lists they all fall to roughly 92–95%, together, with every difference well inside the seed-to-seed noise. The rewired control also still shares half its edges with the real graph, because the sample is so dense that most edge swaps get rejected as duplicates. "Rewired" here is only half-rewired.

That's the honest finding: a 100-neuron, 3,630-weight network solves "is there a 6 in this list" no matter how you wire it. It says something about the task size, not about connectomes.

### What about the whole brain?

I tried. The full graph (164,587 neurons, 25.6 million trainable connections) loads and trains end to end, but it needs about 4.6 GiB of activations at batch size 2 and takes 4.5 seconds per optimizer step on the laptop GPU. In a 45-minute budget it saw 2,428 examples and learned nothing beyond the class prior. The small sample needed on the order of 10,000 presentations to converge, which works out to about five hours per condition on this machine. The timing and memory numbers are in [MILESTONES.md](MILESTONES.md) if you have a bigger GPU and want to pick it up.

## Two things that had to change before anything learned

Both are worth knowing if you build on this, because the obvious setup silently learns nothing.

**The initial weights saturate the network.** The library's default initializer gives this dense little circuit a maximum absolute row sum of about 36. Push that through `tanh` on the first recurrent step and every neuron is pinned at ±1 before the input has a say. Forty hyperparameter configurations all sat at the class prior. Rescaling the starting weights so the max row sum is 1.0 (they stay trainable, topology untouched) fixed it instantly.

**The last digit can't reach the readout.** Input and output neurons are disjoint sets, so a digit presented at step *t* first touches an output neuron at step *t+1*. Reading out after exactly 10 steps means the tenth digit is invisible, which caps accuracy at about 96%. Seventeen otherwise-different configurations all stopped at the same 96.5%, which is how I noticed. Two silent zero-drive steps after the digits let the signal settle through, and accuracy went to 100%.

## Watching it think

The viewer is [fly-connectome-template](https://github.com/cobanov/fly-connectome-template), a React/Three.js page that draws 124,289 real neuron positions and plays back a JSON file of activity values over time. It does no computation of its own. It only draws dots and plays the numbers you hand it.

`src/export_replay.py` runs a trained network on four example lists (only a 6, only a 7, both, neither), records every neuron's state at every step, and writes it in the viewer's replay format. Each frame is one recurrent step: ten digits, then the two settle steps. Activity is `|h_t|`, the absolute value of each neuron's leaky-tanh state, which is a fixed rescale to [0, 1] with no per-run min-max. All 100 body ids in the sample map straight onto the viewer's atlas with no remapping.

```bash
cd viewer && npm run dev
# open http://127.0.0.1:5173/, click "Load model JSON", pick results/replays/real_seed0_both.json
```

The four replays are validated three ways: a Python mirror of the viewer's validator, the viewer's own TypeScript `parseReplay` run under Node, and a headless-Chrome test that uploads each file through the real button and checks no error appears.

## Running it yourself

You need Python 3.10+ (this was run on 3.13 with torch 2.14) and Node 22.18+. One script does the setup: it pulls two pinned git submodules, builds a venv, installs `connectorch` from source (it is not on PyPI), and runs `npm ci` in the viewer.

```bash
bash scripts/bootstrap.sh && source .venv/bin/activate
pytest tests/
python -m src.train --milestone 0               # toy graph sanity check, ~10 s
python -m src.train --milestone 1               # real 100-neuron sample, ~30 s
python -m src.train --milestone 2 --seeds 10    # real + both controls, 30 runs, ~20 min on CPU
python -m src.export_replay                     # -> results/replays/*.json
node scripts/validate_replay.mjs results/replays/*.json
cd viewer && npm run dev
```

Two macOS notes. Don't build the venv on `/usr/bin/python3`, the Xcode shim, because it can vanish and take the venv with it (the script prefers Homebrew Python and detects a broken venv). And run `npm ci` on the machine that will run the viewer, because Vite's bundler has a per-platform native binding.

Milestone 4, the full brain, is `python -m src.train --milestone 4 --download --device mps --time-budget 2700`. It downloads about 520 MB on first use, checks `activation_bytes` before allocating anything, and never asks for the dense backend (which would want 100 GiB).

## How it's built

```
src/
├── task.py            # the digit lists and labels
├── model.py           # encoder -> connectome core -> decoder, plus the two control graphs
├── train.py           # training loop and the per-milestone runners
├── evaluate.py        # accuracy / F1 / balanced accuracy per label, baseline, seed aggregation
├── export_replay.py   # trained trajectory -> viewer JSON, with a validator
└── full.py            # the full-graph attempt, with memory and time budgets
scripts/               # bootstrap, replay validators, the hyperparameter sweep
tests/                 # shapes, labels, topology-never-changes, control invariants, replay schema
results/               # every number in this README, as JSON
vendor/connectorch/    # git submodule, pinned
viewer/                # git submodule, pinned, unmodified
```

The model is `connectorch`'s encoder/core/decoder shape. A one-hot digit goes through a `Linear(10, 16)` encoder into 16 input neurons; the connectome core runs one recurrent step per digit (`tanh`, leak 0.4); a `Linear(16, 2)` decoder reads 16 output neurons after the last step and produces two logits. Loss is `BCEWithLogitsLoss`, optimizer is Adam at `1e-2`. The whole `[batch, 12, 16]` sequence goes in as one call, which the library supports natively. A test asserts the edge index is bit-identical before and after training, so the topology really is fixed.

## Credits

This project stands on two other people's work and a very large dataset. Full text in [ATTRIBUTION.md](ATTRIBUTION.md).

- Built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template) by [Mert Cobanov](https://github.com/cobanov), under the [Cobanov Template Attribution License 1.0](viewer/LICENSE). The viewer is used unmodified as a submodule and its on-screen credit is intact.
- Connectome compiled and trained with [ConnecTorch](https://github.com/us/connectorch) (MIT, library code only).
- Anatomy and connectivity: MaleCNS v1.0. Berg, S. et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell 189, 5504–5526.e15 (2026), CC BY 4.0. FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research.

## References

- Lappalainen, J. K. et al. *Connectome-constrained deep mechanistic networks predict neural responses across the fly visual system.* Nature 634, 1132–1140 (2024). The closest published proof that "real connectome plus gradient descent on a task" works.
- *The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction.* [PMC12109256](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/). Direct precedent for treating the fly connectome as a fixed reservoir.
- Eckstein, N. et al. *Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila.* Cell 187, 2574–2594 (2024). Only relevant if you want signed excitatory/inhibitory weights, which this project doesn't use.
- Berg, S. et al. *Sexual dimorphism in the complete Drosophila male central nervous system connectome.* Cell 189, 5504–5526.e15 (2026). The dataset itself.
