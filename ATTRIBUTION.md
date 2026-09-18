# Attribution

Required by the licenses of the two upstream projects this repo builds on. Do not remove or shorten this file, and keep the matching on-screen credit in `viewer/src/components/Attribution.tsx` visible (it may be restyled or moved to an About view, but not hidden).

- Viewer built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template) by [Mert Cobanov](https://github.com/cobanov). Licensed under the Cobanov Template Attribution License 1.0 (source-available, not OSI-approved) — see `viewer/LICENSE`.
- Connectome compiled with [ConnecTorch](https://github.com/us/connectorch), MIT-licensed for the library code (`vendor/connectorch/LICENSE`); the connectome data it loads keeps its own license below, not MIT.
- Anatomy and connectivity: MaleCNS v1.0, Berg, S. et al. *Sexual dimorphism in the complete Drosophila male central nervous system connectome.* Cell 189, 5504–5526.e15 (2026). CC BY 4.0. Credit: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research.
- Neurotransmitter predictions (only relevant if signed weights are used): Eckstein, N. et al. Cell 187, 2574–2594 (2024).

## How each is used here

- The viewer (`viewer/`) is the upstream template at commit `38f5533`, **unmodified**, as a git submodule; this project only feeds it JSON replay files produced by `src/export_replay.py`. The template's required on-screen credit (`viewer/src/components/Attribution.tsx`) is displayed as shipped, and the required README credit is in the main `README.md` under *Licensing and attribution*.
- ConnecTorch (`vendor/connectorch`, commit `4bbfb64`) is used as a library, unmodified. The 100-neuron MaleCNS sample it bundles (`ct.datasets.malecns_sample()`) is a deterministic subgraph of the CC BY 4.0 dataset: body ids, cell types and synapse counts are unchanged from the published data (see `vendor/connectorch/THIRD_PARTY_DATA.md` for the selection procedure). The full graph, when used (Milestone 4), is downloaded by the library from the dataset's public bucket and never redistributed here.
- Trained weights (`results/checkpoints/*.pt`, git-ignored) and replay files (`results/replays/*.json`) are derived from the MaleCNS v1.0 wiring and carry the same CC BY 4.0 attribution.
