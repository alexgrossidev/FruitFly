"""Milestone 4 (stretch): the full MaleCNS graph, 164,587 neurons, 25,563,197 edges.

    python -m src.train --milestone 4 --download --time-budget 1800

This is not a drop-in scale-up of Milestones 1-2 and it is run differently:

- Memory. Backprop through the scatter backend stores two [num_edges, batch]
  tensors per recurrent step, i.e. 2 * 25.5M * batch * steps * 4 bytes. For the
  12-step sequence used here (10 digits + 2 settle steps) that is ~2.3 GiB per
  example in the batch, so the batch is tiny (default 2) and the run is
  budgeted with ``core.activation_bytes`` before anything is allocated. Even a
  no-grad forward pass materialises a transient [num_edges, batch] message
  tensor, so evaluation is also done in small batches.
- Data. There are no epochs. Each optimizer step draws a fresh random batch
  (the task is synthetic and infinite); a fixed seeded held-out set is used
  for the reported metrics.
- Time. The run stops at ``--max-steps`` or ``--time-budget`` seconds,
  whichever comes first, and records the measured cost per step so the
  write-up can say what a longer run would take.
- Controls. ``random_topology`` is vectorised and feasible on this graph
  (``--conditions real,random``). ``degree_preserving_rewire`` in the pinned
  connectorch is a pure-Python loop over ``10 * num_edges`` swap attempts
  (255 million iterations) and is not feasible here; that gap is recorded in
  the results file rather than papered over.
- Input/output populations are chosen by biological role when the metadata
  allows it (README.md's suggestion): input from the optic lobe's visual
  projection neurons, output from descending neurons. Still an arbitrary
  choice for a task with no biological equivalent.
"""

from __future__ import annotations

import collections
import time

import connectorch as ct
import numpy as np
import torch
from connectorch.transforms import random_topology
from torch import nn

from .evaluate import constant_positive_baseline, format_metrics, per_label_metrics
from .model import (
    ModelConfig,
    SettlingEncoder,
    build_decoder,
    build_encoder,
    rescale_initial_weights,
)
from .task import LIST_LENGTH, make_batch, one_hot_sequence
from .train import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, write_json

DEFAULT_INPUT_SUPERCLASS = "visual_projection"
DEFAULT_OUTPUT_SUPERCLASS = "descending_neuron"


def choose_populations(
    brain: ct.Connectome, k_in: int, k_out: int, log=print
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Input/output node ids by role if the metadata has them, else id-order slices."""
    info: dict = {}
    if "superclass" in brain.node_columns:
        counts = collections.Counter(brain.nodes.column("superclass").to_pylist())
        info["superclass_counts"] = dict(counts.most_common())
        log(
            "superclass counts: "
            + ", ".join(f"{k or '(blank)'}={v}" for k, v in counts.most_common(12))
        )
        inputs = brain.where(superclass=DEFAULT_INPUT_SUPERCLASS)
        outputs = brain.where(superclass=DEFAULT_OUTPUT_SUPERCLASS)
        if inputs.size >= k_in and outputs.size >= k_out:
            info["input_population"] = (
                f"superclass={DEFAULT_INPUT_SUPERCLASS} (first {k_in} of {inputs.size} by body id)"
            )
            info["output_population"] = (
                f"superclass={DEFAULT_OUTPUT_SUPERCLASS} (first {k_out} of {outputs.size} by body id)"
            )
            return np.sort(inputs)[:k_in], np.sort(outputs)[:k_out], info
    info["input_population"] = f"first {k_in} node ids"
    info["output_population"] = f"last {k_out} node ids"
    return brain.node_ids[:k_in], brain.node_ids[-k_out:], info


@torch.no_grad()
def evaluate_small_batches(model, x, y, batch: int) -> tuple[float, dict]:
    model.eval()
    logits = torch.cat([model(x[i : i + batch]).cpu() for i in range(0, x.shape[0], batch)])
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
    return loss, per_label_metrics(logits, y)


def run_condition(
    condition: str,
    brain: ct.Connectome,
    args,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> dict:
    seed = args.seed
    torch.manual_seed(seed)
    steps = LIST_LENGTH + model_cfg.settle_steps
    k_in, k_out = model_cfg.num_input_nodes, model_cfg.num_output_nodes
    inputs, outputs, population_info = choose_populations(brain, k_in, k_out)

    t0 = time.perf_counter()
    core = ct.nn.ConnectomeRNN(
        brain,
        input_nodes=inputs,
        output_nodes=outputs,
        weights="trainable",
        initializer=model_cfg.initializer,
        activation=model_cfg.activation,
        leak=model_cfg.leak,
        bias=model_cfg.bias,
        backend="auto",  # never "dense" on this graph
    )
    raw_row_sum = float(core.diagnostics()["max_abs_row_sum"])
    gain = rescale_initial_weights(core, model_cfg.init_row_sum) if model_cfg.init_row_sum else 1.0
    compile_seconds = time.perf_counter() - t0
    diagnostics = core.diagnostics()
    print(f"[{condition}] compiled in {compile_seconds:.1f}s: {core}")
    print(f"[{condition}] diagnostics: {diagnostics}")
    print(f"[{condition}] raw max_abs_row_sum {raw_row_sum:.2f} -> rescaled by {gain:.3g}")

    budget = {b: core.activation_bytes(b, steps) / 2**30 for b in (1, 2, 4, 8, 16)}
    print(
        f"[{condition}] activation GiB per training call, {steps} steps: "
        + ", ".join(f"batch {b}: {g:.2f}" for b, g in budget.items())
    )
    batch = args.batch_size or train_cfg.batch_size
    if batch > 8:
        batch = 2
    while batch > 1 and core.activation_bytes(batch, steps) > args.memory_budget_gib * 2**30:
        batch //= 2
    print(
        f"[{condition}] training batch {batch} ({core.activation_bytes(batch, steps) / 2**30:.2f} GiB of activations)"
    )

    model = ct.nn.ConnectomeModel(
        core,
        encoder=SettlingEncoder(build_encoder(core.num_input_nodes), model_cfg.settle_steps),
        decoder=build_decoder(core.num_output_nodes),
        readout="last",
    ).to(device)
    parameters = ct.nn.count_parameters(model)
    wiring_before = core.edge_index.clone()

    data_gen = torch.Generator().manual_seed(train_cfg.data_seed_offset + seed)
    test_digits, test_y = make_batch(args.test_size, generator=data_gen)
    test_x = one_hot_sequence(test_digits).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.BCEWithLogitsLoss()
    accumulate = max(1, args.accumulate)
    history = []
    start = time.perf_counter()
    step = 0
    examples_seen = 0
    while step < args.max_steps and (time.perf_counter() - start) < args.time_budget:
        model.train()
        optimizer.zero_grad()
        loss_total = 0.0
        for _ in range(accumulate):
            digits, y = make_batch(batch, generator=data_gen)
            x = one_hot_sequence(digits).to(device)
            loss = criterion(model(x), y.to(device)) / accumulate
            loss.backward()
            loss_total += loss.item()
            examples_seen += batch
        if train_cfg.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        optimizer.step()
        step += 1
        elapsed = time.perf_counter() - start
        history.append(
            {"step": step, "loss": loss_total, "seconds": elapsed, "examples_seen": examples_seen}
        )
        if step == 1 or step % args.log_every == 0:
            print(
                f"[{condition}] step {step:>5}  loss {loss_total:.4f}  {elapsed / step:.2f}s/step  {examples_seen} examples  {elapsed:.0f}s"
            )
    train_seconds = time.perf_counter() - start

    print(f"[{condition}] evaluating on {args.test_size} held-out lists ...")
    t_eval = time.perf_counter()
    test_loss, test_metrics = evaluate_small_batches(model, test_x, test_y, args.eval_batch)
    eval_seconds = time.perf_counter() - t_eval
    baseline = constant_positive_baseline(test_y)
    topology_unchanged = bool(torch.equal(wiring_before.cpu(), core.edge_index.cpu()))
    print(f"[{condition}] test:     {format_metrics(test_metrics)}  ({eval_seconds:.0f}s)")
    print(f"[{condition}] baseline: {format_metrics(baseline)}")
    print(f"[{condition}] topology unchanged: {topology_unchanged}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"full_{condition}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in model.state_dict().items()
            },
            "condition": f"full_{condition}",
            "seed": seed,
            "config": model_cfg.to_dict(),
            "fingerprint": core.fingerprint,
            "input_node_ids": [int(i) for i in inputs],
            "output_node_ids": [int(i) for i in outputs],
        },
        ckpt,
    )
    return {
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "nodes": core.num_nodes,
        "edges": core.num_edges,
        "backend": core.backend,
        "populations": population_info,
        "input_node_ids": [int(i) for i in inputs],
        "output_node_ids": [int(i) for i in outputs],
        "parameters": parameters,
        "compile_seconds": compile_seconds,
        "diagnostics_before": diagnostics,
        "raw_max_abs_row_sum": raw_row_sum,
        "init_gain": gain,
        "diagnostics_after": core.diagnostics(),
        "activation_gib_by_batch": budget,
        "batch": batch,
        "accumulate": accumulate,
        "steps": step,
        "examples_seen": examples_seen,
        "seconds_per_step": train_seconds / max(1, step),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "stopped_by": "max_steps" if step >= args.max_steps else "time_budget",
        "history": history,
        "test_loss": test_loss,
        "test": test_metrics,
        "baseline_always_positive_on_test": baseline,
        "topology_unchanged": topology_unchanged,
        "checkpoint": str(ckpt),
    }


def run_full(args, model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    args.max_steps = args.max_steps or 200
    args.time_budget = args.time_budget or 1800.0
    for name, default in (
        ("test_size", 256),
        ("eval_batch", 8),
        ("accumulate", 1),
        ("log_every", 10),
        ("memory_budget_gib", 6.0),
        ("conditions", "real"),
    ):
        if getattr(args, name, None) is None:
            setattr(args, name, default)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    t0 = time.perf_counter()
    brain = ct.datasets.malecns(download=True if args.download else None)
    print(f"loaded in {time.perf_counter() - t0:.1f}s: {brain}")
    print(f"node columns: {brain.node_columns}")

    results = {}
    for condition in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        if condition == "real":
            graph = brain
        elif condition == "random":
            t1 = time.perf_counter()
            graph = random_topology(brain, seed=args.seed)
            print(
                f"random_topology built in {time.perf_counter() - t1:.0f}s: {graph.num_edges:,} edges"
            )
        elif condition == "rewired":
            raise SystemExit(
                "degree_preserving_rewire on 25.5M edges is a 255M-iteration Python loop in the "
                "pinned connectorch; not feasible here. See MILESTONES.md, Milestone 4."
            )
        else:
            raise SystemExit(f"unknown condition {condition!r}")
        results[condition] = run_condition(condition, graph, args, model_cfg, train_cfg, device)
        del graph

    out = {
        "milestone": 4,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "run_args": {k: v for k, v in vars(args).items() if k not in ("milestone",)},
        "controls_note": (
            "degree_preserving_rewire not run: the pinned connectorch implements it as a "
            "pure-Python loop over 10 * num_edges swap attempts (255M iterations on this graph)."
        ),
        "conditions": results,
    }
    write_json(RESULTS_DIR / "milestone4.json", out)
    return out
