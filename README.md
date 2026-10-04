# agent-adoption-kit

An open-source reference implementation of NANTE™, a five-stage method for measuring enterprise AI adoption from the production telemetry agents already emit — no surveys, computed at cohort level.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21943954.svg)](https://doi.org/10.5281/zenodo.21943954)
[![arXiv](https://img.shields.io/badge/arXiv-2608.23617-b31b1b.svg)](https://arxiv.org/abs/2608.23617)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Given an event stream plus a provisioned-user roster, the kit classifies where a population sits in the adoption process, locates the stall point, and flags characteristic failure signatures. Everything runs deterministically on SQLite inside your own environment. Companion code for the paper *Adoption Telemetry: Measuring Enterprise AI Adoption from Production Signals* (see [Paper](#paper)).

Damon A. Young · Polywise Partners · [polywisepartners.com](https://polywisepartners.com)

## Install

Requires Python ≥ 3.11. From a clone of this repo:

```bash
pip install -e .
```

This installs the `aak` command. Two optional extras:

- `pip install -e ".[report]"` — adds the Anthropic SDK, needed only by `aak report` in its default theme. `simulate`, `analyze`, and `--theme paper` never load it.
- `pip install -e ".[dev]"` — adds `pytest` plus the report extra, for running the tests.

PDF and PNG figure export (`--export-figure`) needs the system cairo library, which the Python wheels do not bundle: `apt install libcairo2` on Debian/Ubuntu, `brew install cairo` on macOS.

## Quickstart

Simulate a population showing the paper's headline failure mode, then diagnose one cohort. Both steps finish in a few seconds, need no API key, and produce identical output on every run at the same seed.

```bash
aak simulate --pathology shallow_plateau --seed 42 --out shallow_plateau.db
aak analyze shallow_plateau.db --cohort cohort-1
```

Output:

```text
Generated 33927 events for 500 provisioned users over 150 days -> shallow_plateau.db
=== cohort-1 ===
as_of: 2025-05-30   observation_days: 149
insufficient_window: False   insufficient_sample_size: False
stall_point: navigate
nante_score: 49.9

Stage distribution:
  notice    :    0.0%  [healthy]
  attempt   :    0.6%  [healthy]
  navigate  :   99.4%  [failing]
  transform :    0.0%  [failing]
  embed     :    0.0%  [healthy]

Flags: shallow_plateau

Interventions:
  [Stalled at Navigate] The tool doesn't fit the real workflow yet.
    -> Redesign the workflow so the tool sits inside the actual task sequence, instead of alongside it.
    not this: more licenses
  [Shallow plateau] Breadth is high and retention looks healthy, but depth of use stays flat — the shallow-plateau pattern the brief's ~2% figure describes.
    -> A workflow-embedding intervention: redesign around deeper task integration, not just more usage of the same shallow interaction.
    not this: more usage of the same shallow interaction
```

Nearly everyone in this cohort uses the agent week after week (99.4% at Navigate), but nobody crosses into Transform: sessions stay shallow, so the Navigate → Transform graduation rate reads *failing* and the stall point lands at Navigate. That is what a stall looks like — a wall at one stage boundary, not low usage. The `shallow_plateau` flag names the pattern (breadth high, depth flat), and the intervention rules say what to try and what not to. For contrast, `--pathology healthy` at the same seed reports no stall point and 18% of the cohort at Embed.

The other pathologies you can simulate: `healthy`, `awareness_gap`, `ability_gap`, `reinforcement_decay`, `champion_dependency`. Omit `--cohort` to diagnose every cohort in the database.

## The five stages

Each stage is a computable predicate over a user's events; the cohort diagnosis comes from the graduation rate between adjacent stages.

| Stage | Question it answers |
|---|---|
| **Notice** | Has this provisioned user invoked the agent at all? |
| **Attempt** | Did a first use happen, without yet becoming recurring? |
| **Navigate** | Is use recurring across distinct weeks? |
| **Transform** | Is recurring use deep and mostly successful — multi-step work that lands? |
| **Embed** | Is use continuous enough that withdrawing the agent would disrupt output? |

The thresholds behind these predicates (`aak/analytics/thresholds.yaml`) are proposed defaults, published openly to be tested and disproven — not a calibrated model. Every value is marked PROPOSED in the file. Treat every number the kit reports as a structured hypothesis about a cohort.

## Status

Reference implementation accompanying the paper. Classification is deterministic and operates at cohort level — never per individual. The suite distinguishes six cohort profiles: one healthy trajectory and five failure modes.

The ingestion adapters in `aak/ingest/` are unimplemented stubs. Every result the kit produces today derives from the synthetic populations generated by the simulator. Running NANTE against a live environment requires writing your own extractors, resolving user identity across systems, and mapping events into the canonical schema — none of which ship here.

## Paper

*Adoption Telemetry: Measuring Enterprise AI Adoption from Production Signals* — Damon A. Young, 2026.

- arXiv: https://arxiv.org/abs/2608.23617
- Zenodo concept DOI (resolves to the latest version): https://doi.org/10.5281/zenodo.21943954

```bibtex
@misc{young2026adoptiontelemetry,
  author        = {Young, Damon A.},
  title         = {Adoption Telemetry: Measuring Enterprise {AI} Adoption from Production Signals},
  year          = {2026},
  eprint        = {2608.23617},
  archivePrefix = {arXiv},
  doi           = {10.5281/zenodo.21943954},
  url           = {https://arxiv.org/abs/2608.23617}
}
```

## Reproduce the paper's results

Everything below is deterministic at a fixed seed (42) and needs no API key or network. From the repo root:

```bash
make results   # the six per-pathology diagnoses behind Table 1 (stage distribution, score, stall, flags)
make figures   # Figure 2 — paper-theme comparison, exported to out/comparison.pdf
```

These two targets wrap the `aak` CLI — nothing more. Equivalently, by hand:

```bash
# make results wraps: simulate the six seed-42 populations (one healthy + five failure modes),
# then classify each population's cohort-1.
mkdir -p out
for p in healthy shallow_plateau awareness_gap ability_gap reinforcement_decay champion_dependency; do
  aak simulate --pathology "$p" --seed 42 --out "out/$p.db"
  echo "== $p =="; aak analyze "out/$p.db" --cohort cohort-1
done

# make figures wraps: paper-theme comparison of the healthy vs shallow-plateau run, exported to PDF.
aak report --compare out/healthy.db out/shallow_plateau.db \
  --ref-cohort cohort-1 --observed-cohort cohort-1 \
  --theme paper --out out/comparison.html --export-figure out/comparison.pdf
```

In the paper repo, its own `make results` and `make figures` targets drive these same commands (same seed 42, same `cohort-1`, same `--theme paper`) and additionally format the output into the booktabs `tables/pathologies.tex` (Table 1) and pin `figures/comparison.pdf` (Figure 2).

`aak report` without `--theme paper` renders the default brand theme, which generates cohort commentary through the Anthropic API. It needs the `report` extra installed and `ANTHROPIC_API_KEY` set (see `.env.example`).

## Run the tests

```bash
pytest
```

The suite is fully reproducible: fixed seeds, no API key, no network. CI runs it on every push and pull request (`.github/workflows/ci.yml`).

## License & trademark

Licensed under Apache-2.0 (see `LICENSE`). The project name and its methodology names, including NANTE, are reserved and not licensed as trademarks (see `TRADEMARKS.md`).

## Data

No client data. All results derive from synthetic populations generated by the simulator at a fixed seed.
