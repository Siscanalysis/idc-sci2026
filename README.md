# Diseño industrial óptimo con redes neuronales (SCI 2026)

Reproduction code and data for the results of the paper

> **Diseño industrial óptimo con redes neuronales**
> S. Scala, R. Lopez. *XXI Simposio CEA de Control Inteligente* (SCI 2026), Salamanca, 24–26 June 2026.

The paper presents **Iterative Domain Contraction (IDC)**, a derivative-free
framework for optimizing the operating conditions of systems modelled by
data-trained neural-network surrogates. IDC combines a *sample–select–contract*
loop with an **affine repair operator** that projects each sample onto the
linear constraints before the surrogate is evaluated, giving feasibility by
construction without rejection sampling. It is illustrated on two industrial
case studies:

- **Single-objective** — photostability of a quaternary organic-photovoltaic
  blend (`photo_wf3`, Olympus suite), under a simplex constraint plus a
  functional donor:acceptor window.
- **Multi-objective** — bi-objective design of a sailing-yacht hull (*Yacht
  Hydrodynamics*, Delft series): residuary resistance vs. beam at fixed Froude
  number.

This repository holds the **reproduction of the paper's results only** (code,
data, surrogates, raw outputs, tables and figures); it does not contain the
manuscript sources.

## Repository layout

```
rerun/
  cpp/              IDC drivers (built against opennn_core.lib via CMake)
  surrogates/       PINNED neural-network surrogates used by the paper
                    (photo_wf3_nn.json, yacht_nn.json/.py, yacht.csv)
  pybaseline/       pymoo baselines (CMA-ES / DE / GA / PSO, NSGA-II/III, MOEA/D)
  nn_train/         surrogate-training inputs
  results/          raw per-seed CSV outputs and run logs (artifact of record)
  tables/           LaTeX result tables regenerated from results/
  figures/          figures regenerated from results/ (Pareto front)
  aggregate.py      builds tables/ and figures/ from results/
  run_so_photo_wf3.sh, run_mo_yacht.sh   driver run scripts
```

Build artifacts and Python virtual environments are not tracked (see `.gitignore`).

## Reproducing the results

1. Build the IDC drivers (Visual Studio + CMake) against a prebuilt
   `opennn_core.lib`:
   ```
   cmake -S rerun -B rerun/build -G "Visual Studio 17 2022" -A x64
   cmake --build rerun/build --config Release
   ```
2. Run the single-objective case: `bash rerun/run_so_photo_wf3.sh`.
3. Regenerate tables and figures: `python rerun/aggregate.py`.

The drivers read the **pinned** surrogates in `rerun/surrogates/`, not the
external dataset library, so the reported numbers do not depend on later
retraining of those surrogates. All reported numbers correspond to a single,
fixed seed (`seed 0`), consistently across the single- and multi-objective
results.

### A note on determinism

- **Single-objective (`photo_wf3`)** is fully deterministic: `run_so_photo_wf3.sh`
  reproduces `results/photo_wf3_idc_full.csv` bit-for-bit (the optimum is
  identical across the 21 internal seeds to six significant figures — used in
  the paper as a robustness/convergence check).
- **Multi-objective (`yacht`)** front *generation* is **stochastic**: a fresh
  `run_mo_yacht.sh` produces a different front even for the same seed. The
  paper's MO numbers therefore correspond to the archived
  `results/yacht_mo_idc_fronts.csv`, which `aggregate.py` re-aggregates to the
  exact reported values (seed 0: HV = 0.554, `|P|` = 286). Treat that CSV as the
  artifact of record.

## Relationship to other work

The full methodological development of IDC and its complete validation suite
are reported in a separate journal manuscript, currently under review (Scala
et al., *Neural network response optimization via iterative domain
contraction*). **This repository contains only the two new case studies of the
SCI 2026 symposium paper** and does not reproduce that method paper. IDC is
implemented in the open-source library [OpenNN](https://github.com/Artelnics/opennn).

## Acknowledgements

Funded by the European Union (Marie Skłodowska-Curie Grant Agreement
no. 101169541 — NEUTEN). Views and opinions expressed are however those of the
author(s) only and do not necessarily reflect those of the European Union or the
European Research Executive Agency (REA). Neither the European Union nor the
granting authority can be held responsible for them.
