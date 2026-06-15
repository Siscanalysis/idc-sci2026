# Diseño industrial óptimo con redes neuronales (SCI 2026)

Reproduction code, data, and manuscript sources for the paper

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

## Repository layout

```
paper/
  first_version/    submitted manuscript (LaTeX, official SCI 2023 template)
  second_version/   revised manuscript addressing the first review round
    main.tex        build with: latexmk -pdf main.tex   (MiKTeX / TeX Live)
    Figuras/        figures, incl. the Spanish vector flow diagrams (TikZ)
rerun/
  cpp/              IDC drivers (built against opennn_core.lib via CMake)
  pybaseline/       pymoo baselines (CMA-ES / DE / GA / PSO, NSGA-II/III, MOEA/D)
  nn_train/         surrogate-training inputs
  results/          raw per-seed CSV outputs and run logs
  tables/           LaTeX result tables (\input into the manuscript)
  aggregate.py      builds tables/ and the Pareto figure from results/
  make_gof.py       goodness-of-fit plots
```

Build artifacts, Python virtual environments, and LaTeX auxiliary files are
not tracked (see `.gitignore`).

## Reproducing the results

1. Build the IDC drivers (Visual Studio + CMake) against a prebuilt
   `opennn_core.lib`:
   ```
   cmake -S rerun -B rerun/build -G "Visual Studio 17 2022" -A x64
   cmake --build rerun/build --config Release
   ```
2. Run the single-objective case: `bash rerun/run_so_photo_wf3.sh`.
3. Aggregate into tables and figures: `python rerun/aggregate.py`.

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
