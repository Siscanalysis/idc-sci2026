#!/usr/bin/env bash
# MO yacht: bi-objective IDC (minimize residuary resistance [NN output] vs
# minimize length/beam L/B [input #4], Froude fixed at 0.2875). 21 seeds, two
# configs (idc_mo_starter, idc_mo_canonical). Uses the PINNED surrogate.
#
# NOTE ON REPRODUCIBILITY: the multi-objective driver is *stochastic* — front
# sizes and the best-HV seed vary run to run, so HV, |P| and the recommended
# knee shift between runs. The paper's MO numbers correspond to the archived
# fronts in results/yacht_mo_idc_fronts.csv, which re-aggregate exactly via
# aggregate.py. This script regenerates a *new* (different) draw; keep the
# archived CSV as the artifact of record unless you intend to refresh the table.
set -u
RR="C:/Users/Artelnics/Desktop/experiments/Papers/Simposio_CEA_2026/rerun"
IDC="$RR/build/bin/Release/run_idc_yacht_mo2.exe"
NN="$RR/surrogates/yacht_nn.json"
mkdir -p "$RR/results"
echo "=== yacht bi-objective : IDC on pinned surrogate (stochastic) ==="
# usage: <nn_json> <fronts_csv> <obj2_idx 0..5> <min|max> <froude_fix>
"$IDC" "$NN" "$RR/results/yacht_mo_idc_fronts.csv" 4 min 0.2875
echo "=== MO yacht DONE ==="
