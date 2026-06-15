#!/usr/bin/env bash
# SO photo_wf3: IDC on the full-data surrogate (trained on all 1040 mixtures),
# 21 internal seeds; the aggregator selects a single representative run.
# Simplex equality enforced by IDC's affine repair. 40k-eval budget.
# Uses the PINNED surrogate in surrogates/ (paper-consistent; the live
# benchmark_datasets copy was retrained after the paper and gives different
# numbers). This run is deterministic: it reproduces results/ exactly.
set -u
RR="C:/Users/Artelnics/Desktop/experiments/Papers/Simposio_CEA_2026/rerun"
IDC="$RR/build/bin/Release/run_idc_photo_wf3.exe"
NN="$RR/surrogates/photo_wf3_nn.json"
YAML="$RR/pybaseline/problems/photo_wf3.yaml"
mkdir -p "$RR/results"
echo "=== photo_wf3 : IDC on pinned full-data surrogate ==="
"$IDC" "$NN" "$RR/results/photo_wf3_idc_full.csv" "$YAML"
echo "=== SO photo_wf3 (full-data) DONE ==="
