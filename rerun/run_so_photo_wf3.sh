#!/usr/bin/env bash
# SO photo_wf3: IDC on the full-data surrogate (trained on all 1040 mixtures),
# 21 internal seeds; the aggregator selects a single representative run.
# Simplex equality enforced by IDC's affine repair. 40k-eval budget.
set -u
RR="C:/Users/Artelnics/Desktop/experiments/Papers/Simposio_CEA_2026/rerun"
WF="C:/Users/Artelnics/Desktop/benchmark_datasets/so_realdata/P_olympus_campaigns/olympus_photo_wf3"
IDC="$RR/build/bin/Release/run_idc_photo_wf3.exe"
YAML="$RR/pybaseline/problems/photo_wf3.yaml"
mkdir -p "$RR/results"
echo "=== photo_wf3 : IDC on full-data surrogate ==="
"$IDC" "$WF/nn/photo_wf3_nn.json" "$RR/results/photo_wf3_idc_full.csv" "$YAML"
echo "=== SO photo_wf3 (full-data) DONE ==="
