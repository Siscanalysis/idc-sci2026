"""Goodness-of-fit parity figures for the two surrogates used in the paper.
  - yacht  : full-data yacht_nn (the MO surrogate)         -> fig_gof_yacht.pdf
  - photo  : photo_wf3 top-5% holdout seed_0 (SO surrogate) -> fig_gof_photo_wf3.pdf
Writes PDFs into ../paper/Figuras/. Uses the exported OpenNN .py predictors.
"""
import importlib.util as iu
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BD = Path(r"C:/Users/Artelnics/Desktop/benchmark_datasets")
FIG = Path(__file__).resolve().parent.parent / "paper" / "Figuras"
FIG.mkdir(parents=True, exist_ok=True)

def load_nn(py_path):
    spec = iu.spec_from_file_location("nn_" + Path(py_path).stem + str(abs(hash(py_path))), py_path)
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    return m.NeuralNetwork()

def r2_rmse(y, yh):
    y, yh = np.ravel(y), np.ravel(yh)
    ss_res = float(np.sum((y - yh) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, float(np.sqrt(np.mean((y - yh) ** 2)))

def parity(y, yh, title, fname, unit):
    r2, rmse = r2_rmse(y, yh)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(y, yh, s=10, alpha=0.5, edgecolor="none", color="#1f77b4")
    lo, hi = float(min(y.min(), yh.min())), float(max(y.max(), yh.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"Medido {unit}"); ax.set_ylabel(f"Predicho {unit}")
    ax.set_title(title, fontsize=9)
    ax.text(0.05, 0.92, f"$R^2={r2:.3f}$\nRMSE$={rmse:.3g}$", transform=ax.transAxes,
            va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout(); fig.savefig(FIG / fname, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {fname}: R2={r2:.4f} RMSE={rmse:.4g} (n={len(y)})")

# --- yacht (full-data surrogate, MO) ---
ya = BD / "so_realdata/O_uci_regression/yacht_hydrodynamics"
df = pd.read_csv(ya / "data/yacht.csv")
Xy = df.iloc[:, :6].to_numpy(float); yy = df.iloc[:, 6].to_numpy(float)
nn_y = load_nn(str(ya / "nn/yacht_nn.py"))
pred_y = np.asarray(nn_y.calculate_batch_output(Xy), float).ravel()
parity(yy, pred_y, "Yates: resistencia residual", "fig_gof_yacht.pdf", "(adim.)")

# --- photo_wf3 (holdout seed_0, SO) ---
wf = BD / "so_realdata/P_olympus_campaigns/olympus_photo_wf3"
dp = pd.read_csv(wf / "data/data.csv", header=None)
Xp = dp.iloc[:, :4].to_numpy(float); yp = dp.iloc[:, 4].to_numpy(float)
nn_p = load_nn(str(wf / "nn/top05_holdout/seed_0/neural_network.py"))
pred_p = np.asarray(nn_p.calculate_batch_output(Xp), float).ravel()
parity(yp, pred_p, "Fotodegradación OPV (photo\\_wf3)", "fig_gof_photo_wf3.pdf", "(adim.)")
