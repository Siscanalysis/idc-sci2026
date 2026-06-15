"""Standard literature metrics for SO/MO benchmarking."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class SOResult:
    """Single-objective per-run summary."""
    algorithm: str
    problem: str
    seed: int
    best_f: float            # objective value in user's units (signed per direction)
    feasible: bool
    max_violation: float
    evals: int
    walltime_s: float
    x_best: list[float]
    notes: str = ""

    def gap(self, f_star: float) -> float:
        """Relative gap to known optimum f_star (returns +inf if infeasible)."""
        if not self.feasible or f_star == 0.0:
            return float("inf") if not self.feasible else float("nan")
        return abs(self.best_f - f_star) / abs(f_star)

    def to_row(self) -> dict:
        d = asdict(self)
        for i, v in enumerate(d.pop("x_best")):
            d[f"x_{i}"] = v
        return d


@dataclass
class MOResult:
    """Multi-objective per-run summary."""
    algorithm: str
    problem: str
    seed: int
    n_pareto: int             # how many non-dominated points
    hv: float                 # hypervolume (vs. fixed reference point)
    igd: float                # IGD vs. analytical / reference Pareto front
    feasible_pct: float
    evals: int
    walltime_s: float
    notes: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def best_feasible(F: np.ndarray, G: Optional[np.ndarray], maximize: bool = False
                 ) -> tuple[int, float]:
    """
    Return (index, f_value_in_pymoo_min_form) of the best FEASIBLE row in F.
    F: shape (n, 1) (pymoo min form — already negated if user wanted maximize).
    G: shape (n, m) of inequality constraints g <= 0; or None.

    If no row is feasible, returns the index of the smallest-violation row.
    """
    f = F.ravel()
    if G is None or G.size == 0:
        feasible_mask = np.ones_like(f, dtype=bool)
    else:
        feasible_mask = (G <= 1e-6).all(axis=1)

    if feasible_mask.any():
        feasible_idx = np.where(feasible_mask)[0]
        i = feasible_idx[np.argmin(f[feasible_idx])]
    else:
        # No feasible — return least-violating
        violation = np.maximum(G, 0).sum(axis=1)
        i = int(np.argmin(violation))

    return int(i), float(f[i])
