"""
Runs a pymoo baseline on an opennn-exported NN. Supports both single-objective
and multi-objective problems, picking the right algorithm pool by YAML
'problem.type'. Writes one row per (algo, seed) to a results CSV.

SO algorithms: CMA-ES, DE, GA
MO algorithms: NSGA-II, NSGA-III, MOEA/D

Usage:
    python run_pymoo.py --problem concrete --algo cmaes --seed 42 --budget 20000
    python run_pymoo.py --problem zdt1     --algo all   --seeds 5 --budget 20000
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from branch_a_same_surface.python.nn_loader import load_nn
from branch_a_same_surface.python.problem_pymoo import OpennnProblem
from metrics.standard_metrics import SOResult, MOResult

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


SO_ALGORITHMS = {
    "cmaes": lambda problem: CMAES(restarts=0),
    "de":    lambda problem: DE(),
    "ga":    lambda problem: GA(pop_size=100),
    "pso":   lambda problem: PSO(pop_size=100),
}


def _mo_algorithms(problem: OpennnProblem):
    """MO algorithms — some need a reference-direction set built per-problem."""
    n_obj = problem.n_obj
    # NSGA-III and MOEA/D need uniform reference directions
    n_partitions = max(12 - n_obj, 4)
    ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=n_partitions)
    algos = {
        "nsga2": lambda: NSGA2(pop_size=100),
        "nsga3": lambda: NSGA3(pop_size=100, ref_dirs=ref_dirs),
    }
    # pymoo's MOEA/D does not support constraints — skip it for constrained problems
    if problem.n_ieq_constr == 0:
        algos["moead"] = lambda: MOEAD(ref_dirs=ref_dirs, n_neighbors=15,
                                       prob_neighbor_mating=0.7)
    return algos


def run_so(problem_name: str, algo_name: str, seed: int, budget: int,
           yaml_path: Path) -> SOResult:
    nn = load_nn(problem_name)
    problem = OpennnProblem(yaml_path, nn)
    if algo_name not in SO_ALGORITHMS:
        raise ValueError(f"Unknown SO algo {algo_name!r}; choose {list(SO_ALGORITHMS)}")
    np.random.seed(seed)
    algo = SO_ALGORITHMS[algo_name](problem)

    t0 = time.perf_counter()
    res = minimize(problem, algo,
                   termination=MaximumFunctionCallTermination(budget),
                   seed=seed, save_history=False, verbose=False)
    walltime = time.perf_counter() - t0

    # Best feasible across full evaluated population
    candidates_X, candidates_F, candidates_G = [], [], []
    def _add(pop):
        if pop is None or len(pop) == 0: return
        candidates_X.append(pop.get("X"))
        candidates_F.append(pop.get("F"))
        g = pop.get("G")
        candidates_G.append(g if g is not None and g.size > 0 else None)
    _add(res.pop)
    for h in (res.history or []):
        _add(h.pop)
    if not candidates_X:
        raise RuntimeError("pymoo returned no evaluated population.")

    X_all = np.vstack(candidates_X)
    F_all = np.vstack([f.reshape(-1, 1) for f in candidates_F]).ravel()
    if any(g is not None for g in candidates_G):
        G_parts = [np.zeros((len(x), problem.n_ieq_constr)) if g is None else g
                   for x, g in zip(candidates_X, candidates_G)]
        G_all = np.vstack(G_parts)
        feas_mask = (G_all <= 1e-6).all(axis=1)
    else:
        feas_mask = np.ones_like(F_all, dtype=bool)

    if feas_mask.any():
        i_pool = np.where(feas_mask)[0]
        i = i_pool[int(np.argmin(F_all[i_pool]))]
        notes = ""
    else:
        violation = np.maximum(G_all, 0).sum(axis=1)
        i = int(np.argmin(violation))
        notes = "no feasible point found across full history"

    x_best = X_all[i]
    f_signed = problem.signed_objective(x_best)
    feasible, max_v = problem.feasible(x_best)
    return SOResult(
        algorithm=algo_name, problem=problem_name, seed=seed,
        best_f=f_signed if isinstance(f_signed, float) else float(f_signed[0]),
        feasible=feasible, max_violation=max_v,
        evals=int(res.algorithm.evaluator.n_eval) if res.algorithm else budget,
        walltime_s=walltime,
        x_best=[float(v) for v in x_best],
        notes=notes,
    )


def run_mo(problem_name: str, algo_name: str, seed: int, budget: int,
           yaml_path: Path) -> tuple[MOResult, np.ndarray, np.ndarray]:
    """Returns (summary, F_pareto, X_pareto) — pareto front in pymoo-min units."""
    nn = load_nn(problem_name)
    problem = OpennnProblem(yaml_path, nn)
    mo_algos = _mo_algorithms(problem)
    if algo_name not in mo_algos:
        raise ValueError(f"Unknown MO algo {algo_name!r}; choose {list(mo_algos)}")
    np.random.seed(seed)
    algo = mo_algos[algo_name]()

    t0 = time.perf_counter()
    res = minimize(problem, algo,
                   termination=MaximumFunctionCallTermination(budget),
                   seed=seed, verbose=False)
    walltime = time.perf_counter() - t0

    if res.X is None or res.F is None:
        # No feasible front — try the final population
        F_pareto = np.atleast_2d(res.pop.get("F"))
        X_pareto = np.atleast_2d(res.pop.get("X"))
        notes = "no non-dominated front; using final population"
    else:
        F_pareto = np.atleast_2d(res.F)
        X_pareto = np.atleast_2d(res.X)
        notes = ""

    # Compute HV and IGD using problem-specific reference points / fronts
    spec = problem.spec
    ref_point = _hv_reference_point(spec, F_pareto)
    hv = float(HV(ref_point=ref_point)(F_pareto))
    pf_ref = _reference_pareto_front(spec)
    igd = float(IGD(pf_ref)(F_pareto)) if pf_ref is not None else float("nan")

    # Feasibility
    if problem.n_ieq_constr > 0:
        out = {}
        problem._evaluate(X_pareto, out)
        feas = (out["G"] <= 1e-6).all(axis=1)
        feasible_pct = 100.0 * float(feas.mean())
    else:
        feasible_pct = 100.0

    summary = MOResult(
        algorithm=algo_name, problem=problem_name, seed=seed,
        n_pareto=int(F_pareto.shape[0]),
        hv=hv, igd=igd,
        feasible_pct=feasible_pct,
        evals=int(res.algorithm.evaluator.n_eval) if res.algorithm else budget,
        walltime_s=walltime, notes=notes,
    )
    return summary, F_pareto, X_pareto


def _hv_reference_point(spec: dict, F_pareto: np.ndarray) -> np.ndarray:
    """Per-problem fixed reference point (1.1 × analytical nadir)."""
    name = spec["problem"]["name"]
    if name == "zdt1":
        from analysis import zdt1_analytical as a
        return a.HV_REF_POINT
    if name == "constr":
        from analysis import constr_analytical as a
        return a.HV_REF_POINT
    if name == "moeed13":
        from analysis import moeed13_analytical as a
        return a.HV_REF_POINT
    # Generic fallback: 1.1 × max of observed front (less reproducible)
    return F_pareto.max(axis=0) * 1.1


def _reference_pareto_front(spec: dict) -> np.ndarray | None:
    """Sample of the analytical Pareto front for IGD."""
    name = spec["problem"]["name"]
    if name == "zdt1":
        from analysis import zdt1_analytical as a
        return a.pareto_front(100)
    if name == "constr":
        from analysis import constr_analytical as a
        return a.pareto_front(100)
    return None


def write_so(results: list[SOResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in results]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows: w.writerow(row)
    print(f"[OK] wrote {len(rows)} SO row(s) to {out_csv}")


def write_mo(results: list[MOResult], fronts: list[tuple[str, int, np.ndarray, np.ndarray]],
             out_csv: Path, fronts_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in results]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows: w.writerow(row)
    print(f"[OK] wrote {len(rows)} MO row(s) to {out_csv}")

    # Per-point CSV — every row is a Pareto point with (algo, seed, F..., X...)
    front_rows = []
    for algo, seed, F, X in fronts:
        for i in range(F.shape[0]):
            r = {"algorithm": algo, "seed": seed}
            for j in range(F.shape[1]): r[f"F_{j}"] = float(F[i, j])
            for j in range(X.shape[1]): r[f"x_{j}"] = float(X[i, j])
            front_rows.append(r)
    if front_rows:
        with open(fronts_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(front_rows[0].keys()))
            w.writeheader()
            for row in front_rows: w.writerow(row)
        print(f"[OK] wrote {len(front_rows)} pareto point(s) to {fronts_csv}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--algo", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    yaml_path = REPO_ROOT / "problems" / f"{args.problem}.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] YAML not found: {yaml_path}", file=sys.stderr); return 1

    # Detect SO vs MO from the YAML
    import yaml as _yaml
    spec = _yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    is_mo = (spec["problem"]["type"] == "multi_objective")

    if is_mo:
        algos = list(_mo_algorithms(OpennnProblem(yaml_path, load_nn(args.problem))))
        if args.algo != "all": algos = [args.algo]
    else:
        algos = list(SO_ALGORITHMS) if args.algo == "all" else [args.algo]

    seeds = list(range(42, 42 + args.seeds)) if args.seeds else [args.seed]

    if is_mo:
        results: list[MOResult] = []
        fronts: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        for algo in algos:
            for seed in seeds:
                print(f"[run] algo={algo} seed={seed} budget={args.budget}")
                r, F, X = run_mo(args.problem, algo, seed, args.budget, yaml_path)
                print(f"   n_pareto={r.n_pareto} HV={r.hv:.4f} IGD={r.igd:.4f} "
                      f"feasible%={r.feasible_pct:.1f} t={r.walltime_s:.1f}s")
                results.append(r)
                fronts.append((algo, seed, F, X))
        out = args.out or (REPO_ROOT / "results" / "branch_a"
                           / f"{args.problem}_pymoo.csv")
        fronts_csv = out.with_name(out.stem + "_fronts.csv")
        write_mo(results, fronts, out, fronts_csv)
    else:
        results: list[SOResult] = []
        for algo in algos:
            for seed in seeds:
                print(f"[run] algo={algo} seed={seed} budget={args.budget}")
                r = run_so(args.problem, algo, seed, args.budget, yaml_path)
                print(f"   best_f={r.best_f:.4f} feasible={r.feasible} "
                      f"violation={r.max_violation:.3e} evals={r.evals} t={r.walltime_s:.2f}s")
                results.append(r)
        out = args.out or (REPO_ROOT / "results" / "branch_a"
                           / f"{args.problem}_pymoo.csv")
        write_so(results, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
