"""
OpennnProblem — pymoo Problem class wrapping an opennn-exported NN + a YAML
problem definition. One source of truth (the YAML) for bounds, constraints,
objectives. Both Branch-A's pymoo runner and the C++ run_idc binaries read
the same YAML.

Constraint conventions (pymoo):
  - Inequality: g(x) <= 0
  - Equality: h(x) == 0
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from pymoo.core.problem import Problem


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Restricted globals for eval — only numpy ops, no builtins.
_SAFE_GLOBALS = {"__builtins__": {}, "np": np, "abs": abs, "min": min, "max": max}


def _compile_expression(expr: str, name: str, input_names: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Compile a YAML constraint or objective expression like
        '4*x_0 + 5*x_1 - 3*x_6 + 9*x_7 - 105'  (g07)
        '-T_s + 0.0193*R'                      (pressure_vessel)
    into a callable taking X (shape n_pop, n_var) and returning a 1-D array.
    The expression's variable names must match `input_names` from the YAML.
    """
    code = compile(expr, f"<{name}>", "eval")

    def fn(X: np.ndarray) -> np.ndarray:
        local = {nm: X[:, i] for i, nm in enumerate(input_names)}
        return eval(code, _SAFE_GLOBALS, local)

    return fn


class OpennnProblem(Problem):
    """
    Generic pymoo problem driven by a YAML spec and an exported opennn NN.

    Supports:
      - Continuous bounded inputs
      - Linear-inequality constraints with lower/upper bounds on a weighted sum
      - Nonlinear inequality constraints as string expressions (g(x) <= 0)
      - Single-objective minimize/maximize on:
          * an output of the NN (by name — currently column 0 of the export),
          * an input variable (no NN call needed),
          * or an analytical expression (evaluated like a constraint).
    """

    def __init__(self, yaml_path: Path, nn):
        spec = load_yaml(yaml_path)
        self.spec = spec
        self.nn = nn

        problem = spec["problem"]
        if problem["type"] not in ("single_objective", "multi_objective"):
            raise NotImplementedError(
                f"Problem type {problem['type']!r} not yet supported."
            )
        self._is_mo = (problem["type"] == "multi_objective")

        inputs = spec["inputs"]
        self.input_names: list[str] = [v["name"] for v in inputs]

        # Mixed-variable handling: when any input has `type: categorical`, the
        # decision-variable space (what pymoo sees) is smaller than the NN
        # input space. Each categorical is one continuous decision var in
        # `[0, n_categories]` (floor-to-int picks the category); each numeric
        # is one continuous decision var bounded by its YAML bounds. At
        # evaluate-time, the decision vector is expanded to the NN's actual
        # input shape (one-hot dummies for each categorical, plus numerics)
        # using `nn.input_names` as the canonical NN input order.
        #
        # The expansion is enabled whenever any input declares
        # `type: categorical`; mixed YAMLs (some categorical, some numeric)
        # work uniformly. Numeric-only YAMLs run the legacy path unchanged.
        self._has_categorical = any(v.get("type") == "categorical" for v in inputs)
        self._dec_kinds: list[tuple[str, str]] = []  # ("num"|"cat", name)
        self._cat_specs: dict[str, dict] = {}        # name -> {categories, dummies, reference}

        xl_list, xu_list = [], []
        for v in inputs:
            if v.get("type") == "categorical":
                cats = v["categories"]
                # NN one-hot convention: `nn_dummies` lists the dummies that
                # actually appear in the NN input vector; the category named
                # by `nn_reference` (or implicitly the one not in dummies)
                # maps to an all-zeros sub-vector. If absent, assume all
                # categories are dummies (no reference dropped).
                dummies = v.get("nn_dummies") or cats
                reference = v.get("nn_reference")
                if reference is None and len(dummies) < len(cats):
                    leftover = [c for c in cats if c not in dummies]
                    reference = leftover[0] if leftover else None
                self._cat_specs[v["name"]] = {
                    "categories": list(cats),
                    "dummies": list(dummies),
                    "reference": reference,
                }
                self._dec_kinds.append(("cat", v["name"]))
                # Decision var: continuous in [0, n_categories]; floored to int
                # at eval. Upper-bound is len(cats) so floor() yields 0..n-1.
                xl_list.append(0.0)
                xu_list.append(float(len(cats)) - 1e-9)
            else:
                self._dec_kinds.append(("num", v["name"]))
                xl_list.append(float(v["bounds"][0]))
                xu_list.append(float(v["bounds"][1]))

        n_vars = len(self._dec_kinds)
        xl = np.array(xl_list, dtype=float)
        xu = np.array(xu_list, dtype=float)

        # Map NN input column index -> producer rule. Each NN input is either:
        #   ("num", decision_idx)   copy this decision var into the column
        #   ("dum", decision_idx, category_idx_within_dummies)
        #                            set 1.0 if floor(dec[dec_idx]) matches,
        #                            else 0.0 (so the reference category is
        #                            implicitly the all-zeros row).
        self._nn_input_recipe: list[tuple] = []
        if self._has_categorical:
            nn_inputs = list(getattr(self.nn, "input_names", []))
            if not nn_inputs:
                raise ValueError(
                    "Categorical inputs require nn.input_names to be set; "
                    "regenerate the .py via export_nn_py.exe.")
            dec_idx_by_name = {nm: i for i, (_, nm) in enumerate(self._dec_kinds)}
            # Per-categorical: dummy_name -> position in the dummies list
            cat_dummy_pos = {
                cat_name: {d: i for i, d in enumerate(cs["dummies"])}
                for cat_name, cs in self._cat_specs.items()
            }
            for col_name in nn_inputs:
                # Is this column a one-hot dummy of some categorical?
                owner = None
                for cat_name, cs in self._cat_specs.items():
                    if col_name in cs["dummies"]:
                        owner = (cat_name, cat_dummy_pos[cat_name][col_name])
                        break
                if owner is not None:
                    cat_name, dum_idx = owner
                    self._nn_input_recipe.append(
                        ("dum", dec_idx_by_name[cat_name], dum_idx))
                elif col_name in dec_idx_by_name:
                    self._nn_input_recipe.append(
                        ("num", dec_idx_by_name[col_name]))
                else:
                    raise ValueError(
                        f"NN input {col_name!r} is neither a YAML numeric "
                        f"input nor a dummy of any categorical.")

        # Parse objective(s) FIRST so we know the NN-output name→index map
        # before constraint parsing (output-scoped constraints reference output
        # names that the objective listing pins to NN column indices).
        # Source-kind: "input" if the objective is one of the inputs,
        # "output" if it's an NN output column. The exported NumPy NN's column
        # order matches the order of Target variables in the dataset, which by
        # convention is the order of `objectives` in the YAML.
        if self._is_mo:
            objs_yaml = spec["objectives"]
        else:
            objs_yaml = [spec["objective"]]
        self._objectives: list[tuple[str, str, str, int]] = []
        self._nn_output_to_idx: dict[str, int] = {}
        nn_output_idx = 0
        for o in objs_yaml:
            name = o["variable"]
            direction = o["direction"]
            if direction not in ("maximize", "minimize"):
                raise ValueError(f"Bad objective direction: {direction!r}")
            if name in self.input_names:
                self._objectives.append((name, direction, "input", self.input_names.index(name)))
            else:
                self._objectives.append((name, direction, "output", nn_output_idx))
                self._nn_output_to_idx[name] = nn_output_idx
                nn_output_idx += 1
        n_outputs = nn_output_idx

        # Build two parallel lists of constraint callables (both feasible when value <= 0):
        # - self._constraints       : input-side; callable signature fn(X) -> 1-D.
        # - self._output_constraints: output-side; callable signature fn(Y) -> 1-D.
        # Constraints with `scope: outputs` reference NN output names (e.g.
        # `strength`) — these must wait until _evaluate computes Y. Without
        # `scope:`, defaults to "inputs" (legacy behaviour preserved).
        self._constraints: list[tuple[Callable[[np.ndarray], np.ndarray], str]] = []
        self._output_constraints: list[tuple[Callable[[np.ndarray], np.ndarray], str]] = []
        for c in spec.get("constraints", []) or []:
            ctype = c["type"]
            scope = c.get("scope", "inputs")
            if scope not in ("inputs", "outputs"):
                raise ValueError(f"Constraint {c.get('name')!r}: scope must be 'inputs' or 'outputs', got {scope!r}")
            if ctype == "linear_inequality":
                if scope == "outputs":
                    # Linear combination of NN outputs, with lower/upper.
                    if n_outputs == 0:
                        raise ValueError(f"Constraint {c['name']!r} has scope:outputs but problem has no NN outputs.")
                    coeffs = np.zeros(n_outputs, dtype=float)
                    for k, w in c["coefficients"].items():
                        if k not in self._nn_output_to_idx:
                            raise ValueError(
                                f"Constraint {c['name']!r} (scope:outputs) references {k!r} which is not "
                                f"a declared NN-output objective. Known NN outputs: {list(self._nn_output_to_idx)}.")
                        coeffs[self._nn_output_to_idx[k]] = float(w)
                    lo = float(c.get("lower", -np.inf))
                    hi = float(c.get("upper",  np.inf))
                    if np.isfinite(lo):
                        self._output_constraints.append(
                            ((lambda Y, _co=coeffs, _lo=lo: _lo - Y @ _co),
                             c["name"] + "_lo"))
                    if np.isfinite(hi):
                        self._output_constraints.append(
                            ((lambda Y, _co=coeffs, _hi=hi: Y @ _co - _hi),
                             c["name"] + "_hi"))
                else:
                    coeffs = np.zeros(n_vars, dtype=float)
                    for k, w in c["coefficients"].items():
                        coeffs[self.input_names.index(k)] = float(w)
                    lo = float(c.get("lower", -np.inf))
                    hi = float(c.get("upper",  np.inf))
                    if np.isfinite(lo):
                        self._constraints.append(
                            ((lambda X, _co=coeffs, _lo=lo: _lo - X @ _co),
                             c["name"] + "_lo"))
                    if np.isfinite(hi):
                        self._constraints.append(
                            ((lambda X, _co=coeffs, _hi=hi: X @ _co - _hi),
                             c["name"] + "_hi"))
            elif ctype == "nonlinear_inequality":
                if scope == "outputs":
                    # Compile against NN output names instead of input names.
                    fn_y = _compile_expression(c["expression"], c["name"],
                                               list(self._nn_output_to_idx))
                    self._output_constraints.append((fn_y, c["name"]))
                else:
                    fn = _compile_expression(c["expression"], c["name"], self.input_names)
                    self._constraints.append((fn, c["name"]))
            else:
                raise NotImplementedError(f"Constraint type {ctype!r} not supported.")

        super().__init__(
            n_var=n_vars,
            n_obj=len(self._objectives),
            n_ieq_constr=len(self._constraints) + len(self._output_constraints),
            xl=xl,
            xu=xu,
        )

    def _expand_to_nn_input(self, X_dec: np.ndarray) -> np.ndarray:
        """Decision-space vector → NN-input vector. Identity when there are
        no categoricals."""
        if not self._has_categorical:
            return X_dec
        n_pop = X_dec.shape[0]
        nn_in = np.zeros((n_pop, len(self._nn_input_recipe)), dtype=float)
        for col, rule in enumerate(self._nn_input_recipe):
            if rule[0] == "num":
                nn_in[:, col] = X_dec[:, rule[1]]
            else:  # "dum"
                dec_idx, dum_idx = rule[1], rule[2]
                # Float [0, n_categories) → int category index in [0, n-1]
                picked = np.floor(X_dec[:, dec_idx]).astype(int)
                nn_in[:, col] = (picked == dum_idx).astype(float)
        return nn_in

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        # Compute NN outputs once if any objective OR any output-scoped
        # constraint needs them.
        needs_nn = (any(src == "output" for _, _, src, _ in self._objectives)
                    or bool(self._output_constraints))
        Y = None
        if needs_nn:
            X_nn = self._expand_to_nn_input(X)
            Y = np.asarray(self.nn.calculate_batch_output(X_nn), dtype=float)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)

        # Stack objectives in pymoo-min form (negate maximize). "input"
        # sources read from the decision space (not the NN-expanded space)
        # so they remain interpretable in user units.
        cols = []
        for _, direction, source, idx in self._objectives:
            f = X[:, idx] if source == "input" else Y[:, idx]
            if direction == "maximize":
                f = -f
            cols.append(f)
        out["F"] = np.column_stack(cols)

        # Constraints: input-side first (decision space), then output-side
        # (NN output space). Order must match super().__init__'s
        # n_ieq_constr so pymoo's G columns line up.
        G_cols = []
        for fn, _ in self._constraints:
            G_cols.append(fn(X))
        for fn, _ in self._output_constraints:
            G_cols.append(fn(Y))
        if G_cols:
            out["G"] = np.column_stack(G_cols).astype(float)

    # ---- public helpers used by the runner ----

    def signed_objective(self, x: np.ndarray) -> float | list[float]:
        """Objective in the user's units (sign per direction). Returns a float
        for single-objective and a list[float] for multi-objective."""
        needs_nn = any(src == "output" for _, _, src, _ in self._objectives)
        y = None
        if needs_nn:
            x_nn = self._expand_to_nn_input(x.reshape(1, -1))
            y = np.asarray(self.nn.calculate_batch_output(x_nn), dtype=float).ravel()
        vals = []
        for _, _, source, idx in self._objectives:
            vals.append(float(x[idx]) if source == "input" else float(y[idx]))
        return vals[0] if len(vals) == 1 else vals

    def feasible(self, x: np.ndarray, tol: float = 1e-6) -> tuple[bool, float]:
        """(is_feasible, max_violation). Considers both input-side and output-side
        constraints; the latter require an NN forward pass."""
        if not (self._constraints or self._output_constraints):
            return True, 0.0
        X = x.reshape(1, -1)
        max_v = 0.0
        for fn, _ in self._constraints:
            v = float(fn(X)[0])
            if v > max_v:
                max_v = v
        if self._output_constraints:
            X_nn = self._expand_to_nn_input(X)
            Y = np.asarray(self.nn.calculate_batch_output(X_nn), dtype=float)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            for fn, _ in self._output_constraints:
                v = float(fn(Y)[0])
                if v > max_v:
                    max_v = v
        return (max_v <= tol), max(0.0, max_v)
