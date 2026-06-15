"""Load the exported numpy NeuralNetwork class for a given problem name.

NN artifacts (.py / .bin / .json / .xml / parity CSVs) live in the canonical
catalogue at ``benchmark_datasets/<section>/<entry>/nn/<basename>_nn.*``. This
module discovers the .py file by globbing the catalogue, so callers only need
the short basename (e.g. ``"airfoil"``, ``"concrete_uci"``, ``"pressure_vessel_cat"``).
"""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

BENCHMARK_DATASETS = Path(r"C:\Users\Artelnics\Desktop\benchmark_datasets")


@lru_cache(maxsize=None)
def find_nn_py(problem: str) -> Path | None:
    """Return the canonical ``<problem>_nn.py`` path in benchmark_datasets, or
    None if no match exists. Searches both so_*/ and mo_*/ subtrees."""
    for subtree in ("so_realdata", "so_synth", "mo_realdata", "mo_synth"):
        root = BENCHMARK_DATASETS / subtree
        if not root.exists():
            continue
        for hit in root.rglob(f"{problem}_nn.py"):
            if hit.parent.name == "nn":
                return hit
    return None


def load_nn(problem: str):
    """Import ``<benchmark_datasets>/.../<problem>_nn.py`` and return an
    instantiated NeuralNetwork.

    Path resolution priority:
      1. Env var ``OPENNN_NN_OVERRIDE`` — absolute path to a .py file. Used by
         the holdout runner to substitute a capped surrogate.
      2. Default: ``find_nn_py(problem)`` in the canonical catalogue.
    """
    override = os.environ.get("OPENNN_NN_OVERRIDE")
    if override:
        py_path = Path(override)
    else:
        py_path = find_nn_py(problem)
        if py_path is None:
            raise FileNotFoundError(
                f"No {problem}_nn.py found under {BENCHMARK_DATASETS}. "
                f"Run the matching train_and_export_{problem}.exe first, "
                f"or set OPENNN_NN_OVERRIDE."
            )
    if not py_path.exists():
        raise FileNotFoundError(f"Exported NN not found at {py_path}.")
    mod_name = f"opennn_export_{problem}_{abs(hash(str(py_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "NeuralNetwork"):
        raise AttributeError(f"{py_path} does not define class NeuralNetwork.")
    return module.NeuralNetwork()
