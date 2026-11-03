"""CI check for the interior Tchebycheff table (tab:exact_interior).

Solver-free: re-derives the augmented Tchebycheff Z of every committed interior
optimum from its (C_F, P) and the lambda 0/1 anchors, using the same
normalization builder the exact driver uses, and asserts it matches the stored Z.
This guards the normalization/scalarization pipeline without needing Gurobi.
The expensive exact re-solve lives in code/exact/run_exact_interior.py and is run
offline (it reproduces these same numbers; see verify_interior.py --resolve).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "exact"))

import verify_interior as V

REFERENCE = os.path.join(ROOT, "Outputs", "Exact", "Afyon_heuristic_vs_exact.csv")


def test_interior_Z_reproduced_from_anchors():
    by_profile = V.load_rows(REFERENCE)
    n_ok, n_bad, _n_heur, n_heur_bad, problems = V.check_arithmetic(by_profile)
    assert n_bad == 0 and n_heur_bad == 0, "interior reproduction mismatches:\n" + \
        "\n".join(problems)
    assert n_ok == 45, f"expected 45 interior points, checked {n_ok}"
