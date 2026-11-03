"""verify_interior.py -- re-verification checker for the interior optima table.

Checks a stored interior-optima file (as written by run_exact_interior.py, or the
archived ``*_heuristic_vs_exact.csv``) against the normalization builder, and -- in
``--resolve`` mode -- against a fresh exact re-solve.

Default (solver-free):
    For each profile the anchors are read off the lambda = 0 and lambda = 1 rows
    (cost and latency endpoints), the augmented Tchebycheff Z is recomputed from
    the stored (C_F, P) with normalization.tchebycheff_Z, and compared to the
    stored Z of every interior row.  This re-derives the whole Z column.
    When the file also carries a heuristic column (C_F_heur/P_heur), the
    heuristic-vs-exact "match" is re-checked too.

--resolve (needs Gurobi):
    Re-runs run_exact_interior on the given city/inputs and compares the fresh
    C_F, P and Z to the stored numbers.

Usage:
    python code/exact/verify_interior.py --csv Outputs/Exact/Afyon_heuristic_vs_exact.csv
    python code/exact/verify_interior.py --csv <stored.csv> --resolve --city Afyon --inputs Inputs
"""
import os, sys, csv, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import normalization as norm

ZTOL = 1e-5        # stored Z are rounded to 6 dp
VTOL = 1e-3        # C_F / P tolerance


def _num(row, *names):
    """First present, non-empty numeric column among ``names`` (else None)."""
    for n in names:
        if n in row and row[n] not in ("", None):
            return float(row[n])
    return None


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_profile = defaultdict(dict)       # profile -> lambda -> dict(C_F, P, Z, heur...)
    for r in rows:
        lam = round(float(r["lambda"]), 3)
        by_profile[r["profile"]][lam] = dict(
            C_F=_num(r, "C_F_exact", "C_F"),
            P=_num(r, "P_exact", "P"),
            Z=_num(r, "Z_exact", "Z"),
            C_F_heur=_num(r, "C_F_heur"),
            P_heur=_num(r, "P_heur"),
        )
    return by_profile


def check_arithmetic(by_profile):
    n_ok = n_bad = n_heur = n_heur_bad = 0
    problems = []
    for profile, bylam in sorted(by_profile.items()):
        if 0.0 not in bylam or 1.0 not in bylam:
            problems.append(f"{profile}: missing lambda 0/1 anchor rows")
            continue
        anc = norm.anchors(Cmin=bylam[0.0]["C_F"], P_C=bylam[0.0]["P"],
                           C_P=bylam[1.0]["C_F"], Pmin=bylam[1.0]["P"])
        for lam, rec in sorted(bylam.items()):
            if lam in (0.0, 1.0) or rec["Z"] is None:
                continue
            z = norm.tchebycheff_Z(rec["C_F"], rec["P"], lam, anc)
            if abs(z - rec["Z"]) <= ZTOL:
                n_ok += 1
            else:
                n_bad += 1
                problems.append(f"{profile} lambda={lam}: Z recomputed {z:.6f} "
                                f"!= stored {rec['Z']:.6f}")
            # heuristic re-verification: heur point should match the exact point
            if rec["C_F_heur"] is not None and rec["P_heur"] is not None:
                n_heur += 1
                if (abs(rec["C_F_heur"] - rec["C_F"]) > VTOL
                        or abs(rec["P_heur"] - rec["P"]) > VTOL):
                    n_heur_bad += 1
                    problems.append(f"{profile} lambda={lam}: heuristic "
                                    f"({rec['C_F_heur']}, {rec['P_heur']}) != exact "
                                    f"({rec['C_F']}, {rec['P']})")
    return n_ok, n_bad, n_heur, n_heur_bad, problems


def check_resolve(by_profile, city, inputs, rho):
    import run_exact_interior as drv
    n_ok = n_bad = 0; problems = []
    for profile, bylam in sorted(by_profile.items()):
        anc, rows = drv.run_profile(city, inputs, profile, rho, 1200.0)
        fresh = {round(r["lam"], 3): r for r in rows}
        for lam, rec in sorted(bylam.items()):
            if lam not in fresh:
                continue
            fr = fresh[lam]
            ok = (abs(fr["C_F"] - rec["C_F"]) <= VTOL and abs(fr["P"] - rec["P"]) <= VTOL)
            if ok and rec["Z"] is not None and fr["Z"] is not None:
                ok = abs(fr["Z"] - rec["Z"]) <= ZTOL
            if ok:
                n_ok += 1
            else:
                n_bad += 1
                problems.append(f"{profile} lambda={lam}: fresh "
                                f"(C_F={fr['C_F']}, P={fr['P']}, Z={fr['Z']}) != stored "
                                f"(C_F={rec['C_F']}, P={rec['P']}, Z={rec['Z']})")
    return n_ok, n_bad, problems


def main():
    ap = argparse.ArgumentParser(description="Verify interior optima vs stored numbers.")
    ap.add_argument("--csv", required=True, help="stored interior CSV to verify")
    ap.add_argument("--resolve", action="store_true", help="also re-solve exactly (Gurobi)")
    ap.add_argument("--city", default="Afyon")
    ap.add_argument("--inputs", default="Inputs")
    ap.add_argument("--rho", type=float, default=norm.RHO)
    a = ap.parse_args()

    by_profile = load_rows(a.csv)
    n_ok, n_bad, n_heur, n_heur_bad, problems = check_arithmetic(by_profile)
    print(f"[normalization] Z reproduced: {n_ok} ok, {n_bad} mismatched")
    if n_heur:
        print(f"[heuristic re-verification] exact==heuristic: "
              f"{n_heur - n_heur_bad}/{n_heur} match")

    if a.resolve:
        import experiment  # noqa  (Gurobi licence discovery)
        r_ok, r_bad, r_problems = check_resolve(by_profile, a.city, a.inputs, a.rho)
        print(f"[re-solve] exact reproduced: {r_ok} ok, {r_bad} mismatched")
        problems += r_problems
        n_bad += r_bad

    for p in problems:
        print("  MISMATCH:", p)
    fail = n_bad + n_heur_bad
    print("RESULT:", "PASS" if fail == 0 else f"FAIL ({fail} mismatches)")
    sys.exit(bool(fail))


if __name__ == "__main__":
    main()
