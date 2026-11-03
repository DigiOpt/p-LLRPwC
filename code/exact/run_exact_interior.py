"""run_exact_interior.py -- exact interior Tchebycheff optima (Table tab:exact_interior).

Reproduces the interior preference experiment of paper 6.4.2: for every priority
profile and every lambda in {0.1, ..., 0.9} it solves the normalized augmented
weighted Tchebycheff problem to proven optimality with the commodity-flow MILP
(all Afyon problems are flow; the formulation is exactness preserving).  For each
profile the four lexicographic anchors are computed first (lambda = 0 -> cost
anchor Cmin / P_C; lambda = 1 -> latency anchor Pmin / C_P), the normalization is
built from them (see normalization.py), and the nine interior problems follow.

C_F and P are recomputed from the extracted routes and independently verified
(capacity, duration, depot throughput, coverage, fleet) before anything is
reported -- the objective value is never trusted directly.

Usage:
    python code/exact/run_exact_interior.py --city Afyon
    python code/exact/run_exact_interior.py --city Afyon --profiles Equal,Tiered
    python code/exact/run_exact_interior.py --city Afyon --inputs Inputs --out Outputs/Exact

Outputs (under --out, default Outputs/Exact):
    <city>_exact_validation.md     anchors + per-lambda table per profile
    <city>_heuristic_vs_exact.csv  one row per (profile, lambda) incl. lambda 0/1
    <city>_interior_routes.json    routes and arrivals of every interior optimum
"""
import os, sys, json, csv, time, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import normalization as norm
import run_exact_anchors as R        # recompute(), verify(), route_detail(), FLAGS

PROFILES_ALL = ["Equal", "Tiered", "Demand_Correlated",
                "Negative_Correlated", "Hyper_Critical"]
LAMBDAS = [round(0.1 * k, 1) for k in range(1, 10)]     # 0.1 .. 0.9
MIP_GAP, MIP_GAP_ABS = 1e-9, 1e-6


def _extract_routes(inst, m):
    """Open depots and routes from an optimized flow model (as in run_exact_anchors)."""
    md, x, y = inst.m, m._x, m._y
    I_open = [i for i in inst.I if y[i].X > 0.5]
    routes = []
    for i in inst.I:
        succ = defaultdict(list)
        for (u, v) in x[i].keys():
            if x[i][u, v].X > 0.5:
                succ[u].append(v)
        while succ[i]:
            cur = succ[i].pop(); path = []
            while cur != i:
                path.append(cur - md); cur = succ[cur].pop()
            routes.append((i, tuple(path)))
    return I_open, routes


def _solve_flow(inst, *, lexicographic=None, tchebycheff=None, limit_s=1200.0):
    """Solve one flow model, extract & recompute, and verify the routes."""
    from llrp_model import build_model_flow
    m = build_model_flow(inst, latency_obj=True, enforce_duration=True,
                         lexicographic=lexicographic, tchebycheff=tchebycheff, **R.FLAGS)
    m.Params.OutputFlag = 0
    m.Params.MIPGap = MIP_GAP; m.Params.MIPGapAbs = MIP_GAP_ABS
    m.Params.TimeLimit = limit_s
    t = time.time(); m.optimize(); rt = time.time() - t
    I_open, routes = _extract_routes(inst, m)
    C_F, P, det = R.recompute(inst, I_open, routes)
    fails = R.verify(inst, I_open, routes, C_F, P)
    status = {2: "OPTIMAL", 9: "TIME_LIMIT"}.get(m.Status, str(m.Status))
    return dict(C_F=C_F, P=P, status=status, time_s=round(rt, 2),
                depots="+".join(inst.depot_labels[i] for i in sorted(I_open)),
                n_routes=len(routes), verify_ok=(not fails), verify_failures=fails,
                routes=det)


def run_profile(city, inputs, profile, rho, limit_s):
    from llrp_loader import load_instance
    inst = load_instance(city, inputs, priority_profile=profile, verbose=False)

    cost = _solve_flow(inst, lexicographic="cost", limit_s=limit_s)
    lat = _solve_flow(inst, lexicographic="latency", limit_s=limit_s)
    anc = norm.anchors_from_extremes(cost, lat)

    rows = []
    # anchors recorded as the lambda 0 / lambda 1 endpoints (Z undefined there)
    rows.append(dict(profile=profile, lam=0.0, Z=None, **{k: cost[k] for k in
                ("C_F", "P", "status", "time_s", "depots", "n_routes", "verify_ok")}))
    for lam in LAMBDAS:
        sol = _solve_flow(inst, tchebycheff=dict(lam=lam, rho=rho, **anc), limit_s=limit_s)
        sol["Z"] = norm.tchebycheff_Z(sol["C_F"], sol["P"], lam, anc, rho)
        rows.append(dict(profile=profile, lam=lam, **{k: sol[k] for k in
                    ("Z", "C_F", "P", "status", "time_s", "depots", "n_routes",
                     "verify_ok")}))
        rows[-1]["_routes"] = sol["routes"]
    rows.append(dict(profile=profile, lam=1.0, Z=None, **{k: lat[k] for k in
                ("C_F", "P", "status", "time_s", "depots", "n_routes", "verify_ok")}))
    return anc, rows


def main():
    ap = argparse.ArgumentParser(description="Exact interior Tchebycheff optima.")
    ap.add_argument("--city", default="Afyon")
    ap.add_argument("--inputs", default="Inputs")
    ap.add_argument("--profiles", default=",".join(PROFILES_ALL))
    ap.add_argument("--out", default=os.path.join("Outputs", "Exact"))
    ap.add_argument("--rho", type=float, default=norm.RHO)
    ap.add_argument("--time-limit", type=float, default=1200.0)
    a = ap.parse_args()

    import experiment  # noqa  (local Gurobi licence discovery, as in run_exact_anchors)
    profiles = PROFILES_ALL if a.profiles == "all" else a.profiles.split(",")
    os.makedirs(a.out, exist_ok=True)

    all_rows, routes_json, md = [], {}, []
    md.append(f"# {a.city} exact validation (flow formulation)\n")
    md.append(f"MIPGap={MIP_GAP:g}, MIPGapAbs={MIP_GAP_ABS:g}; flags: "
              "prune_arcs, time_windows, valid_inequalities, strengthen.\n")
    md.append(f"rho={a.rho:g}; lambda in {{0.1..0.9}}. "
              "C_F and P=sum p_j a_j recomputed from routes.\n")
    for profile in profiles:
        anc, rows = run_profile(a.city, a.inputs, profile, a.rho, a.time_limit)
        md.append(f"\n## {profile}\n")
        md.append(f"anchors: Cmin={anc['Cmin']:.3f} P_C={anc['P_C']:.3f} "
                  f"C_P={anc['C_P']:.3f} Pmin={anc['Pmin']:.3f} "
                  f"dC={anc['dC']:.3f} dP={anc['dP']:.3f}\n")
        md.append("| lambda | Z | C_F | P | depots | #routes | status | time(s) | verify |\n")
        md.append("|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            zc = "" if row["Z"] is None else f"{row['Z']:.6f}"
            md.append(f"| {row['lam']} | {zc} | {row['C_F']:.3f} | {row['P']:.3f} | "
                      f"{row['depots']} | {row['n_routes']} | {row['status']} | "
                      f"{row['time_s']} | {'OK' if row['verify_ok'] else 'FAIL'} |\n")
            routes_json.setdefault(profile, {})[str(row["lam"])] = row.pop("_routes", None)
            all_rows.append(row)

    base = os.path.join(a.out, a.city)
    with open(base + "_exact_validation.md", "w", encoding="utf-8") as fh:
        fh.writelines(md)
    with open(base + "_heuristic_vs_exact.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["profile", "lambda", "C_F_exact", "P_exact", "Z_exact",
                    "exact_status", "exact_time_s", "depots", "verify"])
        for r in all_rows:
            w.writerow([r["profile"], r["lam"], r["C_F"], r["P"],
                        "" if r["Z"] is None else round(r["Z"], 6),
                        r["status"], r["time_s"], r["depots"],
                        "OK" if r["verify_ok"] else "FAIL"])
    with open(base + "_interior_routes.json", "w", encoding="utf-8") as fh:
        json.dump(routes_json, fh, indent=2)
    print(f"wrote {base}_exact_validation.md / _heuristic_vs_exact.csv / _interior_routes.json "
          f"({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
