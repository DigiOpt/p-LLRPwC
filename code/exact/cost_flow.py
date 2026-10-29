"""
cost_flow.py
============

Exact single-objective COST solver (minimise financial cost) for the LLRP, via
the lean commodity-flow model solved by Gurobi branch-and-cut:

    build_model_flow(..., latency_obj=False, enforce_duration=<flag>)

With ``--drop-L`` the route-duration limit L is removed from the model (the
pure capacitated LRP is easier) and re-verified on the extracted routes
afterwards; the run reports whether every route still satisfies L.

Writes <out-dir>/<City>_cost.json with the objective breakdown, bounds, open
depots and the INCUMBENT ROUTES (readable: [depot_label, [pharmacy_code, ...]]),
matching the latency-run JSON layout.

    python cost_flow.py --city Canakkale --time 600 --drop-L
"""
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import experiment  # noqa: F401  -- uses the local Gurobi license
from llrp_loader import load_instance
from llrp_model import build_model_flow, _extract_routes, _objective_breakdown


def _routes_readable(inst, routes):
    """[[depot_label, [pharmacy_code, ...]], ...] from _extract_routes output."""
    out = []
    for r in routes:
        out.append([inst.depot_labels[r["depot"]], list(r["path_codes"])])
    return out


def solve_cost(city, input_dir="Case_Study_Inputs", L=None, priority="Equal",
               time_limit=600.0, drop_L=True, mip_focus=None, out_dir="results_hpc",
               verbose=True):
    inst = load_instance(city, input_dir, max_route_duration=L,
                         priority_profile=priority, verbose=False)
    m = build_model_flow(inst, prune_arcs=True, valid_inequalities=True,
                         strengthen=True, latency_obj=False,
                         enforce_duration=not drop_L)
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.MIPGap = 1e-9
    m.Params.TimeLimit = time_limit
    if mip_focus is not None:
        m.Params.MIPFocus = mip_focus
    t = time.time()
    m.optimize()
    wall = time.time() - t

    routes = _extract_routes(m, inst)
    b = _objective_breakdown(m, inst, routes)
    # verify the route-duration limit L on the incumbent (incl. the return trip)
    max_dur = 0.0
    for r in routes:
        prev, dur = r["depot"], 0.0
        for nd in r["path_nodes"]:
            dur += inst.l[prev][nd]; prev = nd
        dur += inst.l[prev][r["depot"]]
        max_dur = max(max_dur, dur)
    L_ok = max_dur <= inst.L + 1e-6

    try:
        gap = m.MIPGap
    except Exception:
        gap = None
    rec = {
        "city": city, "objective": "cost", "priority_profile": inst.priority_profile,
        "n": inst.n, "m": inst.m, "Ng": inst.Ng, "L": inst.L,
        "financial": b["financial_cost"], "latency_at_cost": b["latency_raw"],
        "bound": m.ObjBound, "gap": gap, "status": m.Status,
        "proven": (m.Status == 2) or (gap is not None and gap <= 1e-6),
        "drop_L": drop_L, "max_route_duration": max_dur, "L_ok": bool(L_ok),
        "wall_s": wall,
        "open_depots": [inst.depot_labels[i] for i in sorted({r["depot"] for r in routes})],
        "routes": _routes_readable(inst, routes),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{city}_cost.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\nCOST {city}: financial={rec['financial']:.2f} bound={rec['bound']:.2f} "
          f"gap={gap:.3e} proven={rec['proven']} depots={rec['open_depots']} "
          f"L_ok={rec['L_ok']} wall={wall:.0f}s -> {path}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Canakkale")
    ap.add_argument("--input-dir", default="Case_Study_Inputs")
    ap.add_argument("--L", type=float, default=None,
                    help="route duration override (default: from the Excel sheet)")
    ap.add_argument("--priority", default="Equal",
                    help="cost is priority-independent; kept for a consistent record")
    ap.add_argument("--time", type=float, default=600.0)
    ap.add_argument("--keep-L", action="store_true",
                    help="enforce the duration limit L in the model (default: drop + verify)")
    ap.add_argument("--mip-focus", type=int, default=None)
    ap.add_argument("--out-dir", default="results_hpc")
    args = ap.parse_args()
    solve_cost(args.city, args.input_dir, args.L, args.priority, args.time,
               drop_L=not args.keep_L, mip_focus=args.mip_focus, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
