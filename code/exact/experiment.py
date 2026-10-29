"""
experiment.py
=============

Run the exact method (lower + upper bounds) for the priority-weighted latency
LRP on the case-study cities in ``Case_Study_Inputs/``.

Edit the CONFIG block below (city, alpha, time limits, ...) and run:

    python experiment.py                      # runs the configured CITY
    python experiment.py --city Canakkale     # override city
    python experiment.py --alpha 1.0          # override alpha
    python experiment.py --all                # run every city found in the input folder
    python experiment.py --list               # just list available cities

Upper bound = flow-model incumbent (a feasible, elementary solution).
Lower bound = best of the flow LP bound and the branch-price bound
              (depot-set enumeration + ng-routes + subset-row cuts, compiled).
LB == UB  =>  proven optimal.
"""

import os

# --------------------------------------------------------------------------- #
# Gurobi license: use whatever local license the machine is configured with   #
# (default discovery -- e.g. ~/gurobi.lic or $GRB_LICENSE_FILE). No WLS setup. #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# CONFIG  -- edit these                                                       #
# --------------------------------------------------------------------------- #
CITY = "Afyon"                 # any city in Case_Study_Inputs/ (see --list)
ALPHA = 0.5                    # latency weight in the objective C_F + alpha*latency
INPUT_DIR = "Case_Study_Inputs"
MAX_ROUTE_DURATION = None      # L, seconds; None -> read from the Excel sheet
PRIORITY_PROFILE = "Equal"     # Equal | Tiered | Demand_Correlated |
                               # Negative_Correlated | Hyper_Critical

# exact-method controls
NG_SIZE = 8                    # ng-neighborhood size for the branch-price pricer
USE_SRC = True                 # subset-row cuts in the pricer
T_UB = 120.0                   # seconds for the flow upper-bound solve
T_SUB = 120.0                  # seconds per depot-set lower-bound solve
SHOW_GUROBI_LOG = True         # print Gurobi's optimization log (flow UB solve)
# --------------------------------------------------------------------------- #

import argparse
import csv
import datetime
import warnings

warnings.filterwarnings("ignore")

from llrp_loader import load_instance                           # noqa: E402
from llrp_bpc import exact_solve                                # noqa: E402

LOG_DIR = "results"          # per-run logs + summary.csv go here


def list_cities(input_dir):
    """Discover city names from the *_customers.csv files in the input folder."""
    cities = set()
    if not os.path.isdir(input_dir):
        return []
    for fn in os.listdir(input_dir):
        if fn.endswith("_customers.csv"):
            cities.add(fn[: -len("_customers.csv")])
    return sorted(cities)


def write_run_log(inst, res, settings, log_dir):
    """Write a detailed per-run log and append a row to summary.csv."""
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{inst.city}_alpha{inst.alpha:g}_{stamp}.log")
    D = inst.total_demand
    W = next(iter(inst.W.values()))
    lb = res["lb"]
    lb_txt = "n/a" if lb is None else f"{lb:.4f}"
    flow_lb_txt = "n/a" if res["flow_lb"] is None else f"{res['flow_lb']:.4f}"
    bpc_lb_txt = "n/a" if res["bpc_lb"] is None else f"{res['bpc_lb']:.4f}"
    with open(path, "w") as f:
        f.write(f"Priority-Weighted Latency LRP -- exact-method run\n")
        f.write(f"timestamp : {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"city      : {inst.city}\n")
        f.write(f"alpha     : {inst.alpha:g}\n\n")
        f.write("Instance\n")
        f.write(f"  n (pharmacies)      : {inst.n}\n")
        f.write(f"  m (candidate depots): {inst.m}\n")
        f.write(f"  Ng (open depots)    : {inst.Ng}\n")
        f.write(f"  Nv (fleet)          : {inst.Nv}\n")
        f.write(f"  Q (vehicle cap)     : {inst.Q:g}\n")
        f.write(f"  W (depot cap)       : {W:g}\n")
        f.write(f"  L (route duration)  : {inst.L:g} s\n")
        f.write(f"  D (total demand)    : {D:g}\n\n")
        f.write("Settings\n")
        for k, v in settings.items():
            f.write(f"  {k:20}: {v}\n")
        f.write("\nBounds\n")
        f.write(f"  UPPER BOUND (flow incumbent) : {res['ub']:.4f}\n")
        f.write(f"  LOWER BOUND (best valid)     : {lb_txt}\n")
        f.write(f"    flow LP bound              : {flow_lb_txt}\n")
        f.write(f"    BPC ng+SRC bound           : {bpc_lb_txt}\n")
        g = "n/a" if res["gap"] is None else f"{res['gap']*100:.4f}%"
        f.write(f"  CERTIFIED GAP                : {g}\n")
        f.write(f"  STATUS                       : "
                f"{'PROVEN OPTIMAL' if res['proven'] else 'gap (not proven)'}\n\n")
        f.write("Lower-bound subproblem detail (depot set / root)\n")
        for labels, slb, cuts, conv, t in res["details"]:
            slbt = "none" if slb is None else f"{slb:.2f}"
            f.write(f"  {str(labels):24} LB={slbt:>12}  cuts={cuts:>3}  "
                    f"converged={conv}  t={t:.1f}s\n")

    # append to summary.csv
    csv_path = os.path.join(log_dir, "summary.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as cf:
        w = csv.writer(cf)
        if new:
            w.writerow(["timestamp", "city", "n", "m", "alpha", "UB", "LB",
                        "flow_lb", "bpc_lb", "gap_%", "proven", "log_file"])
        w.writerow([stamp, inst.city, inst.n, inst.m, inst.alpha,
                    f"{res['ub']:.4f}",
                    "" if lb is None else f"{lb:.4f}",
                    "" if res["flow_lb"] is None else f"{res['flow_lb']:.4f}",
                    "" if res["bpc_lb"] is None else f"{res['bpc_lb']:.4f}",
                    "" if res["gap"] is None else f"{res['gap']*100:.4f}",
                    res["proven"], os.path.basename(path)])
    print(f"  log written: {path}")
    return path


def run_city(city, alpha, input_dir, L, ng_size, use_src, t_ub, t_sub,
             gurobi_log=SHOW_GUROBI_LOG, depot_enum_max_n=24,
             priority_profile=PRIORITY_PROFILE):
    print("=" * 70)
    inst = load_instance(city, input_dir, max_route_duration=L, alpha=alpha,
                         priority_profile=priority_profile, verbose=True)
    res = exact_solve(inst, ng_size=ng_size, t_ub=t_ub, t_sub=t_sub,
                      use_src=use_src, gurobi_log=gurobi_log,
                      depot_enum_max_n=depot_enum_max_n, verbose=True)
    settings = {"ng_size": ng_size, "use_src": use_src, "t_ub": t_ub,
                "t_sub": t_sub, "L": inst.L, "input_dir": input_dir,
                "priority_profile": inst.priority_profile,
                "depot_enum_max_n": depot_enum_max_n}
    write_run_log(inst, res, settings, LOG_DIR)
    return res


def main():
    ap = argparse.ArgumentParser(description="Exact LB/UB for the LLRP case studies")
    ap.add_argument("--city", default=CITY)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--L", type=float, default=MAX_ROUTE_DURATION,
                    help="route duration L (default: read from the Excel sheet)")
    ap.add_argument("--priority", default=PRIORITY_PROFILE,
                    help="priority profile: Equal | Tiered | Demand_Correlated | "
                         "Negative_Correlated | Hyper_Critical")
    ap.add_argument("--ng", type=int, default=NG_SIZE)
    ap.add_argument("--t-ub", type=float, default=T_UB)
    ap.add_argument("--t-sub", type=float, default=T_SUB)
    ap.add_argument("--no-src", action="store_true", help="disable subset-row cuts")
    ap.add_argument("--depot-enum-max-n", type=int, default=24,
                    help="use depot-set enumeration when n <= this (else root bound)")
    ap.add_argument("--no-gurobi-log", action="store_true",
                    help="hide Gurobi's optimization log for the flow UB solve")
    ap.add_argument("--all", action="store_true", help="run every city in the folder")
    ap.add_argument("--list", action="store_true", help="list available cities and exit")
    args = ap.parse_args()

    available = list_cities(args.input_dir)
    print(f"Cities available in '{args.input_dir}': {available}")
    if args.list:
        return

    use_src = not args.no_src
    gurobi_log = SHOW_GUROBI_LOG and not args.no_gurobi_log
    cities = available if args.all else [args.city]

    results = []
    for c in cities:
        try:
            r = run_city(c, args.alpha, args.input_dir, args.L, args.ng,
                         use_src, args.t_ub, args.t_sub, gurobi_log=gurobi_log,
                         depot_enum_max_n=args.depot_enum_max_n,
                         priority_profile=args.priority)
            results.append(r)
        except Exception as e:                 # keep going across cities
            print(f"!! {c} failed: {e}")

    print("\n" + "=" * 70)
    print(f"SUMMARY  (alpha = {args.alpha:g})")
    print(f"{'City':12}{'n':>5}{'m':>4}{'LB':>14}{'UB':>14}{'gap':>10}  status")
    for r in results:
        lb = "n/a" if r["lb"] is None else f"{r['lb']:.2f}"
        ub = "n/a" if r["ub"] is None else f"{r['ub']:.2f}"
        gap = "n/a" if r["gap"] is None else f"{r['gap']*100:.3f}%"
        status = "PROVEN OPTIMAL" if r["proven"] else "gap"
        print(f"{r['city']:12}{r['n']:>5}{r['m']:>4}{lb:>14}{ub:>14}{gap:>10}  {status}")


if __name__ == "__main__":
    main()
