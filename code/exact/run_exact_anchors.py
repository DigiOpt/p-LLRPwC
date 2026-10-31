"""
run_exact_anchors.py — exact lexicographic anchors for the bi-objective
priority-weighted latency LRP (Table 6/7 of the manuscript).

Regime-split exact method
-------------------------
For a city and priority profile the four anchors are

    Chat_min = min C_F                         (cheapest financial cost)
    Phat_C   = min P  s.t.  C_F = Chat_min      (latency at the cost optimum)
    Phat_min = min P                            (fastest priority-weighted latency)
    Chat_P   = min C_F  s.t.  P = Phat_min       (cost at the latency optimum)

with ranges dC = Chat_P - Chat_min and dP = Phat_C - Phat_min. C_F and P are always
recomputed from the extracted routes (P = sum_j p_j a_j, a_j = arrival time).

Two engines are used, each on the objective where its relaxation is tight:

  * financial / cost anchors (Chat_min, Phat_C)  -> commodity-flow MILP
        (``llrp_model.build_model_flow`` with the exactness-preserving options),
        solved lexicographically (min C_F, then min P among cost-optimal).
  * latency anchors (Phat_min, Chat_P)           -> branch-and-price
        (``llrp_bnp.branch_and_price``) over the C(m, Ng) depot combinations run
        in parallel with a shared global incumbent; Phat_min minimises P, and
        Chat_P minimises C_F subject to the budget row  P <= Phat_min.

Every saved solution is independently verified from its routes before it is
accepted (coverage-once, per-route Q and L, per-depot throughput W, fleet Nv,
exactly Ng open depots, and C_F / P recomputed = reported).

Requires Gurobi (a working local licence; see README). The GRASP-VND heuristic in
``code/`` does **not** need Gurobi — only this exact part does.

Usage (run from the repository root)
------------------------------------
    python code/exact/run_exact_anchors.py --city Afyon
    python code/exact/run_exact_anchors.py --city Canakkale --profiles Equal,Tiered
    python code/exact/run_exact_anchors.py --city Aydin --inputs Inputs \
        --cost-limit 14400 --bnp-cap 7200 --workers 4

Outputs: Outputs/Exact/<City>_exact_anchors.{json,csv,md}
"""
import os, sys, json, csv, time, argparse, itertools, threading, datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # make the exact modules importable
import numpy as np

PROFILES_ALL = ["Equal", "Tiered", "Demand_Correlated",
                "Negative_Correlated", "Hyper_Critical"]
# exactness-preserving flow options (see llrp_model.build_model_flow)
FLAGS = dict(prune_arcs=True, time_windows=True,
             valid_inequalities=True, strengthen=True)
MIP_GAP, MIP_GAP_ABS = 1e-9, 1e-6


def _jdefault(o):
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, (set, tuple)): return list(o)
    return str(o)


def _r(x):
    return None if x is None else round(x, 3)


# ----------------------------- B&P worker (own process) -------------------- #
def _worker(task):
    import warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, HERE)
    import experiment  # noqa  (local Gurobi licence discovery)
    from llrp_loader import load_instance
    from llrp_bnp import branch_and_price
    key = task["key"]
    try:
        inst = load_instance(task["city"], task["inputs"],
                             priority_profile=task["profile"], verbose=False)
        inst.alpha = 0.0
        gval, glock = task["gval"], task["glock"]
        def inc_get(): return gval.value
        def inc_put(v):
            with glock:
                if v < gval.value:
                    gval.value = v
        r = branch_and_price(inst, list(task["depots"]), ng_size=task["ng"],
                             node_time=task["cap"], total_time=task["cap"],
                             max_nodes=10**9, verbose=False,
                             objective="budget", obj=task["obj"],
                             inc_get=inc_get, inc_put=inc_put)
        return {"key": key, "labels": task["labels"], "depots": list(task["depots"]),
                "lb": r["lb"], "incumbent": r["incumbent"], "proven": r["proven"],
                "nodes": r["n_nodes"], "time_s": r["time_s"],
                "routes": [[i, list(p)] for (i, p) in (r["routes"] or [])],
                "C_F": r["C_F"], "P": r["P"], "error": None}
    except Exception as e:
        import traceback
        return {"key": key, "labels": task.get("labels"),
                "error": repr(e) + "\n" + traceback.format_exc()}


def run_phase(city, inputs, profile, obj, pairs, labels, cap, workers, ng, name):
    """Solve one latency anchor over all depot combinations in parallel with a
    shared global incumbent. Returns the per-combination result list."""
    mgr = mp.Manager()
    gval = mgr.Value("d", float("inf")); glock = mgr.Lock()
    tasks = [{"key": f"{profile}:{name}:{c}", "city": city, "inputs": inputs,
              "profile": profile, "depots": c, "labels": labels[c], "obj": obj,
              "cap": cap, "ng": ng, "gval": gval, "glock": glock} for c in pairs]
    out = []
    try:
        ex = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1)
    except TypeError:
        ex = ProcessPoolExecutor(max_workers=workers)
    with ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for f in as_completed(futs):
            out.append(f.result())
    return out


# ------------------------ route recompute + verification ------------------- #
def route_detail(inst, i, path):
    J, c, l, d, pr = inst.J, inst.c, inst.l, inst.demand, inst.priority
    prev = i; travel = tau = load = 0.0; arrivals = {}; ids = []
    for a in path:
        node = J[a]; travel += c[prev][node]; tau += l[prev][node]; load += d[node]
        arrivals[inst.pharmacy_codes[a]] = round(tau, 6)
        ids.append(inst.pharmacy_codes[a]); prev = node
    dur = tau + l[prev][i]; travel += c[prev][i]
    lat = sum(pr[J[a]] * arrivals[inst.pharmacy_codes[a]] for a in path)
    return (dict(depot=inst.depot_labels[i], pharmacy_ids=ids, load=round(load, 6),
                 duration=round(dur, 6), travel_cost=round(travel + inst.F[i], 6),
                 arrivals=arrivals),
            travel + inst.F[i], load, dur, lat)


def recompute(inst, open_idx, routes):
    details = []; C_F = sum(inst.f[i] for i in open_idx); P = 0.0
    for (i, path) in routes:
        det, cF_i, _, _, lat = route_detail(inst, i, path)
        details.append(det); C_F += cF_i; P += lat
    return round(C_F, 6), round(P, 6), details


def verify(inst, open_idx, routes, C_F, P):
    f = []; J = inst.J; seen = []; dep = defaultdict(float)
    for (i, path) in routes:
        load = sum(inst.demand[J[a]] for a in path); prev = i; tau = 0.0
        for a in path:
            tau += inst.l[prev][J[a]]; prev = J[a]; seen.append(a)
        dur = tau + inst.l[prev][i]
        if load > inst.Q + 1e-6: f.append(f"{inst.depot_labels[i]} load {load:g}>Q")
        if dur > inst.L + 1e-6:  f.append(f"{inst.depot_labels[i]} dur {dur:g}>L")
        dep[i] += load
    if len(set(seen)) != inst.n:      f.append(f"coverage {len(set(seen))}/{inst.n}")
    if len(seen) != len(set(seen)):   f.append("duplicate visit")
    for i, dd in dep.items():
        if dd > inst.W[i] + 1e-6:     f.append(f"{inst.depot_labels[i]} demand {dd:g}>W")
    if len(routes) > inst.Nv + 1e-9:  f.append(f"#routes {len(routes)}>Nv")
    if len(set(open_idx)) != inst.Ng: f.append(f"#open {len(set(open_idx))}!=Ng")
    rC, rP, _ = recompute(inst, open_idx, routes)
    if abs(rC - C_F) > 1e-3: f.append(f"C_F recompute {rC}!={C_F}")
    if abs(rP - P) > 1e-3:   f.append(f"P recompute {rP}!={P}")
    return f


def aggregate(results, inst):
    """Global UB = min incumbent; LB = min valid per-combination LB; verified best."""
    lbs = [r["lb"] for r in results if r.get("lb") is not None]
    LB = min(lbs) if lbs else None
    feas = [r for r in results if r.get("incumbent") is not None and r.get("routes")]
    best = min(feas, key=lambda r: r["incumbent"]) if feas else None
    UB = best["incumbent"] if best else None
    gap = ((UB - LB) / UB) if (UB and LB is not None and UB > 1e-9) else None
    sol = None
    if best:
        routes = [(i, tuple(p)) for i, p in best["routes"]]
        I_open = sorted({i for i, _ in routes})
        C_F, P, det = recompute(inst, I_open, routes)
        fails = verify(inst, I_open, routes, C_F, P)
        sol = dict(open_depots=[inst.depot_labels[i] for i in I_open], n_routes=len(routes),
                   C_F=C_F, P=P, routes=det, best_pair=best["labels"],
                   verify_ok=(not fails), verify_failures=fails)
    return dict(UB=UB, LB=LB, gap=gap,
                lb_certified=all(r.get("lb") is not None for r in results),
                solution=sol,
                per_pair=[{"depots": r.get("labels"), "lb": r.get("lb"),
                           "incumbent": r.get("incumbent"), "proven": r.get("proven"),
                           "time_s": round(r.get("time_s") or 0, 1),
                           "error": r.get("error")} for r in results])


# ------------------------- flow cost anchor (lexicographic) ---------------- #
def flow_cost_anchor(city, inputs, profile, limit_s):
    import experiment  # noqa
    from llrp_loader import load_instance
    from llrp_model import build_model_flow
    inst = load_instance(city, inputs, priority_profile=profile, verbose=False)
    m = build_model_flow(inst, latency_obj=True, enforce_duration=True,
                         lexicographic="cost", **FLAGS)
    m.Params.OutputFlag = 0; m.Params.MIPGap = MIP_GAP
    m.Params.MIPGapAbs = MIP_GAP_ABS; m.Params.TimeLimit = limit_s
    t = time.time(); m.optimize(); rt = time.time() - t
    md, x, y = inst.m, m._x, m._y
    I_open = [i for i in inst.I if y[i].X > 0.5]
    routes = []
    for i in inst.I:
        succ = defaultdict(list)
        for (u, v) in x[i].keys():
            if x[i][u, v].X > 0.5: succ[u].append(v)
        while succ[i]:
            cur = succ[i].pop(); path = []
            while cur != i: path.append(cur - md); cur = succ[cur].pop()
            routes.append((i, tuple(path)))
    C_F, P, det = recompute(inst, I_open, routes)
    fails = verify(inst, I_open, routes, C_F, P)
    status = {2: "OPTIMAL", 9: "TIME_LIMIT"}.get(m.Status, str(m.Status))
    try: gap = round(m.MIPGap, 9)
    except Exception: gap = None
    return inst, dict(Cmin=C_F, P_C=P, status=status, gap=gap, time_s=round(rt, 1),
                      open_depots=[inst.depot_labels[i] for i in I_open],
                      n_routes=len(routes), routes=det,
                      verify_ok=(not fails), verify_failures=fails)


# ----------------------------------- main ---------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Exact lexicographic anchors (Table 6/7).")
    ap.add_argument("--city", required=True)
    ap.add_argument("--inputs", default="Inputs", help="input directory (flat layout)")
    ap.add_argument("--profiles", default=",".join(PROFILES_ALL),
                    help="comma-separated profiles, or 'all'")
    ap.add_argument("--out", default=os.path.join("Outputs", "Exact"))
    ap.add_argument("--ng-size", type=int, default=8, help="ng-route neighbourhood size")
    ap.add_argument("--cost-limit", type=float, default=14400.0, help="flow cost solve cap (s)")
    ap.add_argument("--bnp-cap", type=float, default=7200.0, help="B&P cap per depot combination (s, soft)")
    ap.add_argument("--workers", type=int, default=4, help="parallel depot combinations")
    a = ap.parse_args()
    profiles = PROFILES_ALL if a.profiles == "all" else a.profiles.split(",")
    os.makedirs(a.out, exist_ok=True)

    import experiment  # noqa
    from llrp_loader import load_instance
    base = load_instance(a.city, a.inputs, priority_profile="Equal", verbose=False)
    pairs = list(itertools.combinations(range(base.m), base.Ng))
    labels = {c: [base.depot_labels[i] for i in c] for c in pairs}
    print(f"[{a.city}] m={base.m} n={base.n} Ng={base.Ng} Nv={base.Nv} | "
          f"{len(pairs)} depot combinations | profiles={profiles}", flush=True)

    OUT = {}
    for prof in profiles:
        print(f"\n==== {a.city} / {prof} ====", flush=True)
        t0 = time.time()
        # cost anchor: flow lexicographic cost -> Chat_min, Phat_C
        inst, fc = flow_cost_anchor(a.city, a.inputs, prof, a.cost_limit)
        Cmin, P_C = fc["Cmin"], fc["P_C"]
        print(f"  flow cost: Chat_min={Cmin:.3f} Phat_C={P_C:.3f} {fc['status']} "
              f"gap={fc['gap']} {fc['time_s']}s verify={'OK' if fc['verify_ok'] else fc['verify_failures']}", flush=True)
        # latency anchor stage 1: Phat_min = min P
        res = run_phase(a.city, a.inputs, prof, {"primary": "latency", "eps": None},
                        pairs, labels, a.bnp_cap, a.workers, a.ng_size, "Pmin")
        pm = aggregate(res, inst); Pmin = pm["UB"]
        g = "n/a" if pm["gap"] is None else f"{pm['gap']*100:.3f}%"
        print(f"  B&P  Phat_min: UB={pm['UB']} LB={_r(pm['LB'])} gap={g} "
              f"certified={pm['lb_certified']}", flush=True)
        # latency anchor stage 2: Chat_P = min C_F s.t. P <= Phat_min
        cp = None; C_P = None
        if Pmin is not None:
            res2 = run_phase(a.city, a.inputs, prof, {"primary": "cost", "eps": Pmin},
                             pairs, labels, a.bnp_cap, a.workers, a.ng_size, "C_at_P")
            cp = aggregate(res2, inst); C_P = cp["UB"]
            g2 = "n/a" if cp["gap"] is None else f"{cp['gap']*100:.3f}%"
            print(f"  B&P  Chat_P:   UB={cp['UB']} LB={_r(cp['LB'])} gap={g2} "
                  f"certified={cp['lb_certified']}", flush=True)
        OUT[prof] = dict(
            cost_anchor_flow=fc, lat_anchor_Pmin=pm, lat_anchor_C_at_P=cp,
            anchors=dict(Cmin=Cmin, Pmin=Pmin, P_C=P_C, C_P=C_P,
                         dC=(round(C_P - Cmin, 6) if C_P is not None else None),
                         dP=(round(P_C - Pmin, 6) if (Pmin is not None and P_C is not None) else None)),
            wall_s=round(time.time() - t0, 1))
        json.dump(OUT, open(os.path.join(a.out, f"{a.city}_exact_anchors.json"), "w"),
                  indent=2, default=_jdefault)

    _write_reports(a.city, a.out, OUT, profiles)
    print(f"\nwrote {a.out}/{a.city}_exact_anchors.{{json,csv,md}}", flush=True)


def _write_reports(city, out, OUT, profiles):
    md = [f"# {city} — exact lexicographic anchors\n",
          "Cost anchors by commodity-flow MILP (lexicographic); latency anchors by "
          "branch-and-price (parallel depot combinations, shared incumbent). "
          "UB = best incumbent, LB = min valid per-combination lower bound; the per-"
          "combination B&P cap is soft (checked between column-generation rounds).\n"]
    for prof in profiles:
        s = OUT[prof]["anchors"]; fc = OUT[prof]["cost_anchor_flow"]
        pm = OUT[prof]["lat_anchor_Pmin"]; cp = OUT[prof]["lat_anchor_C_at_P"]
        md.append(f"\n## {prof}\n")
        md.append(f"anchors: Chat_min={s['Cmin']} Phat_C={s['P_C']} Phat_min={s['Pmin']} "
                  f"Chat_P={s['C_P']} dC={s['dC']} dP={s['dP']}\n\n")
        md.append("| anchor | value | solver | gap | verify |\n|---|---|---|---|---|\n")
        md.append(f"| Chat_min,Phat_C | {s['Cmin']}, {s['P_C']} | flow(lex cost) | {fc['gap']} | {'OK' if fc['verify_ok'] else 'FAIL'} |\n")
        g1 = "n/a" if pm["gap"] is None else f"{pm['gap']*100:.3f}%"
        md.append(f"| Phat_min | {s['Pmin']} | B&P | {g1} | {'OK' if (pm.get('solution') or {}).get('verify_ok') else '-'} |\n")
        if cp:
            g2 = "n/a" if cp["gap"] is None else f"{cp['gap']*100:.3f}%"
            md.append(f"| Chat_P | {s['C_P']} | B&P(budget) | {g2} | {'OK' if (cp.get('solution') or {}).get('verify_ok') else '-'} |\n")
    open(os.path.join(out, f"{city}_exact_anchors.md"), "w").write("".join(md))
    with open(os.path.join(out, f"{city}_exact_anchors.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "profile", "Chat_min", "Phat_C", "Phat_min", "Chat_P",
                    "dC", "dP", "Pmin_gap", "C_P_gap", "verify_ok"])
        for prof in profiles:
            s = OUT[prof]["anchors"]; pm = OUT[prof]["lat_anchor_Pmin"]; cp = OUT[prof]["lat_anchor_C_at_P"]
            vok = ((pm.get("solution") or {}).get("verify_ok")
                   and (not cp or (cp.get("solution") or {}).get("verify_ok"))
                   and OUT[prof]["cost_anchor_flow"]["verify_ok"])
            w.writerow([city, prof, s["Cmin"], s["P_C"], s["Pmin"], s["C_P"], s["dC"], s["dP"],
                        pm.get("gap"), (cp or {}).get("gap"), vok])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
