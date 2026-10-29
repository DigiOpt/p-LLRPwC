"""
llrp_bnp.py
===========

Branch-and-price for a FIXED open-depot set: closes the DW/SRC integrality gap
left by ``bpc_lower_bound`` via **arc branching**.

At each node we run column generation (ng + SRC) restricted by a set of forbidden
arcs, get a valid node lower bound, and if the LP is fractional we branch on a
fractional aggregate arc:
    child 0 (forbid): the arc may not be used;
    child 1 (force) : the arc must be used -> forbid all competing arcs out of u
                       and into v (implemented as extra forbids).
A shared column pool is reused across nodes; a column is active at a node iff none
of its arcs is forbidden there. Pruning is by node bound vs. incumbent.

Enforcement lives in the compiled pricer (fb_cc / fb_out / fb_in). This is a
research prototype: correct-by-construction and validated against known optima,
but not tuned for speed.
"""

from __future__ import annotations

import time
from collections import defaultdict
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from llrp_loader import LLRPInstance
from llrp_model import heuristic_solution
from llrp_bpc import _ng_neighborhoods, _price_depot, _route_cost, _route_parts, BIG, EPS


def solution_parts(inst, depots, routes):
    """(C_F, P) recomputed from the route list alone -- no master values.
    C_F = sum_{i in depots} f_i + sum_r (travel + F_i);  P = sum_r latency.
    ``routes`` is a list of (depot_index, path) tuples."""
    C_F = sum(inst.f[i] for i in depots)
    P = 0.0
    for i, path in routes:
        cF, pL, _ = _route_parts(inst, i, path)
        C_F += cF
        P += pL
    return C_F, P


def _greedy_seed(inst, I):
    """A feasible set of routes (nearest-neighbour, respecting Q and L) covering
    every customer, so the initial RMP is fleet-feasible and its duals are sane.
    Each customer goes to its nearest open depot; that depot's customers are
    packed into routes greedily. Returns [(depot_i, [cust_idx, ...]), ...]."""
    J, n = inst.J, inst.n
    assign = {i: [] for i in I}
    for a in range(n):
        i = min(I, key=lambda i: inst.l[i][J[a]])
        assign[i].append(a)
    out = []
    for i in I:
        pool = set(assign[i])
        while pool:
            r, load, dur, prev = [], 0.0, 0.0, i
            while True:
                cand = [a for a in pool
                        if load + inst.demand[J[a]] <= inst.Q
                        and dur + inst.l[prev][J[a]] + inst.l[J[a]][i] <= inst.L + 1e-9]
                if not cand:
                    break
                a = min(cand, key=lambda a: inst.l[prev][J[a]])
                r.append(a); load += inst.demand[J[a]]
                dur += inst.l[prev][J[a]]; prev = J[a]; pool.discard(a)
            if not r:                 # a customer that fits no single-stop route
                break                 # (demand>Q or round-trip>L): leave to artificials
            out.append((i, r))
    return out


def _col_arcs(i, path):
    """Arc keys of a route: ('out',i,a0), ('cc',a,b)..., ('in',ak,i)."""
    arcs = [("out", i, path[0])]
    for a, b in zip(path[:-1], path[1:]):
        arcs.append(("cc", a, b))
    arcs.append(("in", path[-1], i))
    return arcs


def branch_and_price(inst: LLRPInstance, fixed_depots, *, ng_size=8, use_src=True,
                     node_time=150.0, total_time=1800.0, max_nodes=200,
                     incumbent=None, verbose=True, status_cb=None,
                     seed_routes=None, pool_cap=1000, inc_get=None, inc_put=None,
                     cost_mode=False, objective="weighted", obj=None):
    # cost_mode: single-objective COST pricing (use with a cost-only instance,
    # alpha=0). Pricing switches to the bidirectional, fully-elementary cost
    # pricer (price_cost), which collapses the alpha=0 forward-label explosion.
    # SRC is disabled in this mode (the bidirectional pricer carries no cut
    # parity); the elementary LP bound is already strong.
    #
    # objective (Step 3): master/pricing scalarisation. All share one master.
    #   "weighted"    : column obj cF + alpha*pL; wC=1, wP=alpha (default/unchanged).
    #   "budget"      : obj={"primary": "cost"|"latency", "eps": float|None}.
    #                   primary cost   -> min sum cF s.t. sum pL <= eps (if given);
    #                   primary latency-> min sum pL s.t. sum cF <= eps - sum f_i.
    #   "tchebycheff" : obj={"lam","rho","Cmin","Pmin","dC","dP"} -> min Z_lambda.
    if cost_mode:
        use_src = False
    obj = dict(obj or {})
    if objective == "tchebycheff":
        use_src = False          # SRC parity not wired for the augmented objective yet
    I = list(fixed_depots)
    J, n, Nv = inst.J, inst.n, inst.Nv
    t0 = time.time()
    T = {"price": 0.0, "rmh": 0.0, "src": 0.0}   # coarse per-part timing (s)
    CG = {"iters": 0}                            # total CG (pricing-round) iterations
    # Tchebycheff epigraph duals (eta_C, eta_P) -- the convex split the pricer sees
    # (eta_C+eta_P=1 at an optimal master LP). Track the last values and the
    # smallest eta_P reached, to report the "eta_P ~ 0" pricing regime (Step 3).
    TCH = {"eta_C": None, "eta_P": None, "etaP_min": None}
    # ---- objective-mode configuration (Step 3) -------------------------- #
    alpha = float(inst.alpha)
    sum_f = sum(inst.f[i] for i in I)            # fixed depot-opening cost (const)
    if objective == "weighted":
        def _ycoef(i): return inst.f[i]
        def _colcoef(cF, pL): return cF + alpha * pL
        objcon = 0.0
    elif objective == "budget":
        primary = obj["primary"]                 # "cost" or "latency"
        eps = obj.get("eps")                     # budget on the OTHER criterion
        if primary == "cost":
            def _ycoef(i): return inst.f[i]
            def _colcoef(cF, pL): return cF
        else:                                    # latency primary
            def _ycoef(i): return 0.0
            def _colcoef(cF, pL): return pL
        objcon = 0.0
    elif objective == "tchebycheff":
        lam = float(obj["lam"]); rho_t = float(obj["rho"])
        Cmin = float(obj["Cmin"]); Pmin = float(obj["Pmin"])
        dC = float(obj["dC"]); dP = float(obj["dP"])
        aCbar = (1.0 - lam) / dC                 # coeff of cF in Cbar
        aPbar = lam / dP                         # coeff of pL in Pbar
        def _ycoef(i): return rho_t * aCbar * inst.f[i]
        def _colcoef(cF, pL): return rho_t * (aCbar * cF + aPbar * pL)
        # ObjVal = zeta + rho*[(1-lam)Cbar + lam Pbar]; with Cbar/Pbar expanded the
        # constant below makes mdl.ObjVal equal Z_lambda directly.
        objcon = -rho_t * ((1.0 - lam) * Cmin / dC + lam * Pmin / dP)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    def _sol_value(C_F, P):
        """Objective value of a feasible solution (C_F, P) in the active mode --
        used for incumbents, pruning and reporting (recomputed from routes, never
        read from the master). +inf if a budget is violated."""
        if objective == "weighted":
            return C_F + alpha * P
        if objective == "budget":
            if primary == "cost":
                return C_F if (eps is None or P <= eps + 1e-6) else float("inf")
            return P if (eps is None or C_F <= eps + 1e-6) else float("inf")
        Cbar = (C_F - Cmin) / dC
        Pbar = (P - Pmin) / dP
        return max((1.0 - lam) * Cbar, lam * Pbar) \
            + rho_t * ((1.0 - lam) * Cbar + lam * Pbar)

    # Pricing weights (Step 2): weighted-sum default wC=1, wP=alpha. Budget and
    # Tchebycheff recompute them from the master duals each CG iteration.
    wC = 1.0
    wP = alpha
    ng = _ng_neighborhoods(inst, K=ng_size)
    # One pricer per open depot so their labelings can run concurrently: the
    # compiled pricer releases the GIL (njit nogil), and each depot has its own
    # label buffers, so the |I| pricings parallelise across cores. Gurobi/master
    # work stays on the main thread (Gurobi models are not thread-safe).
    nbps = None
    if n <= 64:
        try:
            from llrp_pricer_nb import NbPricer
            nbps = {i: NbPricer(inst, ng) for i in I}
        except Exception:
            nbps = None
    from concurrent.futures import ThreadPoolExecutor
    price_pool = ThreadPoolExecutor(max_workers=len(I)) if (nbps and len(I) > 1) else None

    mdl = gp.Model("BNP")
    mdl.Params.OutputFlag = 0
    mdl.ModelSense = GRB.MINIMIZE
    mdl.ObjCon = objcon
    y = {i: mdl.addVar(lb=1.0, ub=1.0, obj=_ycoef(i)) for i in I}
    art = {a: mdl.addVar(obj=BIG) for a in range(n)}
    cover = {a: mdl.addConstr(art[a] == 1) for a in range(n)}
    dcap = {i: mdl.addConstr(-inst.W[i] * y[i] <= 0) for i in I}
    dlink = {i: mdl.addConstr(-Nv * y[i] <= 0) for i in I}     # sum theta_r <= Nv y_i (upper)
    # paper SP:depveh lower bound: each open depot runs >= 1 route. y_i is fixed
    # to 1 here (all depots in I are open), so the row is sum_{r in R_i} theta_r >= 1.
    # A BIG artificial makes a node where branching removed all of depot i's routes
    # LP-feasible-with-artificials (fathomed) instead of uncertified-infeasible.
    art_dep = {i: mdl.addVar(obj=BIG) for i in I}
    dlink_lo = {i: mdl.addConstr(art_dep[i] >= 1) for i in I}  # sum theta_r + art >= 1
    fleet = mdl.addConstr(gp.quicksum([]) <= Nv)

    # ---- mode-specific rows / variables --------------------------------- #
    zeta = None; epi_cost = epi_lat = None       # Tchebycheff
    budget_row = budget_slack = None             # budget mode
    mode_slacks = []                             # BIG-cost slacks to watch in asum
    if objective == "tchebycheff":
        zeta = mdl.addVar(lb=-GRB.INFINITY, obj=1.0, name="zeta")
        # zeta - aCbar*sum cF_r theta >= aCbar*(sum_f - Cmin)
        epi_cost = mdl.addConstr(zeta >= aCbar * (sum_f - Cmin), "epi_cost")
        # zeta - aPbar*sum pL_r theta >= -aPbar*Pmin
        epi_lat = mdl.addConstr(zeta >= -aPbar * Pmin, "epi_lat")
    elif objective == "budget" and obj.get("eps") is not None:
        budget_slack = mdl.addVar(obj=BIG, name="s_budget")     # >= 0
        mode_slacks.append(budget_slack)
        rhs = eps if primary == "cost" else (eps - sum_f)
        # (cost primary)   sum pL_r theta - s <= eps
        # (latency primary) sum cF_r theta - s <= eps - sum_f
        budget_row = mdl.addConstr(-budget_slack <= rhs, "budget")

    cols = []
    seen = set()
    active_src = []

    def add_column(i, path):
        key = (i, tuple(path))
        if key in seen:
            return None
        seen.add(key)
        cF, pL, dr = _route_parts(inst, i, path)
        cost = _colcoef(cF, pL)
        col = gp.Column()
        for a in path:
            col.addTerms(1.0, cover[a])
        col.addTerms(dr, dcap[i])
        col.addTerms(1.0, dlink[i])
        col.addTerms(1.0, dlink_lo[i])
        col.addTerms(1.0, fleet)
        if objective == "tchebycheff":
            col.addTerms(-aCbar * cF, epi_cost)
            col.addTerms(-aPbar * pL, epi_lat)
        elif budget_row is not None:
            col.addTerms(pL if primary == "cost" else cF, budget_row)
        for s in active_src:
            cf = sum(1 for a in path if (s["mask"] >> a) & 1) // 2
            if cf:
                col.addTerms(cf, s["con"])
        rec = {"var": mdl.addVar(obj=cost, column=col), "i": i,
               "path": tuple(path), "pset": set(path), "cost": cost,
               "cF": cF, "pL": pL,
               "arcs": set(_col_arcs(i, path))}
        cols.append(rec)
        return rec

    # seed columns
    for i in I:
        for a in range(n):
            if inst.demand[J[a]] <= inst.Q and \
                    inst.l[i][J[a]] + inst.l[J[a]][i] <= inst.L + 1e-9:
                add_column(i, [a])
    hs = heuristic_solution(inst)
    if hs is not None:
        for i, routes in hs[1].items():
            if i in y:
                for r in routes:
                    add_column(i, [nd - inst.m for nd in r])
    # Greedy feasible seed (Q- and L-respecting nearest-neighbour routes). Without
    # it, a fleet-limited instance whose heuristic fails is covered only by
    # singletons + artificials, so the first LP's cover duals blow up to BIG and
    # the initial pricing explodes (routes grow to the capacity limit because
    # every extension looks hugely profitable). A handful of long feasible routes
    # keeps the initial duals sane.
    greedy_sol = [(i, tuple(path)) for i, path in _greedy_seed(inst, I)]
    for i, path in greedy_sol:
        add_column(i, path)
    # Warm columns carried from a previous solve (e.g. the previous alpha's routes
    # for this same depot set). Routes are alpha-independent feasibility objects, so
    # they are valid columns here once re-costed; they warm-start CG and hand this
    # node an immediate good incumbent. Each is feasibility-checked (elementary,
    # capacity, duration, depot in this config) since it bypasses the pricer.
    if seed_routes:
        for i, path in seed_routes:
            if i not in y or not path:
                continue
            p = [int(a) for a in path]
            if any(a < 0 or a >= n for a in p) or len(set(p)) != len(p):
                continue
            if sum(inst.demand[J[a]] for a in p) > inst.Q + 1e-9:
                continue
            dur = inst.l[i][J[p[0]]]
            prev = J[p[0]]
            for a in p[1:]:
                dur += inst.l[prev][J[a]]
                prev = J[a]
            dur += inst.l[prev][i]
            if dur > inst.L + 1e-9:
                continue
            add_column(i, p)

    def masks_for(F):
        """Build per-depot forbid arrays from a forbidden-arc set F."""
        fb_cc = np.zeros((n, n), np.int8)
        fb_out = {i: np.zeros(n, np.int8) for i in I}
        fb_in = {i: np.zeros(n, np.int8) for i in I}
        for key in F:
            if key[0] == "cc":
                fb_cc[key[1], key[2]] = 1
            elif key[0] == "out":
                fb_out[key[1]][key[2]] = 1
            else:  # in
                fb_in[key[2]][key[1]] = 1
        return fb_cc, fb_out, fb_in

    def apply_active(F):
        for rec in cols:
            rec["var"].ub = 0.0 if (rec["arcs"] & F) else GRB.INFINITY

    def price_round(F, fb_cc, fb_out, fb_in):
        _t = time.time()
        # Read all duals on the main thread (Gurobi attribute reads are not
        # thread-safe); the compiled labeling below touches no Gurobi state.
        srcs = [(s["mask"], s["con"].Pi) for s in active_src]
        pic = [cover[a].Pi for a in range(n)]
        sig = {i: dcap[i].Pi for i in I}
        # depveh dual = upper (xi_i <= 0) + lower (nu_i >= 0) rows (paper rc_r)
        rho = {i: dlink[i].Pi + dlink_lo[i].Pi for i in I}
        flt = fleet.Pi
        # ---- recompute pricing weights from the current duals (Step 3) ---- #
        nonlocal wC, wP
        if objective == "budget":
            if budget_row is not None:
                om = budget_row.Pi
                om = 0.0 if om > 0.0 else om          # budget dual <= 0; clip noise
            else:
                om = 0.0
            wC, wP = (1.0, -om) if primary == "cost" else (-om, 1.0)
        elif objective == "tchebycheff":
            eC, eP = epi_cost.Pi, epi_lat.Pi
            if verbose and abs(eC + eP - 1.0) > 1e-6:
                print(f"  [tch] WARNING eta_C+eta_P={eC + eP:.6g} != 1", flush=True)
            TCH["eta_C"], TCH["eta_P"] = eC, eP
            TCH["etaP_min"] = eP if TCH["etaP_min"] is None else min(TCH["etaP_min"], eP)
            wC = aCbar * (rho_t + eC)
            wP = aPbar * (rho_t + eP)

        def price_one(i):
            if nbps is not None:
                if cost_mode:
                    found, mr = nbps[i].price_cost(
                        inst, i, pic, sig[i], rho[i], flt, max_cols=40,
                        fb_cc=fb_cc, fb_out=fb_out[i], fb_in=fb_in[i], wC=wC)
                else:
                    found, mr = nbps[i].price(inst, i, pic, sig[i], rho[i], flt,
                                              srcs=srcs, max_cols=40, fb_cc=fb_cc,
                                              fb_out=fb_out[i], fb_in=fb_in[i],
                                              wC=wC, wP=wP)
                if found is not None:
                    return i, found, mr
            found = _price_depot(inst, i, pic, sig[i], rho[i], flt, srcs,
                                 max_cols=40, ng=ng, wC=wC, wP=wP)
            found = [(rc, p) for rc, p in found
                     if not (set(_col_arcs(i, p)) & F)]
            return i, found, (found[0][0] if found else 0.0)

        # Parallel labeling across depots (threads; pricer is nogil), then add
        # the resulting columns to the Gurobi model on the main thread.
        if price_pool is not None:
            results = list(price_pool.map(price_one, I))
        else:
            results = [price_one(i) for i in I]

        cbar = 0.0
        added = 0
        for i, found, mr in results:
            cbar = min(cbar, mr if mr is not None else 0.0)
            for _, path in found:
                r = add_column(i, path)
                if r is not None:
                    r["var"].ub = GRB.INFINITY   # active in current node
                    added += 1
        T["price"] += time.time() - _t
        return added, cbar

    def solve_node(F, deadline):
        """CG (with SRC rounds) at a node. Returns (lb, converged, lp_status)."""
        apply_active(F)
        fb_cc, fb_out, fb_in = masks_for(F)
        best = -float("inf")
        conv = False
        # column generation
        while time.time() < deadline:
            mdl.optimize()
            if mdl.Status != GRB.OPTIMAL:
                return None, False
            z = mdl.ObjVal
            asum = (sum(art[a].X for a in range(n))
                    + sum(art_dep[i].X for i in I)
                    + sum(v.X for v in mode_slacks))
            CG["iters"] += 1
            added, cbar = price_round(F, fb_cc, fb_out, fb_in)
            if asum < 1e-4:
                best = max(best, z + Nv * min(0.0, cbar))
            if added == 0:
                if asum < 1e-4:
                    best = max(best, z)
                conv = True
                break
        # SRC rounds (only when converged, feasible, and cuts requested)
        if use_src and conv and best > -float("inf"):
            for _ in range(4):
                if time.time() >= deadline:
                    conv = False
                    break
                mdl.optimize()
                acols = [(rec, rec["var"].X) for rec in cols
                         if rec["var"].ub > 0.5 and rec["var"].X > 1e-6]
                existing = {s["mask"] for s in active_src}
                viol = []
                for tri in combinations(range(n), 3):
                    mask = (1 << tri[0]) | (1 << tri[1]) | (1 << tri[2])
                    if mask in existing:
                        continue
                    Cs = set(tri)
                    ssum = sum(v for rec, v in acols if len(rec["pset"] & Cs) >= 2)
                    if ssum > 1.0 + 1e-4:
                        viol.append((ssum, mask))
                if not viol:
                    break
                viol.sort(reverse=True)
                for _, mask in viol[:20]:
                    if len(active_src) >= 60:
                        break
                    expr = gp.LinExpr()
                    for rec in cols:
                        cf = sum(1 for a in rec["path"] if (mask >> a) & 1) // 2
                        if cf:
                            expr.addTerms(cf, rec["var"])
                    con = mdl.addConstr(expr <= 1.0)
                    active_src.append({"mask": mask, "con": con})
                conv2 = False
                while time.time() < deadline:
                    mdl.optimize()
                    if mdl.Status != GRB.OPTIMAL:
                        break
                    z = mdl.ObjVal
                    asum = (sum(art[a].X for a in range(n))
                    + sum(art_dep[i].X for i in I)
                    + sum(v.X for v in mode_slacks))
                    added, cbar = price_round(F, fb_cc, fb_out, fb_in)
                    if asum < 1e-4:
                        best = max(best, z + Nv * min(0.0, cbar))
                    if added == 0:
                        if asum < 1e-4:
                            best = max(best, z)
                        conv2 = True
                        break
                conv = conv2
        # Consistency solve: a deadline can break out of a CG loop right after
        # price_round added columns that were never re-optimized, leaving them
        # without a model index. Re-solve so every column has a valid X/ub for
        # frac_arc / rmh_incumbent to read.
        mdl.update()
        mdl.optimize()
        if mdl.Status != GRB.OPTIMAL:
            return None, False
        if best > -float("inf"):
            return best, conv
        # best == -inf: never got an artificial-free RMP.
        if conv:
            # CG converged yet artificials remain -> this node's feasible region
            # is empty (arc-branching forbids over-constrained it). Fathom it
            # (+inf bound); this does NOT taint the optimality certificate.
            return float("inf"), True
        return None, False                  # deadline hit -> genuinely uncertified

    def frac_arc():
        flow = defaultdict(float)
        for rec in cols:
            if rec["var"].ub > 0.5:
                v = rec["var"].X
                if v > 1e-6:
                    for key in rec["arcs"]:
                        flow[key] += v
        best_key, best_d = None, 0.499
        for key, f in flow.items():
            if 1e-4 < f < 1 - 1e-4:
                d = abs(f - 0.5)
                if d <= best_d:
                    best_d, best_key = d, key
        return best_key

    def rmh_incumbent(F):
        """Integer set-partitioning over active ELEMENTARY columns -> feasible UB.
        Non-elementary (ng revisit) and inactive columns are excluded so the
        integer solution is a genuine feasible route set."""
        _t = time.time()
        for a in range(n):
            art[a].ub = 0.0
        for i in I:                              # force a GENUINE feasible solution:
            art_dep[i].ub = 0.0                  #   every open depot runs >=1 route
        for v in mode_slacks:                    #   and the budget (if any) is met
            v.ub = 0.0
        for rec in cols:
            active = not (rec["arcs"] & F)
            elem = len(rec["path"]) == len(set(rec["path"]))
            if active and elem:
                rec["var"].vtype = GRB.BINARY
            else:
                rec["var"].ub = 0.0
        mdl.Params.TimeLimit = 30
        mdl.optimize()
        val = None
        routes = None
        if mdl.SolCount > 0 and mdl.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            routes = [(rec["i"], rec["path"]) for rec in cols
                      if rec["var"].vtype == GRB.BINARY and rec["var"].X > 0.5]
            # Recompute the objective from the routes alone (Step 5) and check it
            # agrees with the master value; use the recomputed one as the incumbent.
            C_F, P = solution_parts(inst, I, routes)
            val = _sol_value(C_F, P)
            if val < float("inf") and abs(val - mdl.ObjVal) > 1e-6 * max(1.0, abs(val)):
                if verbose:
                    print(f"  [rmh] recomputed {val:.6g} != master {mdl.ObjVal:.6g}",
                          flush=True)
        # restore LP relaxation (next node re-masks ubs via apply_active)
        for rec in cols:
            rec["var"].vtype = GRB.CONTINUOUS
            rec["var"].ub = GRB.INFINITY
        for a in range(n):
            art[a].ub = GRB.INFINITY
        for i in I:
            art_dep[i].ub = GRB.INFINITY
        for v in mode_slacks:
            v.ub = GRB.INFINITY
        mdl.Params.TimeLimit = GRB.INFINITY
        T["rmh"] += time.time() - _t
        return val, routes

    # ---- branch-and-bound over forbidden-arc sets (BEST-FIRST) ----
    # The open set is a heap keyed by each node's bound: for an unexplored node
    # that is its parent's converged LP bound (a valid lower bound on the whole
    # subtree). We always expand the lowest-bound open node, so the heap-minimum
    # bound is a valid global LB that rises monotonically as low nodes get
    # replaced by their higher-bound children. When the heap empties (all bounds
    # >= incumbent) the incumbent is proven optimal; on an early stop the
    # heap-minimum is a tight certified lower bound.
    def _feasible(routes):
        """Full feasibility of a route set in THIS depot config -- the guard the
        cost-only recomputes (solution_parts / _sol_value) do NOT perform. Any
        incumbent taken from routes that did not pass through the dcap-constrained
        master (e.g. the greedy seed) must be validated here before it is trusted,
        else a route set that overloads a depot's throughput W can masquerade as a
        better-than-optimal incumbent and never be displaced (Step 5 assert):
          * every customer covered exactly once;
          * each route within vehicle capacity Q and duration limit L;
          * each OPEN depot within its throughput W;  * fleet size within Nv.
        """
        if routes is None:
            return False
        seen = []
        dep_dem = {}
        for (i, path) in routes:
            if i not in y:                       # route from a non-open depot
                return False
            load = 0.0; tau = 0.0; prev = i
            for a in path:
                node = J[a]
                load += inst.demand[node]
                tau += inst.l[prev][node]
                prev = node
                seen.append(a)
            tau += inst.l[prev][i]               # duration incl. return to depot
            if load > inst.Q + 1e-6 or tau > inst.L + 1e-6:
                return False
            dep_dem[i] = dep_dem.get(i, 0.0) + load
        if len(seen) != n or len(set(seen)) != n:
            return False
        if any(dem > inst.W[i] + 1e-6 for i, dem in dep_dem.items()):
            return False
        return len(routes) <= Nv + 1e-9

    best_routes = None                           # route set attaining the incumbent
    if objective == "weighted":
        if incumbent is None:
            incumbent = float("inf")
    else:
        # Step 3.3: in budget/Tchebycheff modes the initial UB is the mode value
        # of a feasible seed solution -- NEVER a weighted-sum UB passed in. The
        # greedy seed packs by Q and L only (it ignores depot throughput W), so it
        # must clear the full feasibility guard before it may seed the incumbent;
        # otherwise start at +inf and let the dcap-constrained RMH supply the first
        # real incumbent.
        incumbent = float("inf")
        if _feasible(greedy_sol):
            _v0 = _sol_value(*solution_parts(inst, I, greedy_sol))
            if _v0 < float("inf"):
                incumbent = _v0
                best_routes = list(greedy_sol)
    import heapq
    ctr = 0                                     # tiebreaker (frozensets uncomparable)
    heap = [(-float("inf"), ctr, frozenset())]
    n_nodes = 0
    all_converged = True
    cur_lb = -float("inf")                       # bound of the node being expanded
    # Shared global incumbent (across parallel depot combinations): we prune/fathom
    # against eff = min(our own incumbent, the global best). So the moment any
    # combination finds a good solution, every other combination whose bound
    # already exceeds it dies immediately instead of proving its own (irrelevant)
    # optimum. With inc_get=None this reduces exactly to the single-solve logic.
    def _glob():
        return inc_get() if inc_get is not None else float("inf")

    def _eff():
        return min(incumbent, _glob())

    def _ptol():
        # Step 3.4: relative pruning / optimality-gap tolerance (Z_lambda is O(1),
        # weighted objective is O(1e4); 1e-6*max(1,|UB|) works for both).
        e = _eff()
        return 1e-6 * max(1.0, abs(e)) if e < float("inf") else 1e-6

    def _improve(val, routes):
        nonlocal incumbent, best_routes
        # An incumbent VALUE is only as trustworthy as the routes behind it: reject
        # any non-None route set that is not fully feasible (coverage/Q/L/W/Nv), so
        # a cost-only recompute can never install a better-than-optimal incumbent.
        if routes is not None and not _feasible(routes):
            return
        if val is not None and val < incumbent - 1e-9:
            incumbent = val
            if routes is not None:
                best_routes = routes
            if inc_put is not None:
                inc_put(val)
        elif (val is not None and routes is not None and best_routes is None
              and val <= incumbent + 1e-6):
            # These routes achieve the current incumbent (e.g. equal to a carried
            # seed UB). Capture them so this combination has its own witnessing
            # solution -- otherwise, when the seed is already optimal, no combo
            # ever "improves" and the optimal routes would be lost.
            best_routes = routes

    while heap and time.time() - t0 < total_time and n_nodes < max_nodes:
        pb, _, F = heapq.heappop(heap)
        if pb >= _eff() - _ptol():             # frontier min >= UB: all remaining
            heap.clear()                           # nodes fathomed -> proven (>= eff)
            break
        n_nodes += 1
        deadline = min(t0 + total_time, time.time() + node_time)
        lb, conv = solve_node(F, deadline)
        if lb == float("inf"):
            continue                                    # infeasible node -> fathom
        if lb is None:
            all_converged = False
            continue
        if not conv:
            all_converged = False
        cur_lb = lb
        if lb >= _eff() - _ptol():
            continue                                    # prune by bound (own or global)
        key = frac_arc()
        if key is None:
            # integer LP -> feasible; refine incumbent via RMH too
            ub, routes = rmh_incumbent(F)
            cand = min(x for x in (lb, ub) if x is not None)
            _improve(cand, routes)
            if verbose:
                print(f"  node {n_nodes}: integer, obj={cand:.2f} "
                      f"incumbent={incumbent:.2f}", flush=True)
            continue
        # try to improve incumbent cheaply before branching. Whenever pricing is
        # cheap (the compiled bidirectional pricer in cost_mode, or the njit pricer
        # under the budget / Tchebycheff scalarisations) the integer RMH solve
        # dominates a node, so running it EVERY node starves the search of nodes
        # (observed: rmh=113s vs price=0.6s over 820 nodes for a tchebycheff pair).
        # Throttle it to every 25th node in those modes; only the plain weighted
        # objective -- whose forward pricing already makes nodes expensive -- keeps
        # the every-node RMH (its verified behaviour is unchanged). Integer-LP nodes
        # still capture their incumbent unconditionally via the frac_arc()-None path.
        if (objective == "weighted" and not cost_mode) or (n_nodes % 25 == 1):
            ub, routes = rmh_incumbent(F)
            _improve(ub, routes)
            if lb >= _eff() - _ptol():
                continue
        # branch on the fractional arc
        if key[0] == "cc":
            _, u, v = key
            forbid = frozenset({("cc", u, v)})
            force = frozenset({("cc", u, w) for w in range(n) if w != v}
                              | {("in", u, i) for i in I}
                              | {("cc", w, v) for w in range(n) if w != u}
                              | {("out", i, v) for i in I})
        elif key[0] == "out":
            _, i, a = key
            forbid = frozenset({("out", i, a)})
            # force out(i,a): a is served as a first stop by depot i, so a may be
            # ENTERED only via (i,a). Restrict at the pharmacy endpoint a only --
            # do NOT forbid depot i's other departures (it may run several routes).
            force = frozenset({("cc", w, a) for w in range(n) if w != a}
                              | {("out", k, a) for k in I if k != i})
        else:
            _, a, i = key
            forbid = frozenset({("in", a, i)})
            # force in(a,i): a is a last stop returning to depot i, so a may LEAVE
            # only via (a,i). Restrict at the pharmacy endpoint a only.
            force = frozenset({("cc", a, w) for w in range(n) if w != a}
                              | {("in", a, k) for k in I if k != i})
        ctr += 1
        heapq.heappush(heap, (lb, ctr, F | forbid))
        ctr += 1
        heapq.heappush(heap, (lb, ctr, F | force))
        if status_cb is not None and n_nodes % 5 == 1:
            gm = heap[0][0] if heap else lb
            status_cb({"nodes": n_nodes,
                       "glb": None if gm == -float("inf") else gm,
                       "incumbent": None if incumbent == float("inf") else incumbent,
                       "open": len(heap), "t": time.time() - t0})
        if verbose and n_nodes % 5 == 1:
            hmin = heap[0][0] if heap else lb
            hmin_txt = "-inf" if hmin == -float("inf") else f"{hmin:.2f}"
            print(f"  node {n_nodes}: lb={lb:.2f} glb={hmin_txt} "
                  f"incumbent={incumbent:.2f} branch {key} "
                  f"|open|={len(heap)}", flush=True)

    if price_pool is not None:
        price_pool.shutdown(wait=False)
    # eff = the effective bound we fathomed against (own incumbent or the shared
    # global, whichever is smaller). Since both only decrease, every pruned/broken
    # node had lb >= eff, so eff is a valid lower bound on this combination's
    # optimum. With inc_get=None, eff == incumbent and this is the original logic.
    eff = min(incumbent, _glob())
    proven = (not heap) and all_converged and eff < float("inf")
    if proven:
        lb_report = eff
    elif heap:
        # valid global LB = min bound over the open frontier
        lb_report = min(heap[0][0], cur_lb)
        if lb_report == -float("inf"):
            lb_report = None
    else:
        lb_report = None
    # own_optimum: we proved THIS combination's OWN optimum -- it must have its own
    # witnessing routes (best_routes) AND its lb must have met that incumbent. A
    # combo merely fathomed against a seed/global it never achieved has
    # best_routes=None (or lb_report < incumbent) and is not own_optimum.
    own_optimum = bool(proven and best_routes is not None and incumbent < float("inf")
                       and lb_report is not None
                       and lb_report >= incumbent - 1e-6 * max(1.0, abs(incumbent)))
    total_s = time.time() - t0
    _CF, _P = (solution_parts(inst, I, best_routes)
               if best_routes is not None else (None, None))
    out = {"fixed_depots": [inst.depot_labels[i] for i in I],
           "incumbent": None if incumbent == float("inf") else incumbent,
           "lb": lb_report,
           "C_F": _CF, "P": _P,            # financial & latency, recomputed from routes
           "routes": best_routes,
           "own_optimum": own_optimum,
           "proven": proven, "n_nodes": n_nodes,
           "stack_left": len(heap), "all_converged": all_converged,
           "cg_iters": CG["iters"], "n_cols": len(cols), "n_src": len(active_src),
           "eta_C": TCH["eta_C"], "eta_P": TCH["eta_P"], "etaP_min": TCH["etaP_min"],
           "time_s": total_s,
           "timing": {"pricing": T["price"], "rmh": T["rmh"],
                      "master_lp_and_other": max(0.0, total_s - T["price"] - T["rmh"]),
                      "total": total_s},
           # Tier-B carry: the cheapest `pool_cap` elementary columns generated
           # here, to warm-start CG for the next solve of this same depot set.
           "pool": [(r["i"], list(r["path"]))
                    for r in sorted((rc for rc in cols
                                     if len(set(rc["path"])) == len(rc["path"])),
                                    key=lambda rc: rc["cost"])[:pool_cap]]}
    if verbose:
        lb_txt = "n/a" if lb_report is None else f"{lb_report:.2f}"
        print(f"BNP {inst.city} depots={out['fixed_depots']}: "
              f"lb={lb_txt} incumbent={out['incumbent']} proven={proven} "
              f"nodes={n_nodes} open_left={len(heap)} "
              f"t={total_s:.0f}s (price={T['price']:.0f}s rmh={T['rmh']:.0f}s)",
              flush=True)
    return out


if __name__ == "__main__":
    import os
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    from llrp_loader import load_instance, subsample_instance
    # quick self-test on the n=12 subsample, fixed to its optimal depots
    parent = load_instance("Elazig", ".", verbose=False)
    sub = subsample_instance(parent, 12, n_depots=4, seed=7, Ng=2, Nv=4)
    branch_and_price(sub, [0, 1], ng_size=8, total_time=600, verbose=True)
