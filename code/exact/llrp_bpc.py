"""
llrp_bpc.py
===========

Branch-price-and-cut style solver for the Priority-Weighted Latency LRP:
column generation + subset-row cuts (SRC) + price-and-branch.

  * Restricted master  : set-partitioning LP over route columns (Gurobi), with
    depot-open (y), exact open-count (N_g), depot capacity (W_i), depot-link,
    and global fleet (N_v) constraints.  Q and L are enforced inside pricing.
  * Pricing subproblem : exact elementary ESPPRC per depot by label-setting.
    The label carries elapsed time tau, load, a visited bitmask, an SRC-parity
    state, and reduced cost.  Priority-weighted latency alpha*p_v*tau is
    accumulated on arrival; tau is monotone, hence a valid dominance resource.
  * Cuts               : limited subset-row cuts on customer triples C:
        sum_{r: |r cap C| >= 2} lambda_r <= 1.
    Non-robust: pricing tracks parity of C-visits and applies the cut dual when
    the count crosses an even value; dominance is corrected accordingly
    (Jepsen et al. 2008).
  * Bounds             : the converged LP (with cuts) is a valid lower bound;
    the integer master over the generated pool is a valid upper bound.

Research prototype (pure-Python pricer) for small instances (n <= ~20).
"""

from __future__ import annotations

import time
from collections import defaultdict
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from llrp_loader import LLRPInstance
from llrp_model import heuristic_solution

BIG = 1e7
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Pricing: exact elementary ESPPRC for one depot, SRC-aware                    #
# --------------------------------------------------------------------------- #
def _ng_neighborhoods(inst, K=8):
    """ng-neighborhood bitmask per customer: the K nearest customers (incl. self),
    by symmetric travel time."""
    n, Jn = inst.n, inst.J
    ng = []
    for a in range(n):
        order = sorted(range(n),
                       key=lambda b: inst.l[Jn[a]][Jn[b]] + inst.l[Jn[b]][Jn[a]])
        mask = 0
        for b in order[:K]:
            mask |= (1 << b)
        mask |= (1 << a)
        ng.append(mask)
    return ng


def _price_depot(inst, i, pi_cov, sig_i, rho_i, mu, srcs, max_cols=40,
                 keep_threshold=-EPS, ng=None, wC=1.0, wP=None):
    """Best reduced-cost routes for depot i.

    ``srcs`` : list of (C_mask, omega) active subset-row cuts (omega = dual <= 0).
    ``ng``   : if given (list of ng-neighborhood bitmasks), price ng-routes
               (bounded memory) instead of elementary routes -- a valid
               relaxation that scales to larger n.
    Routes with reduced cost < ``keep_threshold`` are returned.
    """
    n, m = inst.n, inst.m
    Jn = inst.J
    d = [inst.demand[Jn[a]] for a in range(n)]
    p = [inst.priority[Jn[a]] for a in range(n)]
    l_di = [inst.l[i][Jn[a]] for a in range(n)]
    c_di = [inst.c[i][Jn[a]] for a in range(n)]
    l_id = [inst.l[Jn[a]][i] for a in range(n)]
    c_id = [inst.c[Jn[a]][i] for a in range(n)]
    l_cc = [[inst.l[Jn[a]][Jn[b]] for b in range(n)] for a in range(n)]
    c_cc = [[inst.c[Jn[a]][Jn[b]] for b in range(n)] for a in range(n)]
    Q, L, F = inst.Q, inst.L, inst.F[i]
    if wP is None:                      # weighted-sum default: wC=1, wP=alpha
        wP = inst.alpha

    ncut = len(srcs)
    omega = [srcs[j][1] for j in range(ncut)]
    cust_cuts = [[] for _ in range(n)]         # a -> cut indices whose C contains a
    for j, (Cmask, _) in enumerate(srcs):
        cm = Cmask
        a = 0
        while cm:
            if cm & 1:
                cust_cuts[a].append(j)
            cm >>= 1
            a += 1

    buckets = defaultdict(list)   # a -> [(rc, tau, load, mask, sstate, path)]

    def dominates(A, B):
        rcA, tA, lA, mA, sA, _ = A
        rcB, tB, lB, mB, sB, _ = B
        if (mA & mB) != mA or tA > tB + 1e-9 or lA > lB + 1e-9:
            return False
        pen = 0.0
        diff = sA & ~sB                        # cuts pending in A but not in B
        j = 0
        while diff:
            if diff & 1:
                pen -= omega[j]                # omega <= 0 -> pen >= 0
            diff >>= 1
            j += 1
        return rcA + pen <= rcB + 1e-9

    def try_insert(a, lab):
        lst = buckets[a]
        for ex in lst:
            if dominates(ex, lab):
                return False
        buckets[a] = [ex for ex in lst if not dominates(lab, ex)]
        buckets[a].append(lab)
        return True

    queue = []
    for a in range(n):
        if d[a] > Q:
            continue
        tau = l_di[a]
        if tau + l_id[a] > L + 1e-9:
            continue
        rc = wC * c_di[a] + wP * p[a] * tau - pi_cov[a] - sig_i * d[a]
        sstate = 0
        for j in cust_cuts[a]:                 # first C-member: 0 -> 1, no penalty
            sstate |= (1 << j)
        lab = (rc, tau, d[a], 1 << a, sstate, (a,))
        if try_insert(a, lab):
            queue.append((a, lab))

    while queue:
        a, lab = queue.pop()
        if lab not in buckets[a]:
            continue
        rc, tau, load, mask, sstate, path = lab
        for b in range(n):
            if mask & (1 << b):
                continue
            nload = load + d[b]
            if nload > Q:
                continue
            ntau = tau + l_cc[a][b]
            if ntau + l_id[b] > L + 1e-9:
                continue
            nrc = rc + wC * c_cc[a][b] + wP * p[b] * ntau - pi_cov[b] - sig_i * d[b]
            nstate = sstate
            for j in cust_cuts[b]:
                bit = 1 << j
                if nstate & bit:               # count crosses even -> apply dual
                    nrc -= omega[j]
                    nstate &= ~bit
                else:
                    nstate |= bit
            nmask = (mask & ng[b]) | (1 << b) if ng is not None else mask | (1 << b)
            nlab = (nrc, ntau, nload, nmask, nstate, path + (b,))
            if try_insert(b, nlab):
                queue.append((b, nlab))

    out = []
    for a, lst in buckets.items():
        for (rc, tau, load, mask, sstate, path) in lst:
            rcf = rc + wC * (c_id[a] + F) - rho_i - mu
            if rcf < keep_threshold:
                out.append((rcf, path))
    out.sort(key=lambda z: z[0])
    return out[:max_cols]


def _route_parts(inst, i, path):
    """Split a route's two criteria:
      cF = financial cost = travel (incl. the return arc) + vehicle cost F_i
      pL = priority-weighted latency = sum_j p_j * arrival_j (return arc EXCLUDED)
      d_r = total demand
    Returns (cF, pL, d_r)."""
    Jn = inst.J
    tau = 0.0
    prev = i
    travel = 0.0
    lat = 0.0
    for a in path:
        node = Jn[a]
        travel += inst.c[prev][node]
        tau += inst.l[prev][node]
        lat += inst.priority[node] * tau
        prev = node
    travel += inst.c[prev][i]                 # return arc -> financial only
    dr = sum(inst.demand[Jn[a]] for a in path)
    return travel + inst.F[i], lat, dr


def _route_cost(inst, i, path):
    """Weighted-sum route cost (thin wrapper over _route_parts)."""
    cF, pL, dr = _route_parts(inst, i, path)
    return cF + inst.alpha * pL, dr


# --------------------------------------------------------------------------- #
# Solver                                                                      #
# --------------------------------------------------------------------------- #
def solve_bpc(inst: LLRPInstance, *, time_limit=600.0, cg_time=None,
              use_src=True, max_src_rounds=6, max_src_per_round=25,
              fixed_depots=None, verbose=True) -> dict:
    """Solve by column generation + SRC + price-and-branch.

    ``fixed_depots``: if given, only these depots may be used and all are forced
    open (used by the y-branching wrapper). This removes location fractionality.
    """
    J = inst.J
    n, m = inst.n, inst.m
    Nv = inst.Nv
    I = list(fixed_depots) if fixed_depots is not None else list(inst.I)
    Ng = len(I) if fixed_depots is not None else inst.Ng
    t_start = time.time()
    cg_time = cg_time if cg_time is not None else 0.75 * time_limit

    mdl = gp.Model("LLRP-BPC")
    mdl.Params.OutputFlag = 0
    mdl.ModelSense = GRB.MINIMIZE

    y = {i: mdl.addVar(lb=(1.0 if fixed_depots is not None else 0.0), ub=1.0,
                       obj=inst.f[i], name=f"y{i}") for i in I}
    art = {a: mdl.addVar(obj=BIG, name=f"art{a}") for a in range(n)}
    cover = {a: mdl.addConstr(art[a] == 1, name=f"cov{a}") for a in range(n)}
    dcap = {i: mdl.addConstr(-inst.W[i] * y[i] <= 0, name=f"dcap{i}") for i in I}
    dlink = {i: mdl.addConstr(-Nv * y[i] <= 0, name=f"dlink{i}") for i in I}
    mdl.addConstr(gp.quicksum(y[i] for i in I) == Ng, name="opencount")
    fleet = mdl.addConstr(gp.quicksum([]) <= Nv, name="fleet")

    columns = []
    seen = set()
    active_src = []          # list of {mask, con}

    def add_column(i, path):
        key = (i, tuple(path))
        if key in seen:
            return None
        seen.add(key)
        cost, dr = _route_cost(inst, i, path)
        col = gp.Column()
        for a in path:
            col.addTerms(1.0, cover[a])
        col.addTerms(dr, dcap[i])
        col.addTerms(1.0, dlink[i])
        col.addTerms(1.0, fleet)
        pset = set(path)
        for s in active_src:
            coeff = sum(1 for a in path if (s["mask"] >> a) & 1) // 2
            if coeff:
                col.addTerms(coeff, s["con"])
        var = mdl.addVar(obj=cost, column=col, name=f"r{len(columns)}")
        columns.append({"var": var, "i": i, "path": tuple(path),
                        "cost": cost, "dr": dr, "pset": pset})
        return columns[-1]

    # initial columns: singletons + heuristic routes
    for i in I:
        for a in range(n):
            if inst.demand[J[a]] <= inst.Q and \
                    inst.l[i][J[a]] + inst.l[J[a]][i] <= inst.L + 1e-9:
                add_column(i, [a])
    hs = heuristic_solution(inst)
    if hs is not None:
        for i, routes in hs[1].items():
            if i not in y:                       # depot not allowed in this subset
                continue
            for route in routes:
                add_column(i, [node - m for node in route])

    def run_cg(deadline):
        """Column-generate to convergence (or deadline). Returns converged?"""
        while True:
            mdl.optimize()
            if mdl.Status != GRB.OPTIMAL:
                return False
            srcs = [(s["mask"], s["con"].Pi) for s in active_src]
            pi_cov = [cover[a].Pi for a in range(n)]
            added = 0
            for i in I:
                for _, path in _price_depot(inst, i, pi_cov, dcap[i].Pi,
                                            dlink[i].Pi, fleet.Pi, srcs):
                    if add_column(i, path) is not None:
                        added += 1
            if added == 0:
                return True
            if time.time() > deadline:
                return False

    deadline = t_start + cg_time
    converged = run_cg(deadline)
    root_lp = mdl.ObjVal if mdl.Status == GRB.OPTIMAL else None

    # --- subset-row cut rounds ------------------------------------------- #
    n_cuts = 0
    if use_src:
        for rnd in range(max_src_rounds):
            if time.time() > deadline:
                converged = False
                break
            mdl.optimize()
            active_cols = [(rec, rec["var"].X, rec["pset"])
                           for rec in columns if rec["var"].X > 1e-6]
            existing = {s["mask"] for s in active_src}
            viol = []
            for tri in combinations(range(n), 3):
                mask = (1 << tri[0]) | (1 << tri[1]) | (1 << tri[2])
                if mask in existing:
                    continue
                Cset = set(tri)
                s = sum(v for (_, v, ps) in active_cols if len(ps & Cset) >= 2)
                if s > 1.0 + 1e-4:
                    viol.append((s, mask, tri))
            if not viol:
                break
            viol.sort(reverse=True)
            for _, mask, tri in viol[:max_src_per_round]:
                expr = gp.LinExpr()
                for rec in columns:
                    coeff = sum(1 for a in rec["path"] if (mask >> a) & 1) // 2
                    if coeff:
                        expr.addTerms(coeff, rec["var"])
                con = mdl.addConstr(expr <= 1.0, name=f"src{n_cuts}")
                active_src.append({"mask": mask, "con": con})
                n_cuts += 1
            converged = run_cg(deadline)
        mdl.optimize()

    # A valid lower bound requires CG to have converged (no negative column) and
    # the artificials to be driven out; otherwise the LP objective is NOT a bound.
    art_sum = sum(art[a].X for a in range(n)) if mdl.Status == GRB.OPTIMAL else 1e18
    lb = mdl.ObjVal if (converged and mdl.Status == GRB.OPTIMAL
                        and art_sum < 1e-4) else None

    # --- pool enrichment for a good UB ----------------------------------- #
    if mdl.Status == GRB.OPTIMAL:
        srcs = [(s["mask"], s["con"].Pi) for s in active_src]
        pi_cov = [cover[a].Pi for a in range(n)]
        for i in I:
            for _, path in _price_depot(inst, i, pi_cov, dcap[i].Pi, dlink[i].Pi,
                                        fleet.Pi, srcs, max_cols=1200,
                                        keep_threshold=1e18):
                add_column(i, path)

    # --- integer master over the pool (price-and-branch) ----------------- #
    for a in range(n):
        art[a].ub = 0.0
    for rec in columns:
        rec["var"].vtype = GRB.BINARY
    for i in I:
        y[i].vtype = GRB.BINARY
    mdl.Params.TimeLimit = max(5.0, time_limit - (time.time() - t_start))
    mdl.optimize()

    ub = mdl.ObjVal if mdl.SolCount > 0 else None
    gap = None
    if ub is not None and lb is not None and ub > 1e-9:
        gap = max(0.0, (ub - lb) / ub)

    routes = []
    if mdl.SolCount > 0:
        for rec in columns:
            if rec["var"].X > 0.5:
                routes.append({"depot": rec["i"],
                               "path_nodes": [m + a for a in rec["path"]],
                               "cost": rec["cost"], "load": rec["dr"]})

    result = {
        "city": inst.city, "n": n, "m": m,
        "root_lp_bound": root_lp, "lb": lb, "ub": ub, "gap": gap,
        "n_columns": len(columns), "n_cuts": n_cuts,
        "time_s": time.time() - t_start, "routes": routes,
        "status": "OPTIMAL" if (gap is not None and gap < 1e-6) else
                  ("FEASIBLE" if ub is not None else "NO_SOLUTION"),
    }
    if verbose:
        g = "n/a" if gap is None else f"{gap*100:.3f}%"
        print(f"BPC {inst.city}: root_lp={root_lp:.2f} LB={lb:.2f} UB={ub:.2f} "
              f"gap={g} cuts={n_cuts} cols={len(columns)} time={result['time_s']:.1f}s")
    return result


def solve_bpc_ybranch(inst: LLRPInstance, *, time_limit=600.0, use_src=True,
                      verbose=True) -> dict:
    """Exact BPC via depot-set enumeration (y-branching).

    The root gap is entirely in the location variables y, so we enumerate every
    open-depot set S of size N_g, solve the routing exactly for each with
    column generation + SRC (location fixed => no y fractionality), and take the
    minimum. Valid global bounds:
        global LB = min_S LB(S),   global UB = min_S UB(S).
    If every subset closes, the incumbent is proven optimal.
    """
    subsets = list(combinations(inst.I, inst.Ng))
    per_budget = max(10.0, time_limit / len(subsets))
    t_start = time.time()
    best_ub, best = float("inf"), None
    lb_vals = []
    certified = True                     # every subset gave a valid lower bound?
    for S in subsets:
        remaining = time_limit - (time.time() - t_start)
        if remaining <= 1.0:
            certified = False
            break
        r = solve_bpc(inst, fixed_depots=list(S), use_src=use_src,
                      time_limit=min(per_budget, remaining), verbose=False)
        ub_s = r["ub"] if r["ub"] is not None else float("inf")
        if r["lb"] is None:              # subset not certified (CG not converged)
            certified = False
        else:
            lb_vals.append(r["lb"])
        if ub_s < best_ub:
            best_ub, best = ub_s, r
        if verbose:
            labels = [inst.depot_labels[i] for i in S]
            lbtxt = "none" if r["lb"] is None else f"{r['lb']:.2f}"
            ubtxt = "inf" if ub_s == float("inf") else f"{ub_s:.2f}"
            print(f"  depots {labels}: LB={lbtxt} UB={ubtxt} "
                  f"cuts={r['n_cuts']} t={r['time_s']:.1f}s")

    # Global LB is valid only if EVERY subset was certified.
    global_lb = min(lb_vals) if (certified and lb_vals) else None
    global_ub = best_ub if best_ub < float("inf") else None
    gap = None
    if certified and global_lb is not None and global_ub is not None and global_ub > 1e-9:
        gap = max(0.0, (global_ub - global_lb) / global_ub)
    status = "OPTIMAL" if (gap is not None and gap < 1e-6) else \
             ("FEASIBLE" if global_ub is not None else "NO_SOLUTION")
    out = {"city": inst.city, "n": inst.n, "lb": global_lb, "ub": global_ub,
           "gap": gap, "certified": certified, "status": status,
           "time_s": time.time() - t_start, "best": best}
    if verbose:
        g = "n/a" if gap is None else f"{gap*100:.3f}%"
        lbtxt = "none" if global_lb is None else f"{global_lb:.2f}"
        ubtxt = "none" if global_ub is None else f"{global_ub:.2f}"
        print(f"BPC-Ybranch {inst.city}: LB={lbtxt} UB={ubtxt} gap={g} "
              f"certified={certified} status={status} time={out['time_s']:.1f}s")
    return out


def bpc_lower_bound(inst: LLRPInstance, *, time_limit=120.0, ng_size=8,
                    fixed_depots=None, use_src=False, max_src_rounds=6,
                    max_src_per_round=20, compute_ub=False, ub_time=60.0,
                    verbose=True):
    """Anytime-valid lower bound via column generation (ng + optional SRC).

    After each pricing round the Farley/Lagrangian bound

        LB = z_RMP + N_v * min(0, cbar_min)

    is a valid lower bound on the integer optimum, valid at every iteration.
    ``use_src`` adds subset-row cut rounds (the compiled pricer tracks cut
    parity), which tighten the bound. ``fixed_depots`` restricts/forces the open
    depot set (for depot enumeration).
    """
    J, n, Nv = inst.J, inst.n, inst.Nv
    I = list(fixed_depots) if fixed_depots is not None else list(inst.I)
    Ng_eff = len(I) if fixed_depots is not None else inst.Ng
    ng = _ng_neighborhoods(inst, K=ng_size) if ng_size else None
    nbp = None
    if ng is not None and n <= 64:
        try:
            from llrp_pricer_nb import NbPricer
            nbp = NbPricer(inst, ng)
        except Exception:
            nbp = None
    mdl = gp.Model("BPC-LB")
    mdl.Params.OutputFlag = 0
    mdl.ModelSense = GRB.MINIMIZE
    y = {i: mdl.addVar(lb=(1.0 if fixed_depots is not None else 0.0), ub=1.0,
                       obj=inst.f[i]) for i in I}
    art = {a: mdl.addVar(obj=BIG) for a in range(n)}
    cover = {a: mdl.addConstr(art[a] == 1) for a in range(n)}
    dcap = {i: mdl.addConstr(-inst.W[i] * y[i] <= 0) for i in I}
    dlink = {i: mdl.addConstr(-Nv * y[i] <= 0) for i in I}
    mdl.addConstr(gp.quicksum(y[i] for i in I) == Ng_eff)
    fleet = mdl.addConstr(gp.quicksum([]) <= Nv)

    seen = set()
    cols = []
    active_src = []                              # [{mask, con}]

    def add_column(i, path):
        key = (i, tuple(path))
        if key in seen:
            return False
        seen.add(key)
        cost, dr = _route_cost(inst, i, path)
        col = gp.Column()
        for a in path:
            col.addTerms(1.0, cover[a])
        col.addTerms(dr, dcap[i])
        col.addTerms(1.0, dlink[i])
        col.addTerms(1.0, fleet)
        for s in active_src:
            coeff = sum(1 for a in path if (s["mask"] >> a) & 1) // 2
            if coeff:
                col.addTerms(coeff, s["con"])
        cols.append({"var": mdl.addVar(obj=cost, column=col),
                     "i": i, "path": tuple(path), "pset": set(path)})
        return True

    for i in I:
        for a in range(n):
            if inst.demand[J[a]] <= inst.Q and \
                    inst.l[i][J[a]] + inst.l[J[a]][i] <= inst.L + 1e-9:
                add_column(i, [a])
    hs = heuristic_solution(inst)
    if hs is not None:
        for i, routes in hs[1].items():
            if i not in y:
                continue
            for r in routes:
                add_column(i, [nd - inst.m for nd in r])

    state = {"best_lb": -float("inf")}
    t0 = time.time()

    def run_cg(deadline):
        conv = False
        while time.time() < deadline:
            mdl.optimize()
            if mdl.Status != GRB.OPTIMAL:
                break
            z = mdl.ObjVal
            art_sum = sum(art[a].X for a in range(n))
            pi_cov = [cover[a].Pi for a in range(n)]
            srcs = [(s["mask"], s["con"].Pi) for s in active_src]
            cbar_min = 0.0
            added = 0
            for i in I:
                found, mr = None, None
                if nbp is not None:
                    found, mr = nbp.price(inst, i, pi_cov, dcap[i].Pi, dlink[i].Pi,
                                          fleet.Pi, srcs=srcs, max_cols=40)
                if found is None:
                    found = _price_depot(inst, i, pi_cov, dcap[i].Pi, dlink[i].Pi,
                                         fleet.Pi, srcs, max_cols=40, ng=ng)
                    mr = found[0][0] if found else 0.0
                cbar_min = min(cbar_min, mr)
                for _, path in found:
                    if add_column(i, path):
                        added += 1
            if art_sum < 1e-4:
                state["best_lb"] = max(state["best_lb"], z + Nv * min(0.0, cbar_min))
            if added == 0:
                if art_sum < 1e-4:
                    state["best_lb"] = max(state["best_lb"], z)
                conv = True
                break
        return conv

    deadline = t0 + time_limit
    converged = run_cg(deadline)

    n_cuts = 0
    if use_src:
        for _ in range(max_src_rounds):
            if time.time() >= deadline:
                converged = False
                break
            mdl.optimize()
            active_cols = [(rec["pset"], rec["var"].X) for rec in cols
                           if rec["var"].X > 1e-6]
            existing = {s["mask"] for s in active_src}
            viol = []
            for tri in combinations(range(n), 3):
                mask = (1 << tri[0]) | (1 << tri[1]) | (1 << tri[2])
                if mask in existing:
                    continue
                Cset = set(tri)
                s = sum(v for ps, v in active_cols if len(ps & Cset) >= 2)
                if s > 1.0 + 1e-4:
                    viol.append((s, mask))
            if not viol:
                break
            viol.sort(reverse=True)
            for _, mask in viol[:max_src_per_round]:
                if len(active_src) >= 63:            # cut-parity bitmask limit
                    break
                expr = gp.LinExpr()
                for rec in cols:
                    coeff = sum(1 for a in rec["path"] if (mask >> a) & 1) // 2
                    if coeff:
                        expr.addTerms(coeff, rec["var"])
                con = mdl.addConstr(expr <= 1.0)
                active_src.append({"mask": mask, "con": con})
                n_cuts += 1
            converged = run_cg(deadline)
        mdl.optimize()

    best_lb = state["best_lb"]
    # diagnostics: depot-open (y) values and fractional route count at the final LP
    y_values, n_frac_lambda = {}, None
    try:
        mdl.optimize()
        y_values = {i: round(float(y[i].X), 4) for i in I}
        n_frac_lambda = sum(1 for rec in cols if 1e-6 < rec["var"].X < 1 - 1e-6)
    except Exception:
        pass

    # --- restricted-master-heuristic UPPER bound ------------------------- #
    # Solve the integer set-partitioning over the generated ELEMENTARY columns.
    # A route that revisits a customer (possible with ng) is excluded, so the
    # integer solution is a genuine feasible solution -> a valid upper bound.
    ub = None
    if compute_ub and mdl.Status == GRB.OPTIMAL:
        srcs = [(s["mask"], s["con"].Pi) for s in active_src]
        pi_cov = [cover[a].Pi for a in range(n)]
        for i in I:                                  # enrich with near-optimal cols
            found, _ = (nbp.price(inst, i, pi_cov, dcap[i].Pi, dlink[i].Pi,
                                  fleet.Pi, srcs=srcs, max_cols=1500,
                                  keep_threshold=1e18)) if nbp is not None else (None, None)
            if found is None:
                found = _price_depot(inst, i, pi_cov, dcap[i].Pi, dlink[i].Pi,
                                     fleet.Pi, srcs, max_cols=1500,
                                     keep_threshold=1e18, ng=ng)
            for _, path in found:
                add_column(i, path)
        for a in range(n):
            art[a].ub = 0.0
        for rec in cols:
            if len(rec["path"]) == len(set(rec["path"])):   # elementary
                rec["var"].vtype = GRB.BINARY
            else:
                rec["var"].ub = 0.0                          # exclude revisits
        for i in I:
            y[i].vtype = GRB.BINARY
        mdl.Params.TimeLimit = ub_time
        mdl.optimize()
        ub = mdl.ObjVal if mdl.SolCount > 0 else None

    return {"lb": best_lb if best_lb > -float("inf") else None, "ub": ub,
            "converged": converged, "n_cuts": n_cuts,
            "time_s": time.time() - t0, "n_columns": len(cols),
            "y_values": y_values, "n_frac_lambda": n_frac_lambda}


def certify_gap(inst: LLRPInstance, *, t_lb=120.0, t_ub=120.0, verbose=True):
    """Certified optimality gap: BPC lower bound vs flow-model upper bound."""
    from llrp_model import solve
    lbres = bpc_lower_bound(inst, time_limit=t_lb, verbose=verbose)
    lb = lbres["lb"]
    ubres = solve(inst, formulation="flow", tier1=True, warm_start=True,
                  tuned=True, time_limit=t_ub, verbose=False)
    ub = ubres["objective"]
    gap = None
    if lb is not None and ub is not None and ub > 1e-9:
        gap = max(0.0, (ub - lb) / ub)
    out = {"city": inst.city, "n": inst.n, "lb": lb, "ub": ub, "gap": gap,
           "lb_converged": lbres["converged"], "lb_time_s": lbres["time_s"],
           "lb_cols": lbres["n_columns"]}
    if verbose:
        g = "n/a" if gap is None else f"{gap*100:.3f}%"
        print(f"CERTIFY {inst.city}: LB={lb:.2f} UB={ub:.2f} gap={g} "
              f"(LB converged={lbres['converged']}, {lbres['time_s']:.1f}s)")
    return out


def exact_solve(inst: LLRPInstance, *, ng_size=8, t_ub=120.0, t_sub=120.0,
                use_src=True, enumerate_depots=True, use_bpc=None,
                depot_enum_max_n=24, gurobi_log=False, verbose=True):
    """Exact-method bounds for one instance: upper bound + best valid lower bound.

    UB : flow-model incumbent (a feasible, elementary solution).
    LB : the best of (a) the flow model's own LP bound, and (b) the branch-price
         lower bound.  The BPC bound is obtained by depot-set enumeration only
         for small instances (n <= depot_enum_max_n), where each fixed-depot
         subproblem is tractable; for larger n the depot-fixed pricing frontier
         explodes, so we fall back to the ROOT ng+SRC bound (all depots open --
         customers stay spread, routes short, CG converges) which is looser but
         obtainable.  If LB == UB the instance is proven optimal.
    """
    from math import comb
    from llrp_model import solve

    # --- upper bound + flow LP bound --------------------------------------
    if verbose and gurobi_log:
        print("  [Gurobi log: flow upper-bound solve]")
    ubr = solve(inst, formulation="flow", tier1=True, warm_start=True, tuned=True,
                time_limit=t_ub, verbose=gurobi_log)
    ub = ubr["objective"]
    flow_gap = ubr["mip_gap"] or 0.0
    flow_lb = ub * (1.0 - flow_gap) if ub is not None else None

    # --- branch-price lower bound -----------------------------------------
    # The compiled ng pricer needs n <= 64; beyond that, skip BPC and rely on
    # the flow LP bound (unless the caller forces use_bpc=True).
    if use_bpc is None:
        use_bpc = inst.n <= 64
    do_enum = use_bpc and enumerate_depots and inst.n <= depot_enum_max_n
    details = []
    bpc_lb = None
    enum_certified = None
    conditional_lb = None
    if do_enum:
        subsets = list(combinations(inst.I, inst.Ng))
        if verbose:
            print(f"  depot-set enumeration: {len(subsets)} combinations "
                  f"(C({inst.m},{inst.Ng}))")
        conv_lbs, all_conv = [], True
        for S in subsets:
            r = bpc_lower_bound(inst, fixed_depots=list(S), ng_size=ng_size,
                                use_src=use_src, time_limit=t_sub, verbose=False)
            labels = [inst.depot_labels[i] for i in S]
            # only a CONVERGED subset gives a usable (tight, valid) bound; a
            # non-converged one returns a loose/garbage Farley value -> discard.
            ok = r["converged"] and r["lb"] is not None
            details.append((labels, r["lb"] if ok else None, r["n_cuts"],
                            r["converged"], r["time_s"]))
            if ok:
                conv_lbs.append(r["lb"])
            else:
                all_conv = False
            if verbose:
                lbt = f"{r['lb']:.2f}" if ok else "none (not converged)"
                pr = " PRUNED" if (ok and ub is not None and r["lb"] >= ub - 1e-6) else ""
                print(f"    {labels}: LB={lbt} cuts={r['n_cuts']} "
                      f"conv={r['converged']} t={r['time_s']:.1f}s{pr}")
        enum_certified = all_conv and len(conv_lbs) == len(subsets)
        if enum_certified:
            bpc_lb = min(conv_lbs)                 # valid global lower bound
        else:
            # not every depot set was bounded -> the enumeration min is NOT a
            # valid global bound. Keep the converged subsets' min as CONDITIONAL
            # info, and fall back to the ROOT bound for a valid unconditional LB.
            conditional_lb = min(conv_lbs) if conv_lbs else None
            if verbose:
                print(f"  enumeration incomplete ({len(conv_lbs)}/{len(subsets)} "
                      f"converged) -> falling back to root bound for a valid LB")
            rr = bpc_lower_bound(inst, ng_size=ng_size, use_src=use_src,
                                 time_limit=t_sub, verbose=False)
            details.append((["root(fallback)"], rr["lb"], rr["n_cuts"],
                            rr["converged"], rr["time_s"]))
            if rr["converged"] and rr["lb"] is not None:
                bpc_lb = rr["lb"]
    elif use_bpc:
        if verbose:
            why = (f"n={inst.n} > {depot_enum_max_n}: depot-fixing frontier too "
                   f"large") if inst.n > depot_enum_max_n else "root mode"
            print(f"  BPC root ng+SRC bound ({why})")
        r = bpc_lower_bound(inst, ng_size=ng_size, use_src=use_src,
                            time_limit=t_sub, verbose=verbose)
        details.append((["root"], r["lb"], r["n_cuts"], r["converged"], r["time_s"]))
        bpc_lb = r["lb"]
    elif verbose:
        print(f"  BPC skipped (n={inst.n} > 64): using flow LP bound only")

    # --- best valid lower bound + gap -------------------------------------
    cands = [x for x in (flow_lb, bpc_lb) if x is not None]
    lb = max(cands) if cands else None
    gap = (ub - lb) / ub if (lb is not None and ub and ub > 1e-9) else None
    proven = gap is not None and gap < 1e-6
    out = {"city": inst.city, "n": inst.n, "m": inst.m, "alpha": inst.alpha,
           "ub": ub, "flow_lb": flow_lb, "bpc_lb": bpc_lb, "lb": lb,
           "gap": gap, "proven": proven, "details": details,
           "enum_certified": enum_certified, "conditional_lb": conditional_lb}
    if verbose:
        g = "n/a" if gap is None else f"{gap*100:.3f}%"
        print(f"EXACT {inst.city} (alpha={inst.alpha:g}): "
              f"LB={lb if lb is None else round(lb,2)} UB={round(ub,2)} gap={g} "
              f"{'PROVEN OPTIMAL' if proven else ''}")
        if conditional_lb is not None and ub:
            cg = (ub - conditional_lb) / ub * 100
            print(f"  [conditional] best converged depot-set bound = "
                  f"{conditional_lb:.2f} -> gap {cg:.3f}% if those depots are optimal "
                  f"(NOT certified: some depot sets did not converge)")
    return out


def solve_fixed_depots(inst: LLRPInstance, triples, *, ng_size=8, t_sub=300.0,
                       ub_time=120.0, use_src=True, verbose=True):
    """Tight [LB, UB] per depot triple via BPC (ng+SRC LB + restricted-master UB).

    ``triples`` is a list of open-depot index tuples (the *live* ones, i.e. those
    not already pruned above the incumbent). Returns the global bound:
        global LB = min over triples of LB_t     (valid if pruned sets exceed UB)
        global UB = min over triples of UB_t     (best feasible found)
    """
    rows = []
    for S in triples:
        r = bpc_lower_bound(inst, fixed_depots=list(S), ng_size=ng_size,
                            use_src=use_src, time_limit=t_sub, compute_ub=True,
                            ub_time=ub_time, verbose=False)
        labels = [inst.depot_labels[i] for i in S]
        rows.append((labels, r["lb"], r["ub"], r["converged"], r["time_s"]))
        if verbose:
            lbt = "none" if r["lb"] is None else f"{r['lb']:.2f}"
            ubt = "none" if r["ub"] is None else f"{r['ub']:.2f}"
            tg = ("n/a" if (r["lb"] is None or r["ub"] is None) else
                  f"{(r['ub']-r['lb'])/r['ub']*100:.3f}%")
            print(f"  {labels}: LB={lbt} UB={ubt} triple-gap={tg} "
                  f"conv={r['converged']} t={r['time_s']:.0f}s")
    lbs = [lb for _, lb, _, _, _ in rows if lb is not None]
    ubs = [ub for _, _, ub, _, _ in rows if ub is not None]
    glb = min(lbs) if lbs else None
    gub = min(ubs) if ubs else None
    gap = (gub - glb) / gub if (glb is not None and gub) else None
    proven = gap is not None and gap < 1e-6
    if verbose:
        g = "n/a" if gap is None else f"{gap*100:.3f}%"
        print(f"FIXED-DEPOT {inst.city}: LB={glb} UB={gub} gap={g} "
              f"{'PROVEN OPTIMAL' if proven else ''}")
    return {"city": inst.city, "lb": glb, "ub": gub, "gap": gap,
            "proven": proven, "rows": rows}


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    from llrp_loader import load_instance, subsample_instance
    parent = load_instance(sys.argv[1] if len(sys.argv) > 1 else "Elazig", ".",
                           verbose=False)
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    sub = subsample_instance(parent, nn, n_depots=4, seed=7, Ng=2, Nv=4)
    print(sub.city, f"n={sub.n} m={sub.m} Q={sub.Q:g}")
    solve_bpc_ybranch(sub, time_limit=300, verbose=True)
