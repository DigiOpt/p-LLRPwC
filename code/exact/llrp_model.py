"""
llrp_model.py
=============

Gurobi MILP for the Priority-Weighted Latency Location-Routing Problem,
implementing equations (1)-(21) of ``MathModel.tex`` with two deliberate
reconciliations against the reference implementation (``LLRP_CORE.ipynb``):

  * (Ng)  an EXACT open-depot count is added:      sum_i y_i = Ng
  * (Nv)  the per-depot cap  k_i <= Kbar_i y_i  is replaced by a GLOBAL fleet
          limit  sum_i k_i <= Nv , plus an open-linking cap  k_i <= Nv y_i .
          (The data provides only a global Nv, no per-depot Kbar_i.)

Modelling note on the depot-indexed arc variable x^i_{uv}:
Only arcs meaningful for depot i's route are created -- i.e. the tail/head may
touch depot i but no other depot.  For route i the relevant node set is
{i} U J, giving arcs  i->J, J->i, and J->J.  This is the tight, correct reading
of "a route associated with depot i" and keeps the model compact.

MTZ chains:
  * load  omega_j : capacity + subtour elimination      (eq. 11-12)
  * time  t_j     : route-duration limit + latency term (eq. DL1-DL4)

Requires: gurobipy.  Build with :func:`build_model`, solve with :func:`solve`.
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from llrp_loader import LLRPInstance


def _arrival_lb(I, J, l):
    """Valid lower bound on the arrival time t_j at every pharmacy j.

    The arrival at j along any route is the cumulative travel from its depot
    through the preceding pharmacies, so a sound lower bound is the shortest
    path from any depot to j in the travel-time graph (depot entry + customer
    arcs).  This is <= the direct time ``min_i l_ij`` and -- unlike that direct
    time -- stays valid when the travel-time matrix is non-metric (a two-arc
    path can be faster than the direct arc on real road networks).  Using the
    direct time as the bound would wrongly force t_j up and, in the MTZ model
    where t_j enters the objective, inflate the latency.
    """
    e = {j: min(l[i][j] for i in I) for j in J}       # depot -> j direct (route entry)
    for _ in range(len(J)):                            # Bellman-Ford relaxation
        changed = False
        for u in J:
            du = e[u]
            for v in J:
                if u != v and du + l[u][v] < e[v] - 1e-9:
                    e[v] = du + l[u][v]
                    changed = True
        if not changed:
            break
    return e


def _add_spec_strengthenings(mdl, I, J, x, z):
    """Spec section 3.6 valid inequalities NOT already in the base models:

      * arc-assignment linking:  x^i_uv <= z_iu ,  x^i_uv <= z_iv  (customer arcs);
        x^i_ij <= z_ij ,  x^i_ji <= z_ij  (depot arcs).
      * customer 2-cycle elimination:  sum_i (x^i_uv + x^i_vu) <= 1  for u < v.

    Both tighten the LP relaxation without changing the optimum.
    """
    from collections import defaultdict
    Jset = set(J)
    for i in I:
        for (u, v), var in x[i].items():
            if u in Jset and v in Jset:            # customer -> customer
                mdl.addConstr(var <= z[i, u], f"lnk[{i},{u},{v}]a")
                mdl.addConstr(var <= z[i, v], f"lnk[{i},{u},{v}]b")
            elif u not in Jset and v in Jset:       # depot i -> customer v
                mdl.addConstr(var <= z[i, v], f"lnkd[{i},{v}]")
            elif u in Jset and v not in Jset:       # customer u -> depot i
                mdl.addConstr(var <= z[i, u], f"lnkr[{i},{u}]")
    pair = defaultdict(list)
    for i in I:
        for (u, v), var in x[i].items():
            if u in Jset and v in Jset:
                pair[(u, v) if u < v else (v, u)].append(var)
    for (a, b), vlist in pair.items():
        mdl.addConstr(gp.quicksum(vlist) <= 1, f"twocyc[{a},{b}]")


def build_model(
    inst: LLRPInstance,
    name: str = "LLRP",
    *,
    aggregate_mtz: bool = True,      # paper 3.3: aggregate MTZ over the depot index
                                     #   (set False only to reproduce the old per-depot rows)
    prune_arcs: bool = False,        # T1.2: drop capacity/duration-infeasible arcs
    lifted_mtz: bool = False,        # T1.3a: Desrochers-Laporte lifted load MTZ
    time_windows: bool = False,      # T1.3b: earliest/latest arrival bounds + tight M
    valid_inequalities: bool = False,  # T1.4: fleet-floor and open-depot cuts
    strengthen: bool = False,        # spec 3.6: x<=z linking + 2-cycle elimination
) -> gp.Model:
    """Build the LLRP MILP.

    The constraints match Section 3.3 of the revised paper: the load/time MTZ
    propagation (P:mtzload, P:DL1) is aggregated over the depot index by default
    (``aggregate_mtz=True``), and the arrival-time (DL2) and return-trip (DL4)
    rows are written once per pharmacy, aggregated over depots and big-M free.
    The remaining keyword flags default to False (baseline); each is *exactness
    preserving* -- turning any subset on yields the same optimal objective, only
    faster.  ``solve(..., tier1=True)`` enables all five speed flags.  Set
    ``aggregate_mtz=False`` to reproduce the old per-depot MTZ rows for comparison.
    """
    I, J = inst.I, inst.J
    c, l = inst.c, inst.l
    d, p = inst.demand, inst.priority
    Q, L = inst.Q, inst.L
    W, f, F = inst.W, inst.f, inst.F
    Ng, Nv, alpha = inst.Ng, inst.Nv, inst.alpha

    mdl = gp.Model(name)

    # --- T1.3b: earliest / latest arrival windows ------------------------ #
    # e_j = fastest any route could reach j (direct from nearest depot);
    # L_j = latest j may be reached and still return within L.
    if time_windows:
        e = _arrival_lb(I, J, l)                   # valid arrival LB (non-metric safe)
        Lj = {j: L - min(l[j][i] for i in I) for j in J}
    else:
        e = {j: 0.0 for j in J}
        Lj = {j: L for j in J}

    # --- arc pruning predicates (T1.2) ----------------------------------- #
    def pharm_arc_ok(i, u, v):
        if not prune_arcs:
            return True
        if d[u] + d[v] > Q + 1e-9:                       # capacity: can't be consecutive
            return False
        if l[i][u] + l[u][v] + l[v][i] > L + 1e-9:       # duration: min route i->u->v->i
            return False
        return True

    def depot_arc_ok(i, j):
        if not prune_arcs:
            return True
        return l[i][j] + l[j][i] <= L + 1e-9             # min single-customer route i->j->i

    # depot i can serve j at all only if the round trip i->j->i fits in L
    def can_serve(i, j):
        return (not prune_arcs) or (l[i][j] + l[j][i] <= L + 1e-9)

    # --- arc sets per depot ---------------------------------------------- #
    # A_i : arcs usable by depot i's route  = {i}->J, J->{i}, J->J (u != v)
    arcs_i = {i: [] for i in I}
    for i in I:
        for j in J:
            if depot_arc_ok(i, j):
                arcs_i[i].append((i, j))                 # depot -> pharmacy
                arcs_i[i].append((j, i))                 # pharmacy -> depot
        for u in J:
            for v in J:
                if u != v and pharm_arc_ok(i, u, v):
                    arcs_i[i].append((u, v))

    # --- decision variables ---------------------------------------------- #
    y = mdl.addVars(I, vtype=GRB.BINARY, name="y")                     # (17)
    z = mdl.addVars(I, J, vtype=GRB.BINARY, name="z")                  # (18)
    x = {i: mdl.addVars(arcs_i[i], vtype=GRB.BINARY, name=f"x_{i}")    # (19)
         for i in I}
    k = mdl.addVars(I, vtype=GRB.INTEGER, lb=0, name="k")              # (20)
    omega = mdl.addVars(J, lb=0.0, name="omega")                       # load (21)
    t = mdl.addVars(J, lb=0.0, name="t")                              # time (21)

    # tighten variable bounds from time windows (T1.3b)
    for j in J:
        t[j].LB, t[j].UB = e[j], Lj[j]
        omega[j].LB, omega[j].UB = d[j], Q                            # (12) as bounds

    # fix z_ij = 0 where depot i provably cannot serve j (T1.2)
    if prune_arcs:
        for i in I:
            for j in J:
                if not can_serve(i, j):
                    z[i, j].UB = 0.0

    # --- objective (1) --------------------------------------------------- #
    open_cost = gp.quicksum(f[i] * y[i] for i in I)
    travel_cost = gp.quicksum(c[u][v] * x[i][u, v] for i in I for (u, v) in arcs_i[i])
    fleet_cost = gp.quicksum(F[i] * k[i] for i in I)
    latency = gp.quicksum(p[j] * t[j] for j in J)
    mdl.setObjective(open_cost + travel_cost + fleet_cost + alpha * latency, GRB.MINIMIZE)

    # --- assignment / location (2)-(4) ----------------------------------- #
    mdl.addConstrs((z.sum("*", j) == 1 for j in J), "assign")                       # (2)
    mdl.addConstrs((gp.quicksum(d[j] * z[i, j] for j in J) <= W[i] * y[i]           # (3)
                    for i in I), "depot_cap")
    mdl.addConstrs((z[i, j] <= y[i] for i in I for j in J), "open_link")            # (4)

    # --- degree constraints (5)-(6) -------------------------------------- #
    for i in I:
        for j in J:
            in_arcs = [x[i][u, j] for u in ([i] + J) if u != j and (u, j) in x[i]]
            out_arcs = [x[i][j, v] for v in ([i] + J) if v != j and (j, v) in x[i]]
            mdl.addConstr(gp.quicksum(in_arcs) == z[i, j], f"indeg[{i},{j}]")        # (5)
            mdl.addConstr(gp.quicksum(out_arcs) == z[i, j], f"outdeg[{i},{j}]")      # (6)

    # --- depot balance + vehicle counting (7)-(9)+fixes ------------------ #
    mdl.addConstrs((gp.quicksum(x[i][i, v] for v in J if (i, v) in x[i]) == k[i]     # (7)
                    for i in I), "depart")
    mdl.addConstrs((gp.quicksum(x[i][u, i] for u in J if (u, i) in x[i]) == k[i]     # (8)
                    for i in I), "return")
    mdl.addConstrs((k[i] <= Nv * y[i] for i in I), "veh_open")                       # (9) open-link
    mdl.addConstr(gp.quicksum(k[i] for i in I) <= Nv, "fleet_global")                # (9) GLOBAL Nv
    mdl.addConstr(gp.quicksum(y[i] for i in I) == Ng, "num_open")                    # (Ng) NEW

    # --- aggregate vehicle capacity (10) --------------------------------- #
    mdl.addConstrs((gp.quicksum(d[j] * z[i, j] for j in J) <= Q * k[i] for i in I),  # (10)
                   "veh_cap_agg")
    # (P:vehopen_vl) each OPEN depot dispatches at least one vehicle. This is a
    # model constraint (not a valid inequality), so it is always added.
    mdl.addConstrs((k[i] >= y[i] for i in I), "route_if_open")

    # --- T1.4: valid inequalities (strengthen the root bound) ------------ #
    if valid_inequalities:
        import math
        mdl.addConstr(gp.quicksum(k[i] for i in I) >= math.ceil(inst.total_demand / Q),
                      "fleet_floor")                     # sum k_i >= ceil(D/Q)
        # sum k_i >= Ng is implied by the always-on route_if_open rows; kept only
        # as a (redundant) aggregate cut.
        mdl.addConstr(gp.quicksum(k[i] for i in I) >= Ng, "min_routes")

    # helper: which depots may use pharmacy arc (u,v), given pruning
    def depots_with(u, v):
        return [i for i in I if (u, v) in x[i]]

    # --- MTZ load: capacity + subtour elimination (11)-(12) -------------- #
    # (12) already imposed via omega bounds above.
    pharm_arcs = [(u, v) for u in J for v in J if u != v]
    for (u, v) in pharm_arcs:
        owners = depots_with(u, v)
        if not owners:
            continue
        if aggregate_mtz:
            # single row using X_uv = sum_i x^i_uv  (<= 1, since each arc has one owner)
            Xuv = gp.quicksum(x[i][u, v] for i in owners)
            if lifted_mtz:
                # Desrochers-Laporte lift: add reverse-arc term when (v,u) exists
                rev = depots_with(v, u)
                Xvu = gp.quicksum(x[i][v, u] for i in rev) if rev else 0
                mdl.addConstr(
                    omega[v] - omega[u] >= d[v] - Q * (1 - Xuv) + (Q - d[u] - d[v]) * Xvu,
                    f"mtz_load[{u},{v}]")
            else:
                mdl.addConstr(omega[v] - omega[u] >= d[v] - Q * (1 - Xuv),
                              f"mtz_load[{u},{v}]")
        else:
            for i in owners:                 # per-depot rows (aggregate_mtz=False only)
                mdl.addConstr(omega[v] - omega[u] >= d[v] - Q * (1 - x[i][u, v]),
                              f"mtz_load[{i},{u},{v}]")

    # --- MTZ time / route-length (DL1)-(DL4) ----------------------------- #
    for (u, v) in pharm_arcs:
        owners = depots_with(u, v)
        if not owners:
            continue
        # tight per-arc big-M from the arrival windows: M = l_uv + L_u - e_v (>= 0)
        M = max(0.0, l[u][v] + Lj[u] - e[v]) if time_windows else L
        if aggregate_mtz:
            Xuv = gp.quicksum(x[i][u, v] for i in owners)
            mdl.addConstr(t[v] - t[u] >= l[u][v] - M * (1 - Xuv), f"dl1[{u},{v}]")    # (DL1)
        else:
            for i in owners:
                mdl.addConstr(t[v] - t[u] >= l[u][v] - M * (1 - x[i][u, v]),
                              f"dl1[{i},{u},{v}]")
    # (DL2) initialise the first-stop arrival, aggregated over depots (paper 3.3):
    #   t_j >= sum_i l_ij x^i_ij     (at most one depot->j arc is active)
    mdl.addConstrs((t[j] >= gp.quicksum(l[i][j] * x[i][i, j]
                                        for i in I if (i, j) in x[i])
                    for j in J), "dl2")
    # (DL3) t_j <= L folded into t[j].UB above (= L_j <= L)
    # (DL4) return-trip duration limit, aggregated over depots (paper 3.3), big-M free:
    #   t_j + sum_i l_ji x^i_ji <= L   (at most one j->depot arc is active, so if j is
    #   a route's last stop this gives t_j + l_ji <= L, else it reduces to t_j <= L)
    mdl.addConstrs((t[j] + gp.quicksum(l[j][i] * x[i][j, i]
                                       for i in I if (j, i) in x[i]) <= L
                    for j in J), "dl4")

    # --- spec 3.6 strengthenings (linking + 2-cycle) --------------------- #
    if strengthen:
        _add_spec_strengthenings(mdl, I, J, x, z)

    mdl._inst = inst
    mdl._x = x
    mdl._y = y
    mdl._z = z
    mdl._k = k
    mdl._omega = omega
    mdl._t = t
    return mdl


def build_model_flow(
    inst: LLRPInstance,
    name: str = "LLRP-flow",
    *,
    prune_arcs: bool = False,
    time_windows: bool = False,
    valid_inequalities: bool = False,
    strengthen: bool = False,
    lexicographic: str | None = None,
    latency_obj: bool = True,
    enforce_duration: bool = True,
    tchebycheff: dict | None = None,
) -> gp.Model:
    """Option A -- commodity-flow formulation (exactness preserving).

    Same optimum as :func:`build_model`, but the two big-M MTZ chains are
    replaced by tight, polynomial commodity flows:

      * LOAD  : single-commodity flow  g_uv  (Gavish-Graves).  Enforces vehicle
        capacity and eliminates subtours -- no big-M.  Replaces eq. (11)-(12).
      * LATENCY: a downstream-priority flow  r_uv  = total priority of customers
        still to be served from arc (u,v) onward.  It linearises the objective
        exactly:  sum_j p_j t_j = sum_arcs l_uv r_uv  -- no big-M.  Replaces the
        t_j term of the objective.

    The route-DURATION limit L is still carried by a (time-window tightened)
    arrival-time chain t_j, which is now a pure feasibility constraint and no
    longer couples to the objective.
    """
    import math
    I, J = inst.I, inst.J
    c, l = inst.c, inst.l
    d, p = inst.demand, inst.priority
    Q, L = inst.Q, inst.L
    W, f, F = inst.W, inst.f, inst.F
    Ng, Nv, alpha = inst.Ng, inst.Nv, inst.alpha
    Jset = set(J)
    P = sum(p[j] for j in J)                 # total priority mass (upper bnd on r)

    mdl = gp.Model(name)

    # arrival windows (for the L feasibility chain)
    if time_windows:
        e = _arrival_lb(I, J, l)                   # valid arrival LB (non-metric safe)
        Lj = {j: L - min(l[j][i] for i in I) for j in J}
    else:
        e = {j: 0.0 for j in J}
        Lj = {j: L for j in J}

    def depot_arc_ok(i, j):
        return (not prune_arcs) or (l[i][j] + l[j][i] <= L + 1e-9)

    def pharm_arc_ok(i, u, v):
        if not prune_arcs:
            return True
        if d[u] + d[v] > Q + 1e-9:
            return False
        if l[i][u] + l[u][v] + l[v][i] > L + 1e-9:
            return False
        return True

    # --- arc sets (routing variables identical to the MTZ model) --------- #
    arcs_i = {i: [] for i in I}
    for i in I:
        for j in J:
            if depot_arc_ok(i, j):
                arcs_i[i].append((i, j))
                arcs_i[i].append((j, i))
        for u in J:
            for v in J:
                if u != v and pharm_arc_ok(i, u, v):
                    arcs_i[i].append((u, v))

    y = mdl.addVars(I, vtype=GRB.BINARY, name="y")
    z = mdl.addVars(I, J, vtype=GRB.BINARY, name="z")
    x = {i: mdl.addVars(arcs_i[i], vtype=GRB.BINARY, name=f"x_{i}") for i in I}
    k = mdl.addVars(I, vtype=GRB.INTEGER, lb=0, name="k")
    if enforce_duration:
        t = mdl.addVars(J, lb=0.0, name="t")
        for j in J:
            t[j].LB, t[j].UB = e[j], Lj[j]
    else:
        t = None                     # route-duration limit skipped (verify post-hoc)
    if prune_arcs:
        for i in I:
            for j in J:
                if not depot_arc_ok(i, j):
                    z[i, j].UB = 0.0

    # aggregate owners of each pharmacy arc, and present depot arcs
    owners = {}
    for i in I:
        for (u, v) in arcs_i[i]:
            if u in Jset and v in Jset:
                owners.setdefault((u, v), []).append(i)
    PA = list(owners.keys())                                   # pharmacy arcs
    dep = [(i, j) for i in I for j in J if (i, j) in x[i]]      # depot -> pharmacy
    Xuv = {a: gp.quicksum(x[i][a] for i in owners[a]) for a in PA}

    # --- commodity-flow variables ---------------------------------------- #
    gdep = mdl.addVars(dep, lb=0.0, ub=Q, name="gdep")   # load leaving a depot
    g = mdl.addVars(PA, lb=0.0, ub=Q, name="g")          # load on pharmacy arcs
    if latency_obj:
        rdep = mdl.addVars(dep, lb=0.0, ub=P, name="rdep")   # downstream priority
        r = mdl.addVars(PA, lb=0.0, ub=P, name="r")
    else:
        rdep = r = None                 # latency not in the objective -> no r-flow

    # --- objective (1): latency uses the r-flow, not t ------------------- #
    open_cost = gp.quicksum(f[i] * y[i] for i in I)
    travel_cost = gp.quicksum(c[u][v] * x[i][u, v] for i in I for (u, v) in arcs_i[i])
    fleet_cost = gp.quicksum(F[i] * k[i] for i in I)
    financial = open_cost + travel_cost + fleet_cost
    if not latency_obj:
        # pure-cost model (minimise financial only): a standard capacitated LRP,
        # solved exactly by Gurobi branch-and-cut. No latency flow; duration limit
        # optional (enforce_duration=False -> verify L on the routes afterwards).
        latency = None
        mdl.setObjective(financial, GRB.MINIMIZE)
    else:
        latency = (gp.quicksum(l[i][j] * rdep[i, j] for (i, j) in dep)
                   + gp.quicksum(l[u][v] * r[u, v] for (u, v) in PA))
        if tchebycheff is not None:
            # Normalised augmented weighted Tchebycheff (paper 3.4): minimise
            # Z = max((1-lam)Cbar, lam Pbar) + rho[(1-lam)Cbar + lam Pbar],
            # Cbar=(financial-Cmin)/dC, Pbar=(latency-Pmin)/dP.  Linearised with a
            # free zeta >= each epigraph term.
            lam = float(tchebycheff["lam"]); rho = float(tchebycheff["rho"])
            Cmin = float(tchebycheff["Cmin"]); Pmin = float(tchebycheff["Pmin"])
            dC = float(tchebycheff["dC"]); dP = float(tchebycheff["dP"])
            Cbar = (financial - Cmin) / dC
            Pbar = (latency - Pmin) / dP
            zeta = mdl.addVar(lb=-GRB.INFINITY, name="zeta")
            mdl.addConstr(zeta >= (1.0 - lam) * Cbar, "epi_cost")
            mdl.addConstr(zeta >= lam * Pbar, "epi_lat")
            mdl.setObjective(zeta + rho * ((1.0 - lam) * Cbar + lam * Pbar),
                             GRB.MINIMIZE)
        elif lexicographic in ("cost", "latency"):
            # Hierarchical (lexicographic) objective: optimise the primary first,
            # then the secondary among all primary-optimal solutions. With
            # ObjNRelTol/AbsTol = 0 (default) the primary stays exactly optimal.
            # "cost"    -> min financial, then min latency (tie-break)
            # "latency" -> min latency,   then min financial (tie-break)
            mdl.ModelSense = GRB.MINIMIZE
            primary, secondary = (financial, latency) if lexicographic == "cost" \
                else (latency, financial)
            mdl.setObjectiveN(primary, index=0, priority=2,
                              name=("financial" if lexicographic == "cost" else "latency"))
            mdl.setObjectiveN(secondary, index=1, priority=1,
                              name=("latency" if lexicographic == "cost" else "financial"))
        else:
            mdl.setObjective(financial + alpha * latency, GRB.MINIMIZE)

    # --- assignment / location (2)-(4) ----------------------------------- #
    mdl.addConstrs((z.sum("*", j) == 1 for j in J), "assign")
    mdl.addConstrs((gp.quicksum(d[j] * z[i, j] for j in J) <= W[i] * y[i] for i in I),
                   "depot_cap")
    mdl.addConstrs((z[i, j] <= y[i] for i in I for j in J), "open_link")

    # --- degree (5)-(6) -------------------------------------------------- #
    for i in I:
        for j in J:
            in_arcs = [x[i][u, j] for u in ([i] + J) if u != j and (u, j) in x[i]]
            out_arcs = [x[i][j, v] for v in ([i] + J) if v != j and (j, v) in x[i]]
            mdl.addConstr(gp.quicksum(in_arcs) == z[i, j], f"indeg[{i},{j}]")
            mdl.addConstr(gp.quicksum(out_arcs) == z[i, j], f"outdeg[{i},{j}]")

    # --- depot balance + fleet (7)-(9)+fixes ----------------------------- #
    mdl.addConstrs((gp.quicksum(x[i][i, v] for v in J if (i, v) in x[i]) == k[i]
                    for i in I), "depart")
    mdl.addConstrs((gp.quicksum(x[i][u, i] for u in J if (u, i) in x[i]) == k[i]
                    for i in I), "return")
    mdl.addConstrs((k[i] <= Nv * y[i] for i in I), "veh_open")
    mdl.addConstr(gp.quicksum(k[i] for i in I) <= Nv, "fleet_global")
    mdl.addConstr(gp.quicksum(y[i] for i in I) == Ng, "num_open")
    mdl.addConstrs((gp.quicksum(d[j] * z[i, j] for j in J) <= Q * k[i] for i in I),
                   "veh_cap_agg")
    # (P:vehopen_vl) each OPEN depot dispatches at least one vehicle -- a model
    # constraint, always added (not a valid inequality).
    mdl.addConstrs((k[i] >= y[i] for i in I), "route_if_open")

    # --- LOAD flow: capacity + subtour elimination (replaces 11-12) ------ #
    # conservation: (inflow) - (outflow) = d_v   at every pharmacy v
    for v in J:
        inflow = (gp.quicksum(gdep[i, v] for i in I if (i, v) in gdep)
                  + gp.quicksum(g[u, v] for u in J if (u, v) in g))
        outflow = gp.quicksum(g[v, w] for w in J if (v, w) in g)
        mdl.addConstr(inflow - outflow == d[v], f"load_cons[{v}]")
    # coupling  d_v x <= flow <= Q x   (upper = capacity, lower = tightening)
    for (i, j) in dep:
        mdl.addConstr(gdep[i, j] <= Q * x[i][i, j], f"gdep_ub[{i},{j}]")
        mdl.addConstr(gdep[i, j] >= d[j] * x[i][i, j], f"gdep_lb[{i},{j}]")
    for a in PA:
        u, v = a
        mdl.addConstr(g[a] <= Q * Xuv[a], f"g_ub[{u},{v}]")
        mdl.addConstr(g[a] >= d[v] * Xuv[a], f"g_lb[{u},{v}]")

    # --- LATENCY flow: exact sum_j p_j t_j (replaces the t_j objective) --- #
    # conservation: (inflow) - (outflow) = p_v   (each node drops its priority)
    if latency_obj:
        for v in J:
            inflow = (gp.quicksum(rdep[i, v] for i in I if (i, v) in rdep)
                      + gp.quicksum(r[u, v] for u in J if (u, v) in r))
            outflow = gp.quicksum(r[v, w] for w in J if (v, w) in r)
            mdl.addConstr(inflow - outflow == p[v], f"lat_cons[{v}]")
        for (i, j) in dep:
            mdl.addConstr(rdep[i, j] <= P * x[i][i, j], f"rdep_ub[{i},{j}]")
            mdl.addConstr(rdep[i, j] >= p[j] * x[i][i, j], f"rdep_lb[{i},{j}]")
        for a in PA:
            u, v = a
            mdl.addConstr(r[a] <= P * Xuv[a], f"r_ub[{u},{v}]")
            mdl.addConstr(r[a] >= p[v] * Xuv[a], f"r_lb[{u},{v}]")

    # --- route-DURATION limit L (feasibility-only arrival chain) ---------- #
    if enforce_duration:
        for a in PA:
            u, v = a
            M = max(0.0, l[u][v] + Lj[u] - e[v]) if time_windows else L
            mdl.addConstr(t[v] - t[u] >= l[u][v] - M * (1 - Xuv[a]), f"dl1[{u},{v}]")   # (DL1)
        # (DL2) first-stop arrival, aggregated over depots (paper 3.3):
        mdl.addConstrs((t[j] >= gp.quicksum(l[i][j] * x[i][i, j]
                                            for i in I if (i, j) in x[i])
                        for j in J), "dl2")                                              # (DL2)
        # (DL4) return-trip duration limit, aggregated over depots (paper 3.3), big-M free:
        mdl.addConstrs((t[j] + gp.quicksum(l[j][i] * x[i][j, i]
                                           for i in I if (j, i) in x[i]) <= L
                        for j in J), "dl4")                                              # (DL4)

    # --- valid inequalities (T1.4) --------------------------------------- #
    if valid_inequalities:
        mdl.addConstr(gp.quicksum(k[i] for i in I) >= math.ceil(inst.total_demand / Q),
                      "fleet_floor")
        # route_if_open (k_i >= y_i) is now always added above (model constraint).
        mdl.addConstr(gp.quicksum(k[i] for i in I) >= Ng, "min_routes")

    # --- spec 3.6 strengthenings (linking + 2-cycle) --------------------- #
    if strengthen:
        _add_spec_strengthenings(mdl, I, J, x, z)

    mdl._inst = inst
    mdl._x = x
    mdl._y = y
    mdl._z = z
    mdl._k = k
    mdl._t = t
    mdl._g = g
    mdl._r = r
    return mdl


def _make_rci_callback(inst, x, per_node_cuts=15, total_cuts=600,
                       max_calls=4000, thresholds=(0.08, 0.3, 0.6)):
    """Build a Gurobi callback that separates Rounded Capacity Inequalities.

    For any customer set S the routes must enter it at least ceil(d(S)/Q) times:

        sum_{arcs (u,v): u not in S, v in S}  x_uv  >=  ceil( d(S) / Q ).

    Exact separation is NP-hard, so we use the standard connected-component
    heuristic on the fractional support: at several edge-weight thresholds we
    grow customer components and test each for a violated RCI.  Cuts are added
    with cbCut (they only tighten the LP -- every integer-feasible point already
    satisfies them, so the optimum is unchanged).
    """
    import math
    from collections import defaultdict

    I, J = inst.I, inst.J
    d, Q = inst.demand, inst.Q
    Jset = set(J)

    # precompute, once: pharmacy-arc owners and depot->customer arcs
    pharm_vars = []                       # [((u,v), var), ...]  u,v both customers
    dep_into = defaultdict(list)          # j -> [var(depot i -> j)]
    dep_flat = []                         # [(j, var), ...]  flattened depot->j arcs
    for i in I:
        for (u, v), var in x[i].items():
            if u in Jset and v in Jset:
                pharm_vars.append(((u, v), var))
            elif u not in Jset and v in Jset:
                dep_into[v].append(var)
                dep_flat.append((v, var))
    pharm_var_objs = [var for (_, var) in pharm_vars]
    dep_flat_objs = [var for (_, var) in dep_flat]

    def callback(model, where):
        if where != GRB.Callback.MIPNODE:
            return
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
            return
        model._rci_calls = getattr(model, "_rci_calls", 0) + 1
        if model._rci_calls > max_calls or getattr(model, "_rci_total", 0) >= total_cuts:
            return

        # aggregate fractional arc usage over depots
        pv = model.cbGetNodeRel(pharm_var_objs)
        agg = defaultdict(float)          # (u,v) -> sum_i x*_uv
        for ((uv, _), val) in zip(pharm_vars, pv):
            if val > 1e-7:
                agg[uv] += val
        dep_val = defaultdict(float)      # j -> sum_i x*(depot i -> j)
        for (j, _), val in zip(dep_flat, model.cbGetNodeRel(dep_flat_objs)):
            dep_val[j] += val

        # undirected edge weights for component growing
        uw = defaultdict(float)
        for (u, v), val in agg.items():
            uw[(u, v) if u < v else (v, u)] += val

        seen, added = set(), 0
        for thr in thresholds:
            parent = {j: j for j in J}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for (a, b), val in uw.items():
                if val > thr:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
            comps = defaultdict(list)
            for j in J:
                comps[find(j)].append(j)

            for S in comps.values():
                if not (2 <= len(S) < len(J)):
                    continue
                key = frozenset(S)
                if key in seen:
                    continue
                seen.add(key)
                rhs = math.ceil(sum(d[j] for j in S) / Q - 1e-9)
                if rhs < 2:                       # rhs=1 is implied by assignment
                    continue
                Sset = set(S)
                lhs = sum(dep_val[j] for j in S) + sum(
                    val for (u, v), val in agg.items() if v in Sset and u not in Sset)
                if lhs < rhs - 1e-4:              # violated -> add the cut
                    expr = gp.LinExpr()
                    for j in S:
                        for var in dep_into.get(j, []):
                            expr += var
                    for (u, v), var in pharm_vars:
                        if v in Sset and u not in Sset:
                            expr += var
                    model.cbCut(expr >= rhs)
                    model._rci_total = getattr(model, "_rci_total", 0) + 1
                    added += 1
                    if added >= per_node_cuts or model._rci_total >= total_cuts:
                        return

    return callback


def solve(inst: LLRPInstance, *, time_limit=None, mip_gap=None,
          threads=None, verbose=True, log_file=None,
          tier1: bool = False, warm_start: bool = False, tuned: bool = False,
          formulation: str = "mtz", build_flags: dict | None = None,
          gurobi_params: dict | None = None, rci: bool = False,
          strengthen: bool = False) -> dict:
    """Build, solve, and return a structured solution dictionary.

    Speed flags (all default off -> exact baseline):
      tier1       : enable all five exactness-preserving model reformulations.
      warm_start  : seed a nearest-neighbour heuristic incumbent (Tier 2).
      tuned       : apply feasibility-focused Gurobi parameters (Tier 2).
      formulation : "mtz" (default, build_model) or "flow" (Option A,
                    build_model_flow -- commodity-flow load + latency).
    ``build_flags``  overrides individual Tier-1 flags (e.g. {"prune_arcs": True}).
    ``gurobi_params`` sets raw Gurobi parameters LAST, overriding ``tuned``
    (e.g. {"MIPFocus": 3, "Cuts": 3} to focus on the lower bound).
    """
    flags = dict(aggregate_mtz=tier1, prune_arcs=tier1, lifted_mtz=tier1,
                 time_windows=tier1, valid_inequalities=tier1, strengthen=strengthen)
    if build_flags:
        flags.update(build_flags)

    if formulation == "flow":
        # the flow model subsumes aggregate/lifted MTZ; it only reads these
        fflags = {kk: flags[kk] for kk in ("prune_arcs", "time_windows",
                                           "valid_inequalities", "strengthen")}
        mdl = build_model_flow(inst, **fflags)
    elif formulation == "mtz":
        mdl = build_model(inst, **flags)
    else:
        raise ValueError(f"Unknown formulation '{formulation}' (use 'mtz' or 'flow').")
    if time_limit is not None:
        mdl.Params.TimeLimit = time_limit
    if mip_gap is not None:
        mdl.Params.MIPGap = mip_gap
    if threads is not None:
        mdl.Params.Threads = threads
    if log_file is not None:
        mdl.Params.LogFile = log_file
    mdl.Params.OutputFlag = 1 if verbose else 0

    # --- Tier 2: feasibility-focused parameters -------------------------- #
    if tuned:
        mdl.Params.MIPFocus = 1          # prioritise finding good incumbents
        mdl.Params.Heuristics = 0.3
        mdl.Params.Cuts = 2
        mdl.Params.Presolve = 2
        # Gurobi's no-relaxation heuristic is strong at finding first feasibles
        nrh = 20.0 if time_limit is None else max(5.0, min(30.0, 0.2 * time_limit))
        mdl.Params.NoRelHeurTime = nrh

    # --- Tier 2: warm start ---------------------------------------------- #
    if warm_start:
        ok = _apply_warm_start(mdl, inst, flags, verbose=verbose)
        if verbose:
            print(f" -> warm start {'applied' if ok else 'skipped (no feasible heuristic)'}")

    # --- Option 2: Rounded Capacity Inequality separation ---------------- #
    cb = None
    if rci:
        mdl.Params.PreCrush = 1          # required so user cuts survive presolve
        mdl._rci_total = 0
        cb = _make_rci_callback(inst, mdl._x)

    # --- raw parameter overrides (applied last, win over `tuned`) -------- #
    if gurobi_params:
        for key, val in gurobi_params.items():
            setattr(mdl.Params, key, val)

    mdl.optimize(cb) if cb is not None else mdl.optimize()
    if rci and verbose:
        print(f" -> RCI cuts added: {getattr(mdl, '_rci_total', 0)}")

    result = {
        "city": inst.city,
        "status": mdl.Status,
        "status_str": _status_name(mdl.Status),
        "objective": None,
        "mip_gap": None,
        "runtime_s": mdl.Runtime,
        "open_depots": [],
        "routes": [],
    }
    if mdl.SolCount == 0:
        if verbose:
            print(f"No feasible solution found (status={result['status_str']}).")
        return result

    result["objective"] = mdl.ObjVal
    try:
        result["mip_gap"] = mdl.MIPGap
    except gp.GurobiError:
        pass
    result["open_depots"] = _extract_open_depots(mdl, inst)
    result["routes"] = _extract_routes(mdl, inst)
    result["breakdown"] = _objective_breakdown(mdl, inst, result["routes"])
    if verbose:
        _print_solution(result, inst)
    return result


# --------------------------------------------------------------------------- #
# Tier 2: constructive warm-start heuristic                                   #
# --------------------------------------------------------------------------- #
def heuristic_solution(inst: LLRPInstance):
    """Cluster-first / route-second nearest-neighbour construction.

    Returns (open_depots, routes) with routes = {depot_i: [ [j,...], ... ]}, or
    ``None`` if it cannot build a feasible solution (tight instances may fail;
    the caller then simply skips the warm start).
    """
    I, J = inst.I, inst.J
    l, d = inst.l, inst.demand
    Q, L, W, Ng, Nv = inst.Q, inst.L, inst.W, inst.Ng, inst.Nv

    # 1. open the Ng depots with the smallest mean travel time to all pharmacies
    score = {i: sum(l[i][j] for j in J) / len(J) for i in I}
    open_depots = sorted(I, key=lambda i: score[i])[:Ng]

    # 2. assign each pharmacy to the nearest open depot that (a) can serve it
    #    within L and (b) still has residual W capacity
    residual = {i: W[i] for i in open_depots}
    assign = {i: [] for i in open_depots}
    for j in sorted(J, key=lambda j: -d[j]):             # largest demand first
        cands = sorted(open_depots, key=lambda i: l[i][j])
        placed = False
        for i in cands:
            if l[i][j] + l[j][i] <= L + 1e-9 and residual[i] >= d[j] - 1e-9:
                assign[i].append(j)
                residual[i] -= d[j]
                placed = True
                break
        if not placed:
            return None                                  # cannot place -> give up

    # 3. within each depot, nearest-neighbour routing under Q and L
    routes = {i: [] for i in open_depots}
    total_routes = 0
    for i in open_depots:
        pending = set(assign[i])
        while pending:
            route, load, cur, tcur = [], 0.0, i, 0.0
            while True:
                feas = [
                    j for j in pending
                    if load + d[j] <= Q + 1e-9
                    and tcur + l[cur][j] + l[j][i] <= L + 1e-9
                ]
                if not feas:
                    break
                nxt = min(feas, key=lambda j: l[cur][j])
                route.append(nxt)
                load += d[nxt]
                tcur += l[cur][nxt]
                pending.discard(nxt)
                cur = nxt
            if not route:                                # a lone customer didn't fit
                return None
            routes[i].append(route)
            total_routes += 1
            if total_routes > Nv:
                return None                              # exceeds global fleet -> infeasible
    return open_depots, routes


def _apply_warm_start(mdl, inst, flags, verbose=True):
    sol = heuristic_solution(inst)
    if sol is None:
        return False
    open_depots, routes = sol
    x, y, z, k = mdl._x, mdl._y, mdl._z, mdl._k

    # start from an all-zero integer assignment, then switch on what we use
    for i in inst.I:
        y[i].Start = 1.0 if i in open_depots else 0.0
        k[i].Start = len(routes.get(i, []))
        for j in inst.J:
            z[i, j].Start = 0.0
        for arc in x[i]:
            x[i][arc].Start = 0.0

    for i in open_depots:
        for route in routes[i]:
            nodes = [i] + route + [i]
            for j in route:
                z[i, j].Start = 1.0
            for u, v in zip(nodes[:-1], nodes[1:]):
                if (u, v) not in x[i]:
                    return False                         # arc pruned away -> abort start
                x[i][(u, v)].Start = 1.0
    return True


# --------------------------------------------------------------------------- #
# Solution extraction / reporting                                             #
# --------------------------------------------------------------------------- #
def _status_name(code):
    return {getattr(GRB.status, a): a for a in dir(GRB.status)
            if not a.startswith("_")}.get(code, str(code))


def _extract_open_depots(mdl, inst):
    return [i for i in inst.I if mdl._y[i].X > 0.5]


def _objective_breakdown(mdl, inst, routes):
    """Split the objective (eq. 1) into its four terms, read from the solution.

    financial = depot-opening + arc-travel + fleet cost
    latency   = sum_j p_j t_j  (raw, unweighted) -- computed from the TRUE route
                arrival times, so it is correct for both the MTZ and flow models.
    The reported objective equals  financial + alpha * latency.
    """
    open_cost = sum(inst.f[i] * mdl._y[i].X for i in inst.I)
    travel_cost = sum(inst.c[u][v] * var.X
                      for i in inst.I for (u, v), var in mdl._x[i].items())
    fleet_cost = sum(inst.F[i] * mdl._k[i].X for i in inst.I)
    financial = open_cost + travel_cost + fleet_cost
    latency = sum(inst.priority[j] * arr
                  for r in routes for j, arr in r["arrivals"].items())
    return {
        "alpha": inst.alpha,
        "depot_opening_cost": open_cost,
        "arc_travel_cost": travel_cost,
        "fleet_cost": fleet_cost,
        "financial_cost": financial,        # sum of the three cost terms
        "latency_raw": latency,             # sum_j p_j t_j  (unweighted)
        "latency_weighted": inst.alpha * latency,
        "objective": financial + inst.alpha * latency,
    }


def _extract_routes(mdl, inst):
    """Reconstruct each vehicle path depot -> ... -> depot from the x arcs."""
    routes = []
    for i in inst.I:
        if mdl._y[i].X <= 0.5:
            continue
        # successor map on this depot's used arcs
        succ = {}
        starts = []
        for (u, v), var in mdl._x[i].items():
            if var.X > 0.5:
                succ.setdefault(u, []).append(v)
                if u == i:
                    starts.append(v)
        for first in starts:
            path = [first]
            cur = first
            while cur != i:
                nxts = succ.get(cur, [])
                nxt = next((w for w in nxts if w == i), nxts[0] if nxts else i)
                if nxt == i:
                    break
                path.append(nxt)
                cur = nxt
                if len(path) > inst.n + 1:      # safety against cycles
                    break
            load = sum(inst.demand[j] for j in path)
            # true arrival times along the route (independent of any t_j slack)
            arrivals, cum, prev = {}, 0.0, i
            for node in path:
                cum += inst.l[prev][node]
                arrivals[node] = cum
                prev = node
            routes.append({
                "depot": i,
                "depot_label": inst.depot_labels[i],
                "path_nodes": path,
                "path_codes": [inst.pharmacy_codes[j - inst.m] for j in path],
                "load": load,
                "arrivals": arrivals,
                "max_arrival_s": cum,          # arrival at the last stop
            })
    return routes


def _print_solution(result, inst):
    print("\n" + "=" * 64)
    print(f"  {result['city']}  |  status={result['status_str']}  |  "
          f"obj={result['objective']:.2f}  |  gap={result['mip_gap']}")
    print("=" * 64)
    b = result.get("breakdown")
    if b:
        print("Objective breakdown (eq. 1):")
        print(f"    depot-opening cost   f_i y_i     = {b['depot_opening_cost']:12.2f}")
        print(f"    arc-travel cost      c_uv x_uv^i = {b['arc_travel_cost']:12.2f}")
        print(f"    fleet cost           F_i k_i     = {b['fleet_cost']:12.2f}")
        print(f"    --------------------------------------------------")
        print(f"    FINANCIAL COST (subtotal)        = {b['financial_cost']:12.2f}")
        print(f"    latency  sum_j p_j t_j (raw)     = {b['latency_raw']:12.2f}")
        print(f"    latency  x alpha={b['alpha']:<6g}         = {b['latency_weighted']:12.2f}")
        print(f"    ==================================================")
        print(f"    TOTAL OBJECTIVE                  = {b['objective']:12.2f}")
    print("Open depots:", [inst.depot_labels[i] for i in result["open_depots"]])
    for r in result["routes"]:
        print(f"  {r['depot_label']}: load={r['load']:.0f}/{inst.Q:g}  "
              f"maxArrival={r['max_arrival_s']:.0f}s  "
              f"stops={len(r['path_nodes'])}  -> {r['path_nodes']}")


if __name__ == "__main__":
    import os
    import sys
    from llrp_loader import load_instance

    city = sys.argv[1] if len(sys.argv) > 1 else "Elazig"
    base = os.path.dirname(os.path.abspath(__file__))
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    inst = load_instance(city, base, alpha=alpha)
    solve(inst, time_limit=300, mip_gap=0.01, verbose=True)
