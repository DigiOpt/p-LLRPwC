# Exact method (financial & latency anchors, branch-and-price)

This folder adds the **exact** optimisation code used for the single-objective
anchors and the exact validation reported in the manuscript (Table 6/7). The
GRASP-VND heuristic in `code/` does **not** depend on any of this; only the exact
part needs **Gurobi** (a free academic licence works).

## Modules

| File | Role |
|---|---|
| `llrp_loader.py` | reads a city instance from the de-identified clean CSVs in `Inputs/` (`<City>_customers.csv`, `_depots.csv`, `_distance.csv`, `_duration.csv`, `_params.csv`); exposes depots, demands, priorities, cost/time matrices, and the parameters Ng, Nv, Q, L, W. No coordinates or real identifiers are used; demand is given directly. |
| `llrp_model.py` | compact MILPs: an MTZ model and a tight **commodity-flow** model (`build_model_flow`), with exactness-preserving options (arc pruning, time-window bounds, valid inequalities, x≤z / 2-cycle strengthening) and `lexicographic=` / `tchebycheff=` objectives. |
| `llrp_pricer_nb.py` | Numba-compiled ng-route pricer (forward ESPPRC labelling and a bidirectional cost pricer). |
| `llrp_bpc.py` | column-generation lower bound (ng-routes + subset-row cuts) and `exact_solve` (flow UB + best valid LB by depot-set enumeration). |
| `llrp_bnp.py` | **branch-and-price** (`branch_and_price`): arc-branching best-first search over a fixed depot set, with the weighted, ε-constraint (`budget`) and augmented-Tchebycheff objectives. |
| `cost_flow.py` | lean branch-and-cut financial-cost solver that persists incumbent routes. |
| `experiment.py` | CLI for the single-objective exact LB/UB per city (`python experiment.py --city Afyon`). |
| `run_exact_anchors.py` | **main entry** — computes the four lexicographic anchors per profile (regime split: flow for the cost anchors, branch-and-price for the latency anchors), verifies every solution from its routes, and writes `Outputs/Exact/<City>_exact_anchors.{json,csv,md}`. |

## Requirements

```bash
python -m pip install -r requirements.txt   # includes gurobipy
```

A Gurobi licence must be discoverable (e.g. `~/gurobi.lic` or `$GRB_LICENSE_FILE`);
no licence credentials are embedded in this code.

## Run the exact anchors (from the repository root)

```bash
# all five profiles of one city
python code/exact/run_exact_anchors.py --city Afyon

# selected profiles, custom caps and parallelism
python code/exact/run_exact_anchors.py --city Canakkale --profiles Equal,Tiered \
    --cost-limit 14400 --bnp-cap 7200 --workers 4
```

Anchors produced (recomputed from the extracted routes, not read from the solver):

```
Chat_min = min C_F            Phat_C = min P  s.t. C_F = Chat_min   (cost anchor)
Phat_min = min P              Chat_P = min C_F s.t. P   = Phat_min   (latency anchor)
```

`run_exact_anchors.py` solves the cost anchor by the commodity-flow MILP
(lexicographic) and the latency anchors by branch-and-price over the C(m, Ng)
depot combinations, run **in parallel with a shared global incumbent**. The
per-combination B&P time cap (`--bnp-cap`) is **soft** (checked between
column-generation rounds), so a single hard combination can overshoot it; the
per-profile wall-clock is the maximum over combinations, not their sum.

## Verification

Before any solution is saved it is re-checked from its routes alone: every
pharmacy visited exactly once; each route within Q and L; each open depot within
its throughput W; number of routes ≤ Nv; exactly Ng open depots; and C_F and
P = Σ_j p_j a_j recomputed to match the reported anchor.

## Stored exact outputs

`Outputs/Exact/` holds the exact anchors, full route sets and per-route details
for the manuscript cities, plus the Aydın normalisation set and the independently
**verified** heuristic anchor route sets used for the best-known comparison. See
`docs/EXACT_ANCHORS_STATEMENT.md` for the certification status of each value and
the method notes.
