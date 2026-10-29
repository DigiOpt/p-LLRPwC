# Bi-objective Priority-Weighted Latency Location-Routing (Pharmaceutical Distribution)

**Reproducibility archive for:** *A Bi-objective Priority-Weighted Latency Location-Routing Problem for Pharmaceutical Distribution* — Alperen Bal, Michele Cedolin, Ethem Çanakoğlu. Code and data match the **four-city** manuscript.

> **De-identified public version.** The instances have been fully anonymized for sharing: there are **no geographic coordinates, no real pharmacy/customer identifiers and no commercial sales**. Customers are labelled `C001..` and depots `D01..`; demand is provided directly (as integer vehicle-loads), and the distance/duration matrices are given as plain id-labelled CSVs. The travel matrices were derived from a third-party routing service and remain subject to that provider's terms; redistribute accordingly. See `LICENSE_NOTICE.md`.

## Research question

The study optimizes integrated depot selection and vehicle routes for pharmacy replenishment while balancing financial distribution cost (`CF`) against pharmacy-priority-weighted customer arrival time (`P`). It uses **GRASP-VND-SO** (Extended Clarke–Wright savings, seven VND route neighborhoods, depot replacement, temporary vehicle-capacity oscillation) and anchor-normalized augmented weighted **Tchebycheff** scalarization (`rho = 1e-4`) to explore supported and unsupported trade-offs.

## What is in this repository?

```text
.
├── README.md                       this guide
├── LICENSE_NOTICE.md              redistribution restrictions / licensing status
├── CITATION.cff                    bibliographic metadata (no DOI invented)
├── requirements.txt               reproducibility-oriented dependencies
├── code/
│   ├── LLRP_CORE_PARALLEL_HPC.py   main research heuristic and data loaders
│   ├── RUN_TCHEBY_ANCHORS_HPC_PARALLEL.py
│   ├── RUN_TCHEBY_ANCHORS_PROFILE_HPC_PARALLEL.py
│   ├── RUN_TCHEBY_PRODUCTION_PARALLEL.py
│   ├── legacy_reference/          earlier code snapshot (not the default)
│   └── exact/                     EXACT method (Gurobi): flow MILP, branch-and-price,
│       ├── run_exact_anchors.py     main entry — four lexicographic anchors per profile
│       ├── llrp_model.py            MTZ + commodity-flow MILPs (lexicographic / Tchebycheff)
│       ├── llrp_bnp.py              branch-and-price (arc-branching, budget/Tchebycheff)
│       ├── llrp_bpc.py              CG lower bound + exact_solve (flow UB + valid LB)
│       ├── llrp_pricer_nb.py        Numba ng-route pricer
│       ├── llrp_loader.py           instance loader (reads the clean CSVs in Inputs/)
│       ├── cost_flow.py             lean financial-cost branch-and-cut solver
│       ├── experiment.py            single-objective exact LB/UB CLI
│       └── README.md                exact-method guide
├── Inputs/                         4 cities × 5 clean CSVs (customers, depots,
│                                    distance, duration, params) — de-identified
├── Outputs/
│   ├── *_Anchor_Validation_parallel.xlsx       4 × 5 stored heuristic anchors
│   ├── Tcheby_Production/           20 stored profile workbooks + 20 curves
│   ├── Routing_Cost_Analysis/       route/cost breakdown by city and priority
│   └── Exact/                       exact anchors + full route sets (Table 6/7),
│                                    Aydın normalisation set, verified heuristic anchors
├── benchmarks/
│   ├── instances/                   Barreto .dat files and source workbooks
│   ├── code/                        benchmark-only VND variants
│   ├── results/{baseline,extended,summary}/
│   └── ablation/                    four-configuration audit and supporting output
├── tools/
│   ├── check_package.py            file/data/run-count audit (fast)
│   └── rebuild_tables.py           derive portable CSV summaries (fast)
├── derived_tables/                 output of rebuild_tables.py
├── docs/                           model/experiment/data/audit notes + PDF
│   └── EXACT_ANCHORS_STATEMENT.md  exact-anchor method + per-value certification status
└── MANIFEST_SHA256.csv             per-file integrity hashes
```

## Input data format (de-identified clean CSVs)

Each city `<City>` is described by five flat CSV files in `Inputs/`. Both the
heuristic (`code/LLRP_CORE_PARALLEL_HPC.py`) and the exact method
(`code/exact/llrp_loader.py`) read this layout directly; no coordinates are used.

| File | Columns | Meaning |
|---|---|---|
| `<City>_customers.csv` | `id`, `demand`, `priority_equal`, `priority_tiered`, `priority_demand_correlated`, `priority_negative_correlated`, `priority_hyper_critical` | one row per pharmacy (`C001..`); `demand` is the integer vehicle-load; the five priority columns are the policy-weight profiles. |
| `<City>_depots.csv` | `depot_id`, `at_customer`, `capacity` | candidate depots (`D01..`), each co-located with a customer node (`at_customer`), with fixed capacity `W`. |
| `<City>_distance.csv` | `id`, then one column per customer id | `n × n` financial travel cost per arc (`cost_per_km = 1` ⇒ cost equals km). |
| `<City>_duration.csv` | `id`, then one column per customer id | `n × n` travel time per arc, in **seconds** (used for the route-duration limit `L` and for latency). |
| `<City>_params.csv` | `parameter`, `value` | scalars: `Ng`, `Nv`, `Q`, `L`, `fixed_depot_cost`, `fixed_vehicle_cost`, `cost_per_km`. |

Depots are a subset of customer nodes, so the loaders build the full `(m+n) × (m+n)`
node matrix (depots first) from the customer-only matrices. To run a new instance,
drop in the same five files under a `<City>` prefix.

## Setup (recommended Python 3.9–3.12)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/check_package.py
python tools/verify_manifest.py
python tools/rebuild_tables.py
```

The audit and summary commands inspect the already stored results **without rerunning the optimization**. `derived_tables/` contains machine-readable checks and normalized comparisons to support tables and figures. These fast scripts are useful on a laptop.

> **Gurobi (exact method only).** The heuristic above runs with the `requirements.txt`
> packages alone. The exact code in `code/exact/` additionally needs **Gurobi**
> (`gurobipy`, installed by `requirements.txt`) and a discoverable licence
> (`~/gurobi.lic` or `$GRB_LICENSE_FILE`; free for academics). No licence
> credentials are stored in this repository.

## Rerun cost and latency anchor modes (CPU intensive)

Run all examples from the **repository root**. The original research scripts expect `Inputs/` and `Outputs/` relative to the current working directory and intentionally preserve those conventions. The scripts may write over existing output files; run them on a copy or with backup files when retaining original experimental results.

```bash
# Windows PowerShell (set environment variables with $env:CITY / $env:PROFILE)
# Linux/macOS:
CITY=Afyon WORKERS=4 python code/RUN_TCHEBY_ANCHORS_HPC_PARALLEL.py
CITY=Afyon PROFILE=Tiered WORKERS=4 python code/RUN_TCHEBY_ANCHORS_PROFILE_HPC_PARALLEL.py
```

The common Equal-Priority anchor filename is `Outputs/{City}_Tcheby_Anchor_Validation_parallel.xlsx`; named-profile anchors are stored as `Outputs/{City}_{Profile}_Tcheby_Anchor_Validation_parallel.xlsx`. These are read by the production runner to normalize financial cost and priority-weighted latency.

## Rerun an interior Tchebycheff preference set (CPU intensive)

After producing/retaining both anchors for the intended city/profile:

```bash
CITY=Afyon PROFILE=Equal-Priority MAX_ITER=50 N_SEEDS=30 WORKERS=16 RHO=0.0001 \
  python code/RUN_TCHEBY_PRODUCTION_PARALLEL.py
```

Use `CITY=Isparta` for the largest city. Only the five published interior preferences (`0.1,0.3,0.5,0.7,0.9`) are executed for Isparta in the **publication-aligned runner supplied here**; the other three cities use nine values (`0.1` through `0.9`). **Traceability notice:** the original production runner obtained from Drive used nine interior values for every city, inconsistent with the five Isparta values in the stored output workbooks. We added an explicit Isparta branch to the main runner while preserving the original unmodified source under `code/legacy_reference/RUN_TCHEBY_PRODUCTION_PARALLEL_original.py`. See `docs/REPRODUCIBILITY_LIMITS.md`. A quick smoke run is possible with `MAX_ITER=1 N_SEEDS=1 WORKERS=1`, but it is **not** the published experimental design. The production runner expects compatible exact filenames for anchor workbooks and exports `Best_By_Lambda`, `Run_Level` and curves under `Outputs/Tcheby_Production/`.

## Exact method: financial & latency anchors (Table 6/7)

The `code/exact/` folder contains the exact optimisation code (commodity-flow MILP
+ branch-and-price) used for the single-objective anchors and the exact validation.
**This part requires Gurobi** (a free academic licence is sufficient; install with
`pip install -r requirements.txt`, which now includes `gurobipy`). The GRASP-VND
heuristic above does **not** need Gurobi.

```bash
# all five profiles of a city (regime split: flow for cost anchors, B&P for latency)
python code/exact/run_exact_anchors.py --city Afyon
# selected profiles, custom time caps and parallel depot combinations
python code/exact/run_exact_anchors.py --city Canakkale --profiles Equal,Tiered \
    --cost-limit 14400 --bnp-cap 7200 --workers 4
```

For each profile this produces the four lexicographic anchors — Ĉ_min, P̂_C
(cost anchor, by the flow MILP) and P̂_min, Ĉ_P (latency anchor, by branch-and-price
over the depot combinations with a shared global incumbent) — all **recomputed and
verified from the extracted routes**, written to
`Outputs/Exact/<City>_exact_anchors.{json,csv,md}`. The per-combination B&P cap is
**soft** (checked between column-generation rounds); the per-profile wall-clock is
the maximum over combinations, not their sum. The single-objective exact LB/UB per
city is also available via `python code/exact/experiment.py --city <City>`.

Stored exact outputs for the manuscript cities are in `Outputs/Exact/` (anchors,
full route sets with per-stop arrival times, the Aydın normalisation set, and the
independently verified heuristic anchor routes used for the best-known comparison).
`docs/EXACT_ANCHORS_STATEMENT.md` records the certification status of every value
(certified optimal vs. tight upper bound vs. heuristic best-known) and the method.

## How to locate the results in the paper

| Manuscript result | Source of record in this archive |
|---|---|
| Table 2 and Table 3: 10 benchmark cases | `benchmarks/results/extended/` (10 merged 30-run workbooks), `benchmarks/results/summary/VNDZou.xlsx` |
| Table 4, Appendix I: component ablation | `benchmarks/ablation/analysis/ablation_instance.csv` and `ablation_summary.csv`; method code in `benchmarks/code/` |
| Table 5: four-city characteristics | `Inputs/` and `docs/EXPERIMENT_DESIGN.md` |
| Table 6/7: exact validation | `code/exact/` (flow MILP + branch-and-price) and `Outputs/Exact/` (anchors, routes, Aydın normalisation set); status per value in `docs/EXACT_ANCHORS_STATEMENT.md` |
| Table 8 and Appendix K: cost–latency paths | `Outputs/Routing_Cost_Analysis/{City}/*All_Profiles*` and `Outputs/Tcheby_Production/` |
| Priority profiles, depot switching and robustness | Individual `Outputs/*Anchor*`, `Outputs/Tcheby_Production/*`, routing workbooks and 30-run benchmark records |

### Interpretation traps

- Financial cost includes `5000*Ng + 200*k` in the case studies; travel cost is only the distance-dependent portion (`CF - fixed charges`).
- Return-to-depot time **counts** toward route duration but **does not count** toward latency.
- Priorities are **scenario policy weights**, not clinically validated urgent-care classes.
- Trade-off points may be duplicate or heuristic-dominated. Exclude dominated points when drawing empirical Pareto fronts and distinguish exact/heuristic anchors.
- Aydin has a heuristic cost anchor that the manuscript reports is not globally optimal; normalized cost for some solutions can be negative.
- Fixed stochastic seeds are in stored worksheets and the original scripts; full stochastic bit-for-bit reproducibility depends on execution environment.

## Scope, attribution and limitations

The package contains the **heuristic** source code (`code/`) **and the exact method**
(`code/exact/`: commodity-flow MILP, branch-and-price, lower-bound column generation
and the exact-anchor driver), manuscript-aligned input files, stored heuristic and
exact run outputs (`Outputs/`, `Outputs/Exact/`), benchmark source data and
descriptive analyses. The exact part requires a Gurobi licence (none is embedded in
the code). Formal problem formulations remain in `docs/current_manuscript.pdf`;
the exact-anchor certification status and method notes are in
`docs/EXACT_ANCHORS_STATEMENT.md`. For reproducibility boundaries see
`docs/REPRODUCIBILITY_LIMITS.md`.

**Source provenance:** the instances derive from a real four-city pharmaceutical
distribution study; for this public archive they were **de-identified** — coordinates,
real pharmacy identifiers and commercial sales were removed, customers/depots were
relabelled (`C001..`/`D01..`), and demand (integer vehicle-loads) is stored directly
in place of sales. The stored result files were relabelled with the same mapping, so
every reported number is unchanged. A SHA-256 manifest enumerates included bytes.

## Verification notes

An automated GitHub Actions workflow (`.github/workflows/validate.yml`) checks the source/data inventory, verifies the frozen SHA-256 manifest, and tests two mathematical invariants. It does **not** rerun the expensive GRASP experiments or assert exact optimality. Derived CSV summaries already included in `derived_tables/` can be regenerated with `python tools/rebuild_tables.py`.
