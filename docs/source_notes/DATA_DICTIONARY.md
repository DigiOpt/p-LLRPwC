# Data Dictionary and Provenance

> **Note.** This file documents the **original private research archive** for provenance
> and methodology (how demand was derived, how outputs were structured). It is **not** a
> description of the files shipped in this repository. The published repository contains
> **de-identified** inputs only — the clean CSVs described in the top-level `README.md`
> and `docs/DATA_DICTIONARY.md` (no coordinates, no real identifiers, no sales; demand
> given directly). The raw workbooks named below (`*_index_map.xlsx`, `Case_Study_*`)
> are **not** included here.

## 1. Case_Study_Inputs (original private archive — not shipped)


For each of the six case-study cities (Afyon, Aydin, Canakkale, Isparta, Elazig, Erzurum), three input workbooks are archived.


### <City>_index_map.xlsx
Contains the pharmacy/location-level fields used in instance construction. Depending on the city workbook, fields include:
- pharmacy/customer identifier;
- warehouse/candidate-location identifier where applicable;
- latitude and longitude;
- annual sales / Total_Sales;
- priority information where stored;
- candidate-depot or clustering information in supporting sheets.


Shift-level demand used in the computational instances is derived from the stored sales field using:
    d_j = ceil(Sales_j / (300 * 5))
where 300 is the assumed number of working days per year and 5 is the assumed number of replenishment shifts per working day.


### <City>_distance_matrix_Google.xlsx
Road-network distance matrix used for transportation-cost calculations and location/routing construction.


### <City>_duration_matrix_Google.xlsx
Travel-duration matrix used for latency and route-duration calculations.


## 2. Case_Study_Run_Outputs


The six *_Bulk_FULL_W10.xlsx files are the final stored outputs used for the revised manuscript analysis.


The scenario-definition section includes:
- Experiment_Block;
- Depot_Capacity_Level;
- Vehicle_Capacity_Level;
- Depot_Cap_W;
- Vehicle_Cap_Q;
- Alpha_Weight;
- Priority_Profile;
- Scenario_ID / Scenario_Name;
- Structural_Feasible / Structural_Check.


The run-level section includes:
- City;
- Scenario_ID / Scenario;
- Experiment_Block;
- capacity levels and numerical W/Q;
- Run and Seed;
- Priority_Profile;
- Alpha_Weight;
- Is_Feasible;
- Structural_Feasible;
- Failure_Reason;
- coverage diagnostics;
- Total_Cost;
- Financial_Cost;
- Equity_Penalty;
- number of open depots;
- number of routes;
- CPU_Time_s.


The iteration-history section includes:
- instance/scenario;
- run and seed;
- iteration;
- diversification/intensification mode;
- current and incumbent objective information;
- feasibility and route/depot diagnostics.
These histories support the T=50 convergence analysis.


## 3. Benchmark_Run_Outputs


### VNDZou.xlsx
Stored 30-run results for the extended benchmark configuration, including:
- BKS;
- best latency;
- BKS gap;
- mean latency;
- standard deviation;
- mean CPU time;
- best run/seed;
- depot and route information.


### VND0.xlsx
Stored 30-run results for the baseline benchmark configuration with the same reporting fields needed for comparison.


## 4. Derived statistical variables


Financial cost index:
    FI = 100 * C_F / C_F,baseline
where the baseline is the city-specific moderate-W, moderate-Q, Equal-Priority, alpha=0 configuration.


Priority-weighted latency reduction:
    100 * (P(0) - P(alpha)) / P(0)
within the same city/profile reference.


Financial premium:
    100 * (C_F(alpha) - C_F(0)) / C_F(0)
within the same city/profile reference.


Route increment:
    Routes(alpha) - Routes(0).


Standardized depot-capacity ratio:
    R_W = N_g * W / D_c.


Standardized fleet-capacity ratio:
    R_Q = N_v * Q / D_c.


## 5. Structural-feasibility screening
For a city with N_g required open depots and at most N_v routes, the aggregate serviceable demand is evaluated using:
    U(W,Q) = max sum_i min(W, k_i Q)
subject to k_i >= 1 integer and sum_i k_i <= N_v.
The aggregate pre-check requires d_max <= Q and D_c <= U(W,Q). Passing this check is necessary but not sufficient for full route feasibility.


## 6. Confidentiality
The original `Case_Study_Inputs` workbooks contained detailed operational/geospatial
information (coordinates, pharmacy identifiers, commercial sales) and are **not**
distributed in this repository. The published archive ships only the de-identified
clean CSVs: coordinates, real identifiers and sales have been removed, customers/depots
are relabelled (`C001..`/`D01..`), and demand is stored directly as integer
vehicle-loads. The stored result files were relabelled with the same mapping, leaving
every reported number unchanged.