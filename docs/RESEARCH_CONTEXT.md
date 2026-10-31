# Research context and precise model definition

Based on the 26-page manuscript accompanying this package (not inferred from the older six-city studies).

**Title:** A Bi-objective Priority-Weighted Latency Location-Routing Problem for Pharmaceutical Distribution.

**Decision variables:** select exactly `Ng` depots from `m` candidates, assign each pharmacy to a selected depot, construct at most `Nv` nonempty vehicle routes and order deliveries, subject to route, capacity and fleet constraints. Each open depot must dispatch at least one route, each pharmacy is visited once, each route returns to its originating depot, all travel-time data are directed.

**Cost objective:** `CF = sum_i(f_i*y_i) + sum_routes(sum_arcs(cost_arc)) + sum_i(F_i*k_i)`; in the four city studies, `f_i=5000`, `F_i=200` and travel cost is distance in km. The studied operational parameters force all available vehicles into use, but the algorithm itself is more general.

**Priority-weighted latency:** `P = sum_j p_j*a_j`, where `a_j` is cumulative travel duration from depot to pharmacy `j` in **seconds**. A route's return to depot does not add to `P`, but is counted in route duration `<=L`.

**Normalization:** `Cnorm=(CF-Cmin)/(C_at_Pmin-Cmin)` and `Pnorm=(P-Pmin)/(P_at_Cmin-Pmin)`. Reference values come from best heuristic anchors for the heuristic experiment; the exact anchors are only used for external validation.

**Scalarization:** `Z=max((1-lambda)*Cnorm,lambda*Pnorm)+rho*((1-lambda)*Cnorm+lambda*Pnorm)`, `rho=1e-4`. The two anchor modes optimize only cost or latency; `lambda in (0,1)` is the interior preference. Heuristically found solutions may be dominated and should be screened when plotting an approximate Pareto front.

**Algorithms:** ECWA savings-based randomized construction (RCL 5), 50 GRASP iterations, 5 diversification + 3 intensification; depot replacement, relocate, swap, 2-opt/2-opt*, 2-relocate, node-arc swap, arc-arc swap, Swap*; strategic oscillation permits temporary violations of **vehicle** capacity only, never depot capacity or route duration. 30 independent runs per setting; numerical libraries single-threaded per process and independent runs parallelized across CPU processes.

**Profile definitions:** Equal `p=1`; Tiered `p in {1.0,0.7,0.4}` by seeded probabilities 0.2,0.4,0.4; Demand-Correlated `p=0.5+0.5*(d-dmin)/(dmax-dmin)`; Negative-Correlated `p=1-0.5*(d-dmin)/(dmax-dmin)`; Hyper-Critical two pharmacies with `p=10`, all others `p=1`. The source index-map workbooks store the actual realized coefficients/seeds.

**Data construction:** shift demand `d_j = ceil(annual_sales_j / (300*5))`. Distances and durations are stored as separate Google-road-network-derived directed workbooks. Candidate depots are existing pharmacy sites used as **proxies** for depot locations, not necessarily existing warehouses. See paper Sections 3--5 and Appendices E--L for complete experimental/methodological details.
