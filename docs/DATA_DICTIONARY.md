# Data dictionary and directory provenance

| Location | Contents | Purpose |
|---|---|---|
| `Inputs/{City}_customers.csv` | `id` (`C001..`), `demand` (integer vehicle-load), and five priority-profile columns | Customer nodes, demands and priority weights |
| `Inputs/{City}_depots.csv` | `depot_id` (`D01..`), `at_customer` (co-located node), `capacity` (W) | Candidate depots |
| `Inputs/{City}_distance.csv` | `n × n` id-labelled financial travel cost (`cost_per_km = 1` ⇒ km) | Financial travel cost |
| `Inputs/{City}_duration.csv` | `n × n` id-labelled travel times in seconds | Pharmacy arrival times and route-duration checks |
| `Inputs/{City}_params.csv` | `Ng, Nv, Q, L, fixed_depot_cost, fixed_vehicle_cost, cost_per_km` | Operational parameters |
| `Outputs/{City}[_Profile]_Tcheby_Anchor_Validation_parallel.xlsx` | `Summary`, `Payoff_Table`, `Anchor_Routes`, `Run_Level` | 30-run financial and latency anchor modes, reference pairs and routes |
| `Outputs/Tcheby_Production/{City}_{Profile}_Tcheby_Production.xlsx` | `Summary`, `Best_By_Lambda`, `Run_Level` | 30 runs per non-endpoint lambda and best obtained point |
| `Outputs/Tcheby_Production/*_Curve.csv` | CSV version of the per-lambda preference-path table | Compact machine-readable Pareto analysis |
| `Outputs/Routing_Cost_Analysis/{City}/*` | Profile-specific/aggregated routing travel-cost decompositions and solution routes | Service-priority and travel-cost comparisons |
| `benchmarks/instances/dat` | Original Barreto benchmark .dat files, transported byte-for-byte | Benchmark instance inputs |
| `benchmarks/results/extended` and `/baseline` | 10 instance-specific merged run workbooks each | Benchmark 30-run records and method comparison |
| `benchmarks/ablation` | Code/summary for additional search configurations | Table 4 and Appendix I |
| `docs/current_manuscript.pdf` | User-supplied current 26-page manuscript | Authoritative definition of reported claims |

**De-identification:** these inputs are the anonymized clean CSVs — no coordinates, no real identifiers, no sales. Customers/depots use synthetic labels (`C001..`/`D01..`) and demand is given directly. The stored result files (`Outputs/`) were relabelled with the same mapping, so every reported number is unchanged. The distance/duration matrices derive from a third-party routing service and remain subject to that provider's terms.

Every included artifact has a SHA-256 hash in `MANIFEST_SHA256.csv`.
