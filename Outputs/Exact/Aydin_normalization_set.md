# Aydin normalization set (for the interior Tchebycheff rerun)
Chat_min: certified optimal (single-objective min C_F flow, gap 0). Phat_min: best VERIFIED of B&P and heuristic (30 seeds). Chat_P = cost of that chosen Phat_min solution. Phat_C = latency of the cost-optimal routing under each profile (first-stage evaluation, upper bound). rho=1e-4.

| Profile | Chat_min | Phat_min | source | Phat_C | Chat_P | Delta_C | Delta_P |
|---|---|---|---|---|---|---|---|
| Equal | 16741.297 | 95809.000 | B&P | 150827.000 | 16922.477 | 181.180 | 55018.000 |
| Tiered | 16741.297 | 52436.900 | B&P | 79697.900 | 16930.136 | 188.839 | 27261.000 |
| Demand_Correlated | 16741.297 | 58487.148 | B&P | 91590.963 | 16922.477 | 181.180 | 33103.815 |
| Negative_Correlated | 16741.297 | 84221.130 | heuristic | 134649.537 | 16849.745 | 108.448 | 50428.407 |
| Hyper_Critical | 16741.297 | 140092.000 | B&P | 194378.000 | 16981.612 | 240.315 | 54286.000 |
