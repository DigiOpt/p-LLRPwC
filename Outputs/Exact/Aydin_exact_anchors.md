# Aydin exact anchors (single 4h cost solve + B&P latency)
Cost solve ONCE (profile-independent Cmin): TIME_LIMIT gap=None 14400.4s. P_C per profile = priority-weighted latency of the shared cost-optimal routes under that profile. Latency anchor: B&P Phat_min (parallel pairs, shared incumbent, 2h cap); Chat_P = cost of that latency-optimal solution. Equal reused from aydin_equal_done.json.

## Equal
anchors: Cmin=16741.297 Pmin=95809.0 P_C=150827.0 C_P=16922.477 dC=181.18 dP=55018.0

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin (shared) | flow | 16741.297 | 16741.297 | None | 14400.4 | OK |
| P_C=lat(R*) | derived | 150827.0 | - | - | - | OK |
| Phat_min | B&P | 95809.0 | 95671.7 | 0.143% | 7201.5 | OK |
| Chat_P=cost(Phat_min sol) | B&P | 16922.477 | 16922.477 | 0.000% | - | OK |

## Tiered
anchors: Cmin=16741.297 Pmin=52436.899999999994 P_C=79697.9 C_P=16930.136 dC=188.839 dP=27261.0

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin (shared) | flow | 16741.297 | 16741.297 | None | 14400.4 | OK |
| P_C=lat(R*) | derived | 79697.9 | - | - | - | OK |
| Phat_min | B&P | 52436.899999999994 | 52436.9 | 0.000% | 4124.3 | OK |
| Chat_P=cost(Phat_min sol) | B&P | 16930.136 | 16930.136 | 0.000% | - | OK |

## Demand_Correlated
anchors: Cmin=16741.297 Pmin=58487.148148148146 P_C=91590.962963 C_P=16922.477 dC=181.18 dP=33103.814815

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin (shared) | flow | 16741.297 | 16741.297 | None | 14400.4 | OK |
| P_C=lat(R*) | derived | 91590.962963 | - | - | - | OK |
| Phat_min | B&P | 58487.148148148146 | 58487.1 | 0.000% | 1717.3 | OK |
| Chat_P=cost(Phat_min sol) | B&P | 16922.477 | 16922.477 | 0.000% | - | OK |

## Negative_Correlated
anchors: Cmin=16741.297 Pmin=84313.37037037036 P_C=134649.537037 C_P=16849.667 dC=108.37 dP=50336.166667

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin (shared) | flow | 16741.297 | 16741.297 | None | 14400.4 | OK |
| P_C=lat(R*) | derived | 134649.537037 | - | - | - | OK |
| Phat_min | B&P | 84313.37037037036 | 84058.1 | 0.303% | 7347.4 | OK |
| Chat_P=cost(Phat_min sol) | B&P | 16849.667 | 16849.667 | 0.000% | - | OK |

## Hyper_Critical
anchors: Cmin=16741.297 Pmin=140092.0 P_C=194378.0 C_P=16981.612 dC=240.315 dP=54286.0

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin (shared) | flow | 16741.297 | 16741.297 | None | 14400.4 | OK |
| P_C=lat(R*) | derived | 194378.0 | - | - | - | OK |
| Phat_min | B&P | 140092.0 | 138823.8 | 0.905% | 7201.7 | OK |
| Chat_P=cost(Phat_min sol) | B&P | 16981.612 | 16981.612 | 0.000% | - | OK |

## Extra: flow Tchebycheff MILP, Equal, lambda=0.5
gap=229.0496% (TIME_LIMIT, 1800.0s)
