# Canakkale exact anchors (regime split: flow=cost, B&P=latency)
Cost anchor (Cmin,P_C): flow lexicographic cost (proven). Latency anchor: B&P -- Phat_min (stage1) and Chat_P=minC s.t. P<=Phat_min (stage2, budget row), depot pairs in parallel w/ shared incumbent, 2h/problem. Flow is weak on latency for n=37 (lex-latency time-limited at 17100 vs B&P's proven 16511), hence B&P here.

## Equal
anchors: Cmin=11031.218 Pmin=16511.0 P_C=20505.0 C_P=11062.137 dC=30.919 dP=3994.0

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin,P_C (lex cost) | flow | C=11031.218,P=20505.0 | — | None | 925.5 | OK |
| Phat_min | B&P | 16511.0 | 16511.0 | 0.000% | 1287.8 | OK |
| Chat_P (minC|P<=Pmin) | B&P | 11062.137 | 11062.1 | 0.000% | 11.6 | OK |

## Tiered
anchors: Cmin=11031.218 Pmin=10158.7 P_C=12513.3 C_P=11041.097 dC=9.879 dP=2354.6

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin,P_C (lex cost) | flow | C=11031.218,P=12513.3 | — | None | 1444.0 | OK |
| Phat_min | B&P | 10158.7 | 10158.7 | 0.000% | 1371.0 | OK |
| Chat_P (minC|P<=Pmin) | B&P | 11041.097 | 11041.1 | 0.000% | 66.9 | OK |

## Demand_Correlated
anchors: Cmin=11031.218 Pmin=11083.575 P_C=13265.675 C_P=11064.79 dC=33.572 dP=2182.1

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin,P_C (lex cost) | flow | C=11031.218,P=13265.675 | — | None | 1183.4 | OK |
| Phat_min | B&P | 11083.575 | 11083.6 | 0.000% | 4396.0 | OK |
| Chat_P (minC|P<=Pmin) | B&P | 11064.79 | 11064.8 | 0.000% | 4639.8 | OK |

## Negative_Correlated
anchors: Cmin=11031.218 Pmin=13635.25 P_C=17491.825 C_P=11055.721000000001 dC=24.503 dP=3856.575

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin,P_C (lex cost) | flow | C=11031.218,P=17491.825 | — | None | 1448.2 | OK |
| Phat_min | B&P | 13635.25 | 13635.2 | 0.000% | 891.6 | OK |
| Chat_P (minC|P<=Pmin) | B&P | 11055.721000000001 | 11055.7 | 0.000% | 8.7 | OK |

## Hyper_Critical
anchors: Cmin=11031.218 Pmin=17251.0 P_C=28236.0 C_P=11062.996000000001 dC=31.778 dP=10985.0

| problem | solver | UB | LB | gap | time(s) | verify |
|---|---|---|---|---|---|---|
| Cmin,P_C (lex cost) | flow | C=11031.218,P=28236.0 | — | None | 1800.3 | OK |
| Phat_min | B&P | 17251.0 | 17251.0 | 0.000% | 24622.5 | OK |
| Chat_P (minC|P<=Pmin) | B&P | 11062.996000000001 | 11063.0 | 0.000% | 8.8 | OK |

## Extra: flow Tchebycheff MILP, Equal, lambda=0.5
gap=98.5497% (TIME_LIMIT, 1800.0s) -- flow weak on latency at n=37.
