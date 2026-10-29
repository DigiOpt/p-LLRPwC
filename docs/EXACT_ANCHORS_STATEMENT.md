# Exact Anchor Computation for the Bi-objective LLRP — Statement for Co-authors

**Purpose.** This note documents how the single-objective lexicographic anchor
values (used to normalise the augmented weighted Tchebycheff scalarisation) were
computed exactly for the case-study instances, the certification status of each
value, and one methodological point that affects how we should present the
anchors in the paper.

---

## 1. Anchors and what they certify

For each instance and priority profile we need the four lexicographic anchors

- **Ĉ_min** = min C_F  (cheapest financial cost);
- **P̂_C**  = min P among cost-optimal solutions  (latency at the cost optimum);
- **P̂_min** = min P  (fastest priority-weighted latency);
- **Ĉ_P**  = min C_F among latency-optimal solutions  (cost at the latency optimum),

with ranges Δ_C = Ĉ_P − Ĉ_min and Δ_P = P̂_C − P̂_min. (Ĉ_min, P̂_min) is the
ideal point and (Ĉ_P, P̂_C) the nadir point. P̂_C and Ĉ_P are **lexicographic
second-stage** quantities (a tie-break optimisation over the optimal face of the
first objective). C_F and P are always **recomputed from the extracted routes**
(P = Σ_j p_j a_j, with a_j the arrival time), never read from the solver's
objective.

## 2. Solution method (regime split)

Two exact engines are used, each on the objective where it is strong:

- **Financial / cost side → commodity-flow MILP** (`build_model_flow`, with arc
  pruning, time-window bounds, the valid inequalities and the x≤z / 2-cycle
  strengthening; MIPGap = 1e-9, MIPGapAbs = 1e-6). The flow model has a tight
  (supporting-hyperplane) relaxation for the *linear* cost objective and closes
  Ĉ_min / P̂_C quickly on the smaller instances.
- **Latency side → branch-and-price (B&P)** over the set-partitioning
  reformulation, enumerating the depot combinations **in parallel with a shared
  global incumbent** (a good solution found on one combination fathoms the
  others). The ng-route + subset-row-cut bound is strong for the latency
  objective, which is where the compact flow relaxation is weak.

This split is deliberate: on the larger instances the flow MILP is weak on
*latency* and B&P is weak on *cost* — they are mirror images, so each anchor is
assigned to the engine whose relaxation is tight for it. Afyón (n = 14) is small
enough that the flow alone certifies all four anchors.

**Independent verification.** Every saved solution is re-checked from the routes
alone before it is accepted: each pharmacy visited exactly once; each route within
vehicle capacity Q and duration limit L; each open depot within its throughput
capacity W_i; number of routes ≤ N_v; exactly N_g depots open; and C_F, P
recomputed from the routes match the reported anchor. No verification failures
occurred.

## 3. The nadir components: explicit lexicographic vs. first-stage evaluation

The two nadir anchors (P̂_C, Ĉ_P) are lexicographic **second-stage** quantities.
They were obtained two ways:

- **Afyón, Çanakkale — explicit lexicographic second stage.** P̂_C by flow
  lexicographic (`min C_F`, then `min P` among cost-optimal); Ĉ_P by B&P with a
  budget row (`min C_F s.t. P ≤ P̂_min`). These are certified lexicographic values.
- **Aydín — first-stage evaluation (upper bounds).** To avoid re-solving, we take
  **Ĉ_P = the financial cost of the latency-optimal solution** returned by the
  P̂_min solve, and **P̂_C = the priority-weighted latency of the cost-optimal
  routing R\*** (cost being profile-independent) evaluated under each profile's
  priorities. These are **evaluations of the first-stage solutions**, hence
  **upper bounds on P̂_C and Ĉ_P**, *not* certified lexicographic optima: the
  equality to the true second stage holds only if the cost-optimal and
  latency-optimal solutions are **unique**, and on Aydín **neither first stage is
  itself certified** (Ĉ_min and the P̂_min of three profiles are tight bounds, not
  proven to 0). The nadir point may therefore be **slightly over-estimated**. This
  is harmless for the normalisation (Δ_C, Δ_P are then marginally over-stated, and
  normalised criteria stay well-defined), but it must not be called "exact
  lexicographic".

**Cross-check (Çanakkale).** Where we have both — Çanakkale, whose first stages
*are* certified and whose cost optimum is effectively unique — the explicit
lexicographic second stages and the first-stage-evaluation shortcut **agree to the
last digit for all five profiles** (difference 0.000 on all ten comparisons). This
validates the shortcut *in that setting*; it does not by itself certify the Aydín
values, which remain upper bounds. A one-profile explicit lexicographic solve on
Aydín would make its nadir values airtight and is listed as an open item.

**Implication for the paper.** State Afyón/Çanakkale nadir anchors as certified
lexicographic values, and the Aydín P̂_C/Ĉ_P as **best known feasible (upper-bound)
nadir values obtained by evaluating the first-stage solutions**, cross-validated
against the explicit second stage on Çanakkale. The
shortcut also makes the computation far cheaper (one cost solve instead of one
per profile; no separate cost-minimisation for Ĉ_P).

## 4. Certification status (be precise in the paper)

- **Afyón (n = 14).** All four anchors, all five profiles: **certified optimal**
  (flow, gap 0).
- **Çanakkale (n = 37, N_g = 2).** Latency anchors P̂_min and Ĉ_P: **certified
  optimal** by B&P (gap 0). Cost anchors Ĉ_min and P̂_C: certified for four
  profiles; for Hyper-Critical the flow cost solve hit its time limit, so that
  P̂_C is a **tight upper bound**, not certified.
- **Aydín (n = 31, N_g = 3).** **Ĉ_min = 16 741.297, certified optimal (gap
  0.000 %)** from **one identifiable single-objective `min C_F` flow run**
  (`latency_obj=False`; UB = LB = 16 741.297, Gurobi `ObjBound` = 16 741.297, 2 435 s).
  (The earlier 4 h value came from a *lexicographic*/multi-objective model, for
  which Gurobi returns no bound — the `LB = UB` cell in that run's `.md` was a
  placeholder and should be ignored.) P̂_min: **certified** for Tiered and
  Demand-Correlated (gap 0), within ≤ 0.9 % for Equal/Negative/Hyper. P̂_C and Ĉ_P
  are first-stage-evaluation **upper bounds** (Section 3), cross-validated against
  the explicit second stage only on Çanakkale.

**On the time limit.** The B&P latency anchors use a **2 h cap per depot
combination**, enforced *softly* (checked between column-generation rounds), so a
single hard combination can overshoot it — e.g. Çanakkale Hyper-Critical had one
combination (D02+D05) run ≈ 6.5 h without closing. The **per-profile wall-clock is
the maximum over the combinations run in parallel batches, not their sum**; it can
exceed the per-combination cap because of such a straggler. The appendix should
state the cap this way (soft, per-combination; per-profile wall = parallel max).

Where a value is a tight upper bound rather than certified, this matches the
convention already stated in the manuscript: when an anchor cannot be certified by
an exact method, the best known feasible value is used, and normalised criteria of
later solutions may be marginally outside [0, 1] without affecting the properties.

## 5. Heuristic anchor verification (all three cities)

The GRASP-VND financial- and latency-anchor modes were re-run with routes saved
(30 seeds, `max_iter = 50`, RCL = 5) and **every** best-of-seed anchor was
independently re-checked by the exact checker (coverage/Q/L/W/N_v/N_g and C_F, P
recomputed from the routes). All passed; every heuristic anchor in the paper now
has a verified route set. Files: `results/{Aydin,Canakkale,Afyon}_heuristic_anchors.{json,csv}`.

Two cross-validations come out of this:

- **Çanakkale and Afyón: heuristic anchors reproduce the exact values exactly**
  (all four anchors, all five profiles) — independent confirmation of both the
  exact and the heuristic pipelines.
- **Aydín Negative-Correlated:** the heuristic latency solution **P̂_min = 84 221.13**
  (verified, C_F = 16 849.745) is feasible and **better than the B&P incumbent
  84 313.37**; with the certified lower bound 84 058.1 the best-known gap is
  **0.19 %**. (The earlier inability to confirm it was only because that value had
  no stored route set; the rerun supplies one.) On the other four Aydín profiles
  B&P's P̂_min is better than the heuristic's.

## 6. Aydín normalization set (basis for the interior Tchebycheff rerun)

`results/aydin_normalization_set.{md,csv}`. Ĉ_min is the certified optimum; P̂_min
is the **best verified** of B&P and heuristic; Ĉ_P is the cost of that chosen
P̂_min solution; P̂_C is the first-stage-evaluation upper bound (Section 3); ρ = 1e-4.

| Profile | Ĉ_min | P̂_min | source | P̂_C | Ĉ_P | Δ_C | Δ_P |
|---|---|---|---|---|---|---|---|
| Equal | 16741.297 | 95809.000 | B&P | 150827.0 | 16922.477 | 181.180 | 55018.000 |
| Tiered | 16741.297 | 52436.900 | B&P | 79697.9 | 16930.136 | 188.839 | 27261.000 |
| Demand-Correlated | 16741.297 | 58487.148 | B&P | 91590.963 | 16922.477 | 181.180 | 33103.815 |
| Negative-Correlated | 16741.297 | **84221.130** | **heuristic** | 134649.537 | 16849.745 | 108.448 | 50428.407 |
| Hyper-Critical | 16741.297 | 140092.000 | B&P | 194378.0 | 16981.612 | 240.315 | 54286.000 |

## 7. Suggested wording / design choices

- Present a single anchor table per city × profile (Ĉ_min, P̂_C, Ĉ_P, P̂_min, Δ_C,
  Δ_P), marking each entry as *certified*, *tight UB*, or *heuristic best-known*.
- Describe the exact-anchor method as a **regime-split exact approach** (flow for
  the financial anchors, branch-and-price for the latency anchors); state the
  first-stage-evaluation of the nadir components as a short remark, noting it gives
  upper bounds and that the explicit second stage agrees with it on Çanakkale.
- Keep the full solutions (open depots; per route: depot, ordered pharmacy IDs,
  load, duration, travel cost, per-stop arrival times) in the reproducibility
  package: `results/{afyon,canakkale,aydin}_exact_{anchors.md,anchors.csv,routes.json}`
  and `results/{Aydin,Canakkale,Afyon}_heuristic_anchors.{json,csv}`.

## 8. Open items

- Resolved: every heuristic anchor now has a verified route set; Aydín Ĉ_min is
  certified; the Aydín nadir components are stated as upper bounds (no spot-check
  needed — the paper reports them as such).
- Optional only if a reviewer presses: longer cost solves (or a cost-side B&P) to
  *certify* the Hyper-Critical P̂_C on Çanakkale rather than leave it a tight upper
  bound.
