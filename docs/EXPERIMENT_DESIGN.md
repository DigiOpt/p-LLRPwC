# Experiments in the present manuscript

| City | n | m | Ng | Nv | Total demand | Max demand | L (seconds) | Depot W | Vehicle Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Afyon |14|5|2|4|735|136|10000|435|280|
| Aydin |31|5|3|5|621|40|18900|245|140|
| Canakkale |37|5|2|5|545|29|10000|325|125|
| Isparta |48|5|2|6|400|60|10000|240|75|

- Five priority profiles per city.
- Thirty independent runs for each anchor mode and for each interior preference.
- `Afyon`, `Aydin`, `Canakkale`: 9 interior preferences, `lambda = 0.1 ... 0.9`, yielding 270 interior runs per profile.
- `Isparta`: **5** interior preferences, `lambda = {0.1,0.3,0.5,0.7,0.9}`, yielding 150 interior runs per profile. The unmodified original production runner specified the generic 9-preference grid for *every* city, which does not agree with the stored five-preference Isparta workbooks. The publication-aligned copy under `code/` explicitly filters Isparta; the original is retained in `code/legacy_reference/`.
- Anchor workbooks have the 30 cost runs and 30 priority-latency runs, with a lexicographically selected reference point from the available runs.
- Standard latency-only LLRP benchmark: 10 Barreto-family instances, 30 runs each, without depot capacities, fixed costs, priority weights or route-duration constraints. The baseline/extended implementation is in `benchmarks/code/`; the stored individual instance workbooks are in `benchmarks/results/`.
- Ablation: Base, Base+SO, Base+SO+DR, and GRASP-VND-SO. `benchmarks/ablation/analysis/` has the aggregated all-variant results. The two smaller variant code files are preserved separately; the final extended variant is VNDZou.py in the benchmark module. Do not mistake the exploratory 13-variant analysis for the four configurations actually summarized in manuscript Table 4.
- Exact validation: Afyon and partial Canakkale/Aydin anchor references appear in the manuscript. Exact MILP/branch-and-price source code and their solver files are **not included** at the user's request. Isparta has no exact reference results in the current paper.

**Historical separation:** The Drive folder `Updated Manuscript` describes a different **six-city, 52-scenario** revision archive. Its no-rerun statistical pipeline, extra two cities, and parameter sweeps are not results of the current four-city Tchebycheff manuscript and have not been mixed into this publication package.
