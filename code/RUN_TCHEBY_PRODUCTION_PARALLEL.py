import os
import copy
import json
import time
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import LLRP_CORE_PARALLEL_HPC as llrp


CITY = os.environ["CITY"]
PROFILE = os.environ["PROFILE"]

MAX_ITER = int(os.getenv("MAX_ITER", "50"))
N_SEEDS = int(os.getenv("N_SEEDS", "30"))
WORKERS = int(os.getenv("WORKERS", "16"))
RHO = float(os.getenv("RHO", "1e-4"))

LAMBDAS = [round(x / 10, 1) for x in range(11)]
INTERIOR_LAMBDAS = LAMBDAS[1:-1]
# Publication-alignment correction (see docs/REPRODUCIBILITY_LIMITS.md):
# This revision's saved Isparta runs used five interior preference values.
# The original Drive code applied the nine-value grid to all cities.
# Its exact bytes are preserved in code/legacy_reference/.
if CITY.strip().lower() == "isparta":
    INTERIOR_LAMBDAS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = list(range(N_SEEDS))

OUTPUT_DIR = Path("Outputs/Tcheby_Production")
PLOT_DIR = Path("Outputs/Plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

_BASE_INSTANCE = None


def get_anchor_file(city, profile):
    if profile == "Equal-Priority":
        return Path(
            f"Outputs/{city}_Tcheby_Anchor_Validation_parallel.xlsx"
        )

    return Path(
        f"Outputs/{city}_{profile}_"
        f"Tcheby_Anchor_Validation_parallel.xlsx"
    )


def read_anchors(city, profile):
    f = get_anchor_file(city, profile)

    if not f.exists():
        raise FileNotFoundError(f)

    s = pd.read_excel(
        f,
        sheet_name="Summary",
    ).iloc[0]

    return {
        "C_min": float(s["C_min"]),
        "P_at_C_min": float(s["P_at_C_min"]),
        "C_at_P_min": float(s["C_at_P_min"]),
        "P_min": float(s["P_min"]),
    }


def init_worker(city, profile, anchors):
    global _BASE_INSTANCE

    _, instance = llrp.load_city_instance(
        city,
        base_path="Inputs",
        equity_weight=0.0,
    )

    instance.priorities = (
        instance.priority_profiles[profile].copy()
    )

    instance.anchor_c_min = anchors["C_min"]
    instance.anchor_p_at_c_min = anchors["P_at_C_min"]
    instance.anchor_c_at_p_min = anchors["C_at_P_min"]
    instance.anchor_p_min = anchors["P_min"]

    _BASE_INSTANCE = instance


def routes_json(sol):
    data = []

    for r in sol.routes:
        if not r["path"]:
            continue

        data.append({
            "depot": int(r["depot"]),
            "path": [int(x) for x in r["path"]],
            "load": float(r["load"]),
        })

    return json.dumps(data)


def run_one(task):
    global _BASE_INSTANCE

    lam, seed = task
    instance = copy.deepcopy(_BASE_INSTANCE)

    instance.objective_mode = "tchebycheff"
    instance.tcheby_lambda = float(lam)
    instance.tcheby_rho = RHO

    random.seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()

    sol, _ = llrp.run_grasp_vnd_logged(
        instance,
        max_iter=MAX_ITER,
        alpha_max=5,
        run_id=seed + 1,
        seed=seed,
        instance_name=(
            f"{CITY}_{PROFILE}_L{lam:.1f}_S{seed}"
        ),
    )

    runtime = time.perf_counter() - t0

    feasible = (
        sol is not None
        and sol.check_validity()
    )

    if not feasible:
        return {
            "Lambda": float(lam),
            "Seed": int(seed),
            "Feasible": False,
            "Runtime_s": runtime,
        }

    C = float(sol.financial_cost)
    P = float(sol.equity_penalty)

    C_range = (
        instance.anchor_c_at_p_min
        - instance.anchor_c_min
    )

    P_range = (
        instance.anchor_p_at_c_min
        - instance.anchor_p_min
    )

    C_norm = (
        (C - instance.anchor_c_min)
        / C_range
    )

    P_norm = (
        (P - instance.anchor_p_min)
        / P_range
    )

    return {
        "Lambda": float(lam),
        "Seed": int(seed),
        "Feasible": True,
        "Financial_Cost": C,
        "Priority_Weighted_Latency": P,
        "C_norm": C_norm,
        "P_norm": P_norm,
        "Scalar_Objective": float(sol.cost),
        "Runtime_s": runtime,
        "Open_Depots_JSON": json.dumps(
            sorted(int(x) for x in sol.open_depots)
        ),
        "Routes_JSON": routes_json(sol),
    }


def run_production():
    anchors = read_anchors(
        CITY,
        PROFILE,
    )

    tasks = [
        (lam, seed)
        for lam in INTERIOR_LAMBDAS
        for seed in SEEDS
    ]

    print("=" * 70)
    print("TCHEBY PRODUCTION")
    print("City       :", CITY)
    print("Profile    :", PROFILE)
    print("MAX_ITER   :", MAX_ITER)
    print("Seeds      :", N_SEEDS)
    print("Workers    :", WORKERS)
    print("rho        :", RHO)
    print("Tasks      :", len(tasks))
    print("Anchors    :", anchors)
    print("=" * 70)

    t0 = time.perf_counter()
    results = []

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
        initargs=(CITY, PROFILE, anchors),
    ) as ex:

        futures = [
            ex.submit(run_one, task)
            for task in tasks
        ]

        total = len(futures)

        for k, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            result = future.result()
            results.append(result)

            if k % 10 == 0 or k == total:
                print(
                    f"Completed {k}/{total}"
                )

    wall_time = time.perf_counter() - t0

    run_df = pd.DataFrame(results)

    feasible = run_df[
        run_df["Feasible"] == True
    ].copy()

    if feasible.empty:
        raise RuntimeError(
            "No feasible production solutions."
        )

    best = (
        feasible
        .sort_values([
            "Lambda",
            "Scalar_Objective",
            "Financial_Cost",
            "Priority_Weighted_Latency",
            "Seed",
        ])
        .groupby(
            "Lambda",
            as_index=False,
        )
        .first()
    )

    return (
        anchors,
        run_df,
        best,
        wall_time,
    )


def dominated_flags(df):
    flags = []

    for i, row in df.iterrows():
        C = row["Financial_Cost"]
        P = row["Priority_Weighted_Latency"]

        dominated = False

        for j, other in df.iterrows():
            if i == j:
                continue

            better_or_equal = (
                other["Financial_Cost"] <= C
                and
                other["Priority_Weighted_Latency"] <= P
            )

            strictly_better = (
                other["Financial_Cost"] < C
                or
                other["Priority_Weighted_Latency"] < P
            )

            if better_or_equal and strictly_better:
                dominated = True
                break

        flags.append(dominated)

    return flags


def save_results(
    anchors,
    run_df,
    best,
    wall_time,
):
    endpoint0 = {
        "Lambda": 0.0,
        "Seed": None,
        "Feasible": True,
        "Financial_Cost": anchors["C_min"],
        "Priority_Weighted_Latency":
            anchors["P_at_C_min"],
        "C_norm": 0.0,
        "P_norm": 1.0,
        "Scalar_Objective": 0.0,
        "Source": "Validated financial anchor",
    }

    endpoint1 = {
        "Lambda": 1.0,
        "Seed": None,
        "Feasible": True,
        "Financial_Cost":
            anchors["C_at_P_min"],
        "Priority_Weighted_Latency":
            anchors["P_min"],
        "C_norm": 1.0,
        "P_norm": 0.0,
        "Scalar_Objective": 0.0,
        "Source": "Validated latency anchor",
    }

    best = best.copy()
    best["Source"] = (
        "Best of 30 GRASP-VND seeds"
    )

    curve = pd.concat(
        [
            pd.DataFrame([endpoint0]),
            best,
            pd.DataFrame([endpoint1]),
        ],
        ignore_index=True,
    )

    curve = curve.sort_values(
        "Lambda"
    ).reset_index(drop=True)

    curve["Dominated"] = dominated_flags(
        curve
    )

    stem = (
        f"{CITY}_{PROFILE}_"
        f"Tcheby_Production"
    )

    xlsx = OUTPUT_DIR / f"{stem}.xlsx"
    csv = OUTPUT_DIR / f"{stem}_Curve.csv"

    summary = pd.DataFrame([{
        "City": CITY,
        "Priority_Profile": PROFILE,
        "MAX_ITER": MAX_ITER,
        "N_SEEDS": N_SEEDS,
        "Workers": WORKERS,
        "Rho": RHO,
        "Interior_Lambdas": 9,
        "Total_Runs": len(run_df),
        "Feasible_Runs":
            int(run_df["Feasible"].sum()),
        "Wall_Time_s": wall_time,
    }])

    with pd.ExcelWriter(xlsx) as writer:
        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )
        curve.to_excel(
            writer,
            sheet_name="Best_By_Lambda",
            index=False,
        )
        run_df.to_excel(
            writer,
            sheet_name="Run_Level",
            index=False,
        )

    curve.to_csv(
        csv,
        index=False,
    )

    png = PLOT_DIR / (
        f"{stem}_Tradeoff.png"
    )

    x = curve[
        "Financial_Cost"
    ].to_numpy()

    y = curve[
        "Priority_Weighted_Latency"
    ].to_numpy()

    lam = curve[
        "Lambda"
    ].to_numpy()

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o")

    for xi, yi, li in zip(x, y, lam):
        plt.annotate(
            f"λ={li:.1f}",
            (xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Financial Cost, C")
    plt.ylabel(
        "Priority-Weighted Latency, P"
    )

    plt.title(
        f"{CITY} – {PROFILE}\n"
        "Augmented Weighted "
        "Tchebycheff Trade-off"
    )

    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nFINAL CURVE")
    print(curve.to_string(index=False))
    print("\nSaved:", xlsx)
    print("Saved:", csv)
    print("Saved:", png)


if __name__ == "__main__":
    data = run_production()
    save_results(*data)
