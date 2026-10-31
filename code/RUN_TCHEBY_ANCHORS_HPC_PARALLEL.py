import os

# Prevent nested numerical-library threading.
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[var] = "1"

import random
import time
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import LLRP_CORE_PARALLEL_HPC as llrp


CITY = os.environ.get("CITY", "Afyon")
INPUTS = "Inputs"
SEEDS = list(range(30))
MAX_ITER = 50
RCL_SIZE = 5

_BASE_INSTANCE = None


def _json_safe(value):
    """Convert numpy/Python objects into JSON-safe objects."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def serialize_solution(sol):
    """Preserve depot and route structure for reproducibility."""
    instance = sol.instance

    open_depots = []
    for depot_index in sol.open_depots:
        depot_index = int(depot_index)

        open_depots.append({
            "Depot_Index": depot_index,
            "Depot_Label": str(
                instance.depot_labels[depot_index]
            ),
            "Proxy_Customer_Code": str(
                instance.depot_node_ids[depot_index]
            ),
        })

    routes = []

    for route_no, route in enumerate(sol.routes, start=1):
        path = route.get("path", [])

        if not path:
            continue

        depot_index = int(route["depot"])

        route_record = {
            "Route_No": route_no,
            **_json_safe(route),
            "Depot_Index": depot_index,
            "Depot_Label": str(
                instance.depot_labels[depot_index]
            ),
            "Proxy_Customer_Code": str(
                instance.depot_node_ids[depot_index]
            ),
            "Customer_Codes": [
                str(instance.node_ids[int(node)])
                for node in path
            ],
        }

        routes.append(route_record)

    return (
        json.dumps(open_depots, ensure_ascii=False),
        json.dumps(routes, ensure_ascii=False),
    )


def build_anchor_route_rows(anchor_row, anchor_name):
    routes = json.loads(anchor_row["Routes_JSON"])
    open_depots = json.loads(anchor_row["Open_Depots_JSON"])

    rows = []

    for route in routes:
        row = {
            "Anchor": anchor_name,
            "Seed": int(anchor_row["Seed"]),
            "Open_Depots": json.dumps(
                open_depots,
                ensure_ascii=False,
            ),
        }

        for key, value in route.items():
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                )
            else:
                row[key] = value

        rows.append(row)

    return rows

def init_worker(city_name, inputs_path):
    global _BASE_INSTANCE

    _, instance = llrp.load_city_instance(
        city_name,
        base_path=inputs_path,
        equity_weight=0.0,
    )

    instance.priorities = np.ones(
        instance.num_customers,
        dtype=np.float64,
    )

    _BASE_INSTANCE = instance


def run_one(task):
    global _BASE_INSTANCE

    mode, seed = task

    # Each worker owns its instance, but reset objective-specific fields.
    instance = copy.deepcopy(_BASE_INSTANCE)
    instance.objective_mode = mode

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.perf_counter()

    sol, _ = llrp.run_grasp_vnd_logged(
        instance,
        max_iter=MAX_ITER,
        alpha_max=RCL_SIZE,
        run_id=seed + 1,
        seed=seed,
        instance_name=f"{CITY}_{mode}_anchor_seed{seed}",
    )

    runtime_s = time.perf_counter() - start_time

    feasible = sol is not None and sol.check_validity()

    if not feasible:
        return {
            "Mode": mode,
            "Seed": seed,
            "Feasible": False,
            "Scalar_Objective": None,
            "Runtime_s": runtime_s,
            "Financial_Cost": None,
            "Priority_Weighted_Latency": None,
            "Num_Open_Depots": None,
            "Num_Routes": None,
            "Open_Depots_JSON": None,
            "Routes_JSON": None,
        }

    sol.calculate_cost()
    open_depots_json, routes_json = serialize_solution(sol)

    return {
        "Mode": mode,
        "Seed": seed,
        "Feasible": True,
        "Scalar_Objective": float(sol.cost),
        "Runtime_s": runtime_s,
        "Financial_Cost": float(sol.financial_cost),
        "Priority_Weighted_Latency": float(sol.equity_penalty),
        "Num_Open_Depots": len(sol.open_depots),
        "Num_Routes": len([
            r for r in sol.routes if r["path"]
        ]),
        "Open_Depots_JSON": open_depots_json,
        "Routes_JSON": routes_json,
    }


def select_anchor(df, mode):
    x = df[
        (df["Mode"] == mode)
        & (df["Feasible"] == True)
    ].copy()

    if x.empty:
        raise RuntimeError(
            f"No feasible {mode} anchor found."
        )

    if mode == "financial":
        x = x.sort_values(
            [
                "Financial_Cost",
                "Priority_Weighted_Latency",
                "Seed",
            ],
            ascending=True,
        )
    elif mode == "latency":
        x = x.sort_values(
            [
                "Priority_Weighted_Latency",
                "Financial_Cost",
                "Seed",
            ],
            ascending=True,
        )
    else:
        raise ValueError(mode)

    return x.iloc[0]


def main():
    job_start_time = time.perf_counter()
    workers = int(os.environ.get("NSLOTS", "1"))

    print("=" * 72)
    print("PARALLEL TCHEBYCHEFF ANCHOR VALIDATION")
    print("=" * 72)
    print(f"City       : {CITY}")
    print(f"Seeds      : {len(SEEDS)}")
    print(f"Max iter   : {MAX_ITER}")
    print(f"Workers    : {workers}")
    print(f"Total runs : {2 * len(SEEDS)}")
    print("=" * 72)

    tasks = [
        (mode, seed)
        for mode in ("financial", "latency")
        for seed in SEEDS
    ]

    rows = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(CITY, INPUTS),
    ) as executor:

        futures = {
            executor.submit(run_one, task): task
            for task in tasks
        }

        for future in as_completed(futures):
            mode, seed = futures[future]

            try:
                row = future.result()
            except Exception as exc:
                print(
                    f"ERROR mode={mode}, seed={seed}: {exc}",
                    flush=True,
                )
                raise

            rows.append(row)

            print(
                f"DONE mode={mode:9s} "
                f"seed={seed:02d} "
                f"obj={row['Scalar_Objective']}",
                flush=True,
            )

    results = pd.DataFrame(rows).sort_values(
        ["Mode", "Seed"]
    ).reset_index(drop=True)

    cost_anchor = select_anchor(results, "financial")
    latency_anchor = select_anchor(results, "latency")

    C_min = float(cost_anchor["Financial_Cost"])
    P_at_C_min = float(
        cost_anchor["Priority_Weighted_Latency"]
    )

    P_min = float(
        latency_anchor["Priority_Weighted_Latency"]
    )
    C_at_P_min = float(
        latency_anchor["Financial_Cost"]
    )

    C_range = C_at_P_min - C_min
    P_range = P_at_C_min - P_min

    payoff = pd.DataFrame([
        {
            "Anchor": "Cost",
            "Financial_Cost": C_min,
            "Priority_Weighted_Latency": P_at_C_min,
            "Seed": int(cost_anchor["Seed"]),
            "Open_Depots_JSON": cost_anchor["Open_Depots_JSON"],
        },
        {
            "Anchor": "Latency",
            "Financial_Cost": C_at_P_min,
            "Priority_Weighted_Latency": P_min,
            "Seed": int(latency_anchor["Seed"]),
            "Open_Depots_JSON": latency_anchor["Open_Depots_JSON"],
        },
    ])

    anchor_route_rows = []
    anchor_route_rows.extend(
        build_anchor_route_rows(
            cost_anchor,
            "Cost",
        )
    )
    anchor_route_rows.extend(
        build_anchor_route_rows(
            latency_anchor,
            "Latency",
        )
    )

    anchor_routes = pd.DataFrame(anchor_route_rows)

    financial_runtime = results.loc[
        (results["Mode"] == "financial")
        & (results["Feasible"] == True),
        "Runtime_s",
    ]

    latency_runtime = results.loc[
        (results["Mode"] == "latency")
        & (results["Feasible"] == True),
        "Runtime_s",
    ]

    total_wall_clock_s = (
        time.perf_counter() - job_start_time
    )

    summary = pd.DataFrame([
        {
            "City": CITY,
            "Seeds": len(SEEDS),
            "Max_Iter": MAX_ITER,
            "Workers": workers,
            "C_min": C_min,
            "P_at_C_min": P_at_C_min,
            "C_at_P_min": C_at_P_min,
            "P_min": P_min,
            "C_range": C_range,
            "P_range": P_range,
            "Total_Wall_Clock_s": total_wall_clock_s,

            "Financial_Anchor_Runtime_s": float(
                cost_anchor["Runtime_s"]
            ),
            "Financial_Runtime_Mean_s": float(
                financial_runtime.mean()
            ),
            "Financial_Runtime_Median_s": float(
                financial_runtime.median()
            ),
            "Financial_Runtime_Std_s": float(
                financial_runtime.std()
            ),
            "Financial_Runtime_Min_s": float(
                financial_runtime.min()
            ),
            "Financial_Runtime_Max_s": float(
                financial_runtime.max()
            ),

            "Latency_Anchor_Runtime_s": float(
                latency_anchor["Runtime_s"]
            ),
            "Latency_Runtime_Mean_s": float(
                latency_runtime.mean()
            ),
            "Latency_Runtime_Median_s": float(
                latency_runtime.median()
            ),
            "Latency_Runtime_Std_s": float(
                latency_runtime.std()
            ),
            "Latency_Runtime_Min_s": float(
                latency_runtime.min()
            ),
            "Latency_Runtime_Max_s": float(
                latency_runtime.max()
            ),
        }
    ])

    print("\n" + "=" * 72)
    print(f"{CITY.upper()} PAYOFF TABLE — PARALLEL HEURISTIC VALIDATION")
    print("=" * 72)

    print("\nCOST ANCHOR")
    print(f"C_min        = {C_min:.6f}")
    print(f"P_at_C_min   = {P_at_C_min:.6f}")
    print(f"Seed         = {int(cost_anchor['Seed'])}")

    print("\nLATENCY ANCHOR")
    print(f"C_at_P_min   = {C_at_P_min:.6f}")
    print(f"P_min        = {P_min:.6f}")
    print(f"Seed         = {int(latency_anchor['Seed'])}")

    print("\nNORMALIZATION RANGES")
    print(f"C_range      = {C_range:.6f}")
    print(f"P_range      = {P_range:.6f}")

    if C_range <= 0 or P_range <= 0:
        raise RuntimeError(
            "Invalid normalization range."
        )

    Path("Outputs").mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        f"Outputs/{CITY}_Tcheby_Anchor_Validation_parallel.xlsx"
    )

    with pd.ExcelWriter(output_file) as writer:
        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )
        payoff.to_excel(
            writer,
            sheet_name="Payoff_Table",
            index=False,
        )
        anchor_routes.to_excel(
            writer,
            sheet_name="Anchor_Routes",
            index=False,
        )
        results.to_excel(
            writer,
            sheet_name="Run_Level",
            index=False,
        )

    print("\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()
