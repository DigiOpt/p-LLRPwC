import os

# One OS process is used per scenario. Keep numerical libraries single-threaded
# inside each process so workers do not oversubscribe the laptop CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import math
import time
import copy
import random
import re
import concurrent.futures
import multiprocessing as mp
import colorsys
import html
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from numba import njit
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore", message="Unknown extension is not supported and will be removed")
warnings.filterwarnings("ignore", message="Conditional Formatting extension is not supported and will be removed")



DEFAULT_BASE_DATA_PATH = str(Path(__file__).resolve().parent)

DEFAULT_TRAVEL_METRIC = "duration"
DEFAULT_MAX_ROUTE_DURATION_SECONDS = None  # None/"auto" => city-aware value
DEFAULT_FIXED_DEPOT_COST = 5000
DEFAULT_FIXED_VEHICLE_COST = 200
DEFAULT_EQUITY_WEIGHT = 0.5
DEFAULT_COST_PER_KM = 1.0

# Manuscript-specific capacity design. Other cities are derived automatically.
CITY_CAPACITY_LEVEL_OVERRIDES = {
    "Afyon": {
        "depot": {"Low": 400.0, "Moderate": 435.0, "High": 500.0},
        "vehicle": {"Low": 240.0, "Moderate": 280.0, "High": 320.0},
    }
}


def normalize_schema_token(value):
    """Normalize workbook labels for robust schema matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def round_up(value, step=5):
    return float(math.ceil(float(value) / step) * step)


def round_down(value, step=5):
    return float(math.floor(float(value) / step) * step)


def list_available_cities(base_path=DEFAULT_BASE_DATA_PATH):
    """Return cities available in the flat Ablation/Inputs directory."""
    base = Path(base_path)
    if not base.is_dir():
        raise FileNotFoundError(f"Inputs directory not found: {base}")

    suffix = "_index_map.xlsx"
    return sorted(
        path.name[:-len(suffix)]
        for path in base.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    )


def discover_city_files(city_name, base_path=DEFAULT_BASE_DATA_PATH):
    """Return the three exact input workbooks for one city."""
    city = str(city_name).strip()
    if not city:
        raise ValueError("CITY_NAME cannot be empty.")

    base = Path(base_path)
    files = {
        "index": base / f"{city}_index_map.xlsx",
        "distance": base / f"{city}_distance_matrix_Google.xlsx",
        "duration": base / f"{city}_duration_matrix_Google.xlsx",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        available = ", ".join(list_available_cities(base_path)) or "none"
        raise FileNotFoundError(
            f"Missing input file(s) for city '{city}': {missing}. "
            f"Available cities: {available}"
        )

    return {
        "city": city,
        "city_dir": str(base.resolve()),
        **{role: str(path.resolve()) for role, path in files.items()},
    }


_INDEX_COLUMN_ALIASES = {
    "Latitude": {"latitude", "lat"},
    "Longitude": {"longitude", "lon", "lng"},
    "WH_Code": {"whcode", "warehousecode"},
    "Customer_Code": {"customercode", "customerid", "id"},
    "Total_Sales": {"totalsales", "sales", "annualtotalsales"},
    "Priority": {"priority", "priorityweight"},
}

_CANDIDATE_COLUMN_ALIASES = {
    "Candidate_Depot_ID": {"candidatedepotid", "depotid"},
    "Customer_Code": {"customercode", "customerid"},
    "WH_Code": {"whcode", "warehousecode"},
    "Latitude": {"latitude", "lat"},
    "Longitude": {"longitude", "lon", "lng"},
    "Fixed_Depot_Capacity": {
        "fixeddepotcapacity", "depotcapacity", "fixedcapacityw"
    },
}

_PARAMETER_ALIASES = {
    "candidate_depots": {
        "candidatedepotlocationsm", "candidatedepotsm", "candidatedepots"
    },
    "open_depots": {
        "requiredopendepotsng", "opendepotsng", "requiredopendepots"
    },
    "max_vehicles": {
        "maximumvehiclesroutesnv", "maximumvehiclesnv",
        "vehiclesroutesnv", "maxvehicles"
    },
    "vehicle_cap": {"vehiclecapacityq", "vehiclecapq"},
    "depot_cap": {
        "depotcapacityperopendepotw",
        "fixedcapacitypercandidatedepotw",
        "depotcapacityw",
    },
    "total_demand": {"totalshiftdemandd", "totaldemandd"},
    "customer_count": {"numberofcustomersn", "customersn"},
}


def _canonicalize_columns(columns, aliases):
    rename = {}
    for column in columns:
        token = normalize_schema_token(column)
        for canonical, accepted in aliases.items():
            if token in accepted:
                rename[column] = canonical
                break
    return rename


def _find_table_header(file_path, aliases, required_columns, preferred_sheets=()):
    """Locate a sheet/header row by semantic column names."""
    workbook = pd.ExcelFile(file_path)
    normalized_preferences = {
        normalize_schema_token(name) for name in preferred_sheets
    }
    sheets = sorted(
        workbook.sheet_names,
        key=lambda name: (
            normalize_schema_token(name) not in normalized_preferences,
            workbook.sheet_names.index(name),
        ),
    )

    required_columns = set(required_columns)
    for sheet_name in sheets:
        preview = pd.read_excel(
            file_path, sheet_name=sheet_name, header=None, nrows=30
        )
        for header_row in range(len(preview)):
            values = preview.iloc[header_row].tolist()
            detected = set()
            for value in values:
                token = normalize_schema_token(value)
                for canonical, accepted in aliases.items():
                    if token in accepted:
                        detected.add(canonical)
            if required_columns.issubset(detected):
                return sheet_name, header_row

    raise ValueError(
        f"Could not locate columns {sorted(required_columns)} in "
        f"{os.path.basename(file_path)}."
    )


def read_city_customer_table(index_file):
    sheet, header_row = _find_table_header(
        index_file,
        _INDEX_COLUMN_ALIASES,
        required_columns={"Latitude", "Longitude", "Customer_Code", "Total_Sales"},
        preferred_sheets=("Sheet1", "Customers", "Data"),
    )
    frame = pd.read_excel(index_file, sheet_name=sheet, header=header_row)
    frame.rename(
        columns=_canonicalize_columns(frame.columns, _INDEX_COLUMN_ALIASES),
        inplace=True,
    )
    required = ["Latitude", "Longitude", "Customer_Code", "Total_Sales"]
    frame = frame.dropna(subset=required).copy()
    frame["Customer_Code"] = frame["Customer_Code"].astype(str).str.strip()
    return frame, sheet


def read_candidate_depot_table(index_file):
    sheet, header_row = _find_table_header(
        index_file,
        _CANDIDATE_COLUMN_ALIASES,
        required_columns={
            "Candidate_Depot_ID", "Customer_Code", "Latitude",
            "Longitude", "Fixed_Depot_Capacity",
        },
        preferred_sheets=("Sheet3", "Candidate Depots", "Depots"),
    )
    frame = pd.read_excel(index_file, sheet_name=sheet, header=header_row)
    frame.rename(
        columns=_canonicalize_columns(frame.columns, _CANDIDATE_COLUMN_ALIASES),
        inplace=True,
    )
    required = [
        "Candidate_Depot_ID", "Customer_Code", "Latitude",
        "Longitude", "Fixed_Depot_Capacity",
    ]
    frame = frame.dropna(subset=required).copy()
    frame["Candidate_Depot_ID"] = (
        frame["Candidate_Depot_ID"].astype(str).str.strip()
    )
    frame["Customer_Code"] = frame["Customer_Code"].astype(str).str.strip()
    return frame, sheet, header_row


def read_city_parameter_map(index_file):
    """Scan workbook labels so tab names and row positions need not be fixed."""
    workbook = pd.ExcelFile(index_file)
    preferred = sorted(
        workbook.sheet_names,
        key=lambda name: (
            normalize_schema_token(name) not in {"sheet2", "parameters", "params"},
            workbook.sheet_names.index(name),
        ),
    )
    found = {}
    for sheet_name in preferred:
        raw = pd.read_excel(index_file, sheet_name=sheet_name, header=None)
        for _, row in raw.iterrows():
            values = row.tolist()
            for column_index in range(max(0, len(values) - 1)):
                label = values[column_index]
                value = values[column_index + 1]
                if pd.isna(label) or pd.isna(value):
                    continue
                token = normalize_schema_token(label)
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                for canonical, accepted in _PARAMETER_ALIASES.items():
                    if token in accepted and canonical not in found:
                        found[canonical] = numeric_value
        if {"candidate_depots", "open_depots", "max_vehicles",
            "vehicle_cap", "depot_cap"}.issubset(found):
            break

    required = {
        "candidate_depots", "open_depots", "max_vehicles",
        "vehicle_cap", "depot_cap"
    }
    missing = sorted(required - set(found))
    if missing:
        raise ValueError(
            f"Missing LLRP parameters {missing} in "
            f"{os.path.basename(index_file)}."
        )
    return found


# =============================================================================
# MULTIOBJECTIVE SCALARIZATION
# =============================================================================

def evaluate_scalar_objective(financial_cost, latency, instance):
    """
    Evaluate the scalar objective used by the search.

    Modes
    -----
    legacy:
        C_F + alpha * P
    financial:
        C_F
    latency:
        P
    tchebycheff:
        Normalized augmented weighted Tchebycheff scalarization.
    """
    financial_cost = float(financial_cost)
    latency = float(latency)

    mode = str(
        getattr(instance, "objective_mode", "legacy")
    ).strip().lower()

    if mode == "legacy":
        return (
            financial_cost
            + float(instance.equity_weight) * latency
        )

    if mode == "financial":
        return financial_cost

    if mode == "latency":
        return latency

    if mode != "tchebycheff":
        raise ValueError(
            f"Unknown objective_mode={mode!r}. "
            "Expected legacy, financial, latency, or tchebycheff."
        )

    lam = float(getattr(instance, "tcheby_lambda", 0.5))
    rho = float(getattr(instance, "tcheby_rho", 1e-4))

    if not (0.0 < lam < 1.0):
        raise ValueError(
            "tcheby_lambda must satisfy 0 < lambda < 1. "
            "Use objective_mode='financial' or 'latency' for anchors."
        )

    if rho <= 0.0:
        raise ValueError("tcheby_rho must be strictly positive.")

    required = (
        "anchor_c_min",
        "anchor_c_at_p_min",
        "anchor_p_min",
        "anchor_p_at_c_min",
    )

    missing = [
        name for name in required
        if getattr(instance, name, None) is None
    ]
    if missing:
        raise ValueError(
            "Tchebycheff objective requires fixed anchor values: "
            + ", ".join(missing)
        )

    c_min = float(instance.anchor_c_min)
    c_at_p_min = float(instance.anchor_c_at_p_min)
    p_min = float(instance.anchor_p_min)
    p_at_c_min = float(instance.anchor_p_at_c_min)

    c_range = c_at_p_min - c_min
    p_range = p_at_c_min - p_min

    if c_range <= 0.0:
        raise ValueError(
            f"Invalid financial anchor range: {c_range}. "
            "Require C_at_P_min > C_min."
        )

    if p_range <= 0.0:
        raise ValueError(
            f"Invalid latency anchor range: {p_range}. "
            "Require P_at_C_min > P_min."
        )

    c_norm = (financial_cost - c_min) / c_range
    p_norm = (latency - p_min) / p_range

    weighted_c = (1.0 - lam) * c_norm
    weighted_p = lam * p_norm

    tcheby_term = max(weighted_c, weighted_p)
    augmentation = rho * (weighted_c + weighted_p)

    return float(tcheby_term + augmentation)



def evaluate_route_search_objective(financial_cost, latency, instance):
    """
    Scalar route value used by construction/local-search components.

    Legacy, financial, and latency modes are additive across routes.

    Tchebycheff is intentionally rejected here because its max operator is
    not route-additive. Tchebycheff neighborhood evaluation must use global
    candidate-solution deltas.
    """
    mode = str(
        getattr(instance, "objective_mode", "legacy")
    ).strip().lower()

    if mode == "tchebycheff":
        raise RuntimeError(
            "Route-local Tchebycheff evaluation is invalid. "
            "Use global candidate-solution objective evaluation."
        )

    return evaluate_scalar_objective(
        financial_cost,
        latency,
        instance,
    )


class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.routes = []
        self.open_depots = set()
        self.financial_cost = 0
        self.equity_penalty = 0
        self.cost = float("inf")

    def coverage_status(self):
        served = []

        for route in self.routes:
            if route["path"]:
                served.extend(route["path"])

        num_served = len(served)
        num_unique_served = len(set(served))
        num_customers = self.instance.num_customers

        missing_count = num_customers - num_unique_served
        duplicate_count = num_served - num_unique_served

        coverage_ok = (
            num_served == num_customers and
            num_unique_served == num_customers
        )

        return coverage_ok, num_served, num_unique_served, missing_count, duplicate_count

    def calculate_cost(self):
        coverage_ok, _, _, _, _ = self.coverage_status()

        if not coverage_ok:
            self.cost = float("inf")
            return self.cost

        travel_cost = 0.0
        vehicle_cost = 0.0
        total_latency_penalty = 0.0
        self.open_depots = set()

        for route in self.routes:
            if not route["path"]:
                continue

            self.open_depots.add(route["depot"])
            vehicle_cost += self.instance.fixed_vehicle_cost

            financial_travel_cost, latency_penalty = evaluate_route(
                route["depot"], route["path"], self.instance
            )
            real_distance_m, real_duration_s = self.instance.route_real_metrics(
                route["depot"], route["path"]
            )

            travel_cost += financial_travel_cost
            total_latency_penalty += latency_penalty

            route["financial_travel_cost"] = financial_travel_cost
            route["distance_meters"] = real_distance_m
            route["duration_seconds"] = real_duration_s

            # Backward-compatible route-duration field used by ECWA/validity.
            route["dist"] = real_duration_s

        depot_cost = len(self.open_depots) * self.instance.fixed_depot_cost

        self.financial_cost = travel_cost + vehicle_cost + depot_cost
        self.equity_penalty = total_latency_penalty
        self.cost = evaluate_scalar_objective(
            self.financial_cost,
            self.equity_penalty,
            self.instance,
        )

        return self.cost

    def check_validity(self):
        self.calculate_cost()

        coverage_ok, _, _, _, _ = self.coverage_status()
        if not coverage_ok:
            return False

        nonempty_routes = [route for route in self.routes if route["path"]]

        # Famous Barreto-style instance controls loaded from Sheet2.
        if len(self.open_depots) != self.instance.required_open_depots:
            return False

        if len(nonempty_routes) > self.instance.max_vehicles:
            return False

        depot_loads = {depot: 0.0 for depot in self.open_depots}

        for route in nonempty_routes:
            depot = route["depot"]

            if depot < 0 or depot >= self.instance.num_depots:
                return False

            expected_load = float(np.sum(self.instance.demands[route["path"]]))
            if abs(expected_load - float(route["load"])) > 1e-4:
                return False

            depot_loads[depot] += float(route["load"])

        for depot, load in depot_loads.items():
            if load > self.instance.get_depot_capacity(depot) + 1e-4:
                return False

        for route in nonempty_routes:
            if route["load"] > self.instance.vehicle_cap + 1e-4:
                return False

            if route.get("duration_seconds", route.get("dist", float("inf"))) > self.instance.max_route_dist + 1e-4:
                return False

        return True


@njit
def evaluate_route_components_fast(
    depot_idx,
    path_arr,
    financial_cost_matrix,
    duration_matrix,
    num_depots,
    priorities,
):
    """
    Closed-route evaluation used by the pharmaceutical case-study model.

    Returns
    -------
    financial_travel_cost:
        Sum of distance-based arc costs, including the return to the depot.
    priority_weighted_latency:
        Sum of priority-weighted customer arrival times, measured with the
        real duration matrix. The return leg is not part of customer latency.
    route_duration:
        Total depot-to-depot route duration, including the return leg.
    """
    if len(path_arr) == 0:
        return 0.0, 0.0, 0.0

    financial_cost = 0.0
    latency = 0.0
    elapsed = 0.0
    previous = depot_idx

    for k in range(len(path_arr)):
        customer_global = num_depots + path_arr[k]

        financial_cost += financial_cost_matrix[previous][customer_global]

        arc_duration = duration_matrix[previous][customer_global]
        elapsed += arc_duration
        latency += elapsed * priorities[path_arr[k]]

        previous = customer_global

    financial_cost += financial_cost_matrix[previous][depot_idx]
    elapsed += duration_matrix[previous][depot_idx]

    return financial_cost, latency, elapsed


def evaluate_route(depot_idx, path, instance):
    path_arr = np.asarray(path, dtype=np.int64)
    financial_cost, latency, route_duration = evaluate_route_components_fast(
        depot_idx,
        path_arr,
        instance.financial_cost_matrix,
        instance.duration_matrix,
        instance.num_depots,
        instance.priorities,
    )
    return float(financial_cost), float(latency)


def evaluate_route_duration(depot_idx, path, instance):
    path_arr = np.asarray(path, dtype=np.int64)
    _, _, route_duration = evaluate_route_components_fast(
        depot_idx,
        path_arr,
        instance.financial_cost_matrix,
        instance.duration_matrix,
        instance.num_depots,
        instance.priorities,
    )
    return float(route_duration)


def get_route_obj(depot_idx, path, instance):
    """
    Route contribution to the manuscript objective:
        distance-based financial travel cost
        + alpha * priority-weighted arrival-time latency.
    """
    financial_cost, latency = evaluate_route(depot_idx, path, instance)
    return evaluate_route_search_objective(
        financial_cost, latency, instance
    )

@njit
def evaluate_route_swap_delta(depot_idx, path_arr, i, new_node, matrix, num_depots, priorities):
    """
    Evaluates the route if path_arr[i] were replaced by new_node,
    without copying the array or re-calculating the prefix before index i.
    """
    dist = 0.0
    latency = 0.0
    curr_time = 0.0

    # 1. Process the unchanged prefix before index i
    for k in range(i):
        u = depot_idx if k == 0 else num_depots + path_arr[k-1]
        v = num_depots + path_arr[k]
        t = matrix[u][v]
        curr_time += t
        dist += t
        latency += curr_time * priorities[path_arr[k]]

    # 2. Process the substituted node at index i
    u_swap = depot_idx if i == 0 else num_depots + path_arr[i-1]
    v_swap = num_depots + new_node
    t_swap = matrix[u_swap][v_swap]
    curr_time += t_swap
    dist += t_swap
    latency += curr_time * priorities[new_node]

    # 3. Process the suffix after index i (with shifted arrival times)
    for k in range(i + 1, len(path_arr)):
        u = num_depots + (new_node if k == i + 1 else path_arr[k-1])
        v = num_depots + path_arr[k]
        t = matrix[u][v]
        curr_time += t
        dist += t
        latency += curr_time * priorities[path_arr[k]]

    # 4. Complete the return leg to the depot
    last_node = num_depots + (new_node if i == len(path_arr) - 1 else path_arr[-1])
    dist += matrix[last_node][depot_idx]

    return dist, latency

@njit
def evaluate_route_remove_delta(depot_idx, path_arr, idx_to_remove, matrix, num_depots, priorities):
    """
    Evaluates a route after removing the element at path_arr[idx_to_remove]
    without copying arrays or recreating lists.
    """
    if len(path_arr) <= 1:
        return 0.0, 0.0

    dist = 0.0
    latency = 0.0
    curr_time = 0.0

    # Process unchanged prefix
    for k in range(idx_to_remove):
        u = depot_idx if k == 0 else num_depots + path_arr[k-1]
        v = num_depots + path_arr[k]
        t = matrix[u][v]
        curr_time += t
        dist += t
        latency += curr_time * priorities[path_arr[k]]

    # Bridge the gap left by the removed node
    if idx_to_remove < len(path_arr) - 1:
        u_bridge = depot_idx if idx_to_remove == 0 else num_depots + path_arr[idx_to_remove - 1]
        v_bridge = num_depots + path_arr[idx_to_remove + 1]
        t_bridge = matrix[u_bridge][v_bridge]
        curr_time += t_bridge
        dist += t_bridge
        latency += curr_time * priorities[path_arr[idx_to_remove + 1]]

        # Process the remaining suffix
        for k in range(idx_to_remove + 2, len(path_arr)):
            u = num_depots + path_arr[k-1]
            v = num_depots + path_arr[k]
            t = matrix[u][v]
            curr_time += t
            dist += t
            latency += curr_time * priorities[path_arr[k]]

    # Return to depot leg
    if idx_to_remove == len(path_arr) - 1:
        last_node = depot_idx if len(path_arr) == 1 else num_depots + path_arr[len(path_arr) - 2]
    else:
        last_node = num_depots + path_arr[-1]

    if last_node != depot_idx:
        dist += matrix[last_node][depot_idx]

    return dist, latency

@njit
def evaluate_route_insert_delta(depot_idx, path_arr, insert_idx, new_node, matrix, num_depots, priorities):
    """
    Evaluates a route after inserting new_node at position insert_idx
    into path_arr without allocating new array memory inside loops.
    """
    dist = 0.0
    latency = 0.0
    curr_time = 0.0

    # 1. Process prefix up to insertion point
    for k in range(insert_idx):
        u = depot_idx if k == 0 else num_depots + path_arr[k-1]
        v = num_depots + path_arr[k]
        t = matrix[u][v]
        curr_time += t
        dist += t
        latency += curr_time * priorities[path_arr[k]]

    # 2. Insert the relocated node
    u_ins = depot_idx if insert_idx == 0 else num_depots + path_arr[insert_idx - 1]
    v_ins = num_depots + new_node
    t_ins = matrix[u_ins][v_ins]
    curr_time += t_ins
    dist += t_ins
    latency += curr_time * priorities[new_node]

    # 3. Process the shifted remaining nodes
    if insert_idx < len(path_arr):
        # First shifted node connection
        u_shift = num_depots + new_node
        v_shift = num_depots + path_arr[insert_idx]
        t_shift = matrix[u_shift][v_shift]
        curr_time += t_shift
        dist += t_shift
        latency += curr_time * priorities[path_arr[insert_idx]]

        # Suffix handling
        for k in range(insert_idx + 1, len(path_arr)):
            u = num_depots + path_arr[k-1]
            v = num_depots + path_arr[k]
            t = matrix[u][v]
            curr_time += t
            dist += t
            latency += curr_time * priorities[path_arr[k]]

    # 4. Return to depot
    last_node = num_depots + (new_node if insert_idx == len(path_arr) else path_arr[-1])
    dist += matrix[last_node][depot_idx]

    return dist, latency

def ECWA(instance, allowed_depots, rcl_size=5):
    """
    Stochastic Extended Clarke & Wright Heuristic (ECWA)
    with 4-way orientation merging, incremental distance tracking,
    integrated depot reassignment, and strict feasibility initialization.
    """

    allowed_depots = sorted(set(int(d) for d in allowed_depots))

    if len(allowed_depots) != instance.required_open_depots:
        sol = Solution(instance)
        sol.cost = float("inf")
        return sol

    if any(d < 0 or d >= instance.num_depots for d in allowed_depots):
        sol = Solution(instance)
        sol.cost = float("inf")
        return sol

    sol = Solution(instance)
    depot_loads = {d: 0 for d in range(instance.num_depots)}

    # ============================================================
    # PHASE 1: CAPACITY-AWARE INITIALIZATION
    # Build trivial 1-customer round trips. Customers that have fewer
    # feasible depot alternatives are assigned first to avoid filling
    # central depots before geographically difficult customers are served.
    # ============================================================
    customer_candidates = []

    for j in range(instance.num_customers):
        demand = instance.demands[j]
        cust_node = instance.num_depots + j

        if demand > instance.vehicle_cap or demand > max(instance.depot_capacities):
            sol.cost = float("inf")
            return sol

        feasible_options = []
        for d in allowed_depots:
            # Google road times can be asymmetric; therefore, use the actual
            # depot-customer-depot duration rather than twice the outbound leg.
            round_trip = (
                instance.get_dist(d, cust_node)
                + instance.get_dist(cust_node, d)
            )
            if (
                round_trip <= instance.max_route_dist and
                demand <= instance.get_depot_capacity(d)
            ):
                feasible_options.append((d, round_trip))

        if not feasible_options:
            sol.cost = float("inf")
            return sol

        customer_candidates.append((
            len(feasible_options),
            -demand,
            -min(value for _, value in feasible_options),
            j,
            feasible_options,
        ))

    customer_candidates.sort()

    for _, _, _, j, feasible_options in customer_candidates:
        demand = instance.demands[j]
        best_depot = -1
        best_score = float("inf")
        best_round_trip = float("inf")

        for d, round_trip in feasible_options:
            capacity = instance.get_depot_capacity(d)
            new_load = depot_loads[d] + demand
            if new_load > capacity:
                continue

            # Mild utilization penalty prevents premature saturation of the
            # geographically central depot while retaining travel proximity.
            utilization = new_load / max(capacity, 1e-9)
            score = round_trip * (1.0 + 0.35 * utilization * utilization)

            if score < best_score:
                best_score = score
                best_depot = d
                best_round_trip = round_trip

        if best_depot == -1:
            sol.cost = float("inf")
            return sol

        depot_loads[best_depot] += demand
        sol.routes.append({
            "depot": best_depot,
            "path": [j],
            "load": demand,
            "dist": best_round_trip,
            "internal_dist": 0.0
        })

    # ============================================================
    # PHASE 2: INTEGRATED ROUTE MERGING & DEPOT REASSIGNMENT
    # 4-way orientation merge
    # ============================================================
    while True:
        savings = []
        n_routes = len(sol.routes)

        for i in range(n_routes):
            for j in range(n_routes):
                if i == j:
                    continue

                r1 = sol.routes[i]
                r2 = sol.routes[j]

                if r1["load"] + r2["load"] > instance.vehicle_cap:
                    continue

                d1 = r1["depot"]
                d2 = r2["depot"]

                r1_nodes = [instance.num_depots + n for n in r1["path"]]
                r2_nodes = [instance.num_depots + n for n in r2["path"]]

                r1_start = r1_nodes[0]
                r1_end = r1_nodes[-1]
                r2_start = r2_nodes[0]
                r2_end = r2_nodes[-1]

                orientations = [
                    (
                        instance.get_dist(r1_end, r2_start),
                        r1["path"] + r2["path"],
                        r1["internal_dist"] + r2["internal_dist"]
                    ),
                    (
                        instance.get_dist(r1_start, r2_start),
                        list(reversed(r1["path"])) + r2["path"],
                        r1["internal_dist"] + r2["internal_dist"]
                    ),
                    (
                        instance.get_dist(r1_end, r2_end),
                        r1["path"] + list(reversed(r2["path"])),
                        r1["internal_dist"] + r2["internal_dist"]
                    ),
                    (
                        instance.get_dist(r1_start, r2_end),
                        list(reversed(r1["path"])) + list(reversed(r2["path"])),
                        r1["internal_dist"] + r2["internal_dist"]
                    )
                ]

                best_merge_saving = -float("inf")
                best_target_depot = -1
                best_total_new_dist = float("inf")
                best_chosen_path = None
                best_chosen_internal = 0.0

                for target_d in (d1, d2):

                    if (
                        target_d == d1 and
                        d1 != d2 and
                        depot_loads[d1] + r2["load"] > instance.get_depot_capacity(d1)
                    ):
                        continue

                    if (
                        target_d == d2 and
                        d1 != d2 and
                        depot_loads[d2] + r1["load"] > instance.get_depot_capacity(d2)
                    ):
                        continue

                    for link_dist, combined_path, base_internal in orientations:
                        new_start = instance.num_depots + combined_path[0]
                        new_end = instance.num_depots + combined_path[-1]

                        dist_endpoints = (
                            instance.get_dist(target_d, new_start) +
                            instance.get_dist(new_end, target_d)
                        )

                        total_internal_dist = base_internal + link_dist
                        total_new_dist = dist_endpoints + total_internal_dist

                        if total_new_dist > instance.max_route_dist:
                            continue

                        saving = r1["dist"] + r2["dist"] - total_new_dist

                        if saving > best_merge_saving:
                            best_merge_saving = saving
                            best_target_depot = target_d
                            best_total_new_dist = total_new_dist
                            best_chosen_path = combined_path
                            best_chosen_internal = total_internal_dist

                must_reduce_fleet = n_routes > instance.max_vehicles

                if (
                    best_target_depot != -1 and
                    (best_merge_saving > 0 or must_reduce_fleet)
                ):
                    savings.append({
                        "i": i,
                        "j": j,
                        "saving": best_merge_saving,
                        "new_depot": best_target_depot,
                        "new_dist": best_total_new_dist,
                        "new_path": best_chosen_path,
                        "new_internal": best_chosen_internal
                    })

        if not savings:
            break

        savings.sort(key=lambda x: x["saving"], reverse=True)
        actual_rcl = min(len(savings), rcl_size)
        candidate = random.choice(savings[:actual_rcl])

        idx1 = candidate["i"]
        idx2 = candidate["j"]
        target_depot = candidate["new_depot"]

        r1 = sol.routes[idx1]
        r2 = sol.routes[idx2]

        depot_loads[r1["depot"]] -= r1["load"]
        depot_loads[r2["depot"]] -= r2["load"]
        depot_loads[target_depot] += r1["load"] + r2["load"]

        new_route = {
            "depot": target_depot,
            "path": candidate["new_path"],
            "load": r1["load"] + r2["load"],
            "dist": candidate["new_dist"],
            "internal_dist": candidate["new_internal"]
        }

        for index in sorted([idx1, idx2], reverse=True):
            sol.routes.pop(index)

        sol.routes.append(new_route)

    sol.calculate_cost()
    return sol



# =============================================================================
# VND_ZOU WITH STRATEGIC OSCILLATION
# Adapted from Zou et al. (2024) to the manuscript objective.
#
# Preserved case-study constraints:
#   - customer coverage,
#   - depot capacity,
#   - maximum route duration,
#   - fixed depot and vehicle costs.
#
# Strategic oscillation permits temporary VEHICLE-capacity violations only.
# Depot-capacity and route-duration constraints remain strict.
# =============================================================================

SO_WINDOW = 4
SO_EPS = 1e-4
SO_BETA_MIN = 1e-6
SO_BETA_MAX = 1e9
TCHEBY_MOVE_EPS = 1e-8


def _is_tcheby_so(instance):
    return str(
        getattr(instance, "objective_mode", "legacy")
    ).strip().lower() == "tchebycheff"


def _so_move_eps(instance):
    return TCHEBY_MOVE_EPS if _is_tcheby_so(instance) else SO_EPS


def _path_load_so(path, instance):
    return float(sum(instance.demands[c] for c in path))


def _update_route_so(route, new_path, instance):
    route["path"] = list(new_path)
    route["load"] = _path_load_so(route["path"], instance)
    financial_cost, latency = evaluate_route(
        route["depot"], route["path"], instance
    )
    route["financial_travel_cost"] = financial_cost
    route["latency"] = latency
    route["dist"] = evaluate_route_duration(
        route["depot"], route["path"], instance
    )
    route["duration_seconds"] = route["dist"]


def _vehicle_capacity_violation_so(load, instance):
    return max(0.0, float(load) - float(instance.vehicle_cap))


def capacity_violation_so(sol, instance):
    return sum(
        _vehicle_capacity_violation_so(r["load"], instance)
        for r in sol.routes if r["path"]
    )


def _route_penalized_value_so(depot, path, load, instance, beta):
    """
    Route contribution to the strategic-oscillation objective.

    Vehicle-capacity violations are penalized, while maximum route duration
    remains a hard constraint. Fixed depot/vehicle costs are omitted here
    because all implemented moves preserve the number of open depots and
    nonempty routes.
    """
    if not path:
        return float("inf")

    financial_cost, latency = evaluate_route(depot, path, instance)
    route_duration = evaluate_route_duration(depot, path, instance)

    if route_duration > instance.max_route_dist + SO_EPS:
        return float("inf")

    route_objective = evaluate_route_search_objective(
        financial_cost, latency, instance
    )
    return route_objective + beta * _vehicle_capacity_violation_so(load, instance)



def _tcheby_candidate_value_so(sol, instance, beta, replacements):
    """Global Tcheby value after replacing affected routes."""
    sol.calculate_cost()

    C = float(sol.financial_cost)
    P = float(sol.equity_penalty)
    violation = capacity_violation_so(sol, instance)

    for ridx, (new_depot, new_path) in replacements.items():
        old = sol.routes[ridx]

        old_C, old_P = evaluate_route(
            old["depot"], old["path"], instance
        )

        if evaluate_route_duration(
            new_depot, new_path, instance
        ) > instance.max_route_dist + SO_EPS:
            return float("inf")

        new_C, new_P = evaluate_route(
            new_depot, new_path, instance
        )

        C += new_C - old_C
        P += new_P - old_P

        violation -= _vehicle_capacity_violation_so(
            old["load"], instance
        )
        violation += _vehicle_capacity_violation_so(
            _path_load_so(new_path, instance), instance
        )

    return (
        evaluate_scalar_objective(C, P, instance)
        + beta * violation
    )


def _depot_loads_so(sol, instance):
    loads = {d: 0.0 for d in range(instance.num_depots)}
    for route in sol.routes:
        if route["path"]:
            loads[route["depot"]] += float(route["load"])
    return loads


def _depot_capacity_ok_so(base_loads, changes, instance):
    """
    Checks depot capacity after replacing one or more route assignments.

    Each change is:
        (old_depot, old_load, new_depot, new_load)
    """
    loads = dict(base_loads)

    for old_depot, old_load, new_depot, new_load in changes:
        loads[old_depot] = loads.get(old_depot, 0.0) - float(old_load)
        loads[new_depot] = loads.get(new_depot, 0.0) + float(new_load)

    return all(
        load <= instance.get_depot_capacity(depot) + SO_EPS and load >= -SO_EPS
        for depot, load in loads.items()
    )


def initial_beta_so(sol, instance):
    total_demand = float(np.sum(instance.demands))
    sol.calculate_cost()

    if total_demand <= 0 or not np.isfinite(sol.cost):
        return 1.0

    if _is_tcheby_so(instance):
        return max(
            SO_BETA_MIN,
            abs(float(sol.cost)) / total_demand
        )

    return max(1.0, float(sol.cost) / total_demand)


def depot_swap_perturbation_so(sol, instance, beta):
    """
    Replaces one open depot with one closed depot and reassigns all routes
    of the removed depot to the new depot.
    """
    sol.routes = [r for r in sol.routes if r["path"]]
    sol.calculate_cost()

    open_depots = list(sol.open_depots)
    closed_depots = [
        d for d in range(instance.num_depots)
        if d not in sol.open_depots
    ]

    if not open_depots or not closed_depots:
        return False

    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        current_global = (
            sol.cost + beta * capacity_violation_so(sol, instance)
        )
        best_value = current_global

    best_move = None
    best_delta = 0.0

    for old_depot in open_depots:
        affected = [
            idx for idx, route in enumerate(sol.routes)
            if route["depot"] == old_depot
        ]

        if not affected:
            continue

        total_load = sum(sol.routes[idx]["load"] for idx in affected)

        for new_depot in closed_depots:
            if depot_loads.get(new_depot, 0.0) + total_load > instance.get_depot_capacity(new_depot) + SO_EPS:
                continue

            if tcheby:
                replacements = {
                    ridx: (
                        new_depot,
                        sol.routes[ridx]["path"]
                    )
                    for ridx in affected
                }

                candidate = _tcheby_candidate_value_so(
                    sol, instance, beta, replacements
                )

                if candidate < best_value - _so_move_eps(instance):
                    best_value = candidate
                    best_move = (affected, new_depot)

                continue

            old_local = 0.0
            new_local = 0.0
            feasible_move = True

            for ridx in affected:
                route = sol.routes[ridx]

                old_value = _route_penalized_value_so(
                    old_depot, route["path"], route["load"], instance, beta
                )
                new_value = _route_penalized_value_so(
                    new_depot, route["path"], route["load"], instance, beta
                )

                if not np.isfinite(new_value):
                    feasible_move = False
                    break

                old_local += old_value
                new_local += new_value

            if not feasible_move:
                continue

            delta = new_local - old_local

            if delta < best_delta - SO_EPS:
                best_delta = delta
                best_move = (affected, new_depot)

    if best_move is None:
        return False

    affected, new_depot = best_move

    for ridx in affected:
        sol.routes[ridx]["depot"] = new_depot
        _update_route_so(sol.routes[ridx], sol.routes[ridx]["path"], instance)

    sol.calculate_cost()
    return True


def apply_insertion_so(sol, instance, beta):
    """Relocates one customer within or between routes."""
    sol.routes = [r for r in sol.routes if r["path"]]
    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        sol.calculate_cost()
        current_global = (
            sol.cost + beta * capacity_violation_so(sol, instance)
        )

    for r1_idx, r1 in enumerate(sol.routes):
        for i, node in enumerate(r1["path"]):
            node_load = float(instance.demands[node])

            for r2_idx, r2 in enumerate(sol.routes):

                # Intra-route relocate
                if r1_idx == r2_idx:
                    old_local = None
                    if not tcheby:
                        old_local = _route_penalized_value_so(
                            r1["depot"], r1["path"], r1["load"], instance, beta
                        )

                    removed = r1["path"][:i] + r1["path"][i + 1:]

                    for pos in range(len(removed) + 1):
                        if pos == i:
                            continue

                        new_path = removed[:pos] + [node] + removed[pos:]
                        if tcheby:
                            new_local = _tcheby_candidate_value_so(
                                sol, instance, beta,
                                {r1_idx: (r1["depot"], new_path)}
                            )
                            accept = new_local < current_global - _so_move_eps(instance)
                        else:
                            new_local = _route_penalized_value_so(
                                r1["depot"], new_path, r1["load"], instance, beta
                            )
                            accept = new_local < old_local - SO_EPS

                        if accept:
                            _update_route_so(r1, new_path, instance)
                            sol.calculate_cost()
                            return True

                # Inter-route relocate
                else:
                    if len(r1["path"]) <= 1:
                        continue

                    new_r1_path = r1["path"][:i] + r1["path"][i + 1:]
                    new_r1_load = r1["load"] - node_load

                    old_local = None
                    if not tcheby:
                        old_local = (
                            _route_penalized_value_so(
                                r1["depot"], r1["path"], r1["load"], instance, beta
                            )
                            + _route_penalized_value_so(
                                r2["depot"], r2["path"], r2["load"], instance, beta
                            )
                        )

                    for pos in range(len(r2["path"]) + 1):
                        new_r2_path = r2["path"][:pos] + [node] + r2["path"][pos:]
                        new_r2_load = r2["load"] + node_load

                        if not _depot_capacity_ok_so(
                            depot_loads,
                            [
                                (r1["depot"], r1["load"], r1["depot"], new_r1_load),
                                (r2["depot"], r2["load"], r2["depot"], new_r2_load),
                            ],
                            instance,
                        ):
                            continue

                        if tcheby:
                            new_local = _tcheby_candidate_value_so(
                                sol, instance, beta,
                                {
                                    r1_idx: (r1["depot"], new_r1_path),
                                    r2_idx: (r2["depot"], new_r2_path),
                                }
                            )
                            accept = new_local < current_global - _so_move_eps(instance)
                        else:
                            new_local = (
                                _route_penalized_value_so(
                                    r1["depot"], new_r1_path, new_r1_load, instance, beta
                                )
                                + _route_penalized_value_so(
                                    r2["depot"], new_r2_path, new_r2_load, instance, beta
                                )
                            )
                            accept = new_local < old_local - SO_EPS

                        if accept:
                            _update_route_so(r1, new_r1_path, instance)
                            _update_route_so(r2, new_r2_path, instance)
                            sol.calculate_cost()
                            return True

    return False


def apply_swap_so(sol, instance, beta):
    """
    Swap neighborhood:
      1) depot--depot swap between routes,
      2) customer--customer swap within or between routes.
    """
    sol.routes = [r for r in sol.routes if r["path"]]
    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        sol.calculate_cost()
        current_global = (
            sol.cost + beta * capacity_violation_so(sol, instance)
        )

    # 1) Depot--depot swap
    for r1_idx, r1 in enumerate(sol.routes):
        for r2_idx in range(r1_idx + 1, len(sol.routes)):
            r2 = sol.routes[r2_idx]

            if r1["depot"] == r2["depot"]:
                continue

            if not _depot_capacity_ok_so(
                depot_loads,
                [
                    (r1["depot"], r1["load"], r2["depot"], r1["load"]),
                    (r2["depot"], r2["load"], r1["depot"], r2["load"]),
                ],
                instance,
            ):
                continue

            if tcheby:
                new_local = _tcheby_candidate_value_so(
                    sol, instance, beta,
                    {
                        r1_idx: (r2["depot"], r1["path"]),
                        r2_idx: (r1["depot"], r2["path"]),
                    }
                )
                accept = new_local < current_global - _so_move_eps(instance)
            else:
                old_local = (
                    _route_penalized_value_so(
                        r1["depot"], r1["path"], r1["load"], instance, beta
                    )
                    + _route_penalized_value_so(
                        r2["depot"], r2["path"], r2["load"], instance, beta
                    )
                )
                new_local = (
                    _route_penalized_value_so(
                        r2["depot"], r1["path"], r1["load"], instance, beta
                    )
                    + _route_penalized_value_so(
                        r1["depot"], r2["path"], r2["load"], instance, beta
                    )
                )
                accept = new_local < old_local - SO_EPS

            if accept:
                r1["depot"], r2["depot"] = r2["depot"], r1["depot"]
                _update_route_so(r1, r1["path"], instance)
                _update_route_so(r2, r2["path"], instance)
                sol.calculate_cost()
                return True

    # 2) Customer--customer swap
    for r1_idx, r1 in enumerate(sol.routes):
        for r2_idx in range(r1_idx, len(sol.routes)):
            r2 = sol.routes[r2_idx]

            for i in range(len(r1["path"])):
                for j in range(len(r2["path"])):

                    # Intra-route
                    if r1_idx == r2_idx:
                        if i >= j:
                            continue

                        new_path = list(r1["path"])
                        new_path[i], new_path[j] = new_path[j], new_path[i]

                        if tcheby:
                            new_local = _tcheby_candidate_value_so(
                                sol, instance, beta,
                                {r1_idx: (r1["depot"], new_path)}
                            )
                            accept = new_local < current_global - _so_move_eps(instance)
                        else:
                            old_local = _route_penalized_value_so(
                                r1["depot"], r1["path"], r1["load"], instance, beta
                            )
                            new_local = _route_penalized_value_so(
                                r1["depot"], new_path, r1["load"], instance, beta
                            )
                            accept = new_local < old_local - SO_EPS

                        if accept:
                            _update_route_so(r1, new_path, instance)
                            sol.calculate_cost()
                            return True

                    # Inter-route
                    else:
                        n1 = r1["path"][i]
                        n2 = r2["path"][j]

                        new_r1_path = list(r1["path"])
                        new_r2_path = list(r2["path"])
                        new_r1_path[i] = n2
                        new_r2_path[j] = n1

                        new_r1_load = (
                            r1["load"] - instance.demands[n1] + instance.demands[n2]
                        )
                        new_r2_load = (
                            r2["load"] - instance.demands[n2] + instance.demands[n1]
                        )

                        if not _depot_capacity_ok_so(
                            depot_loads,
                            [
                                (r1["depot"], r1["load"], r1["depot"], new_r1_load),
                                (r2["depot"], r2["load"], r2["depot"], new_r2_load),
                            ],
                            instance,
                        ):
                            continue

                        if tcheby:
                            new_local = _tcheby_candidate_value_so(
                                sol, instance, beta,
                                {
                                    r1_idx: (r1["depot"], new_r1_path),
                                    r2_idx: (r2["depot"], new_r2_path),
                                }
                            )
                            accept = new_local < current_global - _so_move_eps(instance)
                        else:
                            old_local = (
                                _route_penalized_value_so(
                                    r1["depot"], r1["path"], r1["load"], instance, beta
                                )
                                + _route_penalized_value_so(
                                    r2["depot"], r2["path"], r2["load"], instance, beta
                                )
                            )
                            new_local = (
                                _route_penalized_value_so(
                                    r1["depot"], new_r1_path, new_r1_load, instance, beta
                                )
                                + _route_penalized_value_so(
                                    r2["depot"], new_r2_path, new_r2_load, instance, beta
                                )
                            )
                            accept = new_local < old_local - SO_EPS

                        if accept:
                            _update_route_so(r1, new_r1_path, instance)
                            _update_route_so(r2, new_r2_path, instance)
                            sol.calculate_cost()
                            return True

    return False


def apply_2opt_so(sol, instance, beta):
    """Applies intra-route 2-opt and inter-route 2-opt*."""
    sol.routes = [r for r in sol.routes if r["path"]]
    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        sol.calculate_cost()
        current_global = sol.cost + beta * capacity_violation_so(sol, instance)

    # Intra-route 2-opt
    for route_idx, route in enumerate(sol.routes):
        if len(route["path"]) < 3:
            continue

        old_local = None
        if not tcheby:
            old_local = _route_penalized_value_so(
                route["depot"], route["path"], route["load"], instance, beta
            )

        for i in range(len(route["path"]) - 1):
            for j in range(i + 2, len(route["path"]) + 1):
                new_path = (
                    route["path"][:i]
                    + route["path"][i:j][::-1]
                    + route["path"][j:]
                )

                if tcheby:
                    new_local = _tcheby_candidate_value_so(
                        sol, instance, beta,
                        {route_idx: (route["depot"], new_path)}
                    )
                    accept = new_local < current_global - _so_move_eps(instance)
                else:
                    new_local = _route_penalized_value_so(
                        route["depot"], new_path, route["load"], instance, beta
                    )
                    accept = new_local < old_local - SO_EPS

                if accept:
                    _update_route_so(route, new_path, instance)
                    sol.calculate_cost()
                    return True

    # Inter-route 2-opt*
    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 2:
            continue

        for r2_idx in range(r1_idx + 1, len(sol.routes)):
            r2 = sol.routes[r2_idx]

            if len(r2["path"]) < 2:
                continue

            old_local = None
            if not tcheby:
                old_local = (
                    _route_penalized_value_so(
                        r1["depot"], r1["path"], r1["load"], instance, beta
                    )
                    + _route_penalized_value_so(
                        r2["depot"], r2["path"], r2["load"], instance, beta
                    )
                )

            for cut1 in range(1, len(r1["path"])):
                for cut2 in range(1, len(r2["path"])):
                    new_r1_path = r1["path"][:cut1] + r2["path"][cut2:]
                    new_r2_path = r2["path"][:cut2] + r1["path"][cut1:]

                    new_r1_load = _path_load_so(new_r1_path, instance)
                    new_r2_load = _path_load_so(new_r2_path, instance)

                    if not _depot_capacity_ok_so(
                        depot_loads,
                        [
                            (r1["depot"], r1["load"], r1["depot"], new_r1_load),
                            (r2["depot"], r2["load"], r2["depot"], new_r2_load),
                        ],
                        instance,
                    ):
                        continue

                    if tcheby:
                        new_local = _tcheby_candidate_value_so(
                            sol, instance, beta,
                            {
                                r1_idx: (r1["depot"], new_r1_path),
                                r2_idx: (r2["depot"], new_r2_path),
                            }
                        )
                        accept = new_local < current_global - _so_move_eps(instance)
                    else:
                        new_local = (
                            _route_penalized_value_so(
                                r1["depot"], new_r1_path, new_r1_load, instance, beta
                            )
                            + _route_penalized_value_so(
                                r2["depot"], new_r2_path, new_r2_load, instance, beta
                            )
                        )
                        accept = new_local < old_local - SO_EPS

                    if accept:
                        _update_route_so(r1, new_r1_path, instance)
                        _update_route_so(r2, new_r2_path, instance)
                        sol.calculate_cost()
                        return True

    return False


def apply_2relocate_so(sol, instance, beta):
    """Relocates two consecutive customers within or between routes."""
    sol.routes = [r for r in sol.routes if r["path"]]
    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        sol.calculate_cost()
        current_global = sol.cost + beta * capacity_violation_so(sol, instance)

    # Intra-route 2-relocate
    for route_idx, route in enumerate(sol.routes):
        if len(route["path"]) < 3:
            continue

        old_local = None
        if not tcheby:
            old_local = _route_penalized_value_so(
                route["depot"], route["path"], route["load"], instance, beta
            )

        for i in range(len(route["path"]) - 1):
            block = route["path"][i:i + 2]
            remaining = route["path"][:i] + route["path"][i + 2:]

            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue

                new_path = remaining[:pos] + block + remaining[pos:]

                if new_path == route["path"]:
                    continue

                if tcheby:
                    new_local = _tcheby_candidate_value_so(
                        sol, instance, beta,
                        {route_idx: (route["depot"], new_path)}
                    )
                    accept = new_local < current_global - _so_move_eps(instance)
                else:
                    new_local = _route_penalized_value_so(
                        route["depot"], new_path, route["load"], instance, beta
                    )
                    accept = new_local < old_local - SO_EPS

                if accept:
                    _update_route_so(route, new_path, instance)
                    sol.calculate_cost()
                    return True

    # Inter-route 2-relocate
    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 3:
            continue

        for r2_idx, r2 in enumerate(sol.routes):
            if r1_idx == r2_idx:
                continue

            old_local = None
            if not tcheby:
                old_local = (
                    _route_penalized_value_so(
                        r1["depot"], r1["path"], r1["load"], instance, beta
                    )
                    + _route_penalized_value_so(
                        r2["depot"], r2["path"], r2["load"], instance, beta
                    )
                )

            for i in range(len(r1["path"]) - 1):
                block = r1["path"][i:i + 2]
                block_load = float(sum(instance.demands[c] for c in block))

                new_r1_path = r1["path"][:i] + r1["path"][i + 2:]
                new_r1_load = r1["load"] - block_load

                if not new_r1_path:
                    continue

                for pos in range(len(r2["path"]) + 1):
                    new_r2_path = r2["path"][:pos] + block + r2["path"][pos:]
                    new_r2_load = r2["load"] + block_load

                    if not _depot_capacity_ok_so(
                        depot_loads,
                        [
                            (r1["depot"], r1["load"], r1["depot"], new_r1_load),
                            (r2["depot"], r2["load"], r2["depot"], new_r2_load),
                        ],
                        instance,
                    ):
                        continue

                    if tcheby:
                        new_local = _tcheby_candidate_value_so(
                            sol, instance, beta,
                            {
                                r1_idx: (r1["depot"], new_r1_path),
                                r2_idx: (r2["depot"], new_r2_path),
                            }
                        )
                        accept = new_local < current_global - _so_move_eps(instance)
                    else:
                        new_local = (
                            _route_penalized_value_so(
                                r1["depot"], new_r1_path, new_r1_load, instance, beta
                            )
                            + _route_penalized_value_so(
                                r2["depot"], new_r2_path, new_r2_load, instance, beta
                            )
                        )
                        accept = new_local < old_local - SO_EPS

                    if accept:
                        _update_route_so(r1, new_r1_path, instance)
                        _update_route_so(r2, new_r2_path, instance)
                        sol.calculate_cost()
                        return True

    return False


def apply_shift2_1_so(sol, instance, beta):
    """Exchanges one customer node with an arc of two consecutive customers."""
    sol.routes = [r for r in sol.routes if r["path"]]
    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        sol.calculate_cost()
        current_global = sol.cost + beta * capacity_violation_so(sol, instance)

    # Intra-route node--arc swap
    for route_idx, route in enumerate(sol.routes):
        path = route["path"]

        if len(path) < 3:
            continue

        old_local = None
        if not tcheby:
            old_local = _route_penalized_value_so(
                route["depot"], path, route["load"], instance, beta
            )

        for i in range(len(path)):
            single = path[i]

            for j in range(len(path) - 1):
                if i == j or i == j + 1:
                    continue

                arc = path[j:j + 2]

                if i < j:
                    new_path = (
                        path[:i] + arc + path[i + 1:j]
                        + [single] + path[j + 2:]
                    )
                else:
                    new_path = (
                        path[:j] + [single] + path[j + 2:i]
                        + arc + path[i + 1:]
                    )

                if new_path == path:
                    continue

                if tcheby:
                    new_local = _tcheby_candidate_value_so(
                        sol, instance, beta,
                        {route_idx: (route["depot"], new_path)}
                    )
                    accept = new_local < current_global - _so_move_eps(instance)
                else:
                    new_local = _route_penalized_value_so(
                        route["depot"], new_path, route["load"], instance, beta
                    )
                    accept = new_local < old_local - SO_EPS

                if accept:
                    _update_route_so(route, new_path, instance)
                    sol.calculate_cost()
                    return True

    # Inter-route node--arc swap
    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 2:
            continue

        for r2_idx, r2 in enumerate(sol.routes):
            if r1_idx == r2_idx or len(r2["path"]) < 1:
                continue

            old_local = None
            if not tcheby:
                old_local = (
                    _route_penalized_value_so(
                        r1["depot"], r1["path"], r1["load"], instance, beta
                    )
                    + _route_penalized_value_so(
                        r2["depot"], r2["path"], r2["load"], instance, beta
                    )
                )

            for i in range(len(r1["path"]) - 1):
                arc = r1["path"][i:i + 2]
                arc_load = float(sum(instance.demands[c] for c in arc))

                for j, single in enumerate(r2["path"]):
                    single_load = float(instance.demands[single])

                    new_r1_path = r1["path"][:i] + [single] + r1["path"][i + 2:]
                    new_r2_path = r2["path"][:j] + arc + r2["path"][j + 1:]
                    new_r1_load = r1["load"] - arc_load + single_load
                    new_r2_load = r2["load"] - single_load + arc_load

                    if not _depot_capacity_ok_so(
                        depot_loads,
                        [
                            (r1["depot"], r1["load"], r1["depot"], new_r1_load),
                            (r2["depot"], r2["load"], r2["depot"], new_r2_load),
                        ],
                        instance,
                    ):
                        continue

                    if tcheby:
                        new_local = _tcheby_candidate_value_so(
                            sol, instance, beta,
                            {
                                r1_idx: (r1["depot"], new_r1_path),
                                r2_idx: (r2["depot"], new_r2_path),
                            }
                        )
                        accept = new_local < current_global - _so_move_eps(instance)
                    else:
                        new_local = (
                            _route_penalized_value_so(
                                r1["depot"], new_r1_path, new_r1_load, instance, beta
                            )
                            + _route_penalized_value_so(
                                r2["depot"], new_r2_path, new_r2_load, instance, beta
                            )
                        )
                        accept = new_local < old_local - SO_EPS

                    if accept:
                        _update_route_so(r1, new_r1_path, instance)
                        _update_route_so(r2, new_r2_path, instance)
                        sol.calculate_cost()
                        return True

    return False


def apply_arc_swap_so(sol, instance, beta):
    """Exchanges two arcs of consecutive customers."""
    sol.routes = [r for r in sol.routes if r["path"]]
    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        sol.calculate_cost()
        current_global = sol.cost + beta * capacity_violation_so(sol, instance)

    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 2:
            continue

        for r2_idx in range(r1_idx, len(sol.routes)):
            r2 = sol.routes[r2_idx]

            if len(r2["path"]) < 2:
                continue

            for i in range(len(r1["path"]) - 1):
                block1 = r1["path"][i:i + 2]

                for j in range(len(r2["path"]) - 1):
                    block2 = r2["path"][j:j + 2]

                    # Intra-route
                    if r1_idx == r2_idx:
                        if abs(i - j) < 2:
                            continue

                        new_path = list(r1["path"])
                        new_path[i:i + 2] = block2
                        new_path[j:j + 2] = block1

                        if tcheby:
                            new_local = _tcheby_candidate_value_so(
                                sol, instance, beta,
                                {r1_idx: (r1["depot"], new_path)}
                            )
                            accept = new_local < current_global - _so_move_eps(instance)
                        else:
                            old_local = _route_penalized_value_so(
                                r1["depot"], r1["path"], r1["load"], instance, beta
                            )
                            new_local = _route_penalized_value_so(
                                r1["depot"], new_path, r1["load"], instance, beta
                            )
                            accept = new_local < old_local - SO_EPS

                        if accept:
                            _update_route_so(r1, new_path, instance)
                            sol.calculate_cost()
                            return True

                    # Inter-route
                    else:
                        new_r1_path = list(r1["path"])
                        new_r2_path = list(r2["path"])
                        new_r1_path[i:i + 2] = block2
                        new_r2_path[j:j + 2] = block1
                        new_r1_load = _path_load_so(new_r1_path, instance)
                        new_r2_load = _path_load_so(new_r2_path, instance)

                        if not _depot_capacity_ok_so(
                            depot_loads,
                            [
                                (r1["depot"], r1["load"], r1["depot"], new_r1_load),
                                (r2["depot"], r2["load"], r2["depot"], new_r2_load),
                            ],
                            instance,
                        ):
                            continue

                        if tcheby:
                            new_local = _tcheby_candidate_value_so(
                                sol, instance, beta,
                                {
                                    r1_idx: (r1["depot"], new_r1_path),
                                    r2_idx: (r2["depot"], new_r2_path),
                                }
                            )
                            accept = new_local < current_global - _so_move_eps(instance)
                        else:
                            old_local = (
                                _route_penalized_value_so(
                                    r1["depot"], r1["path"], r1["load"], instance, beta
                                )
                                + _route_penalized_value_so(
                                    r2["depot"], r2["path"], r2["load"], instance, beta
                                )
                            )
                            new_local = (
                                _route_penalized_value_so(
                                    r1["depot"], new_r1_path, new_r1_load, instance, beta
                                )
                                + _route_penalized_value_so(
                                    r2["depot"], new_r2_path, new_r2_load, instance, beta
                                )
                            )
                            accept = new_local < old_local - SO_EPS

                        if accept:
                            _update_route_so(r1, new_r1_path, instance)
                            _update_route_so(r2, new_r2_path, instance)
                            sol.calculate_cost()
                            return True

    return False


def apply_swapstar_so(sol, instance, beta):
    """
    Swap* removes one customer from each of two routes and reinserts both
    customers in their best positions in the opposite routes.
    """
    sol.routes = [r for r in sol.routes if r["path"]]
    depot_loads = _depot_loads_so(sol, instance)
    tcheby = _is_tcheby_so(instance)

    if tcheby:
        sol.calculate_cost()
        current_global = sol.cost + beta * capacity_violation_so(sol, instance)

    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 2:
            continue

        for r2_idx in range(r1_idx + 1, len(sol.routes)):
            r2 = sol.routes[r2_idx]

            if len(r2["path"]) < 2:
                continue

            if tcheby:
                best_value = current_global
            else:
                best_value = (
                    _route_penalized_value_so(
                        r1["depot"], r1["path"], r1["load"], instance, beta
                    )
                    + _route_penalized_value_so(
                        r2["depot"], r2["path"], r2["load"], instance, beta
                    )
                )
            best_move = None

            for i, c1 in enumerate(r1["path"]):
                load_c1 = float(instance.demands[c1])
                r1_removed = r1["path"][:i] + r1["path"][i + 1:]

                for j, c2 in enumerate(r2["path"]):
                    load_c2 = float(instance.demands[c2])
                    r2_removed = r2["path"][:j] + r2["path"][j + 1:]

                    if not r1_removed or not r2_removed:
                        continue

                    new_r1_load = r1["load"] - load_c1 + load_c2
                    new_r2_load = r2["load"] - load_c2 + load_c1

                    if not _depot_capacity_ok_so(
                        depot_loads,
                        [
                            (r1["depot"], r1["load"], r1["depot"], new_r1_load),
                            (r2["depot"], r2["load"], r2["depot"], new_r2_load),
                        ],
                        instance,
                    ):
                        continue

                    for pos1 in range(len(r1_removed) + 1):
                        new_r1_path = r1_removed[:pos1] + [c2] + r1_removed[pos1:]
                        if not tcheby:
                            value_r1 = _route_penalized_value_so(
                                r1["depot"], new_r1_path,
                                new_r1_load, instance, beta
                            )
                            if not np.isfinite(value_r1):
                                continue

                        for pos2 in range(len(r2_removed) + 1):
                            new_r2_path = r2_removed[:pos2] + [c1] + r2_removed[pos2:]
                            if tcheby:
                                new_value = _tcheby_candidate_value_so(
                                    sol, instance, beta,
                                    {
                                        r1_idx: (r1["depot"], new_r1_path),
                                        r2_idx: (r2["depot"], new_r2_path),
                                    }
                                )
                            else:
                                value_r2 = _route_penalized_value_so(
                                    r2["depot"], new_r2_path,
                                    new_r2_load, instance, beta
                                )
                                new_value = value_r1 + value_r2

                            if new_value < best_value - _so_move_eps(instance):
                                best_value = new_value
                                best_move = (new_r1_path, new_r2_path)

            if best_move is not None:
                _update_route_so(r1, best_move[0], instance)
                _update_route_so(r2, best_move[1], instance)
                sol.calculate_cost()
                return True

    return False


def _update_so_state(sol, instance, best_feasible, best_feasible_cost,
                     feasible_counter, infeasible_counter, beta):
    """Updates the feasible incumbent, oscillation counters, and beta."""
    if sol.check_validity():
        feasible_counter += 1
        infeasible_counter = 0

        if sol.cost < best_feasible_cost - SO_EPS:
            best_feasible = copy.deepcopy(sol)
            best_feasible_cost = sol.cost
    else:
        infeasible_counter += 1
        feasible_counter = 0

    if infeasible_counter >= SO_WINDOW:
        beta = min(SO_BETA_MAX, beta * random.choice([1.5, 2.5]))
        infeasible_counter = 0

    if feasible_counter >= SO_WINDOW:
        beta = max(SO_BETA_MIN, beta / random.choice([1.5, 2.5]))
        feasible_counter = 0

    return (
        best_feasible,
        best_feasible_cost,
        feasible_counter,
        infeasible_counter,
        beta,
    )


def vnd_local_search_so(sol, instance):
    """
    Seven-neighborhood VND adapted from Zou et al. (2024), with depot perturbation and strategic
    oscillation. Temporary vehicle-capacity violations are permitted through
    an adaptive penalty, while depot capacity and route duration stay strict.
    """
    if sol is None:
        return sol

    sol.routes = [r for r in sol.routes if r["path"]]
    sol.calculate_cost()

    if not np.isfinite(sol.cost):
        return sol

    beta = initial_beta_so(sol, instance)
    feasible_counter = 0
    infeasible_counter = 0

    if sol.check_validity():
        best_feasible = copy.deepcopy(sol)
        best_feasible_cost = sol.cost
    else:
        best_feasible = None
        best_feasible_cost = float("inf")

    neighborhoods = [
        ("relocate", apply_insertion_so),
        ("swap", apply_swap_so),
        ("2-opt", apply_2opt_so),
        ("2-relocate", apply_2relocate_so),
        ("node-arc-swap", apply_shift2_1_so),
        ("arc-arc-swap", apply_arc_swap_so),
        ("swapstar", apply_swapstar_so),
    ]

    while True:
        pass_improved = False

        if depot_swap_perturbation_so(sol, instance, beta):
            pass_improved = True
            (
                best_feasible,
                best_feasible_cost,
                feasible_counter,
                infeasible_counter,
                beta,
            ) = _update_so_state(
                sol,
                instance,
                best_feasible,
                best_feasible_cost,
                feasible_counter,
                infeasible_counter,
                beta,
            )

        for _, operator in neighborhoods:
            improved = True

            while improved:
                sol.routes = [r for r in sol.routes if r["path"]]
                sol.calculate_cost()
                improved = operator(sol, instance, beta)

                if improved:
                    pass_improved = True
                    (
                        best_feasible,
                        best_feasible_cost,
                        feasible_counter,
                        infeasible_counter,
                        beta,
                    ) = _update_so_state(
                        sol,
                        instance,
                        best_feasible,
                        best_feasible_cost,
                        feasible_counter,
                        infeasible_counter,
                        beta,
                    )

        if not pass_improved:
            break

    if best_feasible is not None:
        best_feasible.routes = [
            r for r in best_feasible.routes if r["path"]
        ]
        best_feasible.calculate_cost()
        return best_feasible

    sol.routes = [r for r in sol.routes if r["path"]]
    sol.calculate_cost()
    return sol


# Backward-compatible alias: all existing run, reporting, and plotting cells
# continue to call vnd_local_search without further structural changes.
vnd_local_search = vnd_local_search_so


def run_grasp_vnd_logged(instance, max_iter=50, alpha_max=5, run_id=1, seed=0, instance_name=""):
    """
    Executes GRASP-VNDZou-SO using exactly Ng open depots selected from the
    fixed candidate set recorded in Sheet3.
    """
    best_sol = None
    best_cost = float("inf")
    iteration_logs = []

    div_mode = True
    max_it_div = 5
    max_it_int = 3
    period_counter = 0

    best_depots_div = None
    best_cost_div = float("inf")

    for k in range(max_iter):
        if div_mode or not best_depots_div:
            allowed_depots = random.sample(
                range(instance.num_depots),
                instance.required_open_depots,
            )
        else:
            allowed_depots = sorted(best_depots_div)

            if len(allowed_depots) != instance.required_open_depots:
                allowed_depots = random.sample(
                    range(instance.num_depots),
                    instance.required_open_depots,
                )

        current_sol = ECWA(instance, allowed_depots, rcl_size=alpha_max)
        current_sol = vnd_local_search(current_sol, instance)

        feasible = current_sol.check_validity() if current_sol else False
        improved = False

        if feasible and current_sol.cost < best_cost:
            best_sol = copy.deepcopy(current_sol)
            best_cost = current_sol.cost
            improved = True
            print(
                f"[GRASP-VNDZou-SO] Run {run_id}, Iter {k}: "
                f"New Best Objective Cost = {best_cost:.2f}"
            )

        if div_mode and feasible and current_sol.cost < best_cost_div:
            best_cost_div = current_sol.cost
            best_depots_div = set(current_sol.open_depots)

        open_indices = sorted(current_sol.open_depots) if current_sol else []
        open_codes = [
            instance.depot_node_ids[index]
            for index in open_indices
        ]

        iteration_logs.append({
            "Instance": instance_name,
            "Algorithm": "GRASP-VNDZou-SO",
            "Run": run_id,
            "Seed": seed,
            "Iteration": k,
            "Mode": "Diversification" if div_mode else "Intensification",
            "Feasible": feasible,
            "Current Total Cost": current_sol.cost if (current_sol and feasible) else None,
            "Current Financial Cost": current_sol.financial_cost if (current_sol and feasible) else None,
            "Current Equity Penalty": current_sol.equity_penalty if (current_sol and feasible) else None,
            "Best Cost So Far": best_cost if best_cost != float("inf") else None,
            "Improved Best": improved,
            "Allowed Depot Indices": str(sorted(allowed_depots)),
            "Open Depot Indices": str(open_indices),
            "Open Depot Customer Codes": str(open_codes),
            "Number of Routes": len([
                route for route in current_sol.routes if route["path"]
            ]) if current_sol else None,
            "Required Open Depots (Ng)": instance.required_open_depots,
            "Maximum Vehicles (Nv)": instance.max_vehicles,
            "Coverage OK": current_sol.coverage_status()[0] if current_sol else False,
            "Num Served": current_sol.coverage_status()[1] if current_sol else None,
            "Num Unique Served": current_sol.coverage_status()[2] if current_sol else None,
            "Missing Customers": current_sol.coverage_status()[3] if current_sol else None,
            "Duplicate Customers": current_sol.coverage_status()[4] if current_sol else None,
        })

        period_counter += 1
        if div_mode and period_counter >= max_it_div:
            div_mode = False
            period_counter = 0
        elif not div_mode and period_counter >= max_it_int:
            div_mode = True
            period_counter = 0
            best_depots_div = None
            best_cost_div = float("inf")

    return best_sol, iteration_logs


# =========================================================================
# ROUTE EXTRACTOR FOR DETAILED SCHEDULING TABLES
# =========================================================================

def extract_routes(solution, instance_name, algorithm="GRASP-VNDZou-SO"):
    """Flattens the best solution into reporting rows."""
    rows = []
    if solution is None:
        return rows

    instance = solution.instance

    for idx, route in enumerate(solution.routes):
        if not route["path"]:
            continue

        depot_index = int(route["depot"])
        customer_indices = [int(node) for node in route["path"]]
        real_distance_m, real_duration_s = instance.route_real_metrics(
            depot_index, customer_indices
        )

        rows.append({
            "Instance": instance_name,
            "Algorithm": algorithm,
            "Route ID": idx + 1,
            "Depot Node Index": depot_index,
            "Candidate Depot ID": instance.depot_labels[depot_index],
            "Depot Customer Code": instance.depot_node_ids[depot_index],
            "Sequence Path": " -> ".join(map(str, customer_indices)),
            "Sequence Customer Codes": " -> ".join(
                instance.node_ids[node] for node in customer_indices
            ),
            "Number of Customers": len(customer_indices),
            "Total Vehicle Load": route["load"],
            "Objective Travel Metric": instance.travel_metric,
            "Objective Travel Value": route.get("dist", None),
            "Route Duration (seconds)": real_duration_s,
            "Route Duration (minutes)": real_duration_s / 60.0,
            "Route Distance (meters)": real_distance_m,
            "Route Distance (km)": real_distance_m / 1000.0,
        })

    return rows



class Instance:
    def __init__(
        self,
        num_depots,
        num_customers,
        vehicle_cap,
        max_route_dist,
        depot_capacities,
        required_open_depots,
        max_vehicles,
        fixed_depot_cost,
        fixed_vehicle_cost,
        cost_per_km,
        depot_coords,
        cust_coords,
        demands,
        priorities,
        objective_matrix,
        distance_matrix,
        duration_matrix,
        customer_node_ids,
        depot_node_ids,
        depot_labels,
        travel_metric="duration",
        equity_weight=1.0,
        objective_mode="legacy",
        tcheby_lambda=0.5,
        tcheby_rho=1e-4,
        anchor_c_min=None,
        anchor_c_at_p_min=None,
        anchor_p_min=None,
        anchor_p_at_c_min=None,
        city_name=None,
        source_files=None,
        parameter_map=None,
        route_limit_source=None,
        priority_profiles=None,
    ):
        self.num_depots = int(num_depots)
        self.num_candidate_depots = int(num_depots)
        self.num_customers = int(num_customers)
        self.vehicle_cap = float(vehicle_cap)
        self.max_route_dist = float(max_route_dist)
        self.depot_capacities = np.asarray(depot_capacities, dtype=np.float64)
        self.depot_cap = float(self.depot_capacities[0])  # backward compatibility
        self.required_open_depots = int(required_open_depots)
        self.max_vehicles = int(max_vehicles)
        self.fixed_depot_cost = float(fixed_depot_cost)
        self.fixed_vehicle_cost = float(fixed_vehicle_cost)
        self.cost_per_km = float(cost_per_km)
        self.equity_weight = float(equity_weight)

        # Multiobjective scalarization settings.
        self.objective_mode = str(objective_mode).strip().lower()
        self.tcheby_lambda = float(tcheby_lambda)
        self.tcheby_rho = float(tcheby_rho)

        self.anchor_c_min = (
            None if anchor_c_min is None else float(anchor_c_min)
        )
        self.anchor_c_at_p_min = (
            None if anchor_c_at_p_min is None else float(anchor_c_at_p_min)
        )
        self.anchor_p_min = (
            None if anchor_p_min is None else float(anchor_p_min)
        )
        self.anchor_p_at_c_min = (
            None if anchor_p_at_c_min is None else float(anchor_p_at_c_min)
        )

        self.depot_coords = np.asarray(depot_coords, dtype=np.float64)
        self.cust_coords = np.asarray(cust_coords, dtype=np.float64)
        self.demands = np.asarray(demands, dtype=np.float64)
        self.priorities = np.asarray(priorities, dtype=np.float64)

        self.priority_profiles = {
            str(name): np.asarray(values, dtype=np.float64).copy()
            for name, values in dict(priority_profiles or {}).items()
        }

        self.financial_cost_matrix = np.asarray(objective_matrix, dtype=np.float64)
        # Backward-compatible alias: construction feasibility uses duration.
        self.dist_matrix = np.asarray(duration_matrix, dtype=np.float64)
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        self.duration_matrix = np.asarray(duration_matrix, dtype=np.float64)
        self.node_ids = list(customer_node_ids)
        self.customer_node_ids = list(customer_node_ids)
        self.depot_node_ids = list(depot_node_ids)
        self.depot_labels = list(depot_labels)
        self.travel_metric = str(travel_metric).lower().strip()
        self.city_name = city_name
        self.source_files = dict(source_files or {})
        self.parameter_map = dict(parameter_map or {})
        self.route_limit_source = route_limit_source

    def get_dist(self, i, j):
        """Return real travel duration in seconds for route construction."""
        return self.duration_matrix[i][j]

    def get_financial_cost(self, i, j):
        """Return distance-based financial travel cost for one arc."""
        return self.financial_cost_matrix[i][j]

    def get_distance_meters(self, i, j):
        return self.distance_matrix[i][j]

    def get_duration_seconds(self, i, j):
        return self.duration_matrix[i][j]

    def get_depot_capacity(self, depot_index):
        return float(self.depot_capacities[int(depot_index)])

    def route_real_metrics(self, depot_index, customer_path):
        """Return actual Google road distance (m) and duration (s)."""
        if not customer_path:
            return 0.0, 0.0

        nodes = [int(depot_index)]
        nodes.extend(self.num_depots + int(customer) for customer in customer_path)
        nodes.append(int(depot_index))

        total_distance = 0.0
        total_duration = 0.0
        for origin, destination in zip(nodes[:-1], nodes[1:]):
            total_distance += self.get_distance_meters(origin, destination)
            total_duration += self.get_duration_seconds(origin, destination)
        return total_distance, total_duration

    @staticmethod
    def _read_labeled_matrix(file_path, ordered_ids):
        """Read a row/column-labelled Google matrix and reorder by customer ID."""
        workbook = pd.ExcelFile(file_path)
        preferred = sorted(
            workbook.sheet_names,
            key=lambda name: (
                normalize_schema_token(name) not in {"sheet1", "matrix", "data"},
                workbook.sheet_names.index(name),
            ),
        )

        matrix_df = None
        selected_sheet = None
        for sheet_name in preferred:
            candidate = pd.read_excel(file_path, sheet_name=sheet_name)
            candidate.columns = [str(column).strip() for column in candidate.columns]
            id_columns = [
                column for column in candidate.columns
                if normalize_schema_token(column) in {"id", "customercode", "customerid"}
            ]
            if id_columns:
                candidate.rename(columns={id_columns[0]: "id"}, inplace=True)
                matrix_df = candidate
                selected_sheet = sheet_name
                break

        if matrix_df is None:
            raise ValueError(
                f"No labelled matrix table with an id column was found in "
                f"{os.path.basename(file_path)}."
            )

        matrix_df["id"] = matrix_df["id"].astype(str).str.strip()
        matrix_df.set_index("id", inplace=True)
        matrix_df.columns = [str(column).strip() for column in matrix_df.columns]

        if matrix_df.index.duplicated().any():
            duplicate_rows = matrix_df.index[matrix_df.index.duplicated()].tolist()
            raise ValueError(
                f"Duplicate row IDs in {os.path.basename(file_path)}: "
                f"{duplicate_rows[:10]}"
            )
        if pd.Index(matrix_df.columns).duplicated().any():
            raise ValueError(
                f"Duplicate column IDs in {os.path.basename(file_path)}."
            )

        missing_rows = sorted(set(ordered_ids) - set(matrix_df.index))
        missing_cols = sorted(set(ordered_ids) - set(matrix_df.columns))
        if missing_rows or missing_cols:
            raise ValueError(
                f"Matrix/customer-code mismatch in {os.path.basename(file_path)} "
                f"(sheet={selected_sheet}). Missing rows={missing_rows[:20]}, "
                f"missing columns={missing_cols[:20]}"
            )

        result = matrix_df.reindex(
            index=ordered_ids,
            columns=ordered_ids,
        ).to_numpy(dtype=np.float64)

        if result.shape != (len(ordered_ids), len(ordered_ids)):
            raise ValueError(
                f"Unexpected matrix shape in {os.path.basename(file_path)}: "
                f"{result.shape}; expected {(len(ordered_ids), len(ordered_ids))}."
            )
        if not np.isfinite(result).all():
            raise ValueError(
                f"Non-finite matrix entries found in {os.path.basename(file_path)}."
            )
        if np.any(result < 0):
            raise ValueError(
                f"Negative matrix entries found in {os.path.basename(file_path)}."
            )
        return result

    @staticmethod
    def _build_combined_matrix(base_matrix, customer_ids, depot_ids):
        customer_position = {
            customer_code: index
            for index, customer_code in enumerate(customer_ids)
        }
        depot_positions = [customer_position[code] for code in depot_ids]

        depot_to_depot = base_matrix[np.ix_(depot_positions, depot_positions)]
        depot_to_customer = base_matrix[np.ix_(depot_positions, range(len(customer_ids)))]
        customer_to_depot = base_matrix[np.ix_(range(len(customer_ids)), depot_positions)]
        customer_to_customer = base_matrix

        return np.block([
            [depot_to_depot, depot_to_customer],
            [customer_to_depot, customer_to_customer],
        ])

    @staticmethod
    def _maximum_deliverable(ng, nv, w, q):
        """
        Maximum throughput possible with exactly ng used depots, at most nv
        routes, per-depot capacity w, and per-route capacity q.
        """
        ng = int(ng)
        nv = int(nv)
        w = float(w)
        q = float(q)
        if ng <= 0 or nv < ng:
            return 0.0

        depot_load_limits = [min(w, q) for _ in range(ng)]
        remaining_routes = nv - ng
        for _ in range(remaining_routes):
            marginal_gains = [
                max(0.0, min(q, w - current))
                for current in depot_load_limits
            ]
            best = int(np.argmax(marginal_gains))
            gain = marginal_gains[best]
            if gain <= 1e-12:
                break
            depot_load_limits[best] += gain
        return float(sum(depot_load_limits))

    @classmethod
    def from_real_data(
        cls,
        city_name,
        base_path,
        vehicle_cap=None,
        max_route_dist=None,
        depot_cap=None,
        required_open_depots=None,
        max_vehicles=None,
        fixed_depot_cost=5000,
        fixed_vehicle_cost=200,
        cost_per_km=DEFAULT_COST_PER_KM,
        equity_weight=0.5,
        travel_metric="duration",
        candidate_depot_ids=None,
    ):
        files = discover_city_files(city_name, base_path)
        resolved_city = files["city"]
        print(f"Loading real data for city: {resolved_city}")

        print("Loading files:")
        print(f" -> Index, parameters, depots: {os.path.basename(files['index'])}")
        print(f" -> Real road distance: {os.path.basename(files['distance'])}")
        print(f" -> Real road duration: {os.path.basename(files['duration'])}")

        df_index, customer_sheet = read_city_customer_table(files["index"])
        parameter_map = read_city_parameter_map(files["index"])
        candidate_df, candidate_sheet, candidate_header_row = (
            read_candidate_depot_table(files["index"])
        )

        customer_ids = df_index["Customer_Code"].tolist()
        if len(set(customer_ids)) != len(customer_ids):
            duplicates = (
                df_index.loc[
                    df_index["Customer_Code"].duplicated(), "Customer_Code"
                ].tolist()
            )
            raise ValueError(
                f"Duplicate customer codes in {resolved_city}: {duplicates[:20]}"
            )

        expected_candidate_count = int(float(parameter_map["candidate_depots"]))
        recommended_open_depots = int(float(parameter_map["open_depots"]))
        recommended_max_vehicles = int(float(parameter_map["max_vehicles"]))
        recommended_vehicle_cap = float(parameter_map["vehicle_cap"])
        recommended_depot_cap = float(parameter_map["depot_cap"])

        if candidate_depot_ids is not None:
            requested = [str(value).strip() for value in candidate_depot_ids]
            by_code = candidate_df.set_index("Customer_Code", drop=False)
            by_label = candidate_df.set_index("Candidate_Depot_ID", drop=False)
            if all(value in by_code.index for value in requested):
                candidate_df = by_code.loc[requested].reset_index(drop=True)
            elif all(value in by_label.index for value in requested):
                candidate_df = by_label.loc[requested].reset_index(drop=True)
            else:
                missing = [
                    value for value in requested
                    if value not in by_code.index and value not in by_label.index
                ]
                raise ValueError(
                    f"Unknown candidate depot codes/labels for {resolved_city}: {missing}"
                )
            print(
                f" -> Custom candidate subset selected: {len(candidate_df)} "
                f"of {expected_candidate_count}"
            )
        elif len(candidate_df) != expected_candidate_count:
            raise ValueError(
                f"Parameters specify {expected_candidate_count} candidate depots, "
                f"but candidate sheet contains {len(candidate_df)} rows."
            )

        depot_ids = candidate_df["Customer_Code"].tolist()
        depot_labels = candidate_df["Candidate_Depot_ID"].tolist()

        if len(set(depot_ids)) != len(depot_ids):
            raise ValueError("Candidate-depot customer codes are not unique.")

        unknown_depots = sorted(set(depot_ids) - set(customer_ids))
        if unknown_depots:
            raise ValueError(
                f"Candidate depots not found in customer table: {unknown_depots}"
            )

        sheet_capacities = candidate_df[
            "Fixed_Depot_Capacity"
        ].to_numpy(dtype=np.float64)
        if candidate_depot_ids is None and not np.allclose(
            sheet_capacities, recommended_depot_cap
        ):
            raise ValueError(
                "Candidate-depot capacities do not match the fixed W parameter."
            )

        vehicle_cap = (
            recommended_vehicle_cap if vehicle_cap is None else float(vehicle_cap)
        )
        depot_cap = (
            recommended_depot_cap if depot_cap is None else float(depot_cap)
        )
        required_open_depots = (
            recommended_open_depots
            if required_open_depots is None
            else int(required_open_depots)
        )
        max_vehicles = (
            recommended_max_vehicles
            if max_vehicles is None
            else int(max_vehicles)
        )

        if required_open_depots > len(depot_ids):
            raise ValueError(
                f"Ng={required_open_depots} exceeds selected candidate depots "
                f"m={len(depot_ids)}."
            )
        if max_vehicles < required_open_depots:
            raise ValueError(
                f"Nv={max_vehicles} is smaller than Ng={required_open_depots}; "
                "each open depot needs at least one nonempty route."
            )

        # All candidate depots use the same fixed capacity, matching Ch/Gaskell.
        depot_capacities = np.full(len(depot_ids), depot_cap, dtype=np.float64)

        cust_coords = df_index[
            ["Latitude", "Longitude"]
        ].to_numpy(dtype=np.float64)
        depot_coords = candidate_df[
            ["Latitude", "Longitude"]
        ].to_numpy(dtype=np.float64)

        annual_demands = df_index["Total_Sales"].to_numpy(dtype=np.float64)
        demands = np.ceil(annual_demands / (300 * 5))

        priority_column_map = {
            "Equal-Priority": "Priority_Equal",
            "Tiered": "Priority_Tiered",
            "Demand-Correlated": "Priority_Demand_Correlated",
            "Negative-Correlated": "Priority_Negative_Correlated",
            "Hyper-Critical": "Priority_Hyper_Critical",
        }

        missing_priority_columns = [
            col for col in priority_column_map.values()
            if col not in df_index.columns
        ]

        if missing_priority_columns:
            raise ValueError(
                f"{resolved_city}: missing priority columns in "
                f"{customer_sheet}: {missing_priority_columns}"
            )

        priority_profiles = {}

        for profile_name, column_name in priority_column_map.items():
            values = df_index[column_name].to_numpy(dtype=np.float64)

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{resolved_city}: non-finite values in {column_name}"
                )

            if np.any(values <= 0):
                raise ValueError(
                    f"{resolved_city}: priorities must be > 0 in {column_name}"
                )

            priority_profiles[profile_name] = values.copy()

        priorities = priority_profiles["Equal-Priority"].copy()

        print(
            " -> Priority profiles loaded from Sheet1: "
            + ", ".join(priority_column_map.values())
        )

        base_distance_matrix = cls._read_labeled_matrix(
            files["distance"], customer_ids
        )
        base_duration_matrix = cls._read_labeled_matrix(
            files["duration"], customer_ids
        )

        distance_matrix = cls._build_combined_matrix(
            base_distance_matrix, customer_ids, depot_ids
        )
        duration_matrix = cls._build_combined_matrix(
            base_duration_matrix, customer_ids, depot_ids
        )

        # Manuscript-consistent separation:
        #   c_uv = distance-based financial cost,
        #   l_uv = real travel duration used for latency and route limit.
        travel_metric = "duration"
        cost_per_km = float(cost_per_km)
        objective_matrix = (distance_matrix / 1000.0) * cost_per_km
        metric_unit = "seconds"

        if max_route_dist is None or (
            isinstance(max_route_dist, str)
            and max_route_dist.strip().lower() == "auto"
        ):
            # Big L is a fixed instance parameter stored in Sheet2.
            # The duration matrix supplies travel times only; it does not
            # determine the route-duration limit during optimization.
            sheet2_raw = pd.read_excel(
                files["index"],
                sheet_name="Sheet2",
                header=None,
            )

            excel_big_l = None

            for _, row in sheet2_raw.iterrows():
                if len(row) < 2:
                    continue

                label = row.iloc[0]

                if pd.isna(label):
                    continue

                normalized_label = normalize_schema_token(label)

                if normalized_label.startswith("maximumrouteduration"):
                    value = row.iloc[1]

                    try:
                        excel_big_l = float(value)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"{resolved_city}: invalid Big L value in Sheet2: "
                            f"{value!r}"
                        )

                    break

            if excel_big_l is None:
                raise ValueError(
                    f"{resolved_city}: Sheet2 does not contain "
                    "'Maximum route duration (L, seconds)'."
                )

            if not np.isfinite(excel_big_l) or excel_big_l <= 0:
                raise ValueError(
                    f"{resolved_city}: Big L must be a positive finite value; "
                    f"found {excel_big_l!r}."
                )

            max_route_dist = excel_big_l
            route_limit_source = "Sheet2 Big L"

        else:
            max_route_dist = float(max_route_dist)

            if not np.isfinite(max_route_dist) or max_route_dist <= 0:
                raise ValueError(
                    "Runner max_route_duration override must be "
                    "a positive finite value."
                )

            route_limit_source = "runner override"

        total_demand = float(np.sum(demands))
        total_vehicle_capacity = float(max_vehicles * vehicle_cap)
        max_customer_demand = float(np.max(demands))
        maximum_deliverable = cls._maximum_deliverable(
            required_open_depots,
            max_vehicles,
            depot_cap,
            vehicle_cap,
        )

        if max_customer_demand > vehicle_cap + 1e-9:
            raise ValueError(
                f"Q={vehicle_cap:g} is below maximum customer demand "
                f"{max_customer_demand:g}."
            )
        if max_customer_demand > depot_cap + 1e-9:
            raise ValueError(
                f"W={depot_cap:g} is below maximum customer demand "
                f"{max_customer_demand:g}."
            )
        if total_demand > required_open_depots * depot_cap + 1e-9:
            raise ValueError(
                "Open-depot capacity is insufficient: "
                f"D={total_demand:g}, Ng*W={required_open_depots * depot_cap:g}."
            )
        if total_demand > total_vehicle_capacity + 1e-9:
            raise ValueError(
                "Fleet capacity is insufficient: "
                f"D={total_demand:g}, Nv*Q={total_vehicle_capacity:g}."
            )
        if total_demand > maximum_deliverable + 1e-9:
            raise ValueError(
                "Combined route/depot capacities are insufficient even though "
                "aggregate checks pass: "
                f"D={total_demand:g}, maximum deliverable={maximum_deliverable:g}."
            )

        print(f"Customer sheet: {customer_sheet}")
        print(
            f"Candidate sheet: {candidate_sheet} "
            f"(header row {candidate_header_row + 1})"
        )
        print(f"Loaded customers (n): {len(customer_ids)}")
        print(f"Selected candidate depots (m): {len(depot_ids)}")
        print(f"Required open depots (Ng): {required_open_depots}")
        print(f"Maximum vehicles/routes (Nv): {max_vehicles}")
        print(f"Vehicle capacity (Q): {vehicle_cap:g}")
        print(f"Fixed capacity per depot (W): {depot_cap:g}")
        print(f"Total shift demand (D): {total_demand:g}")
        print(
            f"Maximum route duration: {max_route_dist:g} "
            f"{metric_unit} ({route_limit_source})"
        )
        print(f"Financial travel cost: distance_km × {cost_per_km:g}")
        print("Latency and route limit: real Google duration (seconds)")
        print("Candidate depot customer codes:")
        for label, code in zip(depot_labels, depot_ids):
            print(f" -> {label}: {code}")

        return cls(
            num_depots=len(depot_ids),
            num_customers=len(customer_ids),
            vehicle_cap=vehicle_cap,
            max_route_dist=max_route_dist,
            depot_capacities=depot_capacities,
            required_open_depots=required_open_depots,
            max_vehicles=max_vehicles,
            fixed_depot_cost=fixed_depot_cost,
            fixed_vehicle_cost=fixed_vehicle_cost,
            cost_per_km=cost_per_km,
            depot_coords=depot_coords,
            cust_coords=cust_coords,
            demands=demands,
            priorities=priorities,
            priority_profiles=priority_profiles,
            objective_matrix=objective_matrix,
            distance_matrix=distance_matrix,
            duration_matrix=duration_matrix,
            customer_node_ids=customer_ids,
            depot_node_ids=depot_ids,
            depot_labels=depot_labels,
            travel_metric=travel_metric,
            equity_weight=equity_weight,
            city_name=resolved_city,
            source_files=files,
            parameter_map=parameter_map,
            route_limit_source=route_limit_source,
        )



def build_candidate_depot_table(instance):
    # Coordinates are intentionally excluded from the de-identified build.
    return pd.DataFrame({
        "Candidate Depot ID": instance.depot_labels,
        "Customer Code": instance.depot_node_ids,
        "Fixed Capacity W": instance.depot_capacities,
    })


def inspect_city_configuration(city_name, base_path=DEFAULT_BASE_DATA_PATH):
    """Read city parameters and candidate depots without loading full matrices."""
    files = discover_city_files(city_name, base_path)
    customer_df, customer_sheet = read_city_customer_table(files["index"])
    parameters = read_city_parameter_map(files["index"])
    candidate_df, candidate_sheet, candidate_header_row = (
        read_candidate_depot_table(files["index"])
    )

    return {
        "City": files["city"],
        "Customers (n)": len(customer_df),
        "Candidate Depots (m)": len(candidate_df),
        "Sheet2 m": int(float(parameters["candidate_depots"])),
        "Required Open Depots (Ng)": int(float(parameters["open_depots"])),
        "Maximum Vehicles (Nv)": int(float(parameters["max_vehicles"])),
        "Vehicle Capacity (Q)": float(parameters["vehicle_cap"]),
        "Fixed Depot Capacity (W)": float(parameters["depot_cap"]),
        "Index File": os.path.basename(files["index"]),
        "Distance File": os.path.basename(files["distance"]),
        "Duration File": os.path.basename(files["duration"]),
        "Customer Sheet": customer_sheet,
        "Candidate Sheet": candidate_sheet,
        "Candidate Header Row": candidate_header_row + 1,
        "Candidate Customer Codes": candidate_df["Customer_Code"].tolist(),
    }


def build_city_catalog(base_path=DEFAULT_BASE_DATA_PATH):
    """Return a compact catalogue of every selectable city in Cleaned."""
    records = []
    for city in list_available_cities(base_path):
        try:
            record = inspect_city_configuration(city, base_path)
            record["Status"] = (
                "OK" if record["Candidate Depots (m)"] == record["Sheet2 m"]
                else "Candidate count mismatch"
            )
        except Exception as exc:
            record = {
                "City": city,
                "Status": f"ERROR: {type(exc).__name__}: {exc}",
            }
        records.append(record)

    catalog = pd.DataFrame(records)
    preferred_columns = [
        "City", "Customers (n)", "Candidate Depots (m)",
        "Required Open Depots (Ng)", "Maximum Vehicles (Nv)",
        "Vehicle Capacity (Q)", "Fixed Depot Capacity (W)",
        "Status", "Index File", "Distance File", "Duration File",
    ]
    available_columns = [
        column for column in preferred_columns if column in catalog.columns
    ]
    remaining_columns = [
        column for column in catalog.columns if column not in available_columns
    ]
    return catalog[available_columns + remaining_columns]


_CSV_PRIORITY_COLS = {
    "Equal-Priority": "priority_equal",
    "Tiered": "priority_tiered",
    "Demand-Correlated": "priority_demand_correlated",
    "Negative-Correlated": "priority_negative_correlated",
    "Hyper-Critical": "priority_hyper_critical",
}


def _csv_inputs_present(city_name, base_path):
    return os.path.isfile(os.path.join(base_path, f"{city_name}_customers.csv"))


def build_instance_from_csv(city_name, base_path, equity_weight=DEFAULT_EQUITY_WEIGHT,
                            travel_metric=DEFAULT_TRAVEL_METRIC, max_route_duration=None,
                            fixed_depot_cost=DEFAULT_FIXED_DEPOT_COST,
                            fixed_vehicle_cost=DEFAULT_FIXED_VEHICLE_COST,
                            cost_per_km=DEFAULT_COST_PER_KM):
    """Build an Instance from the de-identified clean CSV inputs (no coordinates;
    demand given directly). Produces the same object the xlsx loader would, with
    coordinates set to zeros (coordinates are not used by any algorithm)."""
    def _p(role):
        fp = os.path.join(base_path, f"{city_name}_{role}.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {role} CSV for {city_name}: {fp}")
        return fp
    cust = pd.read_csv(_p("customers")); cust.columns = [str(c).strip() for c in cust.columns]
    dep = pd.read_csv(_p("depots"));     dep.columns = [str(c).strip() for c in dep.columns]
    par = pd.read_csv(_p("params"))
    params = {str(k).strip(): v for k, v in zip(par.iloc[:, 0], par.iloc[:, 1])}

    customer_ids = [str(x).strip() for x in cust["id"].tolist()]
    depot_labels = [str(x).strip() for x in dep["depot_id"].tolist()]
    depot_ids = [str(x).strip() for x in dep["at_customer"].tolist()]
    priority_profiles = {k: cust[v].to_numpy(dtype=np.float64)
                         for k, v in _CSV_PRIORITY_COLS.items() if v in cust.columns}
    priorities = priority_profiles["Equal-Priority"].copy()
    demands = cust["demand"].to_numpy(dtype=np.float64)

    Ng = int(float(params["Ng"])); Nv = int(float(params["Nv"])); Q = float(params["Q"])
    if max_route_duration in (None, "auto"):
        L = float(params["L"])
    else:
        L = float(max_route_duration)
    cap = {str(dep["depot_id"][i]).strip(): float(dep["capacity"][i]) for i in range(len(dep))}
    depot_capacities = np.array([cap[l] for l in depot_labels], dtype=np.float64)
    cost_per_km = float(params.get("cost_per_km", cost_per_km))

    def _mat(role):
        d = pd.read_csv(_p(role)); d.columns = [str(c).strip() for c in d.columns]
        d["id"] = d["id"].astype(str).str.strip(); d.set_index("id", inplace=True)
        d.columns = [str(c).strip() for c in d.columns]
        return d.reindex(index=customer_ids, columns=customer_ids).to_numpy(dtype=np.float64)
    # distance.csv already holds the financial cost per arc (cost_per_km applied upstream);
    # duration.csv holds travel seconds. Combine to the (m+n) node matrices (depots first).
    objective_matrix = Instance._build_combined_matrix(_mat("distance"), customer_ids, depot_ids)
    duration_matrix = Instance._build_combined_matrix(_mat("duration"), customer_ids, depot_ids)
    # distance_matrix is used only for the reported "Route Distance (meters)" columns;
    # recover metres from the financial cost so those columns stay faithful.
    distance_matrix = objective_matrix * (1000.0 / cost_per_km) if cost_per_km else objective_matrix

    n, m = len(customer_ids), len(depot_ids)
    instance = Instance(
        num_depots=m, num_customers=n, vehicle_cap=Q, max_route_dist=L,
        depot_capacities=depot_capacities, required_open_depots=Ng, max_vehicles=Nv,
        fixed_depot_cost=float(params.get("fixed_depot_cost", fixed_depot_cost)),
        fixed_vehicle_cost=float(params.get("fixed_vehicle_cost", fixed_vehicle_cost)),
        cost_per_km=cost_per_km,
        depot_coords=np.zeros((m, 2)), cust_coords=np.zeros((n, 2)),
        demands=demands, priorities=priorities, priority_profiles=priority_profiles,
        objective_matrix=objective_matrix, distance_matrix=distance_matrix,
        duration_matrix=duration_matrix, customer_node_ids=customer_ids,
        depot_node_ids=depot_ids, depot_labels=depot_labels, travel_metric=travel_metric,
        equity_weight=equity_weight, city_name=city_name, source_files={},
        parameter_map=params, route_limit_source="params.csv L")
    return instance


def load_city_instance(
    city_name,
    base_path=DEFAULT_BASE_DATA_PATH,
    equity_weight=DEFAULT_EQUITY_WEIGHT,
    travel_metric=DEFAULT_TRAVEL_METRIC,
    max_route_duration=DEFAULT_MAX_ROUTE_DURATION_SECONDS,
    fixed_depot_cost=DEFAULT_FIXED_DEPOT_COST,
    fixed_vehicle_cost=DEFAULT_FIXED_VEHICLE_COST,
    cost_per_km=DEFAULT_COST_PER_KM,
    vehicle_cap=None,
    depot_cap=None,
    required_open_depots=None,
    max_vehicles=None,
    candidate_depot_ids=None,
):
    # De-identified clean CSV inputs are the default for the public repository.
    if _csv_inputs_present(city_name, base_path):
        instance = build_instance_from_csv(
            city_name, base_path, equity_weight=equity_weight, travel_metric=travel_metric,
            max_route_duration=max_route_duration, fixed_depot_cost=fixed_depot_cost,
            fixed_vehicle_cost=fixed_vehicle_cost, cost_per_km=cost_per_km)
        return instance.city_name, instance
    instance = Instance.from_real_data(
        city_name=city_name,
        base_path=base_path,
        vehicle_cap=vehicle_cap,
        max_route_dist=max_route_duration,
        depot_cap=depot_cap,
        required_open_depots=required_open_depots,
        max_vehicles=max_vehicles,
        fixed_depot_cost=fixed_depot_cost,
        fixed_vehicle_cost=fixed_vehicle_cost,
        cost_per_km=cost_per_km,
        equity_weight=equity_weight,
        travel_metric=travel_metric,
        candidate_depot_ids=candidate_depot_ids,
    )
    return instance.city_name, instance


def run_single_city(
    city_name,
    base_path=DEFAULT_BASE_DATA_PATH,
    runs=5,
    seeds=None,
    max_iter=50,
    alpha_max=5,
    equity_weight=DEFAULT_EQUITY_WEIGHT,
    travel_metric=DEFAULT_TRAVEL_METRIC,
    max_route_duration=DEFAULT_MAX_ROUTE_DURATION_SECONDS,
    fixed_depot_cost=DEFAULT_FIXED_DEPOT_COST,
    fixed_vehicle_cost=DEFAULT_FIXED_VEHICLE_COST,
    cost_per_km=DEFAULT_COST_PER_KM,
    output_file=None,
    candidate_depot_ids=None,
):
    """Run the standard multi-seed experiment for one city and export Excel."""
    if seeds is None:
        seeds = list(range(int(runs)))
    else:
        seeds = [int(seed) for seed in seeds]
        runs = len(seeds)

    resolved_city, instance = load_city_instance(
        city_name=city_name,
        base_path=base_path,
        equity_weight=equity_weight,
        travel_metric=travel_metric,
        max_route_duration=max_route_duration,
        fixed_depot_cost=fixed_depot_cost,
        fixed_vehicle_cost=fixed_vehicle_cost,
        cost_per_km=cost_per_km,
        candidate_depot_ids=candidate_depot_ids,
    )
    candidate_depot_table = build_candidate_depot_table(instance)

    run_logs = []
    iteration_logs_all = []
    route_logs = []
    best_solution = None
    best_cost = float("inf")
    total_time = 0.0

    print(
        f"\nLaunching GRASP-VNDZou-SO for {resolved_city}: "
        f"n={instance.num_customers}, m={instance.num_depots}, "
        f"Ng={instance.required_open_depots}, Nv={instance.max_vehicles}, "
        f"Q={instance.vehicle_cap:g}, W={instance.depot_cap:g}, "
        f"L={instance.max_route_dist:g}"
    )

    for run_number, seed in enumerate(seeds, start=1):
        print(f"  Run {run_number}/{runs} | seed={seed}")
        random.seed(seed)
        np.random.seed(seed)

        start_time = time.time()
        solution, iteration_logs = run_grasp_vnd_logged(
            instance,
            max_iter=max_iter,
            alpha_max=alpha_max,
            run_id=run_number,
            seed=seed,
            instance_name=resolved_city,
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        iteration_logs_all.extend(iteration_logs)

        feasible = solution.check_validity() if solution else False
        if solution:
            solution.calculate_cost()

        cost = solution.cost if solution and feasible else float("inf")
        coverage = (
            solution.coverage_status()
            if solution
            else (False, None, None, None, None)
        )

        if feasible and cost < best_cost:
            best_cost = cost
            best_solution = copy.deepcopy(solution)

        open_indices = sorted(solution.open_depots) if solution else []
        open_codes = [instance.depot_node_ids[index] for index in open_indices]

        run_logs.append({
            "Instance": resolved_city,
            "Algorithm": "GRASP-VNDZou-SO",
            "Run": run_number,
            "Seed": seed,
            "Feasible": bool(feasible),
            "Coverage OK": coverage[0],
            "Num Served": coverage[1],
            "Num Unique Served": coverage[2],
            "Missing Customers": coverage[3],
            "Duplicate Customers": coverage[4],
            "Customers (n)": instance.num_customers,
            "Candidate Depots (m)": instance.num_depots,
            "Required Open Depots (Ng)": instance.required_open_depots,
            "Maximum Vehicles (Nv)": instance.max_vehicles,
            "Vehicle Capacity (Q)": instance.vehicle_cap,
            "Fixed Depot Capacity (W)": instance.depot_cap,
            "Maximum Route Metric": instance.max_route_dist,
            "Route Limit Source": instance.route_limit_source,
            "Total Objective Cost": cost if cost != float("inf") else None,
            "Financial Cost": solution.financial_cost if solution and feasible else None,
            "Priority Penalty": solution.equity_penalty if solution and feasible else None,
            "CPU Time (s)": elapsed,
            "Open Depot Indices": str(open_indices),
            "Open Depot Customer Codes": str(open_codes),
            "Number of Routes": (
                len([route for route in solution.routes if route["path"]])
                if solution else None
            ),
        })

    if best_solution is not None:
        route_logs = extract_routes(
            best_solution,
            resolved_city,
            "GRASP-VNDZou-SO",
        )

    summary = pd.DataFrame([{
        "Instance": resolved_city,
        "Customers (n)": instance.num_customers,
        "Candidate Depots (m)": instance.num_depots,
        "Required Open Depots (Ng)": instance.required_open_depots,
        "Maximum Vehicles (Nv)": instance.max_vehicles,
        "Vehicle Capacity (Q)": instance.vehicle_cap,
        "Fixed Depot Capacity (W)": instance.depot_cap,
        "Total Shift Demand (D)": float(np.sum(instance.demands)),
        "Travel Metric": instance.travel_metric,
        "Maximum Route Metric": instance.max_route_dist,
        "Route Limit Source": instance.route_limit_source,
        "Feasible Runs Success Rate": (
            f"{sum(1 for row in run_logs if row['Feasible'])}/{runs}"
        ),
        "Strict Coverage Success Rate": (
            f"{sum(1 for row in run_logs if row['Coverage OK'])}/{runs}"
        ),
        "Best Cost": (
            round(best_cost, 2)
            if best_cost != float("inf")
            else "Infeasible"
        ),
        "Pure Financial Cost": (
            round(best_solution.financial_cost, 2)
            if best_solution else "N/A"
        ),
        "Priority Penalty": (
            round(best_solution.equity_penalty, 2)
            if best_solution else "N/A"
        ),
        "Avg CPU Time (s)": round(total_time / max(runs, 1), 2),
    }])

    run_df = pd.DataFrame(run_logs)
    iteration_df = pd.DataFrame(iteration_logs_all)
    route_df = pd.DataFrame(route_logs)

    if output_file is None:
        output_file = os.path.join(
            base_path,
            resolved_city,
            f"Analysis_Report_{resolved_city}_VNDZouSO_CandidateDepots.xlsx",
        )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        candidate_depot_table.to_excel(
            writer, sheet_name="Candidate_Depots", index=False
        )
        run_df.to_excel(writer, sheet_name="Run_Level_Details", index=False)
        iteration_df.to_excel(
            writer, sheet_name="Iteration_Level_Details", index=False
        )
        route_df.to_excel(writer, sheet_name="Best_Routes", index=False)

    print(f"\nSingle-city report saved: {output_file}")

    return {
        "city": resolved_city,
        "instance": instance,
        "best_solution": best_solution,
        "summary": summary,
        "runs": run_df,
        "iterations": iteration_df,
        "routes": route_df,
        "candidate_depots": candidate_depot_table,
        "output_file": output_file,
        "source_files": instance.source_files,
    }



GLOBAL_PROFILE_SEED = 42000
DEFAULT_EQUITY_WEIGHTS = [0.0, 0.5, 1.0, 2.0]
DEFAULT_PRIORITY_PROFILES = [
    "Equal-Priority",
    "Tiered",
    "Demand-Correlated",
    "Negative-Correlated",
    "Hyper-Critical",
]


def build_priority_profile(
    profile_name,
    priority_profiles,
    demands=None,
    profile_seed=None,
):
    """
    Select a pharmacy-level priority profile loaded from the city Excel file.

    Priority values are NOT generated inside the optimization code.
    Sheet1 of each <City>_index_map.xlsx file is the single source of truth.
    """
    profile_name = str(profile_name).strip()

    if profile_name == "Observed":
        raise ValueError(
            "The 'Observed' priority profile is no longer supported because "
            "the empirical data do not contain observed pharmacy priorities."
        )

    if not hasattr(priority_profiles, "keys"):
        raise TypeError(
            "priority_profiles must be a dictionary of Excel-loaded profiles."
        )

    if profile_name not in priority_profiles:
        raise ValueError(
            f"Unknown or unavailable priority profile: {profile_name}. "
            f"Available profiles: {list(priority_profiles.keys())}"
        )

    priorities = np.asarray(
        priority_profiles[profile_name],
        dtype=np.float64,
    ).copy()

    if demands is not None and len(priorities) != len(demands):
        raise ValueError(
            f"Priority length mismatch for {profile_name}: "
            f"{len(priorities)} priorities for {len(demands)} customers."
        )

    if not np.all(np.isfinite(priorities)):
        raise ValueError(
            f"Non-finite priority values detected for {profile_name}."
        )

    if np.any(priorities <= 0):
        raise ValueError(
            f"All priority values must be strictly positive for {profile_name}."
        )

    return priorities

def derive_capacity_levels(instance, city_name=None, step=5):
    """
    Return three distinct structurally feasible W and Q levels.

    Afyon uses the manuscript values exactly. For other cities, the Sheet2
    setting remains Moderate whenever a feasible lower level exists. When the
    Sheet2 value already equals the mathematical minimum (for example Bursa Q),
    it becomes the Low/tight level and two higher feasible levels are created.
    """
    resolved = city_name or instance.city_name or ""
    if resolved in CITY_CAPACITY_LEVEL_OVERRIDES:
        override = CITY_CAPACITY_LEVEL_OVERRIDES[resolved]
        return copy.deepcopy(override["depot"]), copy.deepcopy(override["vehicle"])

    w0 = float(instance.depot_cap)
    q0 = float(instance.vehicle_cap)
    total_demand = float(np.sum(instance.demands))
    max_demand = float(np.max(instance.demands))
    ng = int(instance.required_open_depots)
    nv = int(instance.max_vehicles)

    min_w = round_up(max(total_demand / ng, max_demand), step)
    min_q = round_up(max(total_demand / nv, max_demand), step)

    def make_levels(base, minimum, low_ratio, high_ratio, label):
        base = float(base)
        minimum = float(minimum)

        candidate_low = max(minimum, round_down(low_ratio * base, step))
        if candidate_low < base - 1e-9:
            low = candidate_low
            moderate = base
            high = round_up(max(base + step, high_ratio * base), step)
            baseline_position = "Moderate"
        else:
            # The Sheet2 value is already at the tight feasibility boundary.
            low = max(base, minimum)
            moderate = round_up(max(low + step, 1.10 * low), step)
            high = round_up(max(moderate + step, 1.15 * moderate), step)
            baseline_position = "Low"

        if not (low < moderate < high):
            raise ValueError(
                f"Could not create three distinct {label} levels from "
                f"base={base:g}, minimum={minimum:g}."
            )
        return (
            {"Low": float(low), "Moderate": float(moderate), "High": float(high)},
            baseline_position,
        )

    depot_levels, w_position = make_levels(
        w0, min_w, low_ratio=0.90, high_ratio=1.15, label="depot-capacity"
    )
    vehicle_levels, q_position = make_levels(
        q0, min_q, low_ratio=0.85, high_ratio=1.15, label="vehicle-capacity"
    )

    if w_position != "Moderate" or q_position != "Moderate":
        print(
            f" -> Capacity-level note for {resolved}: Sheet2 baseline position "
            f"is W={w_position}, Q={q_position}, because no lower feasible "
            f"level exists for at least one capacity."
        )

    return depot_levels, vehicle_levels


def build_two_block_scenarios(
    depot_capacity_levels,
    vehicle_capacity_levels,
    equity_weights=None,
    priority_profiles=None,
):
    equity_weights = list(
        DEFAULT_EQUITY_WEIGHTS if equity_weights is None else equity_weights
    )
    priority_profiles = list(
        DEFAULT_PRIORITY_PROFILES if priority_profiles is None else priority_profiles
    )

    moderate_w = float(depot_capacity_levels["Moderate"])
    moderate_q = float(vehicle_capacity_levels["Moderate"])
    registry = {}

    def register(
        block_name,
        depot_level,
        vehicle_level,
        w,
        q,
        alpha,
        profile,
    ):
        key = (float(w), float(q), float(alpha), str(profile))
        if key not in registry:
            registry[key] = {
                "Experiment_Block": block_name,
                "Depot_Capacity_Level": depot_level,
                "Vehicle_Capacity_Level": vehicle_level,
                "Depot_Cap_W": float(w),
                "Vehicle_Cap_Q": float(q),
                "Alpha_Weight": float(alpha),
                "Priority_Profile": str(profile),
            }
        elif block_name not in registry[key]["Experiment_Block"].split("+"):
            registry[key]["Experiment_Block"] += f"+{block_name}"

    for depot_level, w in depot_capacity_levels.items():
        for vehicle_level, q in vehicle_capacity_levels.items():
            for alpha in equity_weights:
                register(
                    "Capacity", depot_level, vehicle_level,
                    w, q, alpha, "Equal-Priority"
                )

    for profile in priority_profiles:
        for alpha in equity_weights:
            register(
                "Priority", "Moderate", "Moderate",
                moderate_w, moderate_q, alpha, profile
            )

    profile_tags = {
        "Equal-Priority": "EQ",
        "Observed": "OBS",
        "Tiered": "TIR",
        "Demand-Correlated": "DCO",
        "Negative-Correlated": "NCO",
        "Hyper-Critical": "HCR",
    }

    experiments = []
    for scenario_id, scenario in enumerate(registry.values(), start=1):
        row = dict(scenario)
        row["Scenario_ID"] = scenario_id
        tag = profile_tags.get(row["Priority_Profile"], "CUS")
        row["Scenario_Name"] = (
            f"S{scenario_id:02d}_"
            f"W{int(row['Depot_Cap_W'])}_"
            f"Q{int(row['Vehicle_Cap_Q'])}_"
            f"A{row['Alpha_Weight']:g}_{tag}"
        )
        experiments.append(row)

    return experiments


def structural_feasibility_check(instance, w, q):
    reasons = []
    total_demand = float(np.sum(instance.demands))
    max_demand = float(np.max(instance.demands))

    if instance.required_open_depots * w < total_demand - 1e-9:
        reasons.append(
            f"Ng*W={instance.required_open_depots * w:g} < D={total_demand:g}"
        )
    if instance.max_vehicles * q < total_demand - 1e-9:
        reasons.append(
            f"Nv*Q={instance.max_vehicles * q:g} < D={total_demand:g}"
        )
    if q < max_demand - 1e-9:
        reasons.append(f"Q={q:g} < max demand={max_demand:g}")
    if w < max_demand - 1e-9:
        reasons.append(f"W={w:g} < max demand={max_demand:g}")

    maximum_deliverable = Instance._maximum_deliverable(
        instance.required_open_depots,
        instance.max_vehicles,
        w,
        q,
    )
    if maximum_deliverable < total_demand - 1e-9:
        reasons.append(
            f"combined route/depot deliverable={maximum_deliverable:g} "
            f"< D={total_demand:g}"
        )

    return len(reasons) == 0, "; ".join(reasons)


def run_bulk_experiments(
    city_name,
    base_path=DEFAULT_BASE_DATA_PATH,
    depot_capacity_levels=None,
    vehicle_capacity_levels=None,
    equity_weights=None,
    priority_profiles=None,
    run_seeds=(0, 1, 2, 3, 4),
    max_iter=50,
    alpha_max=5,
    travel_metric=DEFAULT_TRAVEL_METRIC,
    max_route_duration=DEFAULT_MAX_ROUTE_DURATION_SECONDS,
    fixed_depot_cost=DEFAULT_FIXED_DEPOT_COST,
    fixed_vehicle_cost=DEFAULT_FIXED_VEHICLE_COST,
    cost_per_km=DEFAULT_COST_PER_KM,
    output_file=None,
    candidate_depot_ids=None,
):
    """Run the 52-scenario two-block design for any city."""
    resolved_city, base_instance = load_city_instance(
        city_name=city_name,
        base_path=base_path,
        equity_weight=0.0,
        travel_metric=travel_metric,
        max_route_duration=max_route_duration,
        fixed_depot_cost=fixed_depot_cost,
        fixed_vehicle_cost=fixed_vehicle_cost,
        cost_per_km=cost_per_km,
        candidate_depot_ids=candidate_depot_ids,
    )

    if depot_capacity_levels is None or vehicle_capacity_levels is None:
        auto_w, auto_q = derive_capacity_levels(base_instance, resolved_city)
        if depot_capacity_levels is None:
            depot_capacity_levels = auto_w
        if vehicle_capacity_levels is None:
            vehicle_capacity_levels = auto_q

    experiments = build_two_block_scenarios(
        depot_capacity_levels=depot_capacity_levels,
        vehicle_capacity_levels=vehicle_capacity_levels,
        equity_weights=equity_weights,
        priority_profiles=priority_profiles,
    )

    expected_scenarios = 52
    if len(experiments) != expected_scenarios:
        raise RuntimeError(
            f"Expected {expected_scenarios} unique scenarios, "
            f"obtained {len(experiments)}."
        )

    for experiment in experiments:
        ok, reason = structural_feasibility_check(
            base_instance,
            experiment["Depot_Cap_W"],
            experiment["Vehicle_Cap_Q"],
        )
        experiment["Structural_Feasible"] = bool(ok)
        experiment["Structural_Check"] = "Passed" if ok else reason

    scenario_design = pd.DataFrame(experiments)
    run_seeds = [int(seed) for seed in run_seeds]
    results_log = []
    iteration_logs_all = []

    print("=" * 78)
    print(f"{resolved_city.upper()} TWO-BLOCK BULK EXPERIMENT")
    print(f"n={base_instance.num_customers}, m={base_instance.num_depots}, "
          f"Ng={base_instance.required_open_depots}, "
          f"Nv={base_instance.max_vehicles}")
    print(f"W levels: {depot_capacity_levels}")
    print(f"Q levels: {vehicle_capacity_levels}")
    print(f"Unique scenarios: {len(experiments)}")
    print(f"Runs per scenario: {len(run_seeds)}")
    print(f"Total runs: {len(experiments) * len(run_seeds)}")
    print("=" * 78)

    for experiment in experiments:
        print(
            f"\n[{experiment['Scenario_ID']:02d}/{len(experiments)}] "
            f"{experiment['Scenario_Name']}"
        )

        if not experiment["Structural_Feasible"]:
            for run_number, seed in enumerate(run_seeds, start=1):
                results_log.append({
                    "City": resolved_city,
                    "Scenario_ID": experiment["Scenario_ID"],
                    "Scenario": experiment["Scenario_Name"],
                    "Experiment_Block": experiment["Experiment_Block"],
                    "Depot_Capacity_Level": experiment["Depot_Capacity_Level"],
                    "Vehicle_Capacity_Level": experiment["Vehicle_Capacity_Level"],
                    "Run": run_number,
                    "Seed": seed,
                    "Priority_Profile": experiment["Priority_Profile"],
                    "Vehicle_Cap_Q": experiment["Vehicle_Cap_Q"],
                    "Depot_Cap_W": experiment["Depot_Cap_W"],
                    "Alpha_Weight": experiment["Alpha_Weight"],
                    "Is_Feasible": False,
                    "Structural_Feasible": False,
                    "Failure_Reason": experiment["Structural_Check"],
                    "Coverage_OK": False,
                    "Num_Served": None,
                    "Num_Unique_Served": None,
                    "Missing_Customers": None,
                    "Duplicate_Customers": None,
                    "Total_Cost": None,
                    "Financial_Cost": None,
                    "Equity_Penalty": None,
                    "Num_SWH_Opened": None,
                    "Num_Routes": None,
                    "CPU_Time_s": 0.0,
                })
            print(f"  Skipped: {experiment['Structural_Check']}")
            continue

        instance = Instance.from_real_data(
            city_name=resolved_city,
            base_path=base_path,
            vehicle_cap=experiment["Vehicle_Cap_Q"],
            max_route_dist=base_instance.max_route_dist,
            depot_cap=experiment["Depot_Cap_W"],
            required_open_depots=base_instance.required_open_depots,
            max_vehicles=base_instance.max_vehicles,
            fixed_depot_cost=fixed_depot_cost,
            fixed_vehicle_cost=fixed_vehicle_cost,
            cost_per_km=cost_per_km,
            equity_weight=experiment["Alpha_Weight"],
            travel_metric=travel_metric,
            candidate_depot_ids=candidate_depot_ids,
        )
        instance.priorities = build_priority_profile(
            experiment["Priority_Profile"],
            instance.priority_profiles,
            instance.demands,
        )

        scenario_rows = []
        for run_number, seed in enumerate(run_seeds, start=1):
            run_seed = int(seed)
            random.seed(run_seed)
            np.random.seed(run_seed)

            start_time = time.time()
            solution, iteration_logs = run_grasp_vnd_logged(
                instance,
                max_iter=max_iter,
                alpha_max=alpha_max,
                run_id=run_number,
                seed=run_seed,
                instance_name=experiment["Scenario_Name"],
            )
            elapsed = time.time() - start_time
            iteration_logs_all.extend(iteration_logs)

            feasible = solution.check_validity() if solution else False
            if solution:
                solution.calculate_cost()

            coverage = (
                solution.coverage_status()
                if solution
                else (False, None, None, None, None)
            )

            row = {
                "City": resolved_city,
                "Scenario_ID": experiment["Scenario_ID"],
                "Scenario": experiment["Scenario_Name"],
                "Experiment_Block": experiment["Experiment_Block"],
                "Depot_Capacity_Level": experiment["Depot_Capacity_Level"],
                "Vehicle_Capacity_Level": experiment["Vehicle_Capacity_Level"],
                "Run": run_number,
                "Seed": run_seed,
                "Priority_Profile": experiment["Priority_Profile"],
                "Vehicle_Cap_Q": experiment["Vehicle_Cap_Q"],
                "Depot_Cap_W": experiment["Depot_Cap_W"],
                "Alpha_Weight": experiment["Alpha_Weight"],
                "Is_Feasible": bool(feasible),
                "Structural_Feasible": True,
                "Failure_Reason": None if feasible else "No feasible solution found",
                "Coverage_OK": coverage[0],
                "Num_Served": coverage[1],
                "Num_Unique_Served": coverage[2],
                "Missing_Customers": coverage[3],
                "Duplicate_Customers": coverage[4],
                "Total_Cost": (
                    round(float(solution.cost), 2)
                    if solution and feasible else None
                ),
                "Financial_Cost": (
                    round(float(solution.financial_cost), 2)
                    if solution and feasible else None
                ),
                "Equity_Penalty": (
                    round(float(solution.equity_penalty), 2)
                    if solution and feasible else None
                ),
                "Num_SWH_Opened": (
                    len(solution.open_depots)
                    if solution and feasible else None
                ),
                "Num_Routes": (
                    len([r for r in solution.routes if r["path"]])
                    if solution and feasible else None
                ),
                "CPU_Time_s": round(float(elapsed), 2),
            }
            results_log.append(row)
            scenario_rows.append(row)

        feasible_costs = [
            row["Total_Cost"]
            for row in scenario_rows
            if row["Is_Feasible"] and row["Total_Cost"] is not None
        ]
        if feasible_costs:
            print(
                f"  Feasible={len(feasible_costs)}/{len(run_seeds)}, "
                f"mean={np.mean(feasible_costs):.2f}, "
                f"best={np.min(feasible_costs):.2f}"
            )
        else:
            print(f"  Feasible=0/{len(run_seeds)}")

    result_df = pd.DataFrame(results_log)
    iteration_df = pd.DataFrame(iteration_logs_all)

    group_columns = [
        "Scenario_ID", "Scenario", "Experiment_Block",
        "Depot_Capacity_Level", "Vehicle_Capacity_Level",
        "Depot_Cap_W", "Vehicle_Cap_Q", "Alpha_Weight",
        "Priority_Profile",
    ]
    summary_df = (
        result_df.groupby(group_columns, dropna=False)
        .agg(
            Feasible_Runs=("Is_Feasible", "sum"),
            Structural_Feasible=("Structural_Feasible", "all"),
            Coverage_OK_Runs=("Coverage_OK", "sum"),
            Avg_Total_Cost=("Total_Cost", "mean"),
            Std_Total_Cost=("Total_Cost", "std"),
            Best_Total_Cost=("Total_Cost", "min"),
            Worst_Total_Cost=("Total_Cost", "max"),
            Avg_Financial_Cost=("Financial_Cost", "mean"),
            Avg_Equity_Penalty=("Equity_Penalty", "mean"),
            Avg_Num_SWH_Opened=("Num_SWH_Opened", "mean"),
            Avg_Num_Routes=("Num_Routes", "mean"),
            Avg_CPU_Time_s=("CPU_Time_s", "mean"),
        )
        .reset_index()
    )
    summary_df["Feasibility_Rate"] = (
        summary_df["Feasible_Runs"] / len(run_seeds)
    )
    summary_df["Coverage_OK_Rate"] = (
        summary_df["Coverage_OK_Runs"] / len(run_seeds)
    )
    summary_df = summary_df.round(4)

    profile_definitions = pd.DataFrame([
        {
            "Priority_Profile": "Equal-Priority",
            "Definition": "Fixed pharmacy-level values from Sheet1 column Priority_Equal; p_j = 1.0 for all pharmacies.",
        },
        {"Priority_Profile": "Tiered", "Definition": "20% high, 40% medium, 40% low; fixed assignment."},
        {"Priority_Profile": "Demand-Correlated", "Definition": "Priority increases linearly with demand."},
        {"Priority_Profile": "Negative-Correlated", "Definition": "Priority decreases linearly with demand."},
        {"Priority_Profile": "Hyper-Critical", "Definition": "Two fixed customers have priority 10; others 1."},
    ])

    if output_file is None:
        output_file = os.path.join(
            base_path,
            resolved_city,
            f"{resolved_city}_Bulk_Experiment_52Scenarios_GRASP_VNDZouSO.xlsx",
        )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        scenario_design.to_excel(writer, sheet_name="Scenario_Design", index=False)
        profile_definitions.to_excel(
            writer, sheet_name="Priority_Profiles", index=False
        )
        result_df.to_excel(writer, sheet_name="Run_Level_Results", index=False)
        summary_df.to_excel(writer, sheet_name="Scenario_Summary", index=False)
        iteration_df.to_excel(writer, sheet_name="Iteration_Logs", index=False)

    print(f"\nBulk report saved: {output_file}")

    return {
        "city": resolved_city,
        "base_instance": base_instance,
        "depot_capacity_levels": depot_capacity_levels,
        "vehicle_capacity_levels": vehicle_capacity_levels,
        "scenario_design": scenario_design,
        "run_results": result_df,
        "scenario_summary": summary_df,
        "iterations": iteration_df,
        "output_file": output_file,
    }


def run_city(
    city_name,
    mode="single",
    base_path=DEFAULT_BASE_DATA_PATH,
    **kwargs,
):
    """
    One entry point for the runner notebook.

    mode:
      - 'load': only load and validate city files
      - 'single': standard five-seed experiment
      - 'bulk': 52-scenario design
      - 'both': single and bulk
    """
    mode = str(mode).strip().lower()
    valid_modes = {"load", "single", "bulk", "both"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")

    if mode == "load":
        load_keys = {
            "equity_weight", "travel_metric", "max_route_duration",
            "fixed_depot_cost", "fixed_vehicle_cost", "cost_per_km", "vehicle_cap",
            "depot_cap", "required_open_depots", "max_vehicles",
            "candidate_depot_ids",
        }
        load_kwargs = {k: v for k, v in kwargs.items() if k in load_keys}
        city, instance = load_city_instance(
            city_name,
            base_path=base_path,
            **load_kwargs,
        )
        return {
            "city": city,
            "instance": instance,
            "candidate_depots": build_candidate_depot_table(instance),
        }

    outputs = {}
    if mode in {"single", "both"}:
        single_keys = {
            "runs", "seeds", "max_iter", "alpha_max", "equity_weight",
            "travel_metric", "max_route_duration", "fixed_depot_cost",
            "fixed_vehicle_cost", "cost_per_km", "output_file", "candidate_depot_ids",
        }
        single_kwargs = {k: v for k, v in kwargs.items() if k in single_keys}
        outputs["single"] = run_single_city(
            city_name, base_path=base_path, **single_kwargs
        )

    if mode in {"bulk", "both"}:
        bulk_keys = {
            "depot_capacity_levels", "vehicle_capacity_levels",
            "equity_weights", "priority_profiles", "run_seeds",
            "max_iter", "alpha_max", "travel_metric",
            "max_route_duration", "fixed_depot_cost",
            "fixed_vehicle_cost", "cost_per_km", "output_file", "candidate_depot_ids",
        }
        bulk_kwargs = {k: v for k, v in kwargs.items() if k in bulk_keys}
        outputs["bulk"] = run_bulk_experiments(
            city_name, base_path=base_path, **bulk_kwargs
        )

    return outputs


def _bulk_profile_definitions():
    return pd.DataFrame([
        {"Priority_Profile": "Equal-Priority", "Definition": "Sheet1 priorities."},
        {
            "Priority_Profile": "Tiered",
            "Definition": "20% high, 40% medium, 40% low; fixed assignment.",
        },
        {
            "Priority_Profile": "Demand-Correlated",
            "Definition": "Priority increases linearly with demand.",
        },
        {
            "Priority_Profile": "Negative-Correlated",
            "Definition": "Priority decreases linearly with demand.",
        },
        {
            "Priority_Profile": "Hyper-Critical",
            "Definition": "Two fixed customers have priority 10; others 1.",
        },
    ])


def _build_bulk_summary(result_df, number_of_seeds):
    group_columns = [
        "Scenario_ID", "Scenario", "Experiment_Block",
        "Depot_Capacity_Level", "Vehicle_Capacity_Level",
        "Depot_Cap_W", "Vehicle_Cap_Q", "Alpha_Weight",
        "Priority_Profile",
    ]
    summary_df = (
        result_df.groupby(group_columns, dropna=False)
        .agg(
            Feasible_Runs=("Is_Feasible", "sum"),
            Structural_Feasible=("Structural_Feasible", "all"),
            Coverage_OK_Runs=("Coverage_OK", "sum"),
            Avg_Total_Cost=("Total_Cost", "mean"),
            Std_Total_Cost=("Total_Cost", "std"),
            Best_Total_Cost=("Total_Cost", "min"),
            Worst_Total_Cost=("Total_Cost", "max"),
            Avg_Financial_Cost=("Financial_Cost", "mean"),
            Avg_Equity_Penalty=("Equity_Penalty", "mean"),
            Avg_Num_SWH_Opened=("Num_SWH_Opened", "mean"),
            Avg_Num_Routes=("Num_Routes", "mean"),
            Avg_CPU_Time_s=("CPU_Time_s", "mean"),
        )
        .reset_index()
    )
    summary_df["Feasibility_Rate"] = (
        summary_df["Feasible_Runs"] / int(number_of_seeds)
    )
    summary_df["Coverage_OK_Rate"] = (
        summary_df["Coverage_OK_Runs"] / int(number_of_seeds)
    )
    return summary_df.round(4)


def _sort_bulk_tables(result_df, iteration_df):
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["Scenario_ID", "Run"], kind="stable"
        ).reset_index(drop=True)
    if not iteration_df.empty:
        available = [
            column for column in ["Instance", "Run", "Iteration"]
            if column in iteration_df.columns
        ]
        if available:
            iteration_df = iteration_df.sort_values(
                available, kind="stable"
            ).reset_index(drop=True)
    return result_df, iteration_df


def _write_bulk_workbook(
    output_file,
    scenario_design,
    result_df,
    summary_df,
    iteration_df,
):
    """Write one complete workbook atomically; call only in the parent process."""
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.writing{output_path.suffix}"
    )

    try:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            scenario_design.to_excel(
                writer, sheet_name="Scenario_Design", index=False
            )
            _bulk_profile_definitions().to_excel(
                writer, sheet_name="Priority_Profiles", index=False
            )
            result_df.to_excel(
                writer, sheet_name="Run_Level_Results", index=False
            )
            summary_df.to_excel(
                writer, sheet_name="Scenario_Summary", index=False
            )
            iteration_df.to_excel(
                writer, sheet_name="Iteration_Logs", index=False
            )
        os.replace(temporary_path, output_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot replace {output_path}. Close the workbook in Excel and retry."
        ) from exc

    return str(output_path)


def _structurally_infeasible_rows(resolved_city, experiment, run_seeds):
    rows = []
    for run_number, seed in enumerate(run_seeds, start=1):
        rows.append({
            "City": resolved_city,
            "Scenario_ID": experiment["Scenario_ID"],
            "Scenario": experiment["Scenario_Name"],
            "Experiment_Block": experiment["Experiment_Block"],
            "Depot_Capacity_Level": experiment["Depot_Capacity_Level"],
            "Vehicle_Capacity_Level": experiment["Vehicle_Capacity_Level"],
            "Run": run_number,
            "Seed": seed,
            "Priority_Profile": experiment["Priority_Profile"],
            "Vehicle_Cap_Q": experiment["Vehicle_Cap_Q"],
            "Depot_Cap_W": experiment["Depot_Cap_W"],
            "Alpha_Weight": experiment["Alpha_Weight"],
            "Is_Feasible": False,
            "Structural_Feasible": False,
            "Failure_Reason": experiment["Structural_Check"],
            "Coverage_OK": False,
            "Num_Served": None,
            "Num_Unique_Served": None,
            "Missing_Customers": None,
            "Duplicate_Customers": None,
            "Total_Cost": None,
            "Financial_Cost": None,
            "Equity_Penalty": None,
            "Num_SWH_Opened": None,
            "Num_Routes": None,
            "CPU_Time_s": 0.0,
        })
    return rows


def _run_parallel_scenario_worker(task):
    """
    Run all seeds for one scenario.

    This function must remain at module level so Windows spawned processes can
    import it. It returns records only and never writes an Excel workbook.
    """
    experiment = task["experiment"]
    resolved_city = task["resolved_city"]
    run_seeds = task["run_seeds"]

    instance = Instance.from_real_data(
        city_name=resolved_city,
        base_path=task["base_path"],
        vehicle_cap=experiment["Vehicle_Cap_Q"],
        max_route_dist=task["max_route_dist"],
        depot_cap=experiment["Depot_Cap_W"],
        required_open_depots=task["required_open_depots"],
        max_vehicles=task["max_vehicles"],
        fixed_depot_cost=task["fixed_depot_cost"],
        fixed_vehicle_cost=task["fixed_vehicle_cost"],
        cost_per_km=task["cost_per_km"],
        equity_weight=experiment["Alpha_Weight"],
        travel_metric=task["travel_metric"],
        candidate_depot_ids=task["candidate_depot_ids"],
    )
    instance.priorities = build_priority_profile(
        experiment["Priority_Profile"],
        instance.priority_profiles,
        instance.demands,
    )

    rows = []
    iteration_rows = []
    scenario_start = time.time()

    for run_number, seed in enumerate(run_seeds, start=1):
        run_seed = int(seed)
        random.seed(run_seed)
        np.random.seed(run_seed)

        start_time = time.time()
        solution, iteration_logs = run_grasp_vnd_logged(
            instance,
            max_iter=task["max_iter"],
            alpha_max=task["alpha_max"],
            run_id=run_number,
            seed=run_seed,
            instance_name=experiment["Scenario_Name"],
        )
        elapsed = time.time() - start_time
        iteration_rows.extend(iteration_logs)

        feasible = solution.check_validity() if solution else False
        if solution:
            solution.calculate_cost()
        coverage = (
            solution.coverage_status()
            if solution
            else (False, None, None, None, None)
        )

        rows.append({
            "City": resolved_city,
            "Scenario_ID": experiment["Scenario_ID"],
            "Scenario": experiment["Scenario_Name"],
            "Experiment_Block": experiment["Experiment_Block"],
            "Depot_Capacity_Level": experiment["Depot_Capacity_Level"],
            "Vehicle_Capacity_Level": experiment["Vehicle_Capacity_Level"],
            "Run": run_number,
            "Seed": run_seed,
            "Priority_Profile": experiment["Priority_Profile"],
            "Vehicle_Cap_Q": experiment["Vehicle_Cap_Q"],
            "Depot_Cap_W": experiment["Depot_Cap_W"],
            "Alpha_Weight": experiment["Alpha_Weight"],
            "Is_Feasible": bool(feasible),
            "Structural_Feasible": True,
            "Failure_Reason": None if feasible else "No feasible solution found",
            "Coverage_OK": coverage[0],
            "Num_Served": coverage[1],
            "Num_Unique_Served": coverage[2],
            "Missing_Customers": coverage[3],
            "Duplicate_Customers": coverage[4],
            "Total_Cost": (
                round(float(solution.cost), 2)
                if solution and feasible else None
            ),
            "Financial_Cost": (
                round(float(solution.financial_cost), 2)
                if solution and feasible else None
            ),
            "Equity_Penalty": (
                round(float(solution.equity_penalty), 2)
                if solution and feasible else None
            ),
            "Num_SWH_Opened": (
                len(solution.open_depots)
                if solution and feasible else None
            ),
            "Num_Routes": (
                len([route for route in solution.routes if route["path"]])
                if solution and feasible else None
            ),
            "CPU_Time_s": round(float(elapsed), 2),
        })

    return {
        "scenario_id": int(experiment["Scenario_ID"]),
        "scenario_name": experiment["Scenario_Name"],
        "rows": rows,
        "iteration_rows": iteration_rows,
        "scenario_elapsed_s": round(time.time() - scenario_start, 2),
        "worker_pid": os.getpid(),
    }


def _validate_checkpoint_design(checkpoint_file, scenario_design):
    previous_design = pd.read_excel(
        checkpoint_file, sheet_name="Scenario_Design"
    )
    comparison_columns = [
        "Scenario_ID", "Scenario_Name", "Experiment_Block",
        "Depot_Capacity_Level", "Vehicle_Capacity_Level",
        "Vehicle_Cap_Q", "Depot_Cap_W", "Alpha_Weight",
        "Priority_Profile", "Structural_Feasible", "Structural_Check",
    ]
    missing = [
        column for column in comparison_columns
        if column not in previous_design.columns
    ]
    if missing:
        raise ValueError(
            f"Checkpoint is missing design columns: {missing}"
        )
    left = previous_design[comparison_columns].sort_values(
        "Scenario_ID"
    ).reset_index(drop=True)
    right = scenario_design[comparison_columns].sort_values(
        "Scenario_ID"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
    )


def get_allocated_cpu_count():
    """Return the SGE slot allocation, with safe local fallbacks."""
    value = os.environ.get("NSLOTS")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, int(os.cpu_count() or 1))


def run_bulk_experiments_parallel(
    city_name,
    base_path=DEFAULT_BASE_DATA_PATH,
    depot_capacity_levels=None,
    vehicle_capacity_levels=None,
    equity_weights=None,
    priority_profiles=None,
    run_seeds=(0, 1, 2, 3, 4),
    max_iter=50,
    alpha_max=5,
    travel_metric=DEFAULT_TRAVEL_METRIC,
    max_route_duration=DEFAULT_MAX_ROUTE_DURATION_SECONDS,
    fixed_depot_cost=DEFAULT_FIXED_DEPOT_COST,
    fixed_vehicle_cost=DEFAULT_FIXED_VEHICLE_COST,
    cost_per_km=DEFAULT_COST_PER_KM,
    output_file=None,
    candidate_depot_ids=None,
    max_workers=4,
    scenario_ids=None,
    checkpoint_file=None,
    checkpoint_every=1,
    resume=True,
):
    """
    Run independent scenarios in separate processes.

    Each worker executes one scenario's seeds sequentially. Only the parent
    process writes the checkpoint and final workbooks.
    """
    resolved_city, base_instance = load_city_instance(
        city_name=city_name,
        base_path=base_path,
        equity_weight=0.0,
        travel_metric=travel_metric,
        max_route_duration=max_route_duration,
        fixed_depot_cost=fixed_depot_cost,
        fixed_vehicle_cost=fixed_vehicle_cost,
        cost_per_km=cost_per_km,
        candidate_depot_ids=candidate_depot_ids,
    )

    if depot_capacity_levels is None or vehicle_capacity_levels is None:
        auto_w, auto_q = derive_capacity_levels(base_instance, resolved_city)
        if depot_capacity_levels is None:
            depot_capacity_levels = auto_w
        if vehicle_capacity_levels is None:
            vehicle_capacity_levels = auto_q

    all_experiments = build_two_block_scenarios(
        depot_capacity_levels=depot_capacity_levels,
        vehicle_capacity_levels=vehicle_capacity_levels,
        equity_weights=equity_weights,
        priority_profiles=priority_profiles,
    )
    if len(all_experiments) != 52:
        raise RuntimeError(
            f"Expected 52 unique scenarios, obtained {len(all_experiments)}."
        )

    for experiment in all_experiments:
        ok, reason = structural_feasibility_check(
            base_instance,
            experiment["Depot_Cap_W"],
            experiment["Vehicle_Cap_Q"],
        )
        experiment["Structural_Feasible"] = bool(ok)
        experiment["Structural_Check"] = "Passed" if ok else reason

    if scenario_ids is None:
        experiments = all_experiments
    else:
        requested_ids = sorted({int(value) for value in scenario_ids})
        invalid_ids = [
            value for value in requested_ids if value < 1 or value > 52
        ]
        if invalid_ids:
            raise ValueError(
                f"scenario_ids must be between 1 and 52; invalid: {invalid_ids}"
            )
        requested_set = set(requested_ids)
        experiments = [
            experiment for experiment in all_experiments
            if int(experiment["Scenario_ID"]) in requested_set
        ]
        if not experiments:
            raise ValueError("scenario_ids selected no scenarios.")

    scenario_design = pd.DataFrame(experiments)
    run_seeds = [int(seed) for seed in run_seeds]
    if not run_seeds:
        raise ValueError("run_seeds cannot be empty.")
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    checkpoint_every = int(checkpoint_every)
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")

    if output_file is None:
        suffix = (
            "PARALLEL"
            if len(experiments) == 52
            else f"PARALLEL_TEST_{len(experiments)}Scenarios"
        )
        output_file = os.path.join(
            base_path,
            resolved_city,
            f"{resolved_city}_Bulk_Experiment_52Scenarios_"
            f"GRASP_VNDZouSO_{suffix}.xlsx",
        )
    output_file = str(Path(output_file).resolve())
    if checkpoint_file is None:
        output_path = Path(output_file)
        checkpoint_file = str(
            output_path.with_name(
                f"{output_path.stem}_checkpoint{output_path.suffix}"
            )
        )
    checkpoint_file = str(Path(checkpoint_file).resolve())

    results_log = []
    iteration_logs_all = []
    completed_ids = set()

    if resume and Path(checkpoint_file).is_file():
        _validate_checkpoint_design(checkpoint_file, scenario_design)
        previous_results = pd.read_excel(
            checkpoint_file, sheet_name="Run_Level_Results"
        )
        previous_iterations = pd.read_excel(
            checkpoint_file, sheet_name="Iteration_Logs"
        )
        for scenario_id, group in previous_results.groupby("Scenario_ID"):
            runs = {int(value) for value in group["Run"].dropna().tolist()}
            if runs == set(range(1, len(run_seeds) + 1)):
                completed_ids.add(int(scenario_id))
        if completed_ids:
            previous_results = previous_results[
                previous_results["Scenario_ID"].isin(completed_ids)
            ]
            results_log = previous_results.to_dict("records")
            completed_names = set(
                previous_results["Scenario"].dropna().astype(str)
            )
            if not previous_iterations.empty and "Instance" in previous_iterations:
                previous_iterations = previous_iterations[
                    previous_iterations["Instance"].astype(str).isin(completed_names)
                ]
            iteration_logs_all = previous_iterations.to_dict("records")
            print(
                f"Resuming from checkpoint: {len(completed_ids)} "
                f"scenario(s) already complete."
            )

    print("=" * 78)
    print(f"{resolved_city.upper()} PARALLEL TWO-BLOCK BULK EXPERIMENT")
    print(
        f"n={base_instance.num_customers}, m={base_instance.num_depots}, "
        f"Ng={base_instance.required_open_depots}, "
        f"Nv={base_instance.max_vehicles}"
    )
    print(f"W levels: {depot_capacity_levels}")
    print(f"Q levels: {vehicle_capacity_levels}")
    print(f"Selected scenarios: {len(experiments)}")
    print(f"Runs per scenario: {len(run_seeds)}")
    print(f"Selected total runs: {len(experiments) * len(run_seeds)}")
    print(f"Requested workers: {max_workers}")
    print(f"Checkpoint: {checkpoint_file}")
    print("=" * 78)

    def write_checkpoint():
        result_df = pd.DataFrame(results_log)
        iteration_df = pd.DataFrame(iteration_logs_all)
        result_df, iteration_df = _sort_bulk_tables(result_df, iteration_df)
        summary_df = _build_bulk_summary(result_df, len(run_seeds))
        _write_bulk_workbook(
            checkpoint_file,
            scenario_design,
            result_df,
            summary_df,
            iteration_df,
        )

    remaining = [
        experiment for experiment in experiments
        if int(experiment["Scenario_ID"]) not in completed_ids
    ]
    completions_since_checkpoint = 0

    for experiment in list(remaining):
        if experiment["Structural_Feasible"]:
            continue
        results_log.extend(
            _structurally_infeasible_rows(
                resolved_city, experiment, run_seeds
            )
        )
        completed_ids.add(int(experiment["Scenario_ID"]))
        remaining.remove(experiment)
        completions_since_checkpoint += 1
        print(
            f"[{experiment['Scenario_ID']:02d}] skipped: "
            f"{experiment['Structural_Check']}"
        )

    if completions_since_checkpoint:
        write_checkpoint()
        completions_since_checkpoint = 0

    task_common = {
        "resolved_city": resolved_city,
        "base_path": str(Path(base_path).resolve()),
        "run_seeds": run_seeds,
        "max_route_dist": base_instance.max_route_dist,
        "required_open_depots": base_instance.required_open_depots,
        "max_vehicles": base_instance.max_vehicles,
        "fixed_depot_cost": fixed_depot_cost,
        "fixed_vehicle_cost": fixed_vehicle_cost,
        "cost_per_km": cost_per_km,
        "travel_metric": travel_metric,
        "candidate_depot_ids": candidate_depot_ids,
        "max_iter": int(max_iter),
        "alpha_max": int(alpha_max),
    }

    if remaining:
        allocated_cpus = get_allocated_cpu_count()
        actual_workers = min(
            max_workers,
            len(remaining),
            allocated_cpus,
        )
        print(
            f"Starting {actual_workers} worker process(es); "
            f"scheduler/affinity allocation={allocated_cpus} CPU(s)."
        )
        process_context = mp.get_context("spawn")
        futures = {}

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=process_context,
            ) as executor:
                for experiment in remaining:
                    task = dict(task_common)
                    task["experiment"] = dict(experiment)
                    future = executor.submit(
                        _run_parallel_scenario_worker, task
                    )
                    futures[future] = experiment

                for future in concurrent.futures.as_completed(futures):
                    experiment = futures[future]
                    try:
                        scenario_result = future.result()
                    except Exception as exc:
                        for pending in futures:
                            pending.cancel()
                        if results_log:
                            write_checkpoint()
                        raise RuntimeError(
                            "Parallel scenario failed: "
                            f"{experiment['Scenario_Name']}. "
                            f"Checkpoint retained at {checkpoint_file}"
                        ) from exc

                    results_log.extend(scenario_result["rows"])
                    iteration_logs_all.extend(
                        scenario_result["iteration_rows"]
                    )
                    scenario_id = int(scenario_result["scenario_id"])
                    completed_ids.add(scenario_id)
                    completions_since_checkpoint += 1

                    feasible_costs = [
                        row["Total_Cost"]
                        for row in scenario_result["rows"]
                        if row["Is_Feasible"]
                        and row["Total_Cost"] is not None
                    ]
                    status = (
                        f"feasible={len(feasible_costs)}/{len(run_seeds)}, "
                        f"best={min(feasible_costs):.2f}"
                        if feasible_costs
                        else f"feasible=0/{len(run_seeds)}"
                    )
                    print(
                        f"[{scenario_id:02d}/{len(experiments)}] "
                        f"{scenario_result['scenario_name']} complete; "
                        f"{status}; worker PID={scenario_result['worker_pid']}; "
                        f"elapsed={scenario_result['scenario_elapsed_s']:.2f}s"
                    )

                    if completions_since_checkpoint >= checkpoint_every:
                        write_checkpoint()
                        completions_since_checkpoint = 0
        except KeyboardInterrupt:
            if results_log:
                write_checkpoint()
            print(
                "\nInterrupted. Completed scenarios were saved to: "
                f"{checkpoint_file}"
            )
            raise

    result_df = pd.DataFrame(results_log)
    iteration_df = pd.DataFrame(iteration_logs_all)
    result_df, iteration_df = _sort_bulk_tables(result_df, iteration_df)
    summary_df = _build_bulk_summary(result_df, len(run_seeds))
    final_output = _write_bulk_workbook(
        output_file,
        scenario_design,
        result_df,
        summary_df,
        iteration_df,
    )
    # Refresh the checkpoint so it is also a complete recoverable copy.
    _write_bulk_workbook(
        checkpoint_file,
        scenario_design,
        result_df,
        summary_df,
        iteration_df,
    )

    print(f"\nParallel bulk report saved: {final_output}")
    print(f"Complete checkpoint retained: {checkpoint_file}")

    return {
        "city": resolved_city,
        "base_instance": base_instance,
        "depot_capacity_levels": depot_capacity_levels,
        "vehicle_capacity_levels": vehicle_capacity_levels,
        "scenario_design": scenario_design,
        "run_results": result_df,
        "scenario_summary": summary_df,
        "iterations": iteration_df,
        "output_file": final_output,
        "checkpoint_file": checkpoint_file,
        "max_workers": max_workers,
    }


def run_city_parallel(
    city_name,
    mode="single",
    base_path=DEFAULT_BASE_DATA_PATH,
    **kwargs,
):
    """
    Local/PyCharm entry point.

    Load and single modes use the validated serial workflow. Bulk mode uses
    process-level scenario parallelism. Both runs serial single first and then
    parallel bulk.
    """
    mode = str(mode).strip().lower()
    valid_modes = {"load", "single", "bulk", "both"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")

    if mode == "load":
        return run_city(
            city_name=city_name,
            mode="load",
            base_path=base_path,
            **kwargs,
        )

    outputs = {}
    if mode in {"single", "both"}:
        single_keys = {
            "runs", "seeds", "max_iter", "alpha_max", "equity_weight",
            "travel_metric", "max_route_duration", "fixed_depot_cost",
            "fixed_vehicle_cost", "cost_per_km", "output_file",
            "single_output_file",
            "candidate_depot_ids",
        }
        single_kwargs = {
            key: value for key, value in kwargs.items()
            if key in single_keys
        }
        if "single_output_file" in single_kwargs:
            single_kwargs["output_file"] = single_kwargs.pop(
                "single_output_file"
            )
        outputs["single"] = run_single_city(
            city_name, base_path=base_path, **single_kwargs
        )

    if mode in {"bulk", "both"}:
        bulk_keys = {
            "depot_capacity_levels", "vehicle_capacity_levels",
            "equity_weights", "priority_profiles", "run_seeds",
            "max_iter", "alpha_max", "travel_metric",
            "max_route_duration", "fixed_depot_cost",
            "fixed_vehicle_cost", "cost_per_km", "output_file",
            "bulk_output_file",
            "candidate_depot_ids", "max_workers", "scenario_ids",
            "checkpoint_file", "checkpoint_every", "resume",
        }
        bulk_kwargs = {
            key: value for key, value in kwargs.items()
            if key in bulk_keys
        }
        if "bulk_output_file" in bulk_kwargs:
            bulk_kwargs["output_file"] = bulk_kwargs.pop(
                "bulk_output_file"
            )
        outputs["bulk"] = run_bulk_experiments_parallel(
            city_name, base_path=base_path, **bulk_kwargs
        )

    return outputs



print(
    "LLRP_CORE_PARALLEL loaded. Use "
    "run_city_parallel(CITY_NAME, mode=...)."
)
