# -*- coding: utf-8 -*-

import copy, random
import numpy as np, pandas as pd
from numba import njit

def normalize_instance_name(instance_name):
    return str(instance_name).lower().strip().replace("_", "-")

LLRP_PARAMS = {
    # =====================================================
    # Barreto
    # =====================================================
    "ch50-5": {"vehicle_cap": 160, "num_vehicles": 6, "num_open_depots": 2, "bks": 1661.64},
    "ch75-10": {"vehicle_cap": 160, "num_vehicles": 9, "num_open_depots": 3, "bks": 2362.48},
    "ch100-10": {"vehicle_cap": 200, "num_vehicles": 8, "num_open_depots": 2, "bks": 3788.96},
    "gaskell21-5": {"vehicle_cap": 6000, "num_vehicles": 4, "num_open_depots": 2, "bks": 653.48},
    "gaskell29-5": {"vehicle_cap": 4500, "num_vehicles": 4, "num_open_depots": 2, "bks": 1199.33},
    "gaskell32-5": {"vehicle_cap": 11000, "num_vehicles": 3, "num_open_depots": 2, "bks": 1552.84},
    "gaskell36-5": {"vehicle_cap": 250, "num_vehicles": 4, "num_open_depots": 2, "bks": 1627.17},
    "mi27-5": {"vehicle_cap": 2500, "num_vehicles": 4, "num_open_depots": 2, "bks": 5387.55},
    "mi134-8": {"vehicle_cap": 850, "num_vehicles": 11, "num_open_depots": 3, "bks": 21751.97},
    "or117-14": {"vehicle_cap": 150000, "num_vehicles": 7, "num_open_depots": 3, "bks": 53798.53},

    # =====================================================
    # Prodhon / Prins
    # =====================================================
    "20-5-1-5": {"vehicle_cap": 70, "num_vehicles": 5, "num_open_depots": 3, "bks": 330.00},
    "20-5-1b-5": {"vehicle_cap": 150, "num_vehicles": 3, "num_open_depots": 2, "bks": 608.05},
    "20-5-2-5": {"vehicle_cap": 70, "num_vehicles": 5, "num_open_depots": 3, "bks": 301.97},
    "20-5-2b-5": {"vehicle_cap": 150, "num_vehicles": 3, "num_open_depots": 2, "bks": 486.55},

    "50-5-1-5": {"vehicle_cap": 70, "num_vehicles": 12, "num_open_depots": 3, "bks": 843.93},
    "50-5-1b-5": {"vehicle_cap": 150, "num_vehicles": 6, "num_open_depots": 2, "bks": 1293.46},
    "50-5-2-5": {"vehicle_cap": 70, "num_vehicles": 12, "num_open_depots": 3, "bks": 684.13},
    "50-5-2b-5": {"vehicle_cap": 150, "num_vehicles": 6, "num_open_depots": 3, "bks": 953.25},
    "50-5-2bis-5": {"vehicle_cap": 70, "num_vehicles": 12, "num_open_depots": 3, "bks": 945.45},
    "50-5-2bbis-5": {"vehicle_cap": 150, "num_vehicles": 6, "num_open_depots": 2, "bks": 803.90},
    "50-5-3-5": {"vehicle_cap": 70, "num_vehicles": 12, "num_open_depots": 2, "bks": 831.57},
    "50-5-3b-5": {"vehicle_cap": 150, "num_vehicles": 6, "num_open_depots": 2, "bks": 1101.57},

    "100-5-1-5": {"vehicle_cap": 70, "num_vehicles": 24, "num_open_depots": 3, "bks": 1997.29},
    "100-5-1b-5": {"vehicle_cap": 150, "num_vehicles": 12, "num_open_depots": 3, "bks": 2305.65},
    "100-5-2-5": {"vehicle_cap": 70, "num_vehicles": 24, "num_open_depots": 2, "bks": 1126.39},
    "100-5-2b-5": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 2, "bks": 1506.79},
    "100-5-3-5": {"vehicle_cap": 70, "num_vehicles": 24, "num_open_depots": 2, "bks": 1567.62},
    "100-5-3b-5": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 2, "bks": 1932.96},

    "100-10-1-10": {"vehicle_cap": 70, "num_vehicles": 26, "num_open_depots": 3, "bks": 1457.53},
    "100-10-1b-10": {"vehicle_cap": 150, "num_vehicles": 12, "num_open_depots": 3, "bks": 1894.92},
    "100-10-2-10": {"vehicle_cap": 70, "num_vehicles": 24, "num_open_depots": 3, "bks": 1134.80},
    "100-10-2b-10": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 3, "bks": 1555.71},
    "100-10-3-10": {"vehicle_cap": 70, "num_vehicles": 25, "num_open_depots": 3, "bks": 1204.01},
    "100-10-3b-10": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 3, "bks": 1647.85},

    "200-10-1-10": {"vehicle_cap": 70, "num_vehicles": 49, "num_open_depots": 3, "bks": 2770.45},
    "200-10-1b-10": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 3, "bks": 3270.68},
    "200-10-2-10": {"vehicle_cap": 70, "num_vehicles": 49, "num_open_depots": 3, "bks": 1963.32},
    "200-10-2b-10": {"vehicle_cap": 150, "num_vehicles": 23, "num_open_depots": 3, "bks": 2309.30},
    "200-10-3-10": {"vehicle_cap": 70, "num_vehicles": 48, "num_open_depots": 3, "bks": 2719.34},
    "200-10-3b-10": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 3, "bks": 3174.91},

    # =====================================================
    # Tuzun-Burke - BKS values updated according to RLHEA Best column
    # =====================================================
    "p111112-10": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 3, "bks": 3826.78},
    "p111122-20": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 3, "bks": 3597.64},
    "p111212-10": {"vehicle_cap": 150, "num_vehicles": 10, "num_open_depots": 3, "bks": 3901.18},
    "p111222-20": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 2, "bks": 4058.09},

    "p112112-10": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 2, "bks": 2726.41},
    "p112122-20": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 3, "bks": 2056.84},
    "p112212-10": {"vehicle_cap": 150, "num_vehicles": 12, "num_open_depots": 3, "bks": 1394.65},
    "p112222-20": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 2, "bks": 1614.83},

    "p113112-10": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 3, "bks": 2826.52},
    "p113122-20": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 3, "bks": 2772.98},
    "p113212-10": {"vehicle_cap": 150, "num_vehicles": 12, "num_open_depots": 3, "bks": 1815.62},
    "p113222-20": {"vehicle_cap": 150, "num_vehicles": 11, "num_open_depots": 4, "bks": 1874.42},

    "p121112-10": {"vehicle_cap": 150, "num_vehicles": 21, "num_open_depots": 3, "bks": 6499.53},
    "p121122-20": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 4, "bks": 5571.24},
    "p121212-10": {"vehicle_cap": 150, "num_vehicles": 21, "num_open_depots": 3, "bks": 6337.03},
    "p121222-20": {"vehicle_cap": 150, "num_vehicles": 21, "num_open_depots": 3, "bks": 6349.27},

    "p122112-10": {"vehicle_cap": 150, "num_vehicles": 21, "num_open_depots": 3, "bks": 6018.59},
    "p122122-20": {"vehicle_cap": 150, "num_vehicles": 21, "num_open_depots": 3, "bks": 3705.59},
    "p122212-10": {"vehicle_cap": 150, "num_vehicles": 21, "num_open_depots": 2, "bks": 4013.55},
    "p122222-20": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 3, "bks": 2033.34},

    "p123112-10": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 4, "bks": 4842.05},
    "p123122-20": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 4, "bks": 4647.68},
    "p123212-10": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 3, "bks": 5123.41},
    "p123222-20": {"vehicle_cap": 150, "num_vehicles": 22, "num_open_depots": 5, "bks": 2494.58},

    "p131112-10": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 3, "bks": 5405.04},
    "p131122-20": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 3, "bks": 4870.82},
    "p131212-10": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 3, "bks": 5525.91},
    "p131222-20": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 3, "bks": 5039.22},

    "p132112-10": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 2, "bks": 3831.89},
    "p132122-20": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 2, "bks": 3721.93},
    "p132212-10": {"vehicle_cap": 150, "num_vehicles": 17, "num_open_depots": 3, "bks": 2835.25},
    "p132222-20": {"vehicle_cap": 150, "num_vehicles": 17, "num_open_depots": 3, "bks": 1646.45},

    "p133112-10": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 3, "bks": 4556.33},
    "p133122-20": {"vehicle_cap": 150, "num_vehicles": 16, "num_open_depots": 3, "bks": 3208.21},
    "p133212-10": {"vehicle_cap": 150, "num_vehicles": 17, "num_open_depots": 3, "bks": 2896.63},
    "p133222-20": {"vehicle_cap": 150, "num_vehicles": 17, "num_open_depots": 3, "bks": 2484.68},
    }

def _compact_name(instance_name):
    return normalize_instance_name(instance_name).replace("-", "")

def get_llrp_parameters(instance_name):
    """
    Returns a dictionary with:
        vehicle_cap, num_vehicles, num_open_depots, bks

    The function is permissive about common Barreto naming variants.
    """
    raw = normalize_instance_name(instance_name)
    compact = _compact_name(raw)

    if raw in LLRP_PARAMS:
        return dict(LLRP_PARAMS[raw])

    # Common aliases used in Excel sheets / scripts.
    alias_map = {
        "ch505": "ch50-5",
        "christ505": "ch50-5",
        "ch7510": "ch75-10",
        "christ7510": "ch75-10",
        "ch10010": "ch100-10",
        "christ10010": "ch100-10",
        "gaskell215": "gaskell21-5",
        "gaskell295": "gaskell29-5",
        "gaskell325": "gaskell32-5",
        "gaskell325b": "gaskell32-5b",
        "gaskell365": "gaskell36-5",
        "mi275": "mi27-5",
        "mi275": "mi27-5",
        "mi1348": "mi134-8",
        "mi1348": "mi134-8",
        "or11714": "or117-14",
    }

    if compact in alias_map:
        return dict(LLRP_PARAMS[alias_map[compact]])

    # Conservative fallback for non-Barreto or newly named instances.
    # This lets the algorithm run, but the user should set Nv/Ng explicitly
    # for serious benchmark reporting.
    if "50" in compact and "ch" in compact:
        return dict(LLRP_PARAMS["ch50-5"])
    if "75" in compact and "ch" in compact:
        return dict(LLRP_PARAMS["ch75-10"])
    if "100" in compact and "ch" in compact:
        return dict(LLRP_PARAMS["ch100-10"])

    raise ValueError(
        f"Unknown LLRP instance name: {instance_name}. "
        "Add its vehicle_cap, num_vehicles, and num_open_depots to LLRP_PARAMS."
    )

def get_dynamic_parameters(instance_name):
    p = get_llrp_parameters(instance_name)
    return p["vehicle_cap"]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Instance:
    def __init__(
        self,
        num_depots,
        num_customers,
        vehicle_cap,
        depot_coords,
        cust_coords,
        demands,
        num_vehicles=None,
        num_open_depots=None,
        bks=None,
        instance_name="",
    ):
        self.num_depots = int(num_depots)
        self.num_customers = int(num_customers)
        self.vehicle_cap = float(vehicle_cap)

        self.num_vehicles = None if num_vehicles is None else int(num_vehicles)
        self.num_open_depots = None if num_open_depots is None else int(num_open_depots)
        self.bks = bks
        self.instance_name = instance_name

        self.depot_coords = np.asarray(depot_coords, dtype=float)
        self.cust_coords = np.asarray(cust_coords, dtype=float)
        self.demands = np.asarray(demands, dtype=float)

        all_coords = np.vstack((self.depot_coords, self.cust_coords))
        diff = all_coords[:, np.newaxis, :] - all_coords[np.newaxis, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    def get_dist(self, i, j):
        return self.dist_matrix[i][j]

    @classmethod
    def from_excel(
        cls,
        file_path,
        cust_sheet,
        dep_sheet,
        vehicle_cap=None,
        num_vehicles=None,
        num_open_depots=None,
        bks=None,
    ):
        print(f"Loading data from: {file_path}")

        instance_name = str(cust_sheet).replace("-Customer", "")

        try:
            p = get_llrp_parameters(instance_name)
        except ValueError:
            p = None

        if p is not None:
            if vehicle_cap is None:
                vehicle_cap = p["vehicle_cap"]
            if num_vehicles is None:
                num_vehicles = p["num_vehicles"]
            if num_open_depots is None:
                num_open_depots = p["num_open_depots"]
            if bks is None:
                bks = p["bks"]

        if vehicle_cap is None:
            raise ValueError(
                "vehicle_cap could not be determined. "
                "Pass it explicitly or add the instance to LLRP_PARAMS."
            )

        df_cust = pd.read_excel(file_path, sheet_name=cust_sheet)
        df_dep = pd.read_excel(file_path, sheet_name=dep_sheet)

        df_cust.columns = df_cust.columns.astype(str).str.strip()
        df_dep.columns = df_dep.columns.astype(str).str.strip()

        n_customers = len(df_cust)
        n_depots = len(df_dep)

        cust_coords = df_cust[["X", "Y"]].to_numpy(dtype=float)
        depot_coords = df_dep[["X", "Y"]].to_numpy(dtype=float)
        demands = df_cust["Demand"].to_numpy(dtype=float)

        print(f"Loaded: {n_depots} depots, {n_customers} customers")
        print(f"LLRP parameters: Q={vehicle_cap}, Nv={num_vehicles}, Ng={num_open_depots}")
        print("Objective: total customer latency; return-to-depot is not counted")

        return cls(num_depots=n_depots, num_customers=n_customers, vehicle_cap=vehicle_cap, depot_coords=depot_coords, cust_coords=cust_coords, demands=demands, num_vehicles=num_vehicles, num_open_depots=num_open_depots, bks=bks,          instance_name=instance_name)

class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.routes = []
        self.open_depots = set()

        self.latency_cost = 0.0
        self.open_distance = 0.0
        self.cost = float("inf")
        self.gap = None

    def calculate_cost(self):
        total_latency = 0.0
        total_open_distance = 0.0
        self.open_depots = set()

        for route in self.routes:
            if not route["path"]:
                continue

            self.open_depots.add(route["depot"])

            latency = evaluate_route_latency(route["depot"], route["path"], self.instance)
            distance = evaluate_route_distance_open(route["depot"], route["path"], self.instance)

            route["latency"] = latency
            route["dist"] = distance

            total_latency += latency
            total_open_distance += distance

        self.latency_cost = total_latency
        self.open_distance = total_open_distance
        self.cost = total_latency

        if self.instance.bks:
            self.gap = 100.0 * (self.cost - self.instance.bks) / self.instance.bks
        else:
            self.gap = None

        return self.cost
    
    def check_validity(self):
        self.calculate_cost()

        nonempty_routes = [r for r in self.routes if r["path"]]

    # Vehicle number constraint: at most Nv vehicles
        if self.instance.num_vehicles is not None:
            if len(nonempty_routes) > self.instance.num_vehicles:
              return False

        # Depot number constraint: exactly Ng open depots
        if self.instance.num_open_depots is not None:
            if len(self.open_depots) != self.instance.num_open_depots:
                return False

        # Vehicle capacity constraint
        for r in nonempty_routes:
            if r["load"] > self.instance.vehicle_cap + 1e-4:
                return False
    
        # Customer coverage constraint: each customer must be visited exactly once
        visited = []
        for r in nonempty_routes:
            visited.extend(r["path"])
    
        if len(visited) != self.instance.num_customers:
            return False
    
        if len(set(visited)) != self.instance.num_customers:
            return False
    
        if set(visited) != set(range(self.instance.num_customers)):
            return False
    
        return True
    
# =============================================================================
# ROUTE EVALUATION
# =============================================================================

@njit
def evaluate_route_latency_fast(depot_idx, path_arr, matrix, num_depots):
    """
    Total route latency:
        sum of cumulative arrival times at customers.

    Return-to-depot is not counted.
    """
    if len(path_arr) == 0:
        return 0.0

    total_latency = 0.0
    arrival_time = 0.0
    prev_global = depot_idx

    for k in range(len(path_arr)):
        cust_global = num_depots + path_arr[k]
        arrival_time += matrix[prev_global][cust_global]
        total_latency += arrival_time
        prev_global = cust_global

    return total_latency


@njit
def evaluate_route_distance_open_fast(depot_idx, path_arr, matrix, num_depots):
    """
    Open-route travel distance:
        depot -> customer sequence

    Return-to-depot is not counted.
    """
    if len(path_arr) == 0:
        return 0.0

    dist = 0.0
    prev_global = depot_idx

    for k in range(len(path_arr)):
        cust_global = num_depots + path_arr[k]
        dist += matrix[prev_global][cust_global]
        prev_global = cust_global

    return dist


def evaluate_route_latency(depot_idx, path, instance):
    path_arr = np.array(path, dtype=np.int64)
    return float(
        evaluate_route_latency_fast(
            depot_idx,
            path_arr,
            instance.dist_matrix,
            instance.num_depots,
        )
    )


def evaluate_route_distance_open(depot_idx, path, instance):
    path_arr = np.array(path, dtype=np.int64)
    return float(
        evaluate_route_distance_open_fast(
            depot_idx,
            path_arr,
            instance.dist_matrix,
            instance.num_depots,
        )
    )


def evaluate_route(depot_idx, path, instance):
    """
    Backward-compatible wrapper.

    Old CLRP code expected:
        distance, latency_penalty = evaluate_route(...)

    New LLRP meaning:
        open_distance, total_latency = evaluate_route(...)
    """
    return (
        evaluate_route_distance_open(depot_idx, path, instance),
        evaluate_route_latency(depot_idx, path, instance),
    )


def get_route_obj(depot_idx, path, instance):
    """
    LLRP route objective = total customer latency.
    """
    return evaluate_route_latency(depot_idx, path, instance)


def get_arrival_times(depot_idx, path, instance):
    arrivals = []
    arrival_time = 0.0
    prev_global = depot_idx

    for cust in path:
        cust_global = instance.num_depots + cust
        arrival_time += instance.dist_matrix[prev_global][cust_global]
        arrivals.append(float(arrival_time))
        prev_global = cust_global

    return arrivals


# =============================================================================
# CONSTRUCTION HEURISTIC
# =============================================================================

def _route_count(sol):
    return sum(1 for r in sol.routes if r["path"])


def _select_initial_allowed_depots(instance, allowed_depots):
    """
    Keeps allowed depots within Ng if Ng is specified.
    """
    if allowed_depots:
        allowed = list(dict.fromkeys(int(d) for d in allowed_depots))
    else:
        allowed = list(range(instance.num_depots))

    if instance.num_open_depots is not None:
        max_open = max(1, min(instance.num_open_depots, instance.num_depots))
        if len(allowed) > max_open:
            allowed = random.sample(allowed, max_open)

    if not allowed:
        allowed = [random.randrange(instance.num_depots)]

    return allowed

def repair_route_count_by_relocation(sol, instance):
    """
    Reduces the number of routes to Nv by eliminating small routes.
    Customers of a selected route are inserted into other routes one by one.
    This is especially useful for tight-capacity instances such as Ch75-10.
    """
    if instance.num_vehicles is None:
        return sol

    target_routes = instance.num_vehicles

    while _route_count(sol) > target_routes:
        sol.routes = [r for r in sol.routes if r["path"]]
        sol.calculate_cost()

        # Try to eliminate the smallest-load routes first
        route_order = sorted(range(len(sol.routes)), key=lambda idx: sol.routes[idx]["load"])
        eliminated = False

        for ridx in route_order:
            source_route = sol.routes[ridx]

            if not source_route["path"]:
                continue

            customers_to_move = list(source_route["path"])
            temp_routes = copy.deepcopy(sol.routes)

            # Find corresponding source route in copied list
            temp_source = temp_routes[ridx]
            success = True

            # Move larger-demand customers first
            customers_to_move.sort(key=lambda c: instance.demands[c], reverse=True)

            for cust in customers_to_move:
                cust_demand = instance.demands[cust]
                best_move = None
                best_delta = float("inf")

                for tidx, target_route in enumerate(temp_routes):
                    if tidx == ridx:
                        continue

                    if target_route["load"] + cust_demand > instance.vehicle_cap + 1e-4:
                        continue

                    old_latency = evaluate_route_latency(target_route["depot"], target_route["path"], instance)

                    for pos in range(len(target_route["path"]) + 1):
                        new_path = target_route["path"][:pos] + [cust] + target_route["path"][pos:]
                        new_latency = evaluate_route_latency(target_route["depot"], new_path, instance)
                        delta = new_latency - old_latency

                        if delta < best_delta:
                            best_delta = delta
                            best_move = (tidx, pos, new_path, new_latency)

                if best_move is None:
                    success = False
                    break

                tidx, pos, new_path, new_latency = best_move
                temp_routes[tidx]["path"] = new_path
                temp_routes[tidx]["load"] += cust_demand
                temp_routes[tidx]["latency"] = new_latency
                temp_routes[tidx]["dist"] = evaluate_route_distance_open(temp_routes[tidx]["depot"], new_path, instance)

            if success:
                # Remove eliminated source route
                temp_routes[ridx]["path"] = []
                temp_routes[ridx]["load"] = 0.0
                sol.routes = [r for r in temp_routes if r["path"]]
                sol.calculate_cost()
                eliminated = True
                break

        if not eliminated:
            break

    return sol
    
def ECWA(instance, allowed_depots, rcl_size=5):
 
    sol = Solution(instance)
    allowed_depots = _select_initial_allowed_depots(instance, allowed_depots)

    # -------------------------------------------------------------------------
    # Phase 1: one-customer open routes
    # -------------------------------------------------------------------------
    for cust in range(instance.num_customers):
        demand = instance.demands[cust]
        cust_global = instance.num_depots + cust

        best_depot = min(
            allowed_depots,
            key=lambda d: instance.get_dist(d, cust_global),
        )

        one_way_dist = instance.get_dist(best_depot, cust_global)

        sol.routes.append(
            {
                "depot": best_depot,
                "path": [cust],
                "load": float(demand),
                "dist": float(one_way_dist),
                "latency": float(one_way_dist),
            }
        )

    # -------------------------------------------------------------------------
    # Phase 2: merge until vehicle limit is satisfied.
    # -------------------------------------------------------------------------
    target_routes = instance.num_vehicles if instance.num_vehicles is not None else instance.num_customers

    while _route_count(sol) > target_routes:
        candidates = []
        routes = [r for r in sol.routes if r["path"]]
        n_routes = len(routes)

        for i in range(n_routes):
            for j in range(n_routes):
                if i == j:
                    continue

                r1 = routes[i]
                r2 = routes[j]

                combined_load = r1["load"] + r2["load"]
                if combined_load > instance.vehicle_cap + 1e-4:
                    continue

                old_obj = (
                    evaluate_route_latency(r1["depot"], r1["path"], instance)
                    + evaluate_route_latency(r2["depot"], r2["path"], instance)
                )

                # Keep target depot among already allowed/open depots so Ng is not exceeded.
                candidate_depots = list(dict.fromkeys([r1["depot"], r2["depot"]]))
                candidate_paths = [
                    r1["path"] + r2["path"],
                    r2["path"] + r1["path"],
                ]

                for target_depot in candidate_depots:
                    for merged_path in candidate_paths:
                        new_latency = evaluate_route_latency(target_depot, merged_path, instance)
                        new_distance = evaluate_route_distance_open(target_depot, merged_path, instance)
                        delta = new_latency - old_obj

                        candidates.append({
                            "route_i": r1,
                            "route_j": r2,
                            "delta": delta,
                            "new_latency": new_latency,
                            "new_distance": new_distance,
                            "new_depot": target_depot,
                            "new_path": merged_path,
                        })

        if not candidates:
            # No feasible merge exists, usually because vehicle capacity is too tight.
            # The solution will be marked infeasible by check_validity().
            break

        
        candidates.sort(key=lambda x: x["delta"])
        actual_rcl = min(len(candidates), max(1, rcl_size))
        chosen = random.choice(candidates[:actual_rcl])

        r1 = chosen["route_i"]
        r2 = chosen["route_j"]

        # Remove by object identity.
        sol.routes.remove(r1)
        sol.routes.remove(r2)

        sol.routes.append(
            {
                "depot": chosen["new_depot"],
                "path": list(chosen["new_path"]),
                "load": r1["load"] + r2["load"],
                "dist": chosen["new_distance"],
                "latency": chosen["new_latency"],
            }
        )
    # Additional repair after the merge loop
    sol = repair_route_count_by_relocation(sol, instance)
    
# Force exactly Ng depots to be used if possible.
    if instance.num_open_depots is not None:
        required_ng = instance.num_open_depots
        used_depots = set(r["depot"] for r in sol.routes if r["path"])
    
        if len(used_depots) < required_ng:
            unused_allowed = [d for d in allowed_depots if d not in used_depots]
    
            route_indices = sorted(
                range(len(sol.routes)),
                key=lambda idx: sol.routes[idx]["load"]
            )
    
            idx_pos = 0
            while (
                len(used_depots) < required_ng
                and unused_allowed
                and idx_pos < len(route_indices)
            ):
                route_idx = route_indices[idx_pos]
                new_depot = unused_allowed.pop(0)
    
                sol.routes[route_idx]["depot"] = new_depot
                used_depots.add(new_depot)
    
                idx_pos += 1
    
            sol.calculate_cost()
    
    return sol

def _path_load(path, instance):
    return float(sum(instance.demands[c] for c in path))

def _update_route(route, new_path, instance):
    route["path"] = list(new_path)
    route["load"] = _path_load(route["path"], instance)
    route["latency"] = evaluate_route_latency(route["depot"], route["path"], instance)
    route["dist"] = evaluate_route_distance_open(route["depot"], route["path"], instance)

# =============================================================================
# FAST STRATEGIC OSCILLATION HELPERS
# =============================================================================

SO_WINDOW = 4
SO_EPS = 1e-4
SO_BETA_MIN = 1e-6
SO_BETA_MAX = 1e9

def route_capacity_violation_from_load(load, instance):
    return max(0.0, float(load) - instance.vehicle_cap)

def path_load(path, instance):
    return float(sum(instance.demands[c] for c in path))

def route_penalized_value(depot, path, load, instance, beta):
    latency = evaluate_route_latency(depot, path, instance)
    violation = route_capacity_violation_from_load(load, instance)
    return latency + beta * violation

def initial_beta(sol, instance):
    total_demand = float(np.sum(instance.demands))
    sol.calculate_cost()
    if total_demand <= 0:
        return 1.0
    return max(1.0, sol.cost / total_demand)

def capacity_violation(sol, instance):
    violation = 0.0
    for r in sol.routes:
        if r["path"]:
            violation += route_capacity_violation_from_load(r["load"], instance)
    return violation

def depot_swap_perturbation(sol, instance, beta):
    """
    One open depot is replaced by one closed depot.
    Routes assigned to the removed depot are reassigned to the new depot.
    The move is accepted if the penalized latency improves.
    """

    sol.routes = [r for r in sol.routes if r["path"]]
    sol.calculate_cost()

    open_depots = list(sol.open_depots)
    closed_depots = [d for d in range(instance.num_depots) if d not in sol.open_depots]

    if not open_depots or not closed_depots:
        return False

    old_cost = sol.cost + beta * capacity_violation(sol, instance)

    best_move = None
    best_cost = old_cost

    for old_depot in open_depots:
        affected_routes = [idx for idx, r in enumerate(sol.routes) if r["depot"] == old_depot]

        if not affected_routes:
            continue

        for new_depot in closed_depots:
            temp_routes = copy.deepcopy(sol.routes)

            for ridx in affected_routes:
                temp_routes[ridx]["depot"] = new_depot
                _update_route(temp_routes[ridx], temp_routes[ridx]["path"], instance)

            temp_sol = copy.deepcopy(sol)
            temp_sol.routes = temp_routes
            temp_sol.calculate_cost()

            # Ng korunuyor mu?
            if instance.num_open_depots is not None:
                if len(temp_sol.open_depots) != instance.num_open_depots:
                    continue

            new_cost = temp_sol.cost + beta * capacity_violation(temp_sol, instance)

            if new_cost < best_cost - SO_EPS:
                best_cost = new_cost
                best_move = temp_routes

    if best_move is not None:
        sol.routes = best_move
        sol.calculate_cost()
        return True

    return False

def apply_insertion_so(sol, instance, beta):
    sol.routes = [r for r in sol.routes if r["path"]]

    for r1_idx, r1 in enumerate(sol.routes):
        for i in range(len(r1["path"])):
            node = r1["path"][i]
            node_load = instance.demands[node]

            for r2_idx, r2 in enumerate(sol.routes):

                if r1_idx == r2_idx:
                    old_local = route_penalized_value(
                        r1["depot"], r1["path"], r1["load"], instance, beta
                    )

                    base = list(r1["path"])
                    removed = base[:i] + base[i + 1:]

                    for pos in range(len(removed) + 1):
                        if pos == i:
                            continue

                        new_path = removed[:pos] + [node] + removed[pos:]
                        new_load = r1["load"]

                        new_local = route_penalized_value(
                            r1["depot"], new_path, new_load, instance, beta
                        )

                        if new_local < old_local - SO_EPS:
                            _update_route(r1, new_path, instance)
                            sol.calculate_cost()
                            return True

                else:
                    if len(r1["path"]) <= 1:
                        continue

                    old_local = (
                        route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                        + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
                    )

                    new_r1_path = r1["path"][:i] + r1["path"][i + 1:]
                    new_r1_load = r1["load"] - node_load

                    for pos in range(len(r2["path"]) + 1):
                        new_r2_path = r2["path"][:pos] + [node] + r2["path"][pos:]
                        new_r2_load = r2["load"] + node_load

                        new_local = (
                            route_penalized_value(r1["depot"], new_r1_path, new_r1_load, instance, beta)
                            + route_penalized_value(r2["depot"], new_r2_path, new_r2_load, instance, beta)
                        )

                        if new_local < old_local - SO_EPS:
                            _update_route(r1, new_r1_path, instance)
                            _update_route(r2, new_r2_path, instance)
                            sol.calculate_cost()
                            return True

    return False

def apply_swap_so(sol, instance, beta):
    """
    Swap operator:
    1) depot-depot swap between two routes
    2) customer-customer swap within the same route or between different routes

    This is closer to Zou et al.'s Swap neighborhood, where nodes of the same type
    can be swapped: customer with customer, depot with depot.
    Capacity violation is allowed through the penalized objective.
    """

    sol.routes = [r for r in sol.routes if r["path"]]

    # =========================================================
    # 1) Depot-depot swap between two routes
    # =========================================================
    for r1_idx, r1 in enumerate(sol.routes):
        for r2_idx, r2 in enumerate(sol.routes):
            if r2_idx <= r1_idx:
                continue

            if r1["depot"] == r2["depot"]:
                continue

            old_local = (
                route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
            )

            new_local = (
                route_penalized_value(r2["depot"], r1["path"], r1["load"], instance, beta)
                + route_penalized_value(r1["depot"], r2["path"], r2["load"], instance, beta)
            )

            if new_local < old_local - SO_EPS:
                old_depot_r1 = r1["depot"]
                old_depot_r2 = r2["depot"]

                r1["depot"] = old_depot_r2
                r2["depot"] = old_depot_r1

                _update_route(r1, r1["path"], instance)
                _update_route(r2, r2["path"], instance)

                sol.calculate_cost()
                return True

    # =========================================================
    # 2) Customer-customer swap
    # =========================================================
    for r1_idx, r1 in enumerate(sol.routes):
        for r2_idx, r2 in enumerate(sol.routes):
            if r2_idx < r1_idx:
                continue

            for i in range(len(r1["path"])):
                for j in range(len(r2["path"])):

                    # -------------------------------------------------
                    # Intra-route customer swap
                    # -------------------------------------------------
                    if r1_idx == r2_idx:
                        if i >= j:
                            continue

                        old_local = route_penalized_value(
                            r1["depot"], r1["path"], r1["load"], instance, beta
                        )

                        new_path = list(r1["path"])
                        new_path[i], new_path[j] = new_path[j], new_path[i]

                        new_local = route_penalized_value(
                            r1["depot"], new_path, r1["load"], instance, beta
                        )

                        if new_local < old_local - SO_EPS:
                            _update_route(r1, new_path, instance)
                            sol.calculate_cost()
                            return True

                    # -------------------------------------------------
                    # Inter-route customer swap
                    # -------------------------------------------------
                    else:
                        n1 = r1["path"][i]
                        n2 = r2["path"][j]

                        new_r1_path = list(r1["path"])
                        new_r2_path = list(r2["path"])

                        new_r1_path[i] = n2
                        new_r2_path[j] = n1

                        new_r1_load = r1["load"] - instance.demands[n1] + instance.demands[n2]
                        new_r2_load = r2["load"] - instance.demands[n2] + instance.demands[n1]

                        old_local = (
                            route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                            + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
                        )

                        new_local = (
                            route_penalized_value(r1["depot"], new_r1_path, new_r1_load, instance, beta)
                            + route_penalized_value(r2["depot"], new_r2_path, new_r2_load, instance, beta)
                        )

                        if new_local < old_local - SO_EPS:
                            _update_route(r1, new_r1_path, instance)
                            _update_route(r2, new_r2_path, instance)
                            sol.calculate_cost()
                            return True

    return False

def apply_swapstar_so(sol, instance, beta):
    """
    Swap* operator:
    remove one customer from route A and one customer from route B,
    then insert each customer into the best position of the opposite route.
    Capacity violation is allowed through penalized objective.
    """

    sol.routes = [r for r in sol.routes if r["path"]]

    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 1:
            continue

        for r2_idx, r2 in enumerate(sol.routes):
            if r2_idx <= r1_idx or len(r2["path"]) < 1:
                continue

            old_local = (
                route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
            )

            best_move = None
            best_value = old_local

            for i in range(len(r1["path"])):
                c1 = r1["path"][i]
                load_c1 = instance.demands[c1]
                r1_removed = r1["path"][:i] + r1["path"][i + 1:]

                for j in range(len(r2["path"])):
                    c2 = r2["path"][j]
                    load_c2 = instance.demands[c2]
                    r2_removed = r2["path"][:j] + r2["path"][j + 1:]

                    if not r1_removed or not r2_removed:
                        continue

                    base_load_r1 = r1["load"] - load_c1
                    base_load_r2 = r2["load"] - load_c2

                    # insert c2 into route 1 at best position
                    for pos1 in range(len(r1_removed) + 1):
                        new_r1_path = r1_removed[:pos1] + [c2] + r1_removed[pos1:]
                        new_r1_load = base_load_r1 + load_c2

                        val_r1 = route_penalized_value(
                            r1["depot"], new_r1_path, new_r1_load, instance, beta
                        )

                        # insert c1 into route 2 at best position
                        for pos2 in range(len(r2_removed) + 1):
                            new_r2_path = r2_removed[:pos2] + [c1] + r2_removed[pos2:]
                            new_r2_load = base_load_r2 + load_c1

                            val_r2 = route_penalized_value(
                                r2["depot"], new_r2_path, new_r2_load, instance, beta
                            )

                            new_value = val_r1 + val_r2

                            if new_value < best_value - SO_EPS:
                                best_value = new_value
                                best_move = (
                                    new_r1_path,
                                    new_r2_path,
                                )

            if best_move is not None:
                new_r1_path, new_r2_path = best_move
                _update_route(r1, new_r1_path, instance)
                _update_route(r2, new_r2_path, instance)
                sol.calculate_cost()
                return True

    return False

def apply_2opt_so(sol, instance, beta):
    sol.routes = [r for r in sol.routes if r["path"]]

    for r in sol.routes:
        if len(r["path"]) < 3:
            continue

        old_local = route_penalized_value(
            r["depot"], r["path"], r["load"], instance, beta
        )

        for i in range(len(r["path"]) - 1):
            for j in range(i + 2, len(r["path"]) + 1):
                new_path = r["path"][:i] + r["path"][i:j][::-1] + r["path"][j:]

                new_local = route_penalized_value(
                    r["depot"], new_path, r["load"], instance, beta
                )

                if new_local < old_local - SO_EPS:
                    _update_route(r, new_path, instance)
                    sol.calculate_cost()
                    return True

    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 2:
            continue

        for r2_idx, r2 in enumerate(sol.routes):
            if r2_idx <= r1_idx or len(r2["path"]) < 2:
                continue

            old_local = (
                route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
            )

            for cut1 in range(1, len(r1["path"])):
                for cut2 in range(1, len(r2["path"])):
                    new_r1_path = r1["path"][:cut1] + r2["path"][cut2:]
                    new_r2_path = r2["path"][:cut2] + r1["path"][cut1:]

                    if not new_r1_path or not new_r2_path:
                        continue

                    new_r1_load = path_load(new_r1_path, instance)
                    new_r2_load = path_load(new_r2_path, instance)

                    new_local = (
                        route_penalized_value(r1["depot"], new_r1_path, new_r1_load, instance, beta)
                        + route_penalized_value(r2["depot"], new_r2_path, new_r2_load, instance, beta)
                    )

                    if new_local < old_local - SO_EPS:
                        _update_route(r1, new_r1_path, instance)
                        _update_route(r2, new_r2_path, instance)
                        sol.calculate_cost()
                        return True

    return False

def apply_arc_swap_so(sol, instance, beta):
    sol.routes = [r for r in sol.routes if r["path"]]

    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 2:
            continue

        for r2_idx, r2 in enumerate(sol.routes):
            if r2_idx < r1_idx or len(r2["path"]) < 2:
                continue

            for i in range(len(r1["path"]) - 1):
                block1 = r1["path"][i:i + 2]

                for j in range(len(r2["path"]) - 1):
                    block2 = r2["path"][j:j + 2]

                    if r1_idx == r2_idx:
                        if abs(i - j) < 2:
                            continue

                        old_local = route_penalized_value(
                            r1["depot"], r1["path"], r1["load"], instance, beta
                        )

                        new_path = list(r1["path"])
                        new_path[i:i + 2] = block2
                        new_path[j:j + 2] = block1

                        new_local = route_penalized_value(
                            r1["depot"], new_path, r1["load"], instance, beta
                        )

                        if new_local < old_local - SO_EPS:
                            _update_route(r1, new_path, instance)
                            sol.calculate_cost()
                            return True

                    else:
                        old_local = (
                            route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                            + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
                        )

                        new_r1_path = list(r1["path"])
                        new_r2_path = list(r2["path"])

                        new_r1_path[i:i + 2] = block2
                        new_r2_path[j:j + 2] = block1

                        new_r1_load = path_load(new_r1_path, instance)
                        new_r2_load = path_load(new_r2_path, instance)

                        new_local = (
                            route_penalized_value(r1["depot"], new_r1_path, new_r1_load, instance, beta)
                            + route_penalized_value(r2["depot"], new_r2_path, new_r2_load, instance, beta)
                        )

                        if new_local < old_local - SO_EPS:
                            _update_route(r1, new_r1_path, instance)
                            _update_route(r2, new_r2_path, instance)
                            sol.calculate_cost()
                            return True

    return False

def apply_2relocate_so(sol, instance, beta):
    """
    2-relocate operator:
    relocates two consecutive customers either within the same route
    or from one route to another route.
    Capacity violation is allowed through the penalized objective.
    """

    sol.routes = [r for r in sol.routes if r["path"]]

    # =========================================================
    # 1) Intra-route 2-relocate
    # =========================================================
    for r in sol.routes:
        if len(r["path"]) < 3:
            continue

        old_local = route_penalized_value(
            r["depot"], r["path"], r["load"], instance, beta
        )

        n = len(r["path"])

        for i in range(n - 1):
            block = r["path"][i:i + 2]
            remaining = r["path"][:i] + r["path"][i + 2:]

            for pos in range(len(remaining) + 1):
                # Avoid reinserting into the original equivalent position
                if pos == i:
                    continue

                new_path = remaining[:pos] + block + remaining[pos:]

                if new_path == r["path"]:
                    continue

                new_local = route_penalized_value(
                    r["depot"], new_path, r["load"], instance, beta
                )

                if new_local < old_local - SO_EPS:
                    _update_route(r, new_path, instance)
                    sol.calculate_cost()
                    return True

    # =========================================================
    # 2) Inter-route 2-relocate
    # =========================================================
    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 3:
            continue

        for r2_idx, r2 in enumerate(sol.routes):
            if r1_idx == r2_idx:
                continue

            old_local = (
                route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
            )

            for i in range(len(r1["path"]) - 1):
                block = r1["path"][i:i + 2]
                block_load = instance.demands[block[0]] + instance.demands[block[1]]

                new_r1_path = r1["path"][:i] + r1["path"][i + 2:]
                new_r1_load = r1["load"] - block_load

                if not new_r1_path:
                    continue

                for pos in range(len(r2["path"]) + 1):
                    new_r2_path = r2["path"][:pos] + block + r2["path"][pos:]
                    new_r2_load = r2["load"] + block_load

                    new_local = (
                        route_penalized_value(r1["depot"], new_r1_path, new_r1_load, instance, beta)
                        + route_penalized_value(r2["depot"], new_r2_path, new_r2_load, instance, beta)
                    )

                    if new_local < old_local - SO_EPS:
                        _update_route(r1, new_r1_path, instance)
                        _update_route(r2, new_r2_path, instance)
                        sol.calculate_cost()
                        return True

    return False

def apply_shift2_1_so(sol, instance, beta):
    """
    Full node-arc swap operator:
    exchanges one single customer node with one arc, i.e., two consecutive customers.

    This implementation includes:
        1) intra-route node-arc swap
        2) inter-route node-arc swap

    Capacity violation is allowed through the penalized objective.
    """

    sol.routes = [r for r in sol.routes if r["path"]]

    # =========================================================
    # 1) Intra-route node-arc swap
    # =========================================================
    for r in sol.routes:
        path = r["path"]

        if len(path) < 3:
            continue

        old_local = route_penalized_value(
            r["depot"], path, r["load"], instance, beta
        )

        n = len(path)

        for i in range(n):
            single = path[i]

            for j in range(n - 1):
                # arc = path[j], path[j+1]
                # single cannot be inside the selected arc
                if i == j or i == j + 1:
                    continue

                arc = path[j:j + 2]

                if i < j:
                    # Example:
                    # [..., single, middle, arc1, arc2, ...]
                    # -> [..., arc1, arc2, middle, single, ...]
                    new_path = (
                        path[:i]
                        + arc
                        + path[i + 1:j]
                        + [single]
                        + path[j + 2:]
                    )

                else:
                    # j + 1 < i
                    # Example:
                    # [..., arc1, arc2, middle, single, ...]
                    # -> [..., single, middle, arc1, arc2, ...]
                    new_path = (
                        path[:j]
                        + [single]
                        + path[j + 2:i]
                        + arc
                        + path[i + 1:]
                    )

                if new_path == path:
                    continue

                new_local = route_penalized_value(
                    r["depot"], new_path, r["load"], instance, beta
                )

                if new_local < old_local - SO_EPS:
                    _update_route(r, new_path, instance)
                    sol.calculate_cost()
                    return True

    # =========================================================
    # 2) Inter-route node-arc swap
    # =========================================================
    for r1_idx, r1 in enumerate(sol.routes):
        if len(r1["path"]) < 2:
            continue

        for r2_idx, r2 in enumerate(sol.routes):
            if r1_idx == r2_idx or len(r2["path"]) < 1:
                continue

            old_local = (
                route_penalized_value(r1["depot"], r1["path"], r1["load"], instance, beta)
                + route_penalized_value(r2["depot"], r2["path"], r2["load"], instance, beta)
            )

            for i in range(len(r1["path"]) - 1):
                arc = r1["path"][i:i + 2]
                arc_load = instance.demands[arc[0]] + instance.demands[arc[1]]

                for j in range(len(r2["path"])):
                    single = r2["path"][j]
                    single_load = instance.demands[single]

                    new_r1_path = r1["path"][:i] + [single] + r1["path"][i + 2:]
                    new_r2_path = r2["path"][:j] + arc + r2["path"][j + 1:]

                    if not new_r1_path or not new_r2_path:
                        continue

                    new_r1_load = r1["load"] - arc_load + single_load
                    new_r2_load = r2["load"] - single_load + arc_load

                    new_local = (
                        route_penalized_value(r1["depot"], new_r1_path, new_r1_load, instance, beta)
                        + route_penalized_value(r2["depot"], new_r2_path, new_r2_load, instance, beta)
                    )

                    if new_local < old_local - SO_EPS:
                        _update_route(r1, new_r1_path, instance)
                        _update_route(r2, new_r2_path, instance)
                        sol.calculate_cost()
                        return True

    return False

def vnd_local_search_so(sol, instance):
    beta = initial_beta(sol, instance)
    feasible_counter = 0
    infeasible_counter = 0

    best_feasible = copy.deepcopy(sol) if sol.check_validity() else None
    best_feasible_cost = best_feasible.cost if best_feasible else float("inf")

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

        # ---------------------------------------------------------
        # Depot-level strategic perturbation
        # ---------------------------------------------------------
        if depot_swap_perturbation(sol, instance, beta):
            pass_improved = True

            if sol.check_validity():
                feasible_counter += 1
                infeasible_counter = 0

                if sol.cost < best_feasible_cost:
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

        # ---------------------------------------------------------
        # Route-level VND-SO
        # ---------------------------------------------------------
        for name, operator in neighborhoods:
            improved = True

            while improved:
                sol.routes = [r for r in sol.routes if r["path"]]
                sol.calculate_cost()

                improved = operator(sol, instance, beta)

                if improved:
                    pass_improved = True

                    if sol.check_validity():
                        feasible_counter += 1
                        infeasible_counter = 0

                        if sol.cost < best_feasible_cost:
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

        if not pass_improved:
            break

    if best_feasible is not None:
        best_feasible.routes = [r for r in best_feasible.routes if r["path"]]
        best_feasible.calculate_cost()
        return best_feasible

    sol.routes = [r for r in sol.routes if r["path"]]
    sol.calculate_cost()
    return sol

# =============================================================================
# GRASP-VND MAIN LOOPS
# =============================================================================

def _initial_allowed_depots_for_iteration(instance, div_mode, best_depots_div):
    max_open = instance.num_open_depots if instance.num_open_depots is not None else instance.num_depots
    max_open = max(1, min(max_open, instance.num_depots))

    # Diversification: choose exactly Ng depots
    if div_mode or not best_depots_div:
        return random.sample(range(instance.num_depots), max_open)

    # Intensification: reuse best depot set, but enforce exactly Ng depots
    depots = list(best_depots_div)

    if len(depots) > max_open:
        depots = random.sample(depots, max_open)

    elif len(depots) < max_open:
        missing = [d for d in range(instance.num_depots) if d not in depots]
        depots.extend(random.sample(missing, max_open - len(depots)))

    return depots

def run_grasp_vnd(instance, max_iter=50, alpha_max=5):
    best_sol = None
    best_cost = float("inf")

    div_mode = True
    max_it_div = 5
    max_it_int = 3
    period_counter = 0

    best_depots_div = None
    best_cost_div = float("inf")

    for k in range(max_iter):
        allowed_depots = _initial_allowed_depots_for_iteration(instance, div_mode, best_depots_div)

        current_sol = ECWA(instance, allowed_depots, rcl_size=alpha_max)
        #current_sol = vnd_local_search(current_sol, instance)
        current_sol = vnd_local_search_so(current_sol, instance)
        feasible = current_sol.check_validity()

        if feasible and current_sol.cost < best_cost:
            best_sol = copy.deepcopy(current_sol)
            best_cost = current_sol.cost
            print(f"VND GRASP Iter {k}: New Best Latency = {best_cost:.2f}")

        if div_mode and feasible and current_sol.cost < best_cost_div:
            best_cost_div = current_sol.cost
            best_depots_div = set(current_sol.open_depots)

        period_counter += 1

        if div_mode and period_counter >= max_it_div:
            div_mode = False
            period_counter = 0
        elif not div_mode and period_counter >= max_it_int:
            div_mode = True
            period_counter = 0
            best_depots_div = None
            best_cost_div = float("inf")

    return best_sol if best_sol is not None else current_sol


def run_grasp_vnd_logged(instance, max_iter=50, alpha_max=5, run_id=1, seed=0, instance_name=""):
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
        allowed_depots = _initial_allowed_depots_for_iteration(instance, div_mode, best_depots_div)

        current_sol = ECWA(instance, allowed_depots, rcl_size=alpha_max)
        #current_sol = vnd_local_search(current_sol, instance)
        current_sol = vnd_local_search_so(current_sol, instance)

        feasible = current_sol.check_validity() if current_sol else False
        improved = False

        if feasible and current_sol.cost < best_cost:
            best_sol = copy.deepcopy(current_sol)
            best_cost = current_sol.cost
            improved = True
            print(f"[GRASP-VND0-LLRP] Run {run_id}, Iter {k}: New Best Latency = {best_cost:.2f}")

        if div_mode and feasible and current_sol.cost < best_cost_div:
            best_cost_div = current_sol.cost
            best_depots_div = set(current_sol.open_depots)

        iteration_logs.append(
            {
                "Instance": instance_name,
                "Algorithm": "GRASP-VND0-SO-LLRP",
                "Run": run_id,
                "Seed": seed,
                "Iteration": k,
                "Mode": "Diversification" if div_mode else "Intensification",
                "Feasible": feasible,
                "Current Total Latency": current_sol.cost if current_sol else None,
                "Best Latency So Far": best_cost,
                "Improved Best": improved,
                "BKS/Reference": instance.bks,
                "Gap (%)": current_sol.gap if current_sol else None,
                "Open Depots": str(sorted(list(current_sol.open_depots))) if current_sol else None,
                "Number of Routes": len([r for r in current_sol.routes if r["path"]]) if current_sol else None,
                "Nv": instance.num_vehicles,
                "Ng": instance.num_open_depots,
            }
        )

        period_counter += 1

        if div_mode and period_counter >= max_it_div:
            div_mode = False
            period_counter = 0
        elif not div_mode and period_counter >= max_it_int:
            div_mode = True
            period_counter = 0
            best_depots_div = None
            best_cost_div = float("inf")
    
    if best_sol is not None:
        return best_sol, iteration_logs
    else:
        return current_sol, iteration_logs

# Backward-compatible aliases
VNDInstance = Instance
VNDSolution = Solution
run_vnd = run_grasp_vnd

def extract_routes(solution, instance_name, algorithm):
    rows = []

    if solution is None:
        return rows

    for idx, r in enumerate(solution.routes):
        if not r["path"]:
            continue

        arrivals = get_arrival_times(r["depot"], r["path"], solution.instance)
        route_latency = sum(arrivals)
        open_distance = evaluate_route_distance_open(r["depot"], r["path"], solution.instance)

        rows.append({
            "Instance": instance_name,
            "Algorithm": algorithm,
            "Route ID": idx + 1,
            "Depot": r["depot"],
            "Customers": " -> ".join(map(str, r["path"])),
            "Arrival Times": " -> ".join(f"{a:.2f}" for a in arrivals),
            "Number of Customers": len(r["path"]),
            "Load": r["load"],
            "Route Latency": route_latency,
            "Open Route Distance": open_distance,
        })

    return rows