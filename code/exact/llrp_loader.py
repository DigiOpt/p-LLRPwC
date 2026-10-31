"""
Data loader for the de-identified, clean-CSV inputs of the priority-weighted
latency LRP (exact method).

Each city is described by five CSV files in the input directory (flat layout):

    <City>_customers.csv   id, demand, priority_equal, priority_tiered,
                           priority_demand_correlated, priority_negative_correlated,
                           priority_hyper_critical
    <City>_depots.csv      depot_id, at_customer, capacity      (depots co-located
                           with a customer node; capacity = W_i)
    <City>_distance.csv    n x n financial-cost matrix (cost_per_km = 1 => km),
                           first column 'id', columns = customer ids
    <City>_duration.csv    n x n travel-time matrix (seconds), id-labelled
    <City>_params.csv      parameter, value : Ng, Nv, Q, L, fixed_depot_cost,
                           fixed_vehicle_cost, cost_per_km

The data are fully anonymized: customer/depot identifiers are C001.. / D01.., there
are no coordinates, and demand is given directly (no sales). Coordinates are not
used by any algorithm; this loader does not read or require them.

d[j] demand, p[j] priority, c[u][v] financial cost, l[u][v] travel time (seconds,
used for the duration limit L and for latency). Node order: depots 0..m-1 first,
then pharmacies m..m+n-1.
"""
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

DEFAULT_FIXED_DEPOT_COST = 5000.0
DEFAULT_FIXED_VEHICLE_COST = 200.0
DEFAULT_EQUITY_WEIGHT = 0.5

# priority-profile name (short key or full) -> CSV column
PRIORITY_PROFILES = {
    "equal": "priority_equal",
    "tiered": "priority_tiered",
    "demand_correlated": "priority_demand_correlated",
    "negative_correlated": "priority_negative_correlated",
    "hyper_critical": "priority_hyper_critical",
}


def _norm(s):
    return "".join(str(s).lower().split()).replace("-", "").replace("_", "")


def _resolve_priority_column(profile):
    want = _norm(profile)
    for short, col in PRIORITY_PROFILES.items():
        if want in (_norm(short), _norm(col)):
            return col
    raise ValueError(f"Unknown priority_profile {profile!r}. "
                     f"Options: {list(PRIORITY_PROFILES)}")


@dataclass
class LLRPInstance:
    city: str
    I: list          # depot node indices        0 .. m-1
    J: list          # pharmacy node indices     m .. m+n-1
    m: int
    n: int
    demand: dict     # d_j  for j in J
    priority: dict   # p_j  for j in J
    c: np.ndarray    # c_uv  financial cost  (dense (m+n)x(m+n))
    l: np.ndarray    # l_uv  travel duration (seconds)
    Q: float
    W: dict          # W_i  for i in I
    L: float
    Ng: int
    Nv: int
    f: dict          # f_i  fixed opening cost
    F: dict          # F_i  fixed per-vehicle cost
    alpha: float
    depot_labels: list = field(default_factory=list)   # D01 .. D0m
    depot_codes: list = field(default_factory=list)     # co-located customer ids
    pharmacy_codes: list = field(default_factory=list)  # customer ids C001..
    priority_profile: str = "Equal"

    @property
    def V(self):
        return self.I + self.J

    @property
    def total_demand(self):
        return sum(self.demand.values())

    def summary(self) -> str:
        return (f"[{self.city}]  n={self.n} pharmacies, m={self.m} depots | "
                f"Ng={self.Ng}, Nv={self.Nv}, Q={self.Q:g}, "
                f"W={next(iter(self.W.values())):g}, L={self.L:g}s | "
                f"D={self.total_demand:g} | alpha={self.alpha:g}")


def _combine(base, cust_ids, depot_ids):
    """Build the (m+n)x(m+n) matrix (DEPOTS FIRST, then pharmacies). Depots are
    co-located with pharmacies, so a depot's row/col is copied from its co-located
    pharmacy's row/col in the customer-only base matrix."""
    pos = {code: i for i, code in enumerate(cust_ids)}
    dpos = [pos[code] for code in depot_ids]
    n = len(cust_ids)
    d2d = base[np.ix_(dpos, dpos)]
    d2c = base[np.ix_(dpos, range(n))]
    c2d = base[np.ix_(range(n), dpos)]
    return np.block([[d2d, d2c], [c2d, base]])


def _matrix(path, cust_ids):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["id"] = df["id"].astype(str).str.strip()
    df.set_index("id", inplace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df.reindex(index=cust_ids, columns=cust_ids).to_numpy(dtype=np.float64)


def load_instance(city, base_path=".", *, max_route_duration=None,
                  priority_profile="Equal", fixed_depot_cost=None,
                  fixed_vehicle_cost=None, alpha=DEFAULT_EQUITY_WEIGHT, verbose=True):
    """Load one city from the clean CSV inputs and return an LLRPInstance."""
    def p(role):
        fp = os.path.join(base_path, f"{city}_{role}.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {role} CSV for {city}: {fp}")
        return fp

    cust = pd.read_csv(p("customers")); cust.columns = [str(c).strip() for c in cust.columns]
    dep = pd.read_csv(p("depots"));     dep.columns = [str(c).strip() for c in dep.columns]
    par = pd.read_csv(p("params"))
    params = {str(k).strip(): v for k, v in zip(par.iloc[:, 0], par.iloc[:, 1])}

    cust_ids = [str(x).strip() for x in cust["id"].tolist()]
    depot_labels = [str(x).strip() for x in dep["depot_id"].tolist()]
    depot_codes = [str(x).strip() for x in dep["at_customer"].tolist()]
    unknown = sorted(set(depot_codes) - set(cust_ids))
    if unknown:
        raise ValueError(f"Depots reference unknown customers: {unknown}")

    Ng, Nv = int(float(params["Ng"])), int(float(params["Nv"]))
    Q = float(params["Q"])
    L = float(max_route_duration) if max_route_duration is not None else float(params["L"])
    fdc = float(fixed_depot_cost if fixed_depot_cost is not None
                else params.get("fixed_depot_cost", DEFAULT_FIXED_DEPOT_COST))
    fvc = float(fixed_vehicle_cost if fixed_vehicle_cost is not None
                else params.get("fixed_vehicle_cost", DEFAULT_FIXED_VEHICLE_COST))

    col = _resolve_priority_column(priority_profile)
    if col not in cust.columns:
        if col == PRIORITY_PROFILES["equal"]:
            prio_arr = np.ones(len(cust_ids))
        else:
            raise ValueError(f"priority column '{col}' absent in {city}_customers.csv")
    else:
        prio_arr = cust[col].to_numpy(dtype=np.float64)
    demand_arr = cust["demand"].to_numpy(dtype=np.float64)

    c = _combine(_matrix(p("distance"), cust_ids), cust_ids, depot_codes)
    l = _combine(_matrix(p("duration"), cust_ids), cust_ids, depot_codes)

    m, n = len(depot_codes), len(cust_ids)
    I, J = list(range(m)), list(range(m, m + n))
    demand = {m + j: float(demand_arr[j]) for j in range(n)}
    priority = {m + j: float(prio_arr[j]) for j in range(n)}
    cap = {str(dep["depot_id"][i]).strip(): float(dep["capacity"][i]) for i in range(m)}
    W = {i: cap[depot_labels[i]] for i in I}
    f = {i: fdc for i in I}
    F = {i: fvc for i in I}

    inst = LLRPInstance(city=city, I=I, J=J, m=m, n=n, demand=demand, priority=priority,
                        c=c, l=l, Q=Q, W=W, L=L, Ng=Ng, Nv=Nv, f=f, F=F, alpha=alpha,
                        depot_labels=depot_labels, depot_codes=depot_codes,
                        pharmacy_codes=cust_ids, priority_profile=col)

    D = inst.total_demand; max_d = max(demand.values()); Wv = next(iter(W.values()))
    if max_d > Q + 1e-9:         raise ValueError(f"Q={Q:g} < max demand {max_d:g}")
    if max_d > Wv + 1e-9:        raise ValueError(f"W={Wv:g} < max demand {max_d:g}")
    if D > Ng * Wv + 1e-9:       raise ValueError(f"D={D:g} > Ng*W={Ng*Wv:g}")
    if D > Nv * Q + 1e-9:        raise ValueError(f"D={D:g} > Nv*Q={Nv*Q:g}")
    if verbose:
        print(inst.summary())
    return inst


if __name__ == "__main__":
    import sys
    city = sys.argv[1] if len(sys.argv) > 1 else "Afyon"
    base = sys.argv[2] if len(sys.argv) > 2 else "Inputs"
    inst = load_instance(city, base)
    print("Depots:", list(zip(inst.depot_labels, inst.depot_codes)))
