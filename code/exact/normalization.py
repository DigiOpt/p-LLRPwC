"""normalization.py -- anchor normalization and the augmented Tchebycheff scalar.

This is the "normalization builder" for the interior preference experiment
(Table ``tab:exact_interior``, paper 6.4.2).  It is solver-free: given the four
lexicographic anchors of a (city, profile) and a point ``(C_F, P)`` it returns
the normalized objectives and the scalar ``Z`` that the interior MILP minimizes.

Anchor convention (same symbols as the manuscript and llrp_model.build_model_flow):

    Cmin  financial-cost anchor            = min C_F                         (lambda = 0)
    P_C   latency at the cost anchor       = P of the cost-optimal solution  (lambda = 0)
    C_P   cost at the latency anchor       = C_F of the latency-optimal sol. (lambda = 1)
    Pmin  latency anchor                   = min P                           (lambda = 1)

    dC = C_P - Cmin   (cost range across the front)
    dP = P_C - Pmin   (latency range across the front)

Normalized objectives and scalar (paper 3.4, rho = 1e-4):

    Cbar = (C_F - Cmin) / dC ,   Pbar = (P - Pmin) / dP
    Z(lambda) = max((1-lambda) Cbar, lambda Pbar)
                + rho * ((1-lambda) Cbar + lambda Pbar)

The second (rho) term is the augmentation that removes weakly efficient points.
"""
from __future__ import annotations

RHO = 1e-4


def anchors(Cmin: float, P_C: float, C_P: float, Pmin: float) -> dict:
    """Build the anchor/range dict consumed here and by build_model_flow.

    ``dC``/``dP`` are the front ranges; both must be strictly positive for the
    normalization to be well defined.
    """
    dC = C_P - Cmin
    dP = P_C - Pmin
    if dC <= 0 or dP <= 0:
        raise ValueError(f"degenerate anchor range: dC={dC}, dP={dP}")
    return dict(Cmin=float(Cmin), P_C=float(P_C), C_P=float(C_P), Pmin=float(Pmin),
                dC=float(dC), dP=float(dP))


def anchors_from_extremes(cost_anchor: dict, lat_anchor: dict) -> dict:
    """Assemble anchors from the two lexicographic solves.

    ``cost_anchor`` provides ``Cmin`` and ``P_C`` (min C_F, then min P among
    cost-optimal); ``lat_anchor`` provides ``Pmin`` and ``C_P`` (min P, then
    min C_F among latency-optimal).  Both dicts use the keys written by the
    driver: ``C_F`` and ``P``.
    """
    return anchors(Cmin=cost_anchor["C_F"], P_C=cost_anchor["P"],
                   C_P=lat_anchor["C_F"], Pmin=lat_anchor["P"])


def normalize(C_F: float, P: float, anc: dict) -> tuple[float, float]:
    """Return ``(Cbar, Pbar)`` for a point given the anchor dict."""
    return (C_F - anc["Cmin"]) / anc["dC"], (P - anc["Pmin"]) / anc["dP"]


def tchebycheff_Z(C_F: float, P: float, lam: float, anc: dict, rho: float = RHO) -> float:
    """Normalized augmented weighted Tchebycheff scalar ``Z`` of a point."""
    Cbar, Pbar = normalize(C_F, P, anc)
    return max((1.0 - lam) * Cbar, lam * Pbar) + rho * ((1.0 - lam) * Cbar + lam * Pbar)
