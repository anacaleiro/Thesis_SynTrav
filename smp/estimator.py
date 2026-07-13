"""
estimator.py — Fit the three SMP components from episode + trip data.

Components
----------
Router      : per (persona, time_bin) transition matrix, Laplace-smoothed.
Hazard      : per (state, persona, episode_role) duration distribution,
              best of log-normal / Weibull / gamma by AIC.
Distance    : per (state, persona) log-normal marginal on trip distances (km).
              Fallback to persona-level when cell < min_obs_cell.

Public interface
----------------
fit(episodes_df, trips_df, cfg) -> dict
    Returns a fitted model dict consumed by simulator.py.

save_model(model, path) / load_model(path) -> dict
    Pickle-based persistence.
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, lognorm, weibull_min

logger = logging.getLogger(__name__)

_DIST_CLASSES = {
    "lognorm":     lognorm,
    "weibull_min": weibull_min,
    "gamma":       gamma,
}



# Internal helpers


def _time_bin(start_min: float, bin_edges: list[int]) -> int:
    """Return 0-based bin index for start_min given bin_edges."""
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= start_min < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2  # clamp to last bin


def _fit_best_distribution(data: np.ndarray) -> dict:
    """
    Fit log-normal, Weibull, and gamma to data (floc=0).
    Select by AIC (lower = better). Returns dict with dist name + params.
    """
    data = data[data > 0]
    if len(data) < 3:
        return None

    best = None
    best_aic = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, dist in _DIST_CLASSES.items():
            try:
                params  = dist.fit(data, floc=0)
                log_l   = np.sum(dist.logpdf(data, *params))
                aic     = 2 * len(params) - 2 * log_l
                if np.isfinite(aic) and aic < best_aic:
                    best_aic  = aic
                    best      = {"dist": name, "params": params, "aic": aic}
            except Exception:
                continue

    return best


def _sample_from_fit(fit: dict, rng: np.random.Generator) -> float:
    """Draw one sample from a fitted distribution dict."""
    dist   = _DIST_CLASSES[fit["dist"]]
    params = fit["params"]
    return float(dist.rvs(*params, random_state=rng))



# Router


def _fit_router(episodes_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Build per-(persona, time_bin) transition count matrices, then row-normalise
    with Laplace smoothing epsilon.

    Returns
    -------
    dict: (persona, bin_idx) -> np.ndarray shape (n_states, n_states)
    """
    states     = cfg["states"]
    s_idx      = {s: i for i, s in enumerate(states)}
    n_s        = len(states)
    bin_edges  = cfg["router_bin_edges"]
    eps        = cfg["laplace_epsilon"]

    # Build consecutive-state pairs within each person-day
    pairs = []
    grp_cols = ["person_id", "day_of_week"]
    for _, day in episodes_df.groupby(grp_cols, sort=False):
        day_sorted = day.sort_values("start_min")
        persona    = day_sorted["group_key"].iloc[0]
        states_seq = day_sorted["state"].tolist()
        starts_seq = day_sorted["start_min"].tolist()
        for i in range(len(states_seq) - 1):
            pairs.append({
                "persona":  persona,
                "bin":      _time_bin(starts_seq[i], bin_edges),
                "from":     states_seq[i],
                "to":       states_seq[i + 1],
            })

    if not pairs:
        logger.warning("Router: no transition pairs found.")
        return {}

    pairs_df = pd.DataFrame(pairs)
    router   = {}

    for (persona, bin_idx), grp in pairs_df.groupby(["persona", "bin"]):
        mat = np.zeros((n_s, n_s), dtype=float)
        for _, row in grp.iterrows():
            fi = s_idx.get(row["from"])
            ti = s_idx.get(row["to"])
            if fi is not None and ti is not None:
                mat[fi, ti] += 1

        # Laplace smoothing then row-normalise
        mat += eps
        row_sums = mat.sum(axis=1, keepdims=True)
        router[(persona, bin_idx)] = mat / row_sums

    logger.info("Router: fitted %d (persona, bin) matrices.", len(router))
    return router



# Hazard


def _fit_hazard(episodes_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Fit duration distributions per (state, persona, episode_role).
    Falls back to (state, episode_role) persona-pooled fit if cell < min_obs_cell.

    Returns
    -------
    dict: (state, persona, role) -> fit_dict  (or None if insufficient data)
    """
    min_obs  = cfg["min_obs_cell"]
    activity = episodes_df[episodes_df["duration_min"] > 0].copy()

    hazard     = {}
    n_fallback = 0

    # Pre-compute persona-pooled fits as fallback
    pooled = {}
    for (state, role), grp in activity.groupby(["state", "episode_role"]):
        fit = _fit_best_distribution(grp["duration_min"].values)
        if fit:
            pooled[(state, role)] = fit

    personas = activity["group_key"].unique()
    for persona in personas:
        persona_data = activity[activity["group_key"] == persona]
        for (state, role), grp in persona_data.groupby(["state", "episode_role"]):
            if len(grp) >= min_obs:
                fit = _fit_best_distribution(grp["duration_min"].values)
            else:
                fit = pooled.get((state, role))
                if fit:
                    n_fallback += 1
            hazard[(state, persona, role)] = fit

    logger.info(
        "Hazard: fitted %d cells (%d used persona-pooled fallback).",
        len(hazard), n_fallback,
    )
    return hazard



# Distance marginal


def _fit_distance(trips_df: pd.DataFrame, episodes_df: pd.DataFrame,
                  cfg: dict) -> dict:
    """
    Fit log-normal distance marginal per (state, persona) on real trip distances.
    Distance in km = hectometers * 0.1.
    Excludes HOME-destination trips (return trips have no meaningful outbound distance).

    Returns
    -------
    dict: (state, persona) -> {"dist": "lognorm", "params": (...), "aic": float}
    """
    c        = cfg["columns"]
    pmap     = cfg["purpose_map"]
    min_obs  = cfg["min_obs_cell"]

    # Keep only non-HOME trips with valid distances
    dist_km  = pd.to_numeric(trips_df[c["distance_hm"]], errors="coerce") * 0.1
    home_val = cfg["home_destination_value"]
    dest_col = trips_df[c["destination_purpose"]].astype(str).str.strip().str.lower()
    mask     = (
        dist_km.notna()
        & (dist_km > 0)
        & (dest_col != home_val.lower())
        & trips_df[c["motive"]].isin(pmap)
    )
    tdf = trips_df[mask].copy()
    tdf["_dist_km"] = dist_km[mask]
    tdf["_state"]   = tdf[c["motive"]].map(pmap)
    tdf["_persona"] = tdf[c["group_key"]]

    distance   = {}
    n_fallback = 0

    # Persona-pooled fallback
    pooled = {}
    for state, grp in tdf.groupby("_state"):
        fit = _fit_best_distribution(grp["_dist_km"].values)
        if fit:
            pooled[state] = fit

    personas = tdf["_persona"].unique()
    for persona in personas:
        pdata = tdf[tdf["_persona"] == persona]
        for state, grp in pdata.groupby("_state"):
            if len(grp) >= min_obs:
                fit = _fit_best_distribution(grp["_dist_km"].values)
            else:
                fit = pooled.get(state)
                if fit:
                    n_fallback += 1
            distance[(state, persona)] = fit

    logger.info(
        "Distance: fitted %d (state, persona) cells (%d fallback).",
        len(distance), n_fallback,
    )
    return distance



# Public API


def fit(episodes_df: pd.DataFrame, trips_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Fit all three SMP components.

    Parameters
    ----------
    episodes_df : pd.DataFrame
        Output of episode_extractor.extract_episodes().
    trips_df : pd.DataFrame
        Raw ODiN trip-level data (same rows used to build episodes_df).
    cfg : dict
        Loaded from config.yaml.

    Returns
    -------
    dict with keys: router, hazard, distance, states, bin_edges, cfg
    """
    print("[estimator] Fitting router...")
    router   = _fit_router(episodes_df, cfg)

    print("[estimator] Fitting hazard functions...")
    hazard   = _fit_hazard(episodes_df, cfg)

    print("[estimator] Fitting distance marginals...")
    distance = _fit_distance(trips_df, episodes_df, cfg)

    model = {
        "router":    router,
        "hazard":    hazard,
        "distance":  distance,
        "states":    cfg["states"],
        "bin_edges": cfg["router_bin_edges"],
        "cfg":       cfg,
    }
    print(
        f"[estimator] Done. Router: {len(router)} matrices | "
        f"Hazard: {len(hazard)} cells | Distance: {len(distance)} cells"
    )
    return model


def save_model(model: dict, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[estimator] Model saved to {path}")


def load_model(path: str | Path) -> dict:
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[estimator] Model loaded from {path}")
    return model
