"""
simulator.py — Generate synthetic trip sequences using a fitted SMP model.

The simulator steps through a day state-by-state:
  HOME → sample next state (router) → sample duration (hazard) → advance clock
  → repeat until day_length_min is reached → close with LAST_HOME.

Output schema per trip (non-HOME episodes only):
  person_id, persona_group, day_of_week, purpose_state, departure_min,
  arrival_min, duration_min, distance_km

purpose_state uses ODiN Motive label strings (not internal state names) so that
DARD and participation-rate metrics are cross-model comparable with the LLM output.
HOME episodes are excluded from the output list by design — including them would
inflate DailyLoc counts relative to the LLM baseline.

Public interface
----------------
simulate(model, persona_group, day_of_week, n_samples, seed) -> list[dict]
simulate_population(model, persona_list, seed) -> list[dict]
    persona_list: [{"persona_group": str, "day_of_week": str, "n_samples": int}, ...]
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)



# Internal helpers


def _time_bin(t: float, bin_edges: list) -> int:
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= t < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2


def _sample_next_state(model: dict, persona: str, current_state: str,
                       current_time: float, rng: np.random.Generator) -> str:
    """Sample the next state from the router, with global fallback."""
    states    = model["states"]
    s_idx     = {s: i for i, s in enumerate(states)}
    bin_idx   = _time_bin(current_time, model["bin_edges"])

    mat = model["router"].get((persona, bin_idx))
    if mat is None:
        # Fallback: find any matrix for this persona across all bins
        mats = [v for (p, _), v in model["router"].items() if p == persona]
        if not mats:
            # Last resort: uniform over non-HOME states
            non_home = [s for s in states if s != "HOME"]
            return rng.choice(non_home)
        mat = np.mean(mats, axis=0)
        # Re-normalise after averaging
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = mat / np.where(row_sums == 0, 1, row_sums)

    from_idx = s_idx.get(current_state)
    if from_idx is None:
        from_idx = s_idx["HOME"]

    row    = mat[from_idx]
    choice = rng.choice(len(states), p=row / row_sum) if (row_sum := row.sum()) > 0 else rng.choice(len(states))
    return states[choice]


def _sample_duration(model: dict, state: str, persona: str,
                     role: str, rng: np.random.Generator,
                     min_dur: float = 1.0) -> float:
    """Sample dwell duration from fitted hazard; fallback to exponential mean."""
    from scipy.stats import lognorm, weibull_min, gamma as gamma_dist
    _dists = {"lognorm": lognorm, "weibull_min": weibull_min, "gamma": gamma_dist}

    fit = model["hazard"].get((state, persona, role))
    if fit is None:
        # Try any role for this (state, persona)
        for r in ("ACTIVITY", "FIRST_HOME", "MIDDAY_HOME", "LAST_HOME"):
            fit = model["hazard"].get((state, persona, r))
            if fit:
                break
    if fit is None:
        return max(min_dur, float(rng.exponential(60)))

    dist   = _dists[fit["dist"]]
    params = fit["params"]
    for _ in range(10):
        val = float(dist.rvs(*params, random_state=rng))
        if val >= min_dur:
            return val
    return max(min_dur, float(rng.exponential(60)))


def _sample_distance(model: dict, state: str, persona: str,
                     rng: np.random.Generator) -> float:
    """Sample trip distance (km) from fitted distance marginal."""
    from scipy.stats import lognorm, weibull_min, gamma as gamma_dist
    _dists = {"lognorm": lognorm, "weibull_min": weibull_min, "gamma": gamma_dist}

    fit = model["distance"].get((state, persona))
    if fit is None:
        # Fallback: try any persona for this state
        fits = [v for (s, _), v in model["distance"].items() if s == state and v]
        fit  = fits[0] if fits else None
    if fit is None:
        return max(0.1, float(rng.exponential(5.0)))

    dist   = _dists[fit["dist"]]
    params = fit["params"]
    for _ in range(10):
        val = float(dist.rvs(*params, random_state=rng))
        if val > 0:
            return val
    return max(0.1, float(rng.exponential(5.0)))



# Core simulation loop


def _simulate_one_day(model: dict, persona: str, day_of_week: str,
                      person_id: str, rng: np.random.Generator) -> list[dict]:
    """
    Simulate one synthetic person-day. Returns a list of trip dicts
    (non-HOME episodes only).

    The HOME-centred SMP: HOME → ACT_1 → HOME → ACT_2 → HOME → ...
    Each activity is always preceded and followed by a HOME dwell period.
    The router is consulted FROM HOME to choose the next activity; this is a
    simplification that ignores chained activity-to-activity transitions but
    produces a valid baseline for the thesis comparison.
    """
    cfg        = model["cfg"]
    day_len    = cfg.get("day_length_min", 1440)
    label_map  = cfg.get("state_to_odin_label", {})
    max_trips  = 20  # safety ceiling

    trips        = []
    current_time = 0.0

    # ── FIRST_HOME: always sample home dwell before first departure ───────────
    first_home = _sample_duration(model, "HOME", persona, "FIRST_HOME", rng)
    current_time = min(first_home, day_len - 1)

    for _ in range(max_trips):
        if current_time >= day_len:
            break

        # Sample next activity FROM HOME at current departure time
        next_state = _sample_next_state(model, persona, "HOME", current_time, rng)

        if next_state == "HOME":
            # Router sampled HOME→HOME: stay longer, then retry
            extra        = _sample_duration(model, "HOME", persona, "MIDDAY_HOME", rng)
            current_time = min(current_time + extra, day_len)
            continue

        dep_min = current_time
        act_dur = _sample_duration(model, next_state, persona, "ACTIVITY", rng)
        arr_min = min(dep_min + act_dur, day_len)
        dist_km = _sample_distance(model, next_state, persona, rng)

        trips.append({
            "person_id":      person_id,
            "persona_group":  persona,
            "day_of_week":    day_of_week,
            "purpose_state":  label_map.get(next_state, next_state),
            "departure_min":  dep_min,
            "arrival_min":    arr_min,
            "duration_min":   arr_min - dep_min,
            "distance_km":    dist_km,
        })

        current_time = arr_min
        if current_time >= day_len:
            break

        # MIDDAY_HOME: return home and dwell before next activity
        midday = _sample_duration(model, "HOME", persona, "MIDDAY_HOME", rng)
        current_time = min(current_time + midday, day_len)

    return trips



# Public API


def simulate(
    model:       dict,
    persona_group: str,
    day_of_week:   str,
    n_samples:     int,
    seed:          int = 42,
) -> list[dict]:
    """
    Generate n_samples synthetic person-days for one persona group.

    Returns
    -------
    list[dict]  — flat list of trip records (one dict per non-HOME episode).
    """
    rng     = np.random.default_rng(seed)
    results = []
    for i in range(n_samples):
        person_id = f"smp_{persona_group}_{i:04d}"
        trips     = _simulate_one_day(model, persona_group, day_of_week, person_id, rng)
        results.extend(trips)
    return results


def simulate_population(
    model:        dict,
    persona_list: list[dict],
    seed:         int = 42,
) -> list[dict]:
    """
    Generate synthetic trips for a population of personas.

    Parameters
    ----------
    persona_list : list of dicts, each with keys:
        persona_group (str), day_of_week (str), n_samples (int)
    seed : int

    Returns
    -------
    list[dict] — flat list of all trip records across all personas.
    """
    rng     = np.random.default_rng(seed)
    results = []
    counter = 0

    for spec in persona_list:
        persona    = spec["persona_group"]
        dow        = spec["day_of_week"]
        n_samples  = spec["n_samples"]

        for _ in range(n_samples):
            person_id = f"smp_{counter:05d}"
            counter  += 1
            trips     = _simulate_one_day(model, persona, dow, person_id, rng)
            results.extend(trips)

    print(
        f"[simulator] Generated {len(results)} trips "
        f"for {counter} synthetic persons."
    )
    return results
