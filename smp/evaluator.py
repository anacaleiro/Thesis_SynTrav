"""
evaluator.py — Evaluate SMP synthetic trips against real ODiN holdout data.

Wraps the existing Helpers/evaluation.py metric functions and adds:
  - Participation rates per activity state (JSD)
  - Duration distributions per activity state (JSD + Wasserstein)
  - Departure-time histograms (JSD + Wasserstein)
  - Bigram transition frequencies (JSD)
  - Held-out log-likelihood under the fitted SMP
  - All existing metrics: SD, SI, DARD, DailyLoc

Both real and synthetic are filtered to weekday records before any computation.
Output: one-row-per-(persona_group, metric) DataFrame plus an aggregated row,
with a model column = "SMP" for downstream stacking with LLM results.

Public interface
----------------
evaluate_smp(real_df, syn_trips, model, cfg, label, persona_groups) -> pd.DataFrame
prepare_real(trips_df, cfg) -> pd.DataFrame
syn_trips_to_records(syn_trips) -> list[dict]  (adapter for evaluation.py)
"""

import sys
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# Make Helpers importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from Helpers.evaluation import (
    prepare_real_trips,
    jsd_sd, jsd_si, jsd_dard, jsd_dailyloc,
    real_sd, real_si, real_dard, real_dailyloc,
    syn_si, syn_dard,
    SD_BINS, SI_BINS, TIME_BIN_MINUTES, N_TIME_BINS,
)

logger = logging.getLogger(__name__)

_EPS = 1e-10


# Helpers

def _jsd(h1: np.ndarray, h2: np.ndarray) -> float:
    h1 = np.asarray(h1, dtype=float) + _EPS
    h2 = np.asarray(h2, dtype=float) + _EPS
    return float(jensenshannon(h1, h2, base=2) ** 2)


def _hist1d(values, bins):
    counts, _ = np.histogram(values, bins=np.asarray(bins, dtype=float))
    return counts.astype(float)


def prepare_real(trips_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add departure_minutes column and filter to weekday rows."""
    c   = cfg["columns"]
    df  = prepare_real_trips(
        trips_df,
        hour_col=c["departure_hour"],
        min_col=c["departure_min_disp"],
    )
    return df[df[c["day_type"]] == "weekday"].copy()


def syn_trips_to_records(syn_trips: list[dict]) -> list[dict]:
    """
    Convert flat SMP trip list to the syn_records format expected by
    Helpers/evaluation.py (list of person dicts each with a 'trips' list).
    Needed only for SI and DARD which use the per-person structure.
    """
    from collections import defaultdict
    persons = defaultdict(list)
    for t in syn_trips:
        pid = t["person_id"]
        # Map SMP fields to evaluation.py expected keys
        persons[pid].append({
            "time":               _min_to_hhmm(t["departure_min"]),
            "departure_time_class": _min_to_dep_class(t["departure_min"]),
            "purpose":            t["purpose_state"],
            "distance_class":     None,   # SMP uses numeric distance_km
            "distance_km":        t["distance_km"],
            "destination":        "activity",  # never "home" — HOME excluded by design
        })
    return [
        {"person_id": pid, "group_key": trips[0].get("persona_group", ""),
         "trips": trips}
        for pid, trips in persons.items()
    ]


def _min_to_hhmm(minutes: float) -> str:
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h:02d}:{m:02d}"


# Mirrors DEP_TIME_CLASS_RANGE from evaluation.py for SMP departure_min values
_DEP_BINS = [
    (360,  "Before 6:00 AM"),
    (420,  "6:00 AM to 7:00 AM"),
    (480,  "7:00 AM to 8:00 AM"),
    (540,  "8:00 AM to 9:00 AM"),
    (720,  "9am to 12pm"),
    (780,  "12 noon to 1 p.m"),
    (840,  "1:00 PM to 2:00 PM"),
    (960,  "2:00 PM to 4:00 PM"),
    (1020, "4:00 PM to 5:00 PM"),
    (1080, "5:00 PM to 6:00 PM"),
    (1140, "6:00 PM to 7:00 PM"),
    (1200, "7:00 PM to 8:00 PM"),
]

def _min_to_dep_class(minutes: float) -> str:
    for threshold, label in _DEP_BINS:
        if minutes < threshold:
            return label
    return "8 p.m. to midnight"


# Metric extractors for SMP (flat trip list)

def _smp_sd(syn_trips: list[dict]) -> list[float]:
    return [t["distance_km"] for t in syn_trips if t.get("distance_km") is not None]


def _smp_si(syn_trips: list[dict]) -> list[float]:
    """Inter-trip departure gap in minutes, grouped by person_id."""
    from collections import defaultdict
    person_deps = defaultdict(list)
    for t in syn_trips:
        person_deps[t["person_id"]].append(t["departure_min"])
    gaps = []
    for deps in person_deps.values():
        deps_sorted = sorted(deps)
        for i in range(1, len(deps_sorted)):
            g = deps_sorted[i] - deps_sorted[i - 1]
            if g >= 0:
                gaps.append(g)
    return gaps


def _smp_dard(syn_trips: list[dict]) -> list[tuple[int, str]]:
    """(time_bin, purpose_state) tuples for DARD histogram."""
    out = []
    for t in syn_trips:
        dep = t.get("departure_min")
        p   = t.get("purpose_state")
        if dep is None or not p:
            continue
        out.append((int(dep) // TIME_BIN_MINUTES, str(p)))
    return out


def _smp_dailyloc(syn_trips: list[dict]) -> list[int]:
    """Trip count per synthetic person-day."""
    from collections import Counter
    counts = Counter(t["person_id"] for t in syn_trips)
    # Include persons with zero trips (not in syn_trips — they have no records)
    return list(counts.values())


# Additional metrics

def _participation_jsd(
    real_df: pd.DataFrame,
    syn_trips: list[dict],
    cfg: dict,
) -> float:
    """JSD of participation rate (fraction of persons making ≥1 trip to state)."""
    c      = cfg["columns"]
    states = [s for s in cfg["states"] if s != "HOME"]
    label_map = cfg.get("state_to_odin_label", {})
    odin_labels = {label_map.get(s, s) for s in states}

    motive_col = c["motive"]

    real_persons = real_df[c["person_id"]].nunique()
    syn_persons  = len({t["person_id"] for t in syn_trips}) or 1

    real_rates, syn_rates = [], []
    for state in states:
        odin_label = label_map.get(state, state)
        r_count = real_df[real_df[motive_col] == odin_label][c["person_id"]].nunique()
        s_count = sum(1 for t in syn_trips if t.get("purpose_state") == odin_label)
        real_rates.append(r_count / real_persons)
        syn_rates.append(s_count / syn_persons)

    return _jsd(np.array(real_rates), np.array(syn_rates))


def _duration_metrics(
    real_df: pd.DataFrame,
    syn_trips: list[dict],
    cfg: dict,
) -> dict[str, float]:
    """
    Per-state duration JSD and Wasserstein. Aggregated (mean over states).
    Real duration = 'Activity duration (in minutes)' from ODiN trips.
    Synthetic duration = duration_min field in trip dict.
    """
    c         = cfg["columns"]
    label_map = cfg.get("state_to_odin_label", {})
    states    = [s for s in cfg["states"] if s != "HOME"]
    dur_col   = c["activity_duration"]
    dur_bins  = np.array([0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440], dtype=float)

    jsd_vals, wass_vals = [], []

    for state in states:
        odin_label = label_map.get(state, state)

        real_durs = pd.to_numeric(
            real_df.loc[real_df[c["motive"]] == odin_label, dur_col],
            errors="coerce",
        ).dropna().values

        syn_durs = np.array([
            t["duration_min"] for t in syn_trips
            if t.get("purpose_state") == odin_label and t.get("duration_min") is not None
        ])

        if len(real_durs) < 2 or len(syn_durs) < 2:
            continue

        h_real = _hist1d(real_durs, dur_bins)
        h_syn  = _hist1d(syn_durs,  dur_bins)
        jsd_vals.append(_jsd(h_real, h_syn))

        bin_mids = (dur_bins[:-1] + dur_bins[1:]) / 2
        wass_vals.append(wasserstein_distance(
            bin_mids, bin_mids,
            u_weights=h_real / h_real.sum(),
            v_weights=h_syn  / h_syn.sum(),
        ))

    return {
        "duration_jsd":  float(np.mean(jsd_vals))  if jsd_vals  else np.nan,
        "duration_wass": float(np.mean(wass_vals)) if wass_vals else np.nan,
    }


def _departure_metrics(
    real_df: pd.DataFrame,
    syn_trips: list[dict],
    cfg: dict,
) -> dict[str, float]:
    """
    JSD and Wasserstein of departure-time histograms (15-min bins, 0–95).
    """
    c        = cfg["columns"]
    bin_size = 15
    n_bins   = 1440 // bin_size  # 96 bins
    bins     = np.arange(0, 1441, bin_size, dtype=float)

    real_deps = pd.to_numeric(real_df["departure_minutes"], errors="coerce").dropna().values
    syn_deps  = np.array([t["departure_min"] for t in syn_trips if t.get("departure_min") is not None])

    if len(real_deps) < 2 or len(syn_deps) < 2:
        return {"departure_jsd": np.nan, "departure_wass": np.nan}

    h_real = _hist1d(real_deps, bins)
    h_syn  = _hist1d(syn_deps,  bins)

    bin_mids = (bins[:-1] + bins[1:]) / 2
    wass = wasserstein_distance(
        bin_mids, bin_mids,
        u_weights=h_real / h_real.sum(),
        v_weights=h_syn  / h_syn.sum(),
    )
    return {
        "departure_jsd":  _jsd(h_real, h_syn),
        "departure_wass": float(wass),
    }


def _bigram_jsd(
    real_df: pd.DataFrame,
    syn_trips: list[dict],
    cfg: dict,
) -> float:
    """JSD of bigram (consecutive purpose) transition frequencies."""
    c         = cfg["columns"]
    label_map = cfg.get("state_to_odin_label", {})
    states    = [s for s in cfg["states"] if s != "HOME"]
    all_labels = sorted({label_map.get(s, s) for s in states})
    idx        = {l: i for i, l in enumerate(all_labels)}
    n          = len(all_labels)

    h_real = np.zeros(n * n, dtype=float)
    h_syn  = np.zeros(n * n, dtype=float)

    # Real bigrams
    pid_col = c["person_id"]
    mot_col = c["motive"]
    for _, grp in real_df.groupby(pid_col):
        seq = grp.sort_values("departure_minutes")[mot_col].tolist()
        for i in range(len(seq) - 1):
            a, b = idx.get(seq[i]), idx.get(seq[i + 1])
            if a is not None and b is not None:
                h_real[a * n + b] += 1

    # Synthetic bigrams
    from collections import defaultdict
    person_trips = defaultdict(list)
    for t in syn_trips:
        person_trips[t["person_id"]].append(t)
    for trips in person_trips.values():
        seq = [t["purpose_state"] for t in sorted(trips, key=lambda x: x["departure_min"])]
        for i in range(len(seq) - 1):
            a, b = idx.get(seq[i]), idx.get(seq[i + 1])
            if a is not None and b is not None:
                h_syn[a * n + b] += 1

    return _jsd(h_real, h_syn)


def _log_likelihood(
    real_df: pd.DataFrame,
    model: dict,
    cfg: dict,
) -> float:
    """
    Held-out log-likelihood of real trip sequences under the fitted SMP router.
    Averages over all observed transitions.
    Returns nan if group_key column is absent (holdout data without persona labels).
    """
    from smp.episode_extractor import extract_episodes

    c = cfg["columns"]
    if c["group_key"] not in real_df.columns:
        logger.warning(
            "_log_likelihood skipped: '%s' column not in real_df. "
            "Log-likelihood requires persona labels present in training data.",
            c["group_key"],
        )
        return np.nan

    states    = model["states"]
    s_idx     = {s: i for i, s in enumerate(states)}
    bin_edges = model["bin_edges"]

    episodes = extract_episodes(real_df, cfg)
    if episodes.empty:
        return np.nan

    log_liks = []
    grp_cols = ["person_id", "day_of_week"]
    for _, day in episodes.groupby(grp_cols):
        day_sorted = day.sort_values("start_min")
        persona    = day_sorted["group_key"].iloc[0]
        seq        = day_sorted["state"].tolist()
        starts     = day_sorted["start_min"].tolist()

        for i in range(len(seq) - 1):
            bi  = _time_bin_ll(starts[i], bin_edges)
            mat = model["router"].get((persona, bi))
            if mat is None:
                continue
            fi = s_idx.get(seq[i])
            ti = s_idx.get(seq[i + 1])
            if fi is None or ti is None:
                continue
            p = mat[fi, ti]
            if p > 0:
                log_liks.append(np.log(p))

    return float(np.mean(log_liks)) if log_liks else np.nan


def _time_bin_ll(t, bin_edges):
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= t < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2


# Public API

def evaluate_smp(
    real_df:        pd.DataFrame,
    syn_trips:      list[dict],
    model:          dict,
    cfg:            dict,
    label:          str = "",
    n_real_zero:    int = 0,
    seed:           int = 42,
) -> pd.DataFrame:
    """
    Compute all evaluation metrics comparing SMP synthetic trips to real holdout.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real ODiN holdout trips, already processed with prepare_real().
        Must contain 'departure_minutes' column.
    syn_trips : list[dict]
        Output of simulator.simulate() or simulate_population().
    model : dict
        Fitted SMP model from estimator.fit().
    cfg : dict
        Loaded from config.yaml.
    label : str
        Evaluation label (e.g. "holdout_utrecht").
    n_real_zero : int
        Count of zero-trip persons in real holdout (for DailyLoc zero-padding).
    seed : int
        RNG seed for SI/DARD jitter in evaluation.py functions.

    Returns
    -------
    pd.DataFrame
        Columns: model, label, metric, value
        One row per metric, plus an aggregated summary row.
    """
    c = cfg["columns"]
    rng_si   = np.random.default_rng(seed)
    rng_dard = np.random.default_rng(seed)

    # ── Core metrics (SD, SI, DARD, DailyLoc) ────────────────────────────────
    r_sd   = real_sd(real_df, c["distance_hm"])
    # SMP has numeric distance_km — convert to same scale as real (km)
    # real_sd returns hectometers*0.1 = km already; SMP already in km
    s_sd   = _smp_sd(syn_trips)

    r_si   = real_si(real_df, c["person_id"])
    s_si   = _smp_si(syn_trips)

    r_dard = real_dard(real_df, c["motive"])
    s_dard = _smp_dard(syn_trips)

    r_dloc = real_dailyloc(real_df, c["person_id"], n_additional_zeros=n_real_zero)
    s_dloc = _smp_dailyloc(syn_trips)

    # ── Additional metrics ────────────────────────────────────────────────────
    part_jsd   = _participation_jsd(real_df, syn_trips, cfg)
    dur_m      = _duration_metrics(real_df, syn_trips, cfg)
    dep_m      = _departure_metrics(real_df, syn_trips, cfg)
    bigram_jsd = _bigram_jsd(real_df, syn_trips, cfg)
    log_lik    = _log_likelihood(real_df, model, cfg)

    rows = [
        {"model": "SMP", "label": label, "metric": "SD",              "value": jsd_sd(r_sd, s_sd)},
        {"model": "SMP", "label": label, "metric": "SI",              "value": jsd_si(r_si, s_si)},
        {"model": "SMP", "label": label, "metric": "DARD",            "value": jsd_dard(r_dard, s_dard)},
        {"model": "SMP", "label": label, "metric": "DailyLoc",        "value": jsd_dailyloc(r_dloc, s_dloc)},
        {"model": "SMP", "label": label, "metric": "participation_jsd", "value": part_jsd},
        {"model": "SMP", "label": label, "metric": "duration_jsd",    "value": dur_m["duration_jsd"]},
        {"model": "SMP", "label": label, "metric": "duration_wass",   "value": dur_m["duration_wass"]},
        {"model": "SMP", "label": label, "metric": "departure_jsd",   "value": dep_m["departure_jsd"]},
        {"model": "SMP", "label": label, "metric": "departure_wass",  "value": dep_m["departure_wass"]},
        {"model": "SMP", "label": label, "metric": "bigram_jsd",      "value": bigram_jsd},
        {"model": "SMP", "label": label, "metric": "log_likelihood",  "value": log_lik},
    ]

    results_df = pd.DataFrame(rows)
    print(f"\n[evaluator] Results — {label}")
    print(results_df.to_string(index=False))
    return results_df
