"""
evaluation.py — Trajectory evaluation metrics following LLMob (Wang et al., 2024).

Metrics (all as Jensen-Shannon Divergence, base-2, range [0,1]; lower is better):

  SD   — Step Distance         : distribution of per-trip travel distances
  SI   — Step Interval         : distribution of inter-trip time gaps (minutes)
  DARD — Daily Activity Routine: joint (time-bin × purpose) histogram
  STVD — Spatio-Temporal Visits: joint (time-bin × lat-bin × lon-bin) histogram
           requires a postal-code→coordinate lookup; not available for synthetic
           output unless the generation pipeline is extended to emit coordinates.

Usage
-----
    from Helpers.evaluation import (
        evaluate,
        load_syn_records,
        prepare_real_trips,
        build_geo_lookup,
    )

    # Load data
    real_df   = pd.read_csv("odin_cleaned.csv")
    real_df   = prepare_real_trips(real_df)               # adds departure_minutes
    syn_recs  = load_syn_records("trajectories_weekday.json")
    geo_lkp   = build_geo_lookup(pd.read_csv("geonames-postal-code@public.csv", sep=";"))

    # Filter for the evaluation subset (e.g., weekday hold-out)
    wkday_df  = real_df[real_df["DayType"] == "weekday"]

    # Run
    scores = evaluate(wkday_df, syn_recs, geo_lookup=geo_lkp, label="weekday")
    print(scores)
    # → {"label": "weekday", "SD": 0.042, "SI": 0.078, "DARD": 0.131, "STVD": 0.412}
"""

import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from numpy.random import Generator as _RNG


# Constants


TIME_BIN_MINUTES = 10               # LLMob uses 10-min slots → 144 bins / day
N_TIME_BINS      = 1440 // TIME_BIN_MINUTES   # 144

# (lo, hi) minute ranges for each ODiN / LLM departure-time class label.
# syn_dard and syn_si sample uniformly within [lo, hi) to spread synthetic trips
# across the 144 10-min time bins rather than collapsing them to a single midpoint.
# Uniform sampling is the least informative prior within each class interval.
DEP_TIME_CLASS_RANGE: dict[str, tuple[int, int]] = {
    "Before 6:00 AM":       (0,    360),
    "6:00 AM to 7:00 AM":   (360,  420),
    "7:00 AM to 8:00 AM":   (420,  480),
    "7am to 8am":           (420,  480),   # LLM variant
    "8:00 AM to 9:00 AM":   (480,  540),
    "8am to 9am":           (480,  540),   # LLM variant
    "9am to 12pm":          (540,  720),
    "12 noon to 1 p.m":     (720,  780),
    "1:00 PM to 2:00 PM":   (780,  840),
    "2:00 PM to 4:00 PM":   (840,  960),
    "4:00 PM to 5:00 PM":   (960, 1020),
    "5:00 PM to 6:00 PM":  (1020, 1080),
    "6:00 PM to 7:00 PM":  (1080, 1140),
    "7:00 PM to 8:00 PM":  (1140, 1200),
    "8 p.m. to midnight":  (1200, 1440),
    "8:00 PM to midnight": (1200, 1440),   # LLM variant
    "After 7:00 PM":       (1140, 1440),
}

# Bin edges for SD (km) — aligned to the actual ODiN distance class boundaries
SD_BINS = np.array(
    [0, 0.5, 1.0, 2.5, 3.7, 5.0, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 250],
    dtype=float,
)

# Bin edges for SI (minutes between consecutive trips)
SI_BINS = np.array(
    [0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440],
    dtype=float,
)

# Midpoint km for each ODiN / generated distance-class label (labels match exactly
# between decoded ODiN and the LLM generation prompt since the persona profile
# passes the ODiN decoded values as the allowed distance classes).
DISTANCE_CLASS_KM: dict[str, float] = {
    "0.1 to 0.5 km":  0.30,
    "0.5 to 1.0 km":  0.75,
    "1.0 to 2.5 km":  1.75,
    "2.5 to 3.7 km":  3.10,
    "3.7 to 5.0 km":  4.35,
    "5.0 to 7.5 km":  6.25,
    "7.5 to 10 km":   8.75,
    "10 to 15 km":   12.50,
    "15 to 20 km":   17.50,
    "20 to 30 km":   25.00,
    "30 to 40 km":   35.00,
    "40 to 50 km":   45.00,
    "50 to 75 km":   62.50,
    "75 to 100 km":  87.50,
    "100 km or more": 125.00,
    # LLM sometimes generates slight label variants — map them to the correct class
    "2.0 to 2.5 km":                                          1.75,  # within 1.0–2.5
    "2.0 to 2.5 km (falls within the 1.0 to 2.5 km class)":  1.75,
}


# JSD core


def _jsd(h_real: np.ndarray, h_syn: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence in [0, 1] (base-2 log) from two raw count arrays.
    Laplace smoothing (1e-10) prevents zero-probability issues.
    Note: scipy.jensenshannon returns the *distance* (sqrt of JSD); we square it
    to match LLMob's reported metric.
    """
    h_real = np.asarray(h_real, dtype=float) + 1e-10
    h_syn  = np.asarray(h_syn,  dtype=float) + 1e-10
    return float(jensenshannon(h_real, h_syn, base=2) ** 2)


def _hist1d(values: list, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    return counts.astype(float)



# Data helpers


def prepare_real_trips(
    trips_df: pd.DataFrame,
    hour_col: str = "Departure time transfer",
    min_col:  str = "Departure minute displacement",
) -> pd.DataFrame:
    """
    Add a `departure_minutes` column (integer minutes from midnight) to the
    ODiN trip-level DataFrame.  Returns a copy; does not modify in-place.
    """
    df = trips_df.copy()
    df["departure_minutes"] = (
        pd.to_numeric(df[hour_col], errors="coerce") * 60
        + pd.to_numeric(df[min_col], errors="coerce")
    )
    return df


def load_syn_records(json_path: str) -> list[dict]:
    """
    Flatten a generated-trajectories JSON (keyed by group_key) into a flat list
    of person-level records.  Skips the `__summary__` key if present.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for key, val in data.items():
        if key == "__summary__":
            continue
        if isinstance(val, list):
            records.extend(val)
    return records


def build_geo_lookup(
    geo_names_df: pd.DataFrame,
    postal_col: str = "postal code",
    lat_col:    str = "latitude",
    lon_col:    str = "longitude",
) -> dict[str, tuple[float, float]]:
    """
    Build {postal_code_str: (lat, lon)} from the geonames-postal-code CSV.
    Only 4-digit Dutch postal codes are kept (the ODiN prefix, without letters).
    """
    lookup = {}
    for _, row in geo_names_df.dropna(subset=[postal_col, lat_col, lon_col]).iterrows():
        pc = str(row[postal_col]).strip()
        if pc.isdigit():
            lookup[pc] = (float(row[lat_col]), float(row[lon_col]))
    return lookup


def _parse_time(s) -> int | None:
    """Convert 'HH:MM' to minutes from midnight; returns None on failure."""
    try:
        h, m = map(int, str(s).strip().split(":"))
        return h * 60 + m
    except Exception:
        return None



# Feature extractors — REAL data (ODiN trip-level DataFrame)


def real_sd(
    trips_df:     pd.DataFrame,
    distance_col: str = "Travel distance in the Netherlands (in hectometers)",
) -> list[float]:
    """Distance in km per trip (one value per trip row)."""
    vals = pd.to_numeric(trips_df[distance_col], errors="coerce") * 0.1  # hm → km
    return vals.dropna().tolist()


def real_si(
    trips_df:   pd.DataFrame,
    person_col: str = "Person_index",
) -> list[float]:
    """
    Inter-trip time gap in minutes, pooled across all persons.
    Requires `departure_minutes` column (add via prepare_real_trips first).
    Vectorised with groupby + diff — no Python-level row iteration.
    """
    df = (
        trips_df.dropna(subset=["departure_minutes"])
        .sort_values([person_col, "departure_minutes"])
    )
    diffs = df.groupby(person_col)["departure_minutes"].diff().dropna()
    return diffs[diffs >= 0].tolist()


def real_dard(
    trips_df:    pd.DataFrame,
    purpose_col: str = "Motive",
) -> list[tuple[int, str]]:
    """
    (time_bin, purpose) tuples for DARD histogram.
    time_bin = departure_minutes // TIME_BIN_MINUTES  (0–143 for 10-min slots).
    Vectorised — no iterrows.
    """
    df = trips_df.dropna(subset=["departure_minutes", purpose_col])
    time_bins = (df["departure_minutes"].astype(int) // TIME_BIN_MINUTES).tolist()
    purposes  = df[purpose_col].astype(str).tolist()
    return list(zip(time_bins, purposes))


def real_stvd(
    trips_df:   pd.DataFrame,
    geo_lookup: dict,
    postal_col: str = "Postal code of departure point",
) -> list[tuple[int, float, float]]:
    """
    (time_bin, lat, lon) tuples for STVD histogram.
    Postal codes are matched to coordinates via geo_lookup.
    Vectorised — no iterrows.
    """
    df = trips_df.dropna(subset=["departure_minutes"]).copy()
    df["_pc"]     = df[postal_col].astype(str).str.split(".").str[0].str.strip()
    df["_coords"] = df["_pc"].map(geo_lookup)
    df = df.dropna(subset=["_coords"])
    time_bins = (df["departure_minutes"].astype(int) // TIME_BIN_MINUTES).tolist()
    lats      = [c[0] for c in df["_coords"]]
    lons      = [c[1] for c in df["_coords"]]
    return list(zip(time_bins, lats, lons))



# Feature extractors — SYNTHETIC data (loaded JSON records)


def syn_sd(syn_records: list[dict]) -> list[float]:
    """
    Distance in km per trip.

    Handles two formats:
    - New (distance_km): LLM-reasoned numeric km value. Values <= 0 are dropped;
      values > 250 are dropped by np.histogram naturally, consistent with real_sd().
    - Old (distance_class): ODiN class label mapped to midpoint km; used for
      backward compatibility with Json_files_old/ generation outputs.
    """
    out = []
    for rec in syn_records:
        for trip in rec.get("trips", []):
            if trip.get("injected_return_home", False):
                continue
            km = trip.get("distance_km")
            if km is not None:
                try:
                    val = float(km)
                    if val > 0:
                        out.append(val)
                except (ValueError, TypeError):
                    pass
                continue
            km = DISTANCE_CLASS_KM.get(trip.get("distance_class", ""))
            if km is not None:
                out.append(km)
    return out


def syn_sd_from_coords(syn_records: list[dict]) -> list[float]:
    """
    Distance in km per trip, derived from haversine between consecutive
    destination coordinates rather than from the LLM-estimated distance_km.

    Origin for trip 0 is agent_home_lat / agent_home_lon (written by
    allocate_trajectory). Origin for trip N>0 is destination of trip N-1.
    Requires records to have been processed by allocate_trajectory().
    Trips with missing destination coordinates are skipped silently.
    """
    from math import radians, sin, cos, asin, sqrt

    def _hav(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 2 * R * asin(sqrt(max(a, 0.0)))

    out = []
    for rec in syn_records:
        home_lat = rec.get("agent_home_lat")
        home_lon = rec.get("agent_home_lon")
        if home_lat is None or home_lon is None:
            continue
        origin = (float(home_lat), float(home_lon))
        for trip in rec.get("trips", []):
            dest_lat = trip.get("destination_lat")
            dest_lon = trip.get("destination_lon")
            if dest_lat is None or dest_lon is None:
                continue
            km = _hav(origin[0], origin[1], float(dest_lat), float(dest_lon))
            if km > 0:
                out.append(km)
            origin = (float(dest_lat), float(dest_lon))
    return out


def syn_si(syn_records: list[dict], rng: _RNG | None = None) -> list[float]:
    """
    Inter-trip time gap in minutes.  Departure times are sampled uniformly within
    each departure_time_class range (jitter) so that SI uses the same 144-bin scale
    as real data.  Falls back to checkpoint time if the class label is absent.
    rng: numpy Generator for reproducibility; defaults to default_rng(42).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    gaps = []
    for rec in syn_records:
        times = []
        for trip in rec.get("trips", []):
            if trip.get("injected_return_home", False):
                continue
            dep_class = trip.get("departure_time_class", "")
            if dep_class in DEP_TIME_CLASS_RANGE:
                lo, hi = DEP_TIME_CLASS_RANGE[dep_class]
                t = int(rng.integers(lo, hi))
            else:
                t = _parse_time(trip.get("time"))
            if t is not None:
                times.append(t)
        times.sort()
        for i in range(1, len(times)):
            g = times[i] - times[i - 1]
            if g >= 0:
                gaps.append(g)
    return gaps


def syn_dard(syn_records: list[dict], rng: _RNG | None = None) -> list[tuple[int, str]]:
    """
    (time_bin, purpose) tuples from synthetic trips.
    Departure times are sampled uniformly within each departure_time_class range
    (jitter) so that synthetic trips are spread across the 144 10-min bins instead
    of collapsing to a single midpoint.  Falls back to checkpoint time if absent.
    rng: numpy Generator for reproducibility; defaults to default_rng(42).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    out = []
    for rec in syn_records:
        for trip in rec.get("trips", []):
            if trip.get("injected_return_home", False):
                continue
            dep_class = trip.get("departure_time_class", "")
            if dep_class in DEP_TIME_CLASS_RANGE:
                lo, hi = DEP_TIME_CLASS_RANGE[dep_class]
                t = int(rng.integers(lo, hi))
            else:
                t = _parse_time(trip.get("time"))
            p = trip.get("purpose")
            if t is None or not p:
                continue
            out.append((t // TIME_BIN_MINUTES, str(p)))
    return out


def syn_stvd(syn_records: list[dict]):
    """
    (time_bin, lat, lon) tuples for STVD histogram.
    Requires assign_coordinates() from spatial_assignment.py to have been run
    so that each trip carries lat, lon, and time (HH:MM) fields.
    Returns None if no trips have coordinates.
    """
    out = []
    for rec in syn_records:
        for trip in rec.get("trips", []):
            # Accept coordinates from either spatial_assignment.py (lat/lon)
            # or poi_allocator.py (destination_lat/destination_lon).
            lat = trip.get("lat") if trip.get("lat") is not None else trip.get("destination_lat")
            lon = trip.get("lon") if trip.get("lon") is not None else trip.get("destination_lon")
            t   = trip.get("time", "")
            if lat is None or lon is None or not t:
                continue
            try:
                h, m     = map(int, t.split(":"))
                time_bin = (h * 60 + m) // TIME_BIN_MINUTES
                out.append((time_bin, lat, lon))
            except (ValueError, AttributeError):
                continue
    return out if out else None


def real_dailyloc(
    trips_df:            pd.DataFrame,
    person_col:          str = "Person_index",
    n_additional_zeros:  int = 0,
    reconstruct_home:    bool = False,
    dest_col:            str = "Destination/Purpose",
) -> list[int]:
    """
    Number of trips made per person per day (MobAgent's DailyLoc).
    Each trip in ODiN corresponds to visiting a distinct location, so trip count
    is the natural equivalent of MobAgent's 'number of different locations visited
    per day'.

    n_additional_zeros: number of zero-trip persons to append as 0-count entries.
    Pass the count of ODiN respondents in the evaluation split who made no trips
    that day (i.e., total weekday persons minus trip-makers). This makes the real
    distribution comparable to syn_dailyloc, which always includes zero-trip persons.

    reconstruct_home: when True, add 1 trip to each person whose last recorded trip
    did not end at home (Destination/Purpose != "Home"). This corrects for ODiN's
    midnight-cutoff truncation artifact so the distribution is comparable to a
    synthetic pipeline that conditionally closes days at home.
    Note: only DailyLoc is reconstructed here. DARD reconstruction is not implemented
    because no neutral choice of mode/purpose/time exists for the imputed leg —
    report DARD against raw ODiN with this caveat documented.
    """
    counts_series = trips_df.groupby(person_col).size()
    if reconstruct_home and dest_col in trips_df.columns:
        df_sorted = trips_df.sort_values([person_col, "departure_minutes"])
        last_dest = (
            df_sorted.groupby(person_col)[dest_col]
            .last()
            .str.strip()
            .str.lower()
        )
        needs_home = (last_dest != "home").reindex(counts_series.index, fill_value=True)
        counts_series = counts_series + needs_home.astype(int)
    return counts_series.tolist() + [0] * n_additional_zeros


def syn_dailyloc(syn_records: list[dict]) -> list[int]:
    """
    Number of trips per synthetic person (MobAgent's DailyLoc equivalent).
    Zero-trip (atypical) persons contribute 0, consistent with real_dailyloc.
    Injected closure returns (injected_return_home=True) are excluded so that
    the metric reflects LLM-planned trips only, not the implementation-level
    day-closure mechanism.
    """
    return [
        sum(1 for t in rec.get("trips", []) if not t.get("injected_return_home", False))
        for rec in syn_records
    ]



# Metric JSD computations


def jsd_sd(real_vals: list, syn_vals: list) -> float:
    """JSD of the step-distance distributions."""
    return _jsd(_hist1d(real_vals, SD_BINS), _hist1d(syn_vals, SD_BINS))


def jsd_si(real_vals: list, syn_vals: list) -> float:
    """JSD of the step-interval (inter-trip gap) distributions."""
    return _jsd(_hist1d(real_vals, SI_BINS), _hist1d(syn_vals, SI_BINS))


def jsd_dard(
    real_tuples: list[tuple[int, str]],
    syn_tuples:  list[tuple[int, str]],
) -> float:
    """
    JSD of the joint (time-bin × purpose) distribution.
    The vocabulary of purposes is taken as the union of both sets so that
    categories absent in one split still contribute zero-probability mass.
    """
    all_purposes = sorted({p for _, p in real_tuples} | {p for _, p in syn_tuples})
    p_idx = {p: i for i, p in enumerate(all_purposes)}
    n_p   = len(all_purposes)

    h_real = np.zeros((N_TIME_BINS, n_p), dtype=float)
    h_syn  = np.zeros((N_TIME_BINS, n_p), dtype=float)

    for tb, p in real_tuples:
        if 0 <= tb < N_TIME_BINS and p in p_idx:
            h_real[tb, p_idx[p]] += 1
    for tb, p in syn_tuples:
        if 0 <= tb < N_TIME_BINS and p in p_idx:
            h_syn[tb, p_idx[p]] += 1

    return _jsd(h_real.flatten(), h_syn.flatten())


def jsd_dailyloc(real_counts: list[int], syn_counts: list[int], max_trips: int = 15) -> float:
    """
    JSD of the trips-per-person distributions (MobAgent's DailyLoc).
    Bins are integers 0–max_trips; anything above max_trips is clipped into the
    last bin.  Integer counts so no approximation is needed.
    """
    bins = np.arange(0, max_trips + 2)   # edges: 0,1,2,...,max_trips+1
    return _jsd(_hist1d(real_counts, bins), _hist1d(syn_counts, bins))


def jsd_stvd(
    real_tuples: list[tuple[int, float, float]],
    syn_tuples:  list[tuple[int, float, float]],
    lat_bins: int = 20,
    lon_bins: int = 20,
) -> float | None:
    """
    JSD of the joint (time-bin × lat-bin × lon-bin) distribution.
    Bin edges are derived from the combined range of real + synthetic coordinates.
    Returns None if either list is empty.
    """
    if not real_tuples or not syn_tuples:
        return None

    all_pts = real_tuples + syn_tuples
    lat_edges = np.linspace(min(t[1] for t in all_pts), max(t[1] for t in all_pts), lat_bins + 1)
    lon_edges = np.linspace(min(t[2] for t in all_pts), max(t[2] for t in all_pts), lon_bins + 1)

    h_real = np.zeros((N_TIME_BINS, lat_bins, lon_bins), dtype=float)
    h_syn  = np.zeros((N_TIME_BINS, lat_bins, lon_bins), dtype=float)

    def _fill(hist, tuples):
        for tb, lat, lon in tuples:
            li  = min(int(np.searchsorted(lat_edges[1:-1], lat)), lat_bins - 1)
            loi = min(int(np.searchsorted(lon_edges[1:-1], lon)), lon_bins - 1)
            if 0 <= tb < N_TIME_BINS:
                hist[tb, li, loi] += 1

    _fill(h_real, real_tuples)
    _fill(h_syn,  syn_tuples)
    return _jsd(h_real.flatten(), h_syn.flatten())



# Inter-group diversity (ablation metric)


def inter_group_diversity(syn_records: list[dict]) -> dict:
    """
    Measure how behaviourally distinct the synthetic persona groups are from
    each other.  Higher values mean the pipeline produces more differentiated
    behaviour across groups — the key thing the persona flag should deliver.

    Returns
    -------
    dict with keys:
        mode_diversity     : mean pairwise JSD of per-group mode distributions
        purpose_diversity  : mean pairwise JSD of per-group purpose distributions
        trip_count_std     : std dev of per-group mean trip count
        dep_time_std       : std dev of per-group mean departure time (minutes)
        n_groups           : number of groups with at least one trip
    """
    from collections import defaultdict
    import itertools

    # bucket trips by group
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in syn_records:
        gk = rec.get("group_key", "__unknown__")
        groups[gk].extend(rec.get("trips", []))

    # only groups that produced at least one trip
    active = {g: trips for g, trips in groups.items() if trips}
    group_keys = sorted(active.keys())
    n = len(group_keys)

    if n < 2:
        return {"mode_diversity": None, "purpose_diversity": None,
                "trip_count_std": None, "dep_time_std": None, "n_groups": n}

    # --- per-group distributions ---
    all_modes    = sorted({t.get("mode", "") for trips in active.values() for t in trips if t.get("mode")})
    all_purposes = sorted({t.get("purpose", "") for trips in active.values() for t in trips if t.get("purpose")})
    mode_idx = {m: i for i, m in enumerate(all_modes)}
    purp_idx = {p: i for i, p in enumerate(all_purposes)}

    mode_hists: dict[str, np.ndarray] = {}
    purp_hists: dict[str, np.ndarray] = {}
    mean_trips:  dict[str, float] = {}
    mean_dep:    dict[str, float] = {}

    persons_per_group: dict[str, int] = defaultdict(int)
    for rec in syn_records:
        gk = rec.get("group_key", "__unknown__")
        if gk in active:
            persons_per_group[gk] += 1

    for gk in group_keys:
        trips = active[gk]
        mh = np.zeros(len(all_modes),    dtype=float)
        ph = np.zeros(len(all_purposes), dtype=float)
        dep_mins = []

        for t in trips:
            m = t.get("mode", "")
            if m in mode_idx:
                mh[mode_idx[m]] += 1
            p = t.get("purpose", "")
            if p in purp_idx:
                ph[purp_idx[p]] += 1
            dc = t.get("departure_time_class", "")
            if dc in DEP_TIME_CLASS_RANGE:
                lo, hi = DEP_TIME_CLASS_RANGE[dc]
                dep_mins.append((lo + hi) / 2)

        mode_hists[gk] = mh
        purp_hists[gk] = ph
        n_persons = persons_per_group[gk] or 1
        mean_trips[gk] = len(trips) / n_persons
        mean_dep[gk]   = float(np.mean(dep_mins)) if dep_mins else 0.0

    # --- average pairwise JSD ---
    def _avg_pairwise_jsd(hists: dict) -> float:
        keys = sorted(hists.keys())
        jsds = []
        for a, b in itertools.combinations(keys, 2):
            ha = hists[a] + 1e-10
            hb = hists[b] + 1e-10
            jsds.append(float(jensenshannon(ha, hb, base=2) ** 2))
        return float(np.mean(jsds)) if jsds else 0.0

    return {
        "mode_diversity":    _avg_pairwise_jsd(mode_hists),
        "purpose_diversity": _avg_pairwise_jsd(purp_hists),
        "trip_count_std":    float(np.std(list(mean_trips.values()))),
        "dep_time_std":      float(np.std(list(mean_dep.values()))),
        "n_groups":          n,
    }



# Empirical resampling baseline


def build_resampling_baseline(
    train_trips_df:      pd.DataFrame,
    n_persons:           int,
    group_keys:          list[str] | None = None,
    person_col:          str = "Person_index",
    group_col:           str = "group_key",
    motive_col:          str = "Motive",
    mode_col:            str = "Main mode of transport travel",
    distance_class_col:  str = "Travel distance class in the Netherlands",
    dep_time_class_col:  str = "Departure time class",
    seed:                int = 42,
    zero_trip_fraction:  float = 0.0,
) -> list[dict]:
    """
    Build an empirical resampling baseline from ODiN training trips.

    For each of n_persons persons:
      1. Sample a trip count from the empirical distribution of trip counts
         in train_trips_df (trip-makers only; zero-trip persons are added
         separately via zero_trip_fraction).
      2. If group_keys is provided, sample trips only from training trips
         belonging to that group (group-conditioned resampling). Falls back
         to the full training pool when the group has no training trips.
      3. Assemble records in syn_records format so evaluate() can consume them.

    Parameters
    ----------
    train_trips_df : pd.DataFrame
        Training-split trip rows (e.g. odin_train.csv filtered to weekdays).
        Must contain person_col, group_col, motive_col, mode_col,
        distance_class_col, dep_time_class_col — all decoded to string labels.
    n_persons : int
        How many synthetic persons to produce (match len(syn_records)).
    group_keys : list[str] | None
        One group_key per person.  If provided, each person's trips are
        drawn only from training rows with that group_key (group-conditioned
        resampling).  Pass None for unconditional population resampling.
    zero_trip_fraction : float
        Fraction of the n_persons to assign 0 trips (DailyLoc correction).
        E.g. pass n_real_zero_trip / (n_real_zero_trip + n_persons) if you
        want the resampled distribution to include zero-trip persons.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    list[dict]
        Flat list of person records; each has keys 'group_key' and 'trips',
        where each trip dict has 'purpose', 'mode', 'distance_class', and
        'departure_time_class'.
    """
    rng = np.random.default_rng(seed)

    train_trips_df = train_trips_df.dropna(
        subset=[motive_col, distance_class_col, dep_time_class_col]
    ).reset_index(drop=True)

    # empirical trip-count distribution (trip-makers only)
    person_trip_counts = train_trips_df.groupby(person_col).size().values

    # pre-index trips by group as numpy arrays for fast sampling
    if group_keys is not None:
        group_arrays: dict[str, dict[str, np.ndarray]] = {}
        for gk, grp in train_trips_df.groupby(group_col):
            group_arrays[gk] = {
                "motive":    grp[motive_col].values,
                "mode":      grp[mode_col].values,
                "dist":      grp[distance_class_col].values,
                "dep":       grp[dep_time_class_col].values,
            }

    # global arrays for fallback / unconditional sampling
    global_arrays = {
        "motive": train_trips_df[motive_col].values,
        "mode":   train_trips_df[mode_col].values,
        "dist":   train_trips_df[distance_class_col].values,
        "dep":    train_trips_df[dep_time_class_col].values,
    }
    n_global = len(train_trips_df)

    n_zero = int(round(n_persons * zero_trip_fraction))
    n_main = n_persons - n_zero

    records: list[dict] = []

    for i in range(n_main):
        n_trips = int(rng.choice(person_trip_counts))
        gk      = group_keys[i] if group_keys is not None else f"resample_{i}"

        if group_keys is not None:
            arrs = group_arrays.get(gk, global_arrays)
        else:
            arrs = global_arrays

        pool_size = len(arrs["motive"])
        idx = rng.integers(0, pool_size, size=n_trips)

        trips = [
            {
                "purpose":            str(arrs["motive"][j]),
                "mode":               str(arrs["mode"][j]),
                "distance_class":     str(arrs["dist"][j]),
                "departure_time_class": str(arrs["dep"][j]),
            }
            for j in idx
        ]
        records.append({"group_key": gk, "trips": trips})

    # zero-trip persons (DailyLoc correction)
    for i in range(n_zero):
        gk = group_keys[n_main + i] if group_keys is not None else f"resample_zero_{i}"
        records.append({"group_key": gk, "trips": []})

    return records



# Main evaluate()


def evaluate(
    real_trips_df:           pd.DataFrame,
    syn_records:             list[dict],
    geo_lookup:              dict | None = None,
    purpose_col:             str = "Motive",
    distance_col:            str = "Travel distance in the Netherlands (in hectometers)",
    person_col:              str = "Person_index",
    label:                   str = "",
    seed:                    int = 42,
    n_real_zero_trip:        int = 0,
    use_coordinate_distance: bool = False,
    reconstruct_home:        bool = False,
) -> dict:
    """
    Compute SD / SI / DARD / STVD JSD between real ODiN trips and generated
    synthetic trajectories.

    Parameters
    ----------
    real_trips_df : pd.DataFrame
        Trip-level ODiN data already processed with prepare_real_trips() so that
        the `departure_minutes` column exists.
    syn_records : list[dict]
        Flat list of synthetic person records from load_syn_records().
    geo_lookup : dict | None
        {postal_code_str: (lat, lon)} from build_geo_lookup().
        Required for STVD; if None, STVD is skipped and returned as None.
    use_coordinate_distance : bool
        When True, SD is computed from haversine distances between consecutive
        destination coordinates (requires records processed by allocate_trajectory).
        Both coordinate-derived SD and LLM-estimated SD are returned so results
        are directly comparable. When False (default), uses the existing syn_sd()
        which reads distance_km / distance_class from each trip.
    purpose_col : str
        Name of the trip-purpose column in real_trips_df.
    distance_col : str
        Name of the trip-distance column in real_trips_df (hectometers).
    person_col : str
        Name of the person ID column in real_trips_df.
    label : str
        Free-text tag included in the output dict (e.g. 'weekday', 'weekend',
        'transferability_holdout').

    Returns
    -------
    dict
        {
            "label":     str,
            "SD":        float,        # JSD of step-distance distributions
            "SD_llm":    float | None, # SD from LLM distance_km (only when
                                       # use_coordinate_distance=True for comparison)
            "SI":        float,        # JSD of step-interval distributions
            "DARD":      float,        # JSD of (time-bin × purpose) joint distribution
            "DailyLoc":  float,        # JSD of trips-per-person distribution
            "STVD":      float | None, # JSD of (time-bin × lat × lon), or None
            "n_real_trips": int,
            "n_syn_trips":  int,
        }
    """
    # Each call to syn_si / syn_dard needs its own seeded RNG so jitter is
    # deterministic but the two calls don't share state and produce identical sequences.
    rng_si   = np.random.default_rng(seed)
    rng_dard = np.random.default_rng(seed)

    # Extract features
    r_sd   = real_sd(real_trips_df, distance_col)
    if use_coordinate_distance:
        s_sd     = syn_sd_from_coords(syn_records)
        s_sd_llm = syn_sd(syn_records)
    else:
        s_sd     = syn_sd(syn_records)
        s_sd_llm = None
    r_si      = real_si(real_trips_df, person_col)
    s_si      = syn_si(syn_records, rng=rng_si)
    r_dard    = real_dard(real_trips_df, purpose_col)
    s_dard    = syn_dard(syn_records, rng=rng_dard)
    r_dloc    = real_dailyloc(real_trips_df, person_col, n_additional_zeros=n_real_zero_trip, reconstruct_home=reconstruct_home)
    s_dloc    = syn_dailyloc(syn_records)

    results = {
        "label":        label,
        "SD":           jsd_sd(r_sd, s_sd),
        "SD_llm":       jsd_sd(r_sd, s_sd_llm) if s_sd_llm is not None else None,
        "SI":           jsd_si(r_si, s_si),
        "DARD":         jsd_dard(r_dard, s_dard),
        "DailyLoc":     jsd_dailyloc(r_dloc, s_dloc),
        "STVD":         None,
        "n_real_trips": len(r_sd),
        "n_syn_trips":  len(s_sd),
    }

    if geo_lookup is not None:
        r_stvd = real_stvd(real_trips_df, geo_lookup)
        s_stvd = syn_stvd(syn_records)   # None until pipeline emits coordinates
        if s_stvd is not None:
            results["STVD"] = jsd_stvd(r_stvd, s_stvd)

    return results
