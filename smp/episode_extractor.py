"""
episode_extractor.py — Convert ODiN trip rows into activity episode sequences.

Each ODiN trip row represents travel from A to B plus the subsequent dwell at B
(stored in 'Activity duration (in minutes)'). This module reconstructs the full
person-day as a sequence of activity episodes including synthetic HOME episodes.

Public interface
----------------
extract_episodes(trips_df, cfg) -> pd.DataFrame
    Columns: person_id, group_key, day_of_week, state, start_min, end_min,
             duration_min, episode_role
    episode_role in {FIRST_HOME, LAST_HOME, MIDDAY_HOME, ACTIVITY}

load_config(path) -> dict
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_minutes(hour_col: pd.Series, min_col: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour_col, errors="coerce")
    m = pd.to_numeric(min_col, errors="coerce")
    return h * 60 + m


def _build_day_episodes(group: pd.DataFrame, cfg: dict) -> list[dict] | None:
    """
    Build the episode list for one person-day (pre-sorted by departure_min).
    Returns None if the day fails basic sanity or timeline checks.
    """
    c          = cfg["columns"]
    home_val   = cfg["home_destination_value"]
    pmap       = cfg["purpose_map"]
    tol        = cfg.get("timeline_tolerance_min", 30)
    day_len    = cfg.get("day_length_min", 1440)

    dep_mins  = group["_dep_min"].tolist()
    arr_mins  = group["_arr_min"].tolist()
    act_durs  = pd.to_numeric(group[c["activity_duration"]], errors="coerce").tolist()
    motives   = group[c["motive"]].tolist()
    dests     = group[c["destination_purpose"]].tolist()
    n         = len(group)

    #  Basic sanity 
    if any(pd.isna(v) for v in dep_mins + arr_mins):
        return None
    if dep_mins[0] < 0 or arr_mins[-1] > day_len:
        return None
    if any(dep_mins[i] > dep_mins[i + 1] for i in range(n - 1)):
        return None
    if any(arr_mins[i] < dep_mins[i] for i in range(n)):
        return None

    episodes = []

    #  FIRST_HOME (synthetic: midnight → first departure) 
    first_dep = dep_mins[0]
    episodes.append({
        "state":        "HOME",
        "start_min":    0,
        "end_min":      first_dep,
        "duration_min": first_dep,
        "episode_role": "FIRST_HOME",
    })

    #  One episode per trip: the dwell at the trip's destination 
    for i in range(n):
        dest_raw  = str(dests[i]).strip() if not pd.isna(dests[i]) else ""
        at_home   = dest_raw.lower() == home_val.lower()
        act_dur   = act_durs[i]

        start = arr_mins[i]

        if at_home:
            # HOME episode — duration from arrival to next departure (or EOD)
            end = dep_mins[i + 1] if i < n - 1 else day_len
            dur = end - start
            role = "LAST_HOME" if i == n - 1 else "MIDDAY_HOME"
            episodes.append({
                "state":        "HOME",
                "start_min":    start,
                "end_min":      end,
                "duration_min": max(dur, 0),
                "episode_role": role,
            })
        else:
            state = pmap.get(str(motives[i]))
            if state is None:
                continue  # unknown motive: skip episode, keep processing
            if pd.isna(act_dur) or act_dur < 0:
                continue  # unusable activity duration
            end = start + act_dur
            episodes.append({
                "state":        state,
                "start_min":    start,
                "end_min":      end,
                "duration_min": act_dur,
                "episode_role": "ACTIVITY",
            })

    #  Ensure LAST_HOME exists (last trip may not end at home) 
    if not episodes or episodes[-1]["episode_role"] != "LAST_HOME":
        last_end = episodes[-1]["end_min"] if episodes else arr_mins[-1]
        if last_end < day_len:
            episodes.append({
                "state":        "HOME",
                "start_min":    last_end,
                "end_min":      day_len,
                "duration_min": day_len - last_end,
                "episode_role": "LAST_HOME",
            })

    #  Timeline check 
    # Total = FIRST_HOME + all travel times + all activity durs + LAST_HOME
    # = dep_min[0] + sum(arr-dep) + sum(act_dur for non-home) + home dwell + last_home
    # Simplified: sum of (travel_i + act_dur_i) should bridge dep[0] to arr[-1]
    travel_total = sum(arr_mins[i] - dep_mins[i] for i in range(n))
    ep_total     = sum(e["duration_min"] for e in episodes)
    reconstructed = travel_total + ep_total
    if abs(reconstructed - day_len) > tol:
        return None

    return episodes


def extract_episodes(trips_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Convert ODiN trip rows into activity episode sequences.

    Parameters
    ----------
    trips_df : pd.DataFrame
        Trip-level ODiN data filtered to the desired day_type and provinces.
        Must contain all columns referenced in cfg['columns'].
    cfg : dict
        Loaded from config.yaml via load_config().

    Returns
    -------
    pd.DataFrame
        Columns: person_id, group_key, day_of_week, state,
                 start_min, end_min, duration_min, episode_role
        Rows: one per episode across all accepted person-days.
    """
    c   = cfg["columns"]
    df  = trips_df.copy()

    df["_dep_min"] = _to_minutes(df[c["departure_hour"]], df[c["departure_min_disp"]])
    df["_arr_min"] = _to_minutes(df[c["arrival_hour"]],   df[c["arrival_min_disp"]])

    all_episodes   = []
    n_total        = 0
    n_dropped      = 0

    group_keys_day = [c["person_id"], c["day_of_week"]]

    for (person_id, day_of_week), grp in df.groupby(group_keys_day, sort=False):
        grp_sorted = grp.sort_values("_dep_min", kind="stable")
        n_total   += 1

        gk_col    = c["group_key"]
        group_key = grp_sorted[gk_col].iloc[0] if gk_col in grp_sorted.columns else "unknown"
        episodes  = _build_day_episodes(grp_sorted, cfg)

        if episodes is None:
            n_dropped += 1
            continue

        for ep in episodes:
            ep["person_id"]   = person_id
            ep["group_key"]   = group_key
            ep["day_of_week"] = day_of_week

        all_episodes.extend(episodes)

    drop_rate = n_dropped / n_total if n_total else 0.0
    logger.info(
        "Episode extraction: %d person-days processed, %d dropped (%.1f%%)",
        n_total, n_dropped, drop_rate * 100,
    )
    print(
        f"[episode_extractor] {n_total} person-days | "
        f"{n_dropped} dropped ({drop_rate:.1%}) | "
        f"{len(all_episodes)} episodes produced"
    )

    if not all_episodes:
        return pd.DataFrame(columns=[
            "person_id", "group_key", "day_of_week", "state",
            "start_min", "end_min", "duration_min", "episode_role",
        ])

    return pd.DataFrame(all_episodes)[[
        "person_id", "group_key", "day_of_week", "state",
        "start_min", "end_min", "duration_min", "episode_role",
    ]]
