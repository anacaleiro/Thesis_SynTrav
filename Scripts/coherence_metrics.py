"""
coherence_metrics.py — Behavioural coherence checks for SynTravelers (LLM) vs
SMP vs ODiN, weekday trips.

Table A: presence/absence checks (the original Table 5.4) plus three new
rows added for supervisor feedback (2026-07): temporal plausibility of trip
sequences, distance-purpose consistency, and trip-chaining logic.

Table B: the same distance-purpose consistency check broken out by
(occupation, purpose) cell — supporting detail for a thesis appendix table,
with the "retired | Shopping" cell the supervisor asked about printed
explicitly.

Purpose strings are mapped onto the thesis's 8 short categories (Work,
Shopping, Visitors, Leisure, Sports, Services, Pick up/Drop off, Education)
using the same PURPOSE_SHORT labels as every figure in fig_gallery.py, so
results here are directly comparable to those figures. Anything that doesn't
match one of the 8 (free-text LLM purposes, ODiN motives outside the top 8)
is left uncategorised and excluded from purpose-specific checks — same
convention the figures already use.

Known limitations (see inline notes at point of use):
  * The LLM diary has no explicit activity-duration field. Dwell time is
    approximated as the gap to the next stated departure, which includes
    travel time to the next leg — this makes the LLM's dwell proxy an
    overestimate of true dwell, biasing its immediate-return rate downward
    (conservative, not inflated).
  * SMP's HOME-centred simulator excludes HOME episodes from its output
    entirely, so `is_home_dest` is always False for SMP by construction.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from Helpers.visualizations.fig_gallery import PURPOSE_SHORT

#  Paths 
LLM_JSON = 'Json_files/variance/full_seed42.json'
SMP_CSV  = 'smp/results/syn_trips.csv'
ODIN_CSV = 'ODiN (DATA)/DATAVERSE/intermediarie_csvs/odin_cleaned.csv'

EVENING_HOUR   = 18   # >= this hour counts as "evening"
MIN_ODIN_CELL  = 20   # min ODiN sample per (occupation, purpose) cell before trusting its reference band

# Categories treated as "evening leisure" for the temporal-plausibility check.
# The thesis's purpose taxonomy has no separate "social" category — Visitors
# ("Visitors/staying over") is the closest equivalent and is included here.
LEISURE_LIKE = {'Leisure', 'Visitors', 'Sports'}


#  Purpose classification (shared across all three sources) 

def classify_purpose(raw: str | None) -> str | None:
    """
    Map a raw purpose/motive string onto one of the 8 thesis categories via
    substring match against PURPOSE_SHORT's full labels (handles messy /
    multi-purpose / free-text LLM strings the same way the original
    coherence_metrics.py did for its 4 binary checks). Returns None if no
    known category matches.
    """
    if not raw:
        return None
    low = raw.lower()
    for full_label, short in PURPOSE_SHORT.items():
        if full_label.lower() in low:
            return short
    return None


def _parse_hhmm(s: str | None) -> float | None:
    if not s:
        return None
    try:
        h, m = s.strip().split(':')
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


#  Loaders: one tidy per-trip DataFrame per source 
# Common columns: person_id, occupation, trip_order, dep_min, purpose_cat,
# distance_km, dwell_min, is_home_dest

def load_llm(path: str = LLM_JSON) -> pd.DataFrame:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    records = [p for k, v in raw.items() if k != '__summary__' for p in v if isinstance(p, dict)]

    rows = []
    for r in records:
        occ = r.get('group_key', '').split('|')[0].strip()
        trips = [t for t in r.get('trips', []) if isinstance(t, dict)]
        for i, t in enumerate(trips):
            dep = _parse_hhmm(t.get('time'))
            if dep is None:
                continue
            rows.append({
                'person_id':    r['person_id'],
                'occupation':   occ,
                'trip_order':   i,
                'dep_min':      dep,
                'purpose_cat':  classify_purpose(t.get('purpose')),
                'distance_km':  t.get('distance_km'),
                'is_home_dest': 'home' in (t.get('destination') or '').lower(),
            })
    df = pd.DataFrame(rows).sort_values(['person_id', 'trip_order']).reset_index(drop=True)
    # Dwell proxy: gap to the next stated departure (see module docstring limitation).
    df['dwell_min'] = df.groupby('person_id')['dep_min'].shift(-1) - df['dep_min']
    return df


def load_smp(path: str = SMP_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['occupation']  = df['persona_group'].str.split('|').str[0].str.strip()
    df['purpose_cat'] = df['purpose_state'].apply(classify_purpose)
    df = df.sort_values(['person_id', 'departure_min']).reset_index(drop=True)
    df['trip_order']   = df.groupby('person_id').cumcount()
    df['is_home_dest'] = False  # SMP never emits HOME episodes (structural, see module docstring)
    return df.rename(columns={'departure_min': 'dep_min', 'duration_min': 'dwell_min'})


def load_odin(path: str = ODIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df['DayType'] == 'weekday'].copy()
    df['dep_min'] = (
        pd.to_numeric(df['Departure time transfer'], errors='coerce') * 60
        + pd.to_numeric(df['Departure minute displacement'], errors='coerce')
    )
    df['occupation']   = df['Activity_status'].astype(str).str.strip()
    df['purpose_cat']  = df['Motive'].apply(classify_purpose)
    df['distance_km']  = pd.to_numeric(df['Travel distance in the Netherlands (in hectometers)'], errors='coerce') / 10.0
    df['dwell_min']    = pd.to_numeric(df['Activity duration (in minutes)'], errors='coerce')
    df['is_home_dest'] = df['Destination/Purpose'].astype(str).str.strip().str.lower() == 'home'
    df = df.dropna(subset=['dep_min']).sort_values(['Person_index', 'dep_min']).reset_index(drop=True)
    df['trip_order'] = df.groupby('Person_index').cumcount()
    return df.rename(columns={'Person_index': 'person_id'})


#  Table A, rows 1-4: original presence/absence checks 

def presence_checks(df: pd.DataFrame) -> dict:
    persons = df.groupby('person_id').agg(
        occupation=('occupation', 'first'),
        has_work=('purpose_cat', lambda s: (s == 'Work').any()),
        has_edu=('purpose_cat', lambda s: (s == 'Education').any()),
    ).reset_index()

    workers  = persons[persons.occupation == 'employed']
    students = persons[persons.occupation == 'student']
    nonwork  = persons[persons.occupation != 'employed']

    return {
        'Non-workers, no work trip (%)': 100 * (~nonwork.has_work).mean(),
        'Workers, >=1 work trip (%)':     100 * workers.has_work.mean(),
        'Students, >=1 edu trip (%)':     100 * students.has_edu.mean() if len(students) else float('nan'),
    }


def diary_ends_at_home_llm_odin(df: pd.DataFrame) -> float:
    last = df.sort_values('trip_order').groupby('person_id').tail(1)
    return 100 * last['is_home_dest'].mean()


def diary_ends_at_home_smp(smp_raw: pd.DataFrame) -> float:
    # SMP excludes HOME episodes from output; last_arrival < 1440 means the
    # last trip finished before midnight, implying an (unlogged) trip home.
    last_arrival = smp_raw.sort_values('departure_min').groupby('person_id')['arrival_min'].last()
    return 100 * (last_arrival < 1440.0).mean()


#  Table A, row 5: temporal plausibility 
# Supervisor's example: does the LLM ever schedule a work trip after an
# evening leisure trip?

def evening_leisure_then_work(df: pd.DataFrame) -> float:
    flagged, total = 0, 0
    for _, g in df.sort_values('trip_order').groupby('person_id'):
        total += 1
        evening = g.loc[g.purpose_cat.isin(LEISURE_LIKE) & (g.dep_min >= EVENING_HOUR * 60), 'dep_min']
        if evening.empty:
            continue
        later_work = g.loc[(g.purpose_cat == 'Work') & (g.dep_min > evening.min())]
        if not later_work.empty:
            flagged += 1
    return 100 * flagged / total if total else float('nan')


#  Table A, row 6: trip-chaining logic 
# Supervisor's example: are outbound trips followed by purposeful activity,
# rather than an immediate return? A trip whose dwell at the destination
# falls below ODiN's own 5th-percentile activity duration is flagged as an
# "immediate return" regardless of source.

def immediate_return_rate(df: pd.DataFrame, floor_min: float) -> float:
    sub = df[~df['is_home_dest']].dropna(subset=['dwell_min'])
    if not len(sub):
        return float('nan')
    return 100 * (sub['dwell_min'] < floor_min).mean()


#  Table B: distance-purpose consistency by occupation 
# Supervisor's example: do shopping trips cluster in the short-distance range
# for retired personas, as expected?

def distance_profile(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=['distance_km', 'purpose_cat']).groupby(['occupation', 'purpose_cat'])['distance_km']
    return g.agg(n='count', median='median', p5=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95))


def distance_out_of_band_rate(df: pd.DataFrame, odin_ref: pd.DataFrame) -> float:
    """% of df's trips whose distance falls outside the ODiN [p5,p95] band for
    their (occupation, purpose) cell, restricted to cells with enough ODiN
    support (MIN_ODIN_CELL) to trust the reference band."""
    ref = odin_ref[odin_ref['n'] >= MIN_ODIN_CELL][['p5', 'p95']]
    sub = df.dropna(subset=['distance_km', 'purpose_cat']).join(ref, on=['occupation', 'purpose_cat'], how='inner')
    if not len(sub):
        return float('nan')
    out = (sub['distance_km'] < sub['p5']) | (sub['distance_km'] > sub['p95'])
    return 100 * out.mean()


#  Main 

def main():
    print('Loading LLM, SMP, ODiN trip tables...')
    llm  = load_llm()
    smp  = load_smp()
    smp_raw = pd.read_csv(SMP_CSV)
    odin = load_odin()
    print(f'  LLM trips: {len(llm)}   SMP trips: {len(smp)}   ODiN trips: {len(odin)}')
    print()

    # ODiN-derived reference values, used by both the temporal and distance checks
    odin_dwell_floor = odin.loc[~odin.is_home_dest, 'dwell_min'].quantile(0.05)
    odin_dist_ref = distance_profile(odin)

    table_a = {
        'ODiN':         presence_checks(odin),
        'SynTrav (LLM)': presence_checks(llm),
        'SMP':          presence_checks(smp),
    }
    for src, df in (('ODiN', odin), ('SynTrav (LLM)', llm), ('SMP', smp)):
        table_a[src]['Diary ends at home (%)'] = (
            diary_ends_at_home_smp(smp_raw) if src == 'SMP' else diary_ends_at_home_llm_odin(df)
        )
        table_a[src][f'Work trip after {EVENING_HOUR}:00 leisure/visit/sports trip (%)'] = (
            evening_leisure_then_work(df)
        )
        table_a[src][f'Outbound trip, immediate return (< ODiN p5 dwell = {odin_dwell_floor:.0f} min) (%)'] = (
            immediate_return_rate(df, odin_dwell_floor)
        )
        table_a[src]['Trips outside plausible distance for occupation x purpose (%)'] = (
            distance_out_of_band_rate(df, odin_dist_ref)
        )

    print('=== Table A: Behavioural Coherence Metrics (% share) ===\n')
    results_a = pd.DataFrame(table_a)
    print(results_a.round(1).to_string())
    print()

    print("=== Table B: Distance-purpose profile by occupation (supporting detail) ===\n")
    for src, df in (('ODiN', odin), ('SynTrav (LLM)', llm), ('SMP', smp)):
        print(f'-- {src} --')
        print(distance_profile(df).round(1).to_string())
        print()

    print("-- Highlight: retired | Shopping (supervisor's example) --")
    for src, df in (('ODiN', odin), ('SynTrav (LLM)', llm), ('SMP', smp)):
        sub = df[(df.occupation == 'retired') & (df.purpose_cat == 'Shopping')]['distance_km'].dropna()
        if len(sub):
            print(f'  {src:<14} n={len(sub):<5} median={sub.median():.2f} km   share <5km={100 * (sub < 5).mean():.1f}%')
        else:
            print(f'  {src:<14} no trips in this cell')


if __name__ == '__main__':
    main()
