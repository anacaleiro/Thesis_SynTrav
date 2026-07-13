"""
run_smp_rotterdam_allocation.py — Geo-reference the SMP baseline population using
POIAllocator, for the Rotterdam spatial-allocation comparison.

There was previously no script for this (the existing single
trajectories_smp_rotterdam_geo.json was produced by ad-hoc code that wasn't
saved). This rebuilds it, with the same per-seed option as run_geo_big_run.py
so the SynTrav and SMP sides of the Rotterdam comparison can be run with a
matching number of seeds.

Reuses the cached smp_model.pkl (no re-fitting) — same pattern as
smp/run_seeds.py. Only simulation + POI allocation are repeated per seed.

Usage:
    python Scripts/run_smp_rotterdam_allocation.py                  # default: seed 42 only
    python Scripts/run_smp_rotterdam_allocation.py --seeds 42
    python Scripts/run_smp_rotterdam_allocation.py --seeds 3 7 13 27 42   # full 5-seed sweep

Output (per seed):
    Portugal_POI_data/NT_poi/trajectories_smp_rotterdam_geo_seed{N}.json
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smp.episode_extractor import load_config
from smp.estimator         import load_model
from smp.simulator         import simulate_population
from smp.run_smp           import load_data, build_persona_list, CONFIG_PATH, MODEL_PATH, N_SYNTHETIC

from Scripts.run_geo_big_run import _km_to_class
from Helpers.poi_allocator import POIAllocator, allocate_trajectory

POI_CACHE       = "Portugal_POI_data/NT_poi/nl_rotterdam_pois.gpkg"
ZONE_DATA       = "Portugal_POI_data/NT_poi/nl_rotterdam_zones.gpkg"
OUTPUT_TEMPLATE = "Portugal_POI_data/NT_poi/trajectories_smp_rotterdam_geo_seed{seed}.json"


def parse_args():
    p = argparse.ArgumentParser(description="Geo-reference the SMP baseline for Rotterdam")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to simulate + allocate (default: [42] — single-run check; "
                         "pass --seeds 3 7 13 27 42 for the full sweep)")
    return p.parse_args()


def _trips_to_records(syn_trips: list[dict]) -> list[dict]:
    """
    Group simulate_population()'s flat per-trip rows into per-person records
    with a 'trips' list, matching the schema allocate_trajectory() expects
    (same shape as flatten_big_run() in run_geo_big_run.py).
    """
    df = pd.DataFrame(syn_trips)
    records = []
    for person_id, grp in df.groupby("person_id"):
        grp = grp.sort_values("departure_min")
        trips = []
        for _, row in grp.iterrows():
            trips.append({
                "purpose":       row["purpose_state"],
                "destination":   None,
                "distance_km":   float(row["distance_km"]),
                "distance_class": _km_to_class(row["distance_km"]),
                "time":          f"{int(row['departure_min']) // 60:02d}:{int(row['departure_min']) % 60:02d}",
            })
        records.append({"person_id": str(person_id), "trips": trips})
    return records


def main():
    args = parse_args()
    print(f"Allocating {len(args.seeds)} seed(s): {args.seeds}")

    cfg = load_config(CONFIG_PATH)
    train_df, _holdout_df = load_data(cfg)
    persona_list = build_persona_list(train_df, cfg, N_SYNTHETIC)

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"No cached SMP model at {MODEL_PATH}. Run `python smp/run_smp.py` first.")
    print(f"Loading cached model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    for seed in args.seeds:
        print(f"\n[seed {seed}] Simulating SMP population …")
        syn_trips = simulate_population(model, persona_list, seed=seed)
        records = _trips_to_records(syn_trips)
        print(f"[seed {seed}]   {len(records)} person records")

        print(f"[seed {seed}] Loading POIAllocator (NL) …")
        allocator = POIAllocator(
            country="NL",
            poi_cache_path=POI_CACHE,
            zone_data_path=ZONE_DATA,
            seed=seed,
        )

        print(f"[seed {seed}] Allocating coordinates …")
        enriched = []
        n = len(records)
        for i, rec in enumerate(records):
            if i % 200 == 0:
                print(f"[seed {seed}]   {i}/{n} …")
            home = allocator.sample_agent_home()
            enriched.append(allocate_trajectory(rec, allocator, home))

        output_path = OUTPUT_TEMPLATE.format(seed=seed)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump(enriched, f, ensure_ascii=False, indent=None)

        print(f"[seed {seed}] Saved {len(enriched)} records to {output_path}")


if __name__ == "__main__":
    main()
