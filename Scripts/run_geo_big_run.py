"""
run_geo_big_run.py — Geo-reference the big Rotterdam run(s) using POIAllocator.

Adds destination_lat, destination_lon, destination_poi_label, destination_zone_id
to every trip, then saves a flat list JSON per seed.

Usage:
    python Scripts/run_geo_big_run.py                    # default: seed 42 only
    python Scripts/run_geo_big_run.py --seeds 42
    python Scripts/run_geo_big_run.py --seeds 3 7 13 27 42   # full 5-seed sweep

Input (per seed, produced by the "Big Run for Rotterdam" cell in
Notebooks/3_SynTravel_generation.ipynb):
    Json_files/trajectories_weekday_big_seed{N}.json

Output (per seed):
    Portugal_POI_data/NT_poi/trajectories_weekday_rotterdam_geo_pois_big_seed{N}.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Helpers.poi_allocator import POIAllocator, allocate_trajectory

#  Paths 
BIG_RUN_TEMPLATE = "Json_files/trajectories_weekday_big_seed{seed}.json"
POI_CACHE        = "Portugal_POI_data/NT_poi/nl_rotterdam_pois.gpkg"
ZONE_DATA        = "Portugal_POI_data/NT_poi/nl_rotterdam_zones.gpkg"
OUTPUT_TEMPLATE  = "Portugal_POI_data/NT_poi/trajectories_weekday_rotterdam_geo_pois_big_seed{seed}.json"

#  Distance class lookup (matches DISTANCE_CLASS_BOUNDS in poi_allocator.py) 
_DISTANCE_BREAKS = [
    (0.0,   0.1,  "0.1 to 0.5 km"),
    (0.1,   0.5,  "0.1 to 0.5 km"),
    (0.5,   1.0,  "0.5 to 1.0 km"),
    (1.0,   2.5,  "1.0 to 2.5 km"),
    (2.5,   3.7,  "2.5 to 3.7 km"),
    (3.7,   5.0,  "3.7 to 5.0 km"),
    (5.0,   7.5,  "5.0 to 7.5 km"),
    (7.5,  10.0,  "7.5 to 10 km"),
    (10.0, 15.0,  "10 to 15 km"),
    (15.0, 20.0,  "15 to 20 km"),
    (20.0, 30.0,  "20 to 30 km"),
    (30.0, 40.0,  "30 to 40 km"),
    (40.0, 50.0,  "40 to 50 km"),
    (50.0, 75.0,  "50 to 75 km"),
    (75.0, 100.0, "75 to 100 km"),
]


def _km_to_class(km: float) -> str:
    if km is None:
        return "_default"
    for lo, hi, label in _DISTANCE_BREAKS:
        if lo <= km < hi:
            return label
    return "100 km or more"


def flatten_big_run(path: str) -> list[dict]:
    """Convert group-keyed dict → flat list of person records."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for key, persons in data.items():
        if key == "__summary__":
            continue
        for person in persons:
            records.append(person)
    return records


def add_distance_class(records: list[dict]) -> list[dict]:
    """Add distance_class to every trip based on distance_km."""
    for rec in records:
        for trip in rec.get("trips", []):
            if "distance_class" not in trip:
                trip["distance_class"] = _km_to_class(trip.get("distance_km"))
    return records


def parse_args():
    p = argparse.ArgumentParser(description="Geo-reference the big Rotterdam run(s)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to allocate (default: [42] — single-run check; "
                         "pass --seeds 3 7 13 27 42 for the full sweep)")
    return p.parse_args()


def run_one_seed(seed: int) -> None:
    input_path  = BIG_RUN_TEMPLATE.format(seed=seed)
    output_path = OUTPUT_TEMPLATE.format(seed=seed)

    if not Path(input_path).exists():
        print(f"[seed {seed}] SKIP — {input_path} not found "
              f"(run the \"Big Run for Rotterdam\" cell in Notebooks/3_SynTravel_generation.ipynb first)")
        return

    print(f"[seed {seed}] Loading big run: {input_path}")
    records = flatten_big_run(input_path)
    print(f"[seed {seed}]   {len(records)} person records")

    records = add_distance_class(records)

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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=None)

    print(f"[seed {seed}] Saved {len(enriched)} records to {output_path}")


def main():
    args = parse_args()
    print(f"Allocating {len(args.seeds)} seed(s): {args.seeds}")
    for seed in args.seeds:
        run_one_seed(seed)


if __name__ == "__main__":
    main()
