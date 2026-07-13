"""
run_pt_allocation_big.py — Geo-reference the Portugal big run using POIAllocator (Oeiras).

Adds destination_lat, destination_lon, destination_poi_label, destination_zone_id
to every trip, then saves a flat list JSON.

Usage:
    python run_pt_allocation_big.py

Output:
    Portugal_POI_data/PT_poi/trajectories_portugal_weekday_big_geo.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Helpers.poi_allocator import POIAllocator, allocate_trajectory

#  Paths 
BIG_RUN_PATH = "Json_files/trajectories_portugal_weekday_big.json"
POI_CACHE    = "Portugal_POI_data/PT_poi/oeiras_pois.gpkg"
ZONE_DATA    = "Portugal_POI_data/GRID1K21_CONT.gpkg"
OUTPUT_PATH  = "Portugal_POI_data/PT_poi/trajectories_portugal_weekday_big_geo.json"


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


def main():
    print(f"Loading big run: {BIG_RUN_PATH}")
    records = flatten_big_run(BIG_RUN_PATH)
    print(f"  {len(records)} person records")

    print("Loading POIAllocator (PT, Oeiras) …")
    allocator = POIAllocator(
        country="PT",
        poi_cache_path=POI_CACHE,
        zone_data_path=ZONE_DATA,
        seed=42,
    )

    print("Allocating coordinates …")
    enriched = []
    n = len(records)
    for i, rec in enumerate(records):
        if i % 200 == 0:
            print(f"  {i}/{n} …")
        home = allocator.sample_agent_home()
        enriched.append(allocate_trajectory(rec, allocator, home))

    print(f"Done. Saving to {OUTPUT_PATH}")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=None)

    print(f"Saved {len(enriched)} records.")

    # Sanity check: print first agent
    sample = enriched[0]
    print(f"\nSample agent {sample.get('person_id', '?')}  "
          f"home: ({sample['agent_home_lat']:.4f}, {sample['agent_home_lon']:.4f})")
    for t in sample["trips"][:3]:
        lat = t.get("destination_lat")
        lon = t.get("destination_lon")
        coords = f"({lat:.4f}, {lon:.4f})" if lat is not None else "None"
        print(f"  {t.get('time', '?')}  [{t.get('mode', '?')}]  {t.get('purpose', '?')}")
        print(f"    -> {coords}  POI: {t.get('destination_poi_label', '?')}")


if __name__ == "__main__":
    main()
