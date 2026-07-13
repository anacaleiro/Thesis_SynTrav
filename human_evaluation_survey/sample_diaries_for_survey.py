"""
sample_diaries_for_survey.py

Samples 15 synthetic + 15 real ODiN diaries and formats both through an
identical template before shuffling so the reader of the survey will not be influenced
by asthetic differences. 

Pre-survey checklist addressed here:
  - LLM prose in purpose field extracted to a clean label
  - Unified mode display vocabulary applied to both sides
  - Unified purpose display vocabulary applied to both sides
  - Trips sorted chronologically on both sides
  - __summary__ metadata key skipped

Output:
  survey_diaries_all.csv   — share with evaluators (no labels)
  survey_diaries_key.csv   — answer key (keep private)
  survey_diaries_all.txt   — formatted blocks for Google Forms
  survey_audit.txt         — before/after label audit for manual spot-check
"""

import json
import random
import csv
import re
import pandas as pd

# Config 
SOURCE_JSON  = "Json_files/variance/full_seed42.json"
SOURCE_ODIN  = "ODiN (DATA)/DATAVERSE/intermediarie_csvs/odin_cleaned.csv"
N_SYNTHETIC  = 8
N_REAL       = 8
RANDOM_SEED  = 42
OUT_ALL_CSV  = "survey_diaries_all.csv"
OUT_KEY_CSV  = "survey_diaries_key.csv"
OUT_TXT      = "survey_diaries_all.txt"
OUT_AUDIT    = "survey_audit.txt"

# Applied identically to both synthetic and real diaries so it's more intuitive for evaluators
MODE_DISPLAY = {
    "On foot":                              "Walking",
    "Non-electric bicycle":                 "Bicycle",
    "Electric bike":                        "E-bike",
    "Speed pedelec":                        "Speed pedelec",
    "Passenger car":                        "Car",
    "Engine":                               "Motorcycle",
    "Moped":                                "Moped",
    "Skates/inline skates/step":            "Scooter/skates",
    "Bus":                                  "Bus",
    "Tram":                                 "Tram",
    "Subway":                               "Metro",
    "Train":                                "Train",
    "Taxi/Taxi van":                        "Taxi",
    "Delivery van":                         "Delivery van",
    "Agricultural vehicle":                 "Agricultural vehicle",
    "Disabled transport vehicle with motor":"Adapted transport",
    "Disabled transport vehicle without engine": "Adapted transport",
    "Boat":                                 "Boat",
    "Camper":                               "Camper",
    "Coach":                                "Coach",
    "Truck":                                "Truck",
    "Different with engine":                "Other (motorised)",
    "Otherwise without engine":             "Other (non-motorised)",
    # Synthetic variants that deviate slightly from ODiN spelling
    "Electric bicycle":                     "E-bike",
    "Bus/tram/metro":                       "Bus/tram/metro",
    "Public transport":                     "Public transport",
    "Walking":                              "Walking",
    "Car":                                  "Car",
    "Moped/scooter":                        "Moped",
    "Car (driver)":                         "Car",
    "Car (passenger)":                      "Car (passenger)",
}

# ODiN  purpose labels directly extracted from  the codebook
ODIN_PURPOSES = {
    "To and from work",
    "Taking education/course",
    "Shopping/grocery shopping",
    "Services/personal care",
    "Sports/hobbies",
    "Touring/hiking",
    "Other leisure activities",
    "Visitors/staying over",
    "Business visit in a working atmosphere",
    "Professionally",
    "Pick up/drop off people",
    "Collect/deliver goods",
    "Different motive",
}

# Map non-ODiN synthetic labels: closest ODiN purpose
PURPOSE_MAP = {
    "Running errands":                  "Shopping/grocery shopping",
    "Social or family visit destination": "Visitors/staying over",
    "Socializing":                      "Visitors/staying over",
    "Other":                            "Different motive",
    "Leisure":                          "Other leisure activities",
    "Recreation":                       "Other leisure activities",
    "Work":                             "To and from work",
    "Education":                        "Taking education/course",
    "Healthcare":                       "Services/personal care",
}

# Keyword patterns for prose that couldn't be mapped by regex (e.g. truncated strings).
# Each entry: (substring to look for in the raw prose, ODiN label to assign)
PURPOSE_PROSE_KEYWORDS = [
    ("picking up children",  "Pick up/drop off people"),
    ("pick up children",     "Pick up/drop off people"),
    ("school run",           "Pick up/drop off people"),
    ("dropping off",         "Pick up/drop off people"),
    ("picking up kids",      "Pick up/drop off people"),
]

def clean_mode(raw: str) -> str:
    return MODE_DISPLAY.get(raw, raw) if raw else "Unknown"

def clean_purpose(raw: str) -> str:
    """
    Handle three cases:
    1. Already an ODiN label → return as-is
    2. Known non-ODiN label → map via PURPOSE_MAP
    3. LLM prose → extract the label after the last ": " pattern
    """
    if not raw:
        return "Different motive"

    # Comma-joined (e.g. "Pick up/drop off people, Shopping/grocery shopping")
    # LLM was ambiguous, ODiN allows only one purpose per trip so it's map to  Different motive
    if "," in raw and len(raw) < 80:
        return "Different motive"

    # Already canonical
    if raw in ODIN_PURPOSES:
        return raw

    # Known non-ODiN mapping
    if raw in PURPOSE_MAP:
        return PURPOSE_MAP[raw]

    # LLM prose: extract label after last occurrence of ": " or "is: "
    # Pattern: "... closest match is: <Label>" or "... could be: <Label>"
    match = re.search(r':\s*([A-Z][^:\.]{3,50})$', raw)
    if match:
        candidate = match.group(1).strip().rstrip('.')
        if candidate in ODIN_PURPOSES:
            return candidate

    # Prose but no regex match → try keyword scan before falling back
    raw_lower = raw.lower()
    for keyword, label in PURPOSE_PROSE_KEYWORDS:
        if keyword in raw_lower:
            return label

    return "Different motive"

def format_persona_syn(group_key: str) -> str:
    parts = [p.strip() for p in group_key.split("|")]
    occupation = parts[0].replace("_", " ").capitalize()
    age        = f"age {parts[1]}" if len(parts) > 1 else ""
    day        = parts[2].lower() if len(parts) > 2 else ""
    income_raw = parts[3].replace("Income=", "").lower() if len(parts) > 3 else ""
    label      = f"{occupation}, {age}, {day}"
    if income_raw:
        label += f", income: {income_raw}"
    return label

def format_table(trips: list[dict]) -> str:
    header = f"{'Time':<8} {'Mode':<22} {'Purpose':<34} {'Distance':>10}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for t in sorted(trips, key=lambda x: x.get("dep_mins", 9999)):
        time     = t.get("time", "?")
        mode     = t.get("mode", "Unknown")
        purpose  = t.get("purpose", "Unknown")
        dist     = t.get("distance_km")
        dist_str = f"{dist:.1f} km" if dist is not None else "—"
        rows.append(f"{time:<8} {mode:<22} {purpose:<34} {dist_str:>10}")
    if len(rows) == 2:
        rows.append("  (no trips recorded — stay-at-home day)")
    return "\n".join(rows)

def time_to_mins(h, m):
    try:
        return int(h) * 60 + int(m)
    except (ValueError, TypeError):
        return 9999

#Sample synthetic diaries 
def sample_synthetic(path: str, n: int, seed: int):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    audit_lines = ["=== SYNTHETIC PURPOSE CLEANING ===\n"]
    all_traj = []
    for group_key, trajs in data.items():
        if group_key == "__summary__":
            continue
        for traj in trajs:
            if not isinstance(traj, dict):
                continue
            steps = traj.get("reasoning_steps", [])
            trips = []
            for s in steps:
                if not isinstance(s, dict) or not s.get("travel"):
                    continue
                raw_mode    = str(s.get("mode") or "")
                raw_purpose = str(s.get("purpose") or "")
                dist        = s.get("distance_km")
                time_str    = s.get("time", "?")
                h, m        = (time_str.split(":") + ["0"])[:2]
                dep_mins    = time_to_mins(h, m)

                clean_p = clean_purpose(raw_purpose)
                if clean_p != raw_purpose:
                    audit_lines.append(f"  BEFORE: {repr(raw_purpose[:80])}")
                    audit_lines.append(f"  AFTER:  {repr(clean_p)}\n")

                trips.append({
                    "time":        time_str,
                    "dep_mins":    dep_mins,
                    "mode":        clean_mode(raw_mode),
                    "purpose":     clean_p,
                    "distance_km": dist,
                })
            # Skip zero-trip diaries (atypical stay-at-home) — nothing to evaluate
        if len(trips) == 0:
            continue
        all_traj.append((group_key, traj, trips))

    random.seed(seed)
    sampled = random.sample(all_traj, min(n, len(all_traj)))

    diaries = []
    for group_key, traj, trips in sampled:
        diaries.append({
            "label":   "synthetic",
            "persona": format_persona_syn(group_key),
            "trips":   trips,
        })
    return diaries, audit_lines

#Sample real ODiN diaries 
def sample_real(path: str, n: int, seed: int):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["DayType"] == "weekday"].copy()

    pid_col     = "Unique ID for each OP"
    mode_col    = "Main mode of transport travel"
    purpose_col = "Motive"
    dep_h_col   = "Departure time transfer"
    dep_m_col   = "Departure minute displacement"
    dist_col    = "Travel distance in the Netherlands (in hectometers)"
    age_col     = "Age class OP"
    act_col     = "Activity_status"
    inc_col     = "income_level"

    df = df.dropna(subset=[mode_col, dep_h_col])

    person_groups = df.groupby(pid_col)
    person_ids = list(person_groups.groups.keys())
    random.seed(seed + 1)
    random.shuffle(person_ids)

    diaries = []
    for pid in person_ids:
        if len(diaries) >= n:
            break
        person_df = person_groups.get_group(pid)

        age = person_df[age_col].iloc[0] if age_col in person_df else "?"
        act = str(person_df[act_col].iloc[0]).replace("_", " ").capitalize() if act_col in person_df else "?"
        inc = person_df[inc_col].iloc[0] if inc_col in person_df else ""
        persona = f"{act}, age {age}, weekday"
        if pd.notna(inc) and inc:
            persona += f", income: {str(inc).lower()}"

        trips = []
        for _, row in person_df.iterrows():
            dep_h    = row[dep_h_col]
            dep_m    = row.get(dep_m_col, 0)
            dep_mins = time_to_mins(dep_h, dep_m)
            try:
                time_str = f"{int(dep_h):02d}:{int(dep_m if pd.notna(dep_m) else 0):02d}"
            except (ValueError, TypeError):
                time_str = "?"
            mode    = clean_mode(str(row[mode_col]))
            purpose = str(row[purpose_col]) if pd.notna(row.get(purpose_col)) else "Different motive"
            dist_hm = row.get(dist_col)
            dist_km = float(dist_hm) / 10.0 if pd.notna(dist_hm) else None
            trips.append({
                "time":        time_str,
                "dep_mins":    dep_mins,
                "mode":        mode,
                "purpose":     purpose,
                "distance_km": dist_km,
            })

        # Filter: keep only persons with 1–8 trips and all distances present
        if not (1 <= len(trips) <= 8):
            continue
        if any(t["distance_km"] is None for t in trips):
            continue

        diaries.append({
            "label":   "real",
            "persona": persona,
            "trips":   trips,
        })

    return diaries

# Main 
def main():
    print("Sampling synthetic diaries...")
    syn_diaries, audit_lines = sample_synthetic(SOURCE_JSON, N_SYNTHETIC, RANDOM_SEED)
    print(f"  {len(syn_diaries)} synthetic diaries sampled")

    print("Sampling real ODiN diaries...")
    real_diaries = sample_real(SOURCE_ODIN, N_REAL, RANDOM_SEED)
    print(f"  {len(real_diaries)} real diaries sampled")

    all_diaries = syn_diaries + real_diaries
    random.seed(RANDOM_SEED + 2)
    random.shuffle(all_diaries)
    for i, d in enumerate(all_diaries, start=1):
        d["diary_id"] = f"D{i:03d}"

    # Evaluator CSV (with no labels)
    with open(OUT_ALL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["diary_id", "persona", "n_trips", "table"])
        writer.writeheader()
        for d in all_diaries:
            writer.writerow({
                "diary_id": d["diary_id"],
                "persona":  d["persona"],
                "n_trips":  len(d["trips"]),
                "table":    format_table(d["trips"]),
            })

    # Answer key 
    with open(OUT_KEY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["diary_id", "label"])
        writer.writeheader()
        for d in all_diaries:
            writer.writerow({"diary_id": d["diary_id"], "label": d["label"]})

    # Formatted text for Google Forms 
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for d in all_diaries:
            f.write(f"{'='*64}\n")
            f.write(f"DIARY {d['diary_id']}\n")
            f.write(f"{'='*64}\n")
            f.write(f"Person profile: {d['persona']}\n\n")
            f.write(format_table(d["trips"]))
            f.write("\n\n")

    # Audit log 
    with open(OUT_AUDIT, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
        f.write("\n\n=== FINAL MODE VOCABULARY IN SYNTHETIC SAMPLE ===\n")
        all_modes = {t["mode"] for d in syn_diaries for t in d["trips"]}
        for m in sorted(all_modes):
            f.write(f"  {m}\n")
        f.write("\n=== FINAL PURPOSE VOCABULARY IN SYNTHETIC SAMPLE ===\n")
        all_purps = {t["purpose"] for d in syn_diaries for t in d["trips"]}
        for p in sorted(all_purps):
            f.write(f"  {p}\n")
        f.write("\n=== FINAL MODE VOCABULARY IN REAL SAMPLE ===\n")
        real_modes = {t["mode"] for d in real_diaries for t in d["trips"]}
        for m in sorted(real_modes):
            f.write(f"  {m}\n")
        f.write("\n=== FINAL PURPOSE VOCABULARY IN REAL SAMPLE ===\n")
        real_purps = {t["purpose"] for d in real_diaries for t in d["trips"]}
        for p in sorted(real_purps):
            f.write(f"  {p}\n")

    print(f"\nWritten: {OUT_ALL_CSV}")
    print(f"Written: {OUT_KEY_CSV}")
    print(f"Written: {OUT_TXT}")
    print(f"Written: {OUT_AUDIT}  <- check this before building the form")

    print("\n--- Preview: first 3 diaries ---")
    for d in all_diaries[:3]:
        print(f"\n[{d['diary_id']}] LABEL={d['label']}")
        print(f"Profile: {d['persona']}")
        print(format_table(d["trips"]))

if __name__ == "__main__":
    main()
