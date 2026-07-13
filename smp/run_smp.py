"""
run_smp.py — End-to-end SMP baseline pipeline.

Usage
-----
From the project root:

    python smp/run_smp.py

Stages
------
1. Load config and data
2. Extract episodes from training data
3. Fit SMP model (router + hazard + distance)
4. Simulate 535 synthetic persons from national persona distribution
5. Evaluate against:
   a. 10-province training distribution (in-distribution check)
   b. Utrecht holdout
   c. Friesland holdout
6. Save results CSV
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smp.episode_extractor import extract_episodes, load_config
from smp.estimator         import fit, save_model, load_model
from smp.simulator         import simulate_population
from smp.evaluator         import evaluate_smp, prepare_real

# 
# Paths
# 
DATA_DIR        = ROOT / "ODiN (DATA)" / "DATAVERSE" / "intermediarie_csvs"
TRAIN_CSV       = DATA_DIR / "odin_train.csv"
HOLDOUT_CSV     = DATA_DIR / "odin_holdout.csv"
RESPONDENTS_CSV = ROOT / "ODiN (DATA)" / "DATAVERSE" / "ODiN Data (RAW)" / "Respondents.csv"
CONFIG_PATH     = ROOT / "smp" / "config.yaml"
MODEL_PATH      = ROOT / "smp" / "smp_model.pkl"
RESULTS_DIR     = ROOT / "smp" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SYNTHETIC = 570   # total synthetic persons (matches LLM pipeline), if it says 3210 it was the run for the spatial allocation comparison. The pipeline was with 570. 
SEED        = 42


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg["columns"]
    print("[run_smp] Loading training data...")
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    train_df = train_df[train_df[c["day_type"]] == "weekday"].copy()
    print(f"  Train: {len(train_df):,} weekday trip rows")

    print("[run_smp] Loading holdout data...")
    holdout_df = pd.read_csv(HOLDOUT_CSV, low_memory=False)
    print(f"  Holdout: {len(holdout_df):,} trip rows")
    return train_df, holdout_df


def build_persona_list(
    train_df: pd.DataFrame,
    cfg: dict,
    n_synthetic: int,
) -> list[dict]:
    """
    Build the list of persona specs for simulation, proportional to the
    national training distribution. n_synthetic persons total.
    """
    c = cfg["columns"]
    group_counts = (
        train_df[train_df[c["day_type"]] == "weekday"]
        .groupby(c["group_key"])[c["person_id"]]
        .nunique()
    )
    total = group_counts.sum()
    specs = []
    remainder = n_synthetic
    groups = group_counts.index.tolist()
    for i, gk in enumerate(groups):
        if i == len(groups) - 1:
            n = remainder
        else:
            n = max(1, round(n_synthetic * group_counts[gk] / total))
            remainder -= n
        specs.append({
            "persona_group": gk,
            "day_of_week":   "weekday",
            "n_samples":     n,
        })
    print(f"[run_smp] Persona distribution: {len(specs)} groups, {sum(s['n_samples'] for s in specs)} persons")
    return specs


def zero_trip_correction(trips_weekday_df: pd.DataFrame, cfg: dict) -> int:
    """
    Estimate zero-trip persons for DailyLoc correction using the same formula
    as the LLM evaluation notebook:
        n_zero = round(national_ratio * n_persons_with_trips)
    where national_ratio = zero_trip_weekday_respondents / non_zero_trip_weekday_respondents.
    """
    resp = pd.read_csv(RESPONDENTS_CSV, low_memory=False)
    resp_wkday   = resp[resp["Weekdag"].isin([1, 2, 3, 4, 5])]
    zero_count   = (resp_wkday["AantVpl"] == 0).sum()
    nonzero_count = (resp_wkday["AantVpl"] > 0).sum()
    ratio        = zero_count / nonzero_count if nonzero_count else 0.0

    c = cfg["columns"]
    n_persons = trips_weekday_df[c["person_id"]].nunique()
    return round(ratio * n_persons)


def main():
    cfg = load_config(CONFIG_PATH)
    c   = cfg["columns"]

    #  1. Load data 
    train_df, holdout_df = load_data(cfg)

    #  2. Extract episodes 
    print("\n[run_smp] Extracting episodes from training data...")
    episodes_df = extract_episodes(train_df, cfg)
    episodes_df.to_csv(RESULTS_DIR / "episodes_train.csv", index=False)

    #  3. Fit SMP model 
    if MODEL_PATH.exists():
        print(f"\n[run_smp] Loading cached model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    else:
        print("\n[run_smp] Fitting SMP model...")
        model = fit(episodes_df, train_df, cfg)
        save_model(model, MODEL_PATH)

    #  4. Build persona list and simulate 
    print("\n[run_smp] Building persona distribution...")
    persona_list = build_persona_list(train_df, cfg, N_SYNTHETIC)

    print("\n[run_smp] Simulating synthetic population...")
    syn_trips = simulate_population(model, persona_list, seed=SEED)
    pd.DataFrame(syn_trips).to_csv(RESULTS_DIR / "syn_trips.csv", index=False)

    all_results = []

    prov_col = "Province of residential municipality"

    #  5a. In-distribution evaluation (10 training provinces) 
    print("\n[run_smp] Evaluating — in-distribution (training provinces)...")
    train_wkday    = train_df[train_df[c["day_type"]] == "weekday"]
    n_zero_train   = zero_trip_correction(train_wkday, cfg)
    print(f"  Zero-trip correction (train): {n_zero_train}")
    real_train = prepare_real(train_df, cfg)
    res_train  = evaluate_smp(real_train, syn_trips, model, cfg,
                              label="train_10provinces", n_real_zero=n_zero_train)
    all_results.append(res_train)

    #  5b. Utrecht holdout 
    print("\n[run_smp] Evaluating — Utrecht holdout...")
    utrecht_df     = holdout_df[holdout_df[prov_col] == "Utrecht"].copy()
    utrecht_wkday  = utrecht_df[utrecht_df[c["day_type"]] == "weekday"]
    n_zero_utrecht = zero_trip_correction(utrecht_wkday, cfg)
    print(f"  Zero-trip correction (Utrecht): {n_zero_utrecht}")
    real_utrecht = prepare_real(utrecht_df, cfg)
    res_utrecht  = evaluate_smp(real_utrecht, syn_trips, model, cfg,
                                label="holdout_utrecht", n_real_zero=n_zero_utrecht)
    all_results.append(res_utrecht)

    #  5c. Friesland holdout 
    print("\n[run_smp] Evaluating — Friesland holdout...")
    friesland_df      = holdout_df[holdout_df[prov_col] == "Friesland"].copy()
    friesland_wkday   = friesland_df[friesland_df[c["day_type"]] == "weekday"]
    n_zero_friesland  = zero_trip_correction(friesland_wkday, cfg)
    print(f"  Zero-trip correction (Friesland): {n_zero_friesland}")
    real_friesland = prepare_real(friesland_df, cfg)
    res_friesland  = evaluate_smp(real_friesland, syn_trips, model, cfg,
                                  label="holdout_friesland", n_real_zero=n_zero_friesland)
    all_results.append(res_friesland)

    #  6. Save results 
    results_df = pd.concat(all_results, ignore_index=True)
    out_path   = RESULTS_DIR / "smp_evaluation_results_big.csv" ## was renamed for the big generation
    results_df.to_csv(out_path, index=False)
    print(f"\n[run_smp] Results saved to {out_path}")
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
