"""
run_seeds.py — Multi-seed stability analysis for the SMP baseline.

Reuses the cached smp_model.pkl (no re-fitting). Only simulation + evaluation
are repeated, so each seed takes ~seconds.

Usage
-----
    python smp/run_seeds.py                   # default: 20 seeds (0-19)
    python smp/run_seeds.py --n-seeds 50
    python smp/run_seeds.py --seeds 0 1 7 42  # specific seeds

Output
------
  smp/results/seeds_raw.csv     — per-seed, per-label, per-metric values
  smp/results/seeds_summary.csv — mean ± std across seeds
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smp.episode_extractor import load_config
from smp.estimator         import load_model
from smp.simulator         import simulate_population
from smp.evaluator         import evaluate_smp, prepare_real
from smp.run_smp           import (
    load_data, build_persona_list, zero_trip_correction,
    N_SYNTHETIC, CONFIG_PATH, MODEL_PATH, RESULTS_DIR,
    TRAIN_CSV, HOLDOUT_CSV,
)

DATA_DIR        = ROOT / "ODiN (DATA)" / "DATAVERSE" / "intermediarie_csvs"
RESPONDENTS_CSV = ROOT / "ODiN (DATA)" / "DATAVERSE" / "ODiN Data (RAW)" / "Respondents.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed SMP stability analysis")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--n-seeds", type=int, default=20,
                       help="Run seeds 0 … n-1 (default: 20)")
    group.add_argument("--seeds", type=int, nargs="+",
                       help="Explicit list of seeds to run")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds if args.seeds else list(range(args.n_seeds))
    print(f"[run_seeds] Running {len(seeds)} seeds: {seeds[:10]}{'...' if len(seeds) > 10 else ''}")

    cfg = load_config(CONFIG_PATH)
    c   = cfg["columns"]

    #  Load data (once) 
    train_df, holdout_df = load_data(cfg)
    persona_list = build_persona_list(train_df, cfg, N_SYNTHETIC)

    prov_col     = "Province of residential municipality"
    train_wkday  = train_df[train_df[c["day_type"]] == "weekday"]
    utrecht_df   = holdout_df[holdout_df[prov_col] == "Utrecht"].copy()
    friesland_df = holdout_df[holdout_df[prov_col] == "Friesland"].copy()

    real_train    = prepare_real(train_df,    cfg)
    real_utrecht  = prepare_real(utrecht_df,  cfg)
    real_friesland = prepare_real(friesland_df, cfg)

    n_zero_train    = zero_trip_correction(train_wkday, cfg)
    n_zero_utrecht  = zero_trip_correction(utrecht_df[utrecht_df[c["day_type"]] == "weekday"], cfg)
    n_zero_friesland = zero_trip_correction(friesland_df[friesland_df[c["day_type"]] == "weekday"], cfg)

    #  Load cached model 
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No cached model at {MODEL_PATH}. Run smp/run_smp.py first."
        )
    print(f"[run_seeds] Loading cached model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    #  Sweep seeds 
    all_rows = []

    for seed in seeds:
        print(f"\n[run_seeds] Seed {seed} — simulating...")
        syn_trips = simulate_population(model, persona_list, seed=seed)

        for label, real_df, n_zero in [
            ("train_10provinces", real_train,     n_zero_train),
            ("holdout_utrecht",   real_utrecht,   n_zero_utrecht),
            ("holdout_friesland", real_friesland, n_zero_friesland),
        ]:
            res = evaluate_smp(real_df, syn_trips, model, cfg,
                               label=label, n_real_zero=n_zero, seed=seed)
            res["seed"] = seed
            all_rows.append(res)

    raw_df = pd.concat(all_rows, ignore_index=True)
    raw_path = RESULTS_DIR / "seeds_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n[run_seeds] Raw results saved to {raw_path}")

    #  Summary: mean ± std across seeds 
    summary = (
        raw_df.groupby(["label", "metric"])["value"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    summary["range"] = summary["max"] - summary["min"]
    summary["cv"]    = summary["std"] / summary["mean"].abs()   # coefficient of variation

    summary_path = RESULTS_DIR / "seeds_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[run_seeds] Summary saved to {summary_path}")

    #  Print focused view on DARD 
    print("\n" + "=" * 70)
    print("DARD stability across seeds")
    print("=" * 70)
    dard = summary[summary["metric"] == "DARD"][["label", "mean", "std", "min", "max", "cv"]]
    print(dard.to_string(index=False))

    print("\n" + "=" * 70)
    print("All metrics — mean ± std (train_10provinces)")
    print("=" * 70)
    train_sum = summary[summary["label"] == "train_10provinces"][
        ["metric", "mean", "std", "cv"]
    ]
    print(train_sum.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Full summary across {len(seeds)} seeds")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
