"""
variance_evaluation.py — Aggregate SynTrav JSD scores across generation seeds.

instead of a single-run JSD per condition, this py collects one evaluate() score per
(condition, seed) pair from Json_files/variance/*.json and summarizes it as
mean/std/min/max/range/cv — the same schema as smp/results/seeds_summary.csv,
so the SMP baseline and SynTrav results can be concatenated into one table.

Usage
-----
    from Helpers.variance_evaluation import (
        collect_variance_long, summarize_variance,
        format_mean_std_table, save_latex_table,
    )

    df_long = collect_variance_long(
        "Json_files/variance", real_weekday, geo_lookup, n_real_zero_trip,
    )
    df_summary = summarize_variance(df_long)
    display_df, numeric_df = format_mean_std_table(df_summary, order=[...])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Helpers.evaluation import evaluate, load_syn_records


# Constants
SEEDS = [3, 7, 13, 27, 42]

# File-name prefix (Json_files/variance/{prefix}_seed{N}.json) -> display label
CONDITIONS: dict[str, str] = {
    "full":                   "Llama 3.3-70B",
    "openai":                 "GPT-4o-mini",
    "claude":                 "Claude Haiku",
    "no_plan":                "w/o daily plan",
    "no_patterns_personas":   "w/o persona + pattern",
    "distance_ablation":      "with distance reasoning",
    "mode_ablation":          "with mode reasoning",
    "mode_distance_ablation": "with mode + distance reasoning",
}

METRICS = ["SD", "SI", "DARD", "DailyLoc"]



# Collection
def collect_variance_long(
    variance_dir:     str | Path,
    real_trips_df:    pd.DataFrame,
    geo_lookup:       dict | None,
    n_real_zero_trip: int,
    conditions:       dict[str, str] = CONDITIONS,
    seeds:            list[int] = SEEDS,
    reconstruct_home: bool = False,
) -> pd.DataFrame:
    """
    Run evaluate() once per (condition, seed) file and return a long-format
    DataFrame: model, label, metric, value, seed — one row per metric per run.
    Matches the schema of smp/results/seeds_raw.csv.

    Missing files are skipped with a printed warning rather than raising, so a
    partially-complete seed sweep can still be summarized.
    """
    variance_dir = Path(variance_dir)
    rows: list[dict] = []

    for prefix, label in conditions.items():
        for seed in seeds:
            path = variance_dir / f"{prefix}_seed{seed}.json"
            if not path.exists():
                print(f"[variance] missing {path}, skipping")
                continue

            records = load_syn_records(str(path))
            scores = evaluate(
                real_trips_df, records,
                geo_lookup=geo_lookup,
                label=label,
                seed=seed,
                n_real_zero_trip=n_real_zero_trip,
                reconstruct_home=reconstruct_home,
            )
            for metric in METRICS:
                rows.append({
                    "model":  "SynTrav",
                    "label":  label,
                    "metric": metric,
                    "value":  scores[metric],
                    "seed":   seed,
                })

    return pd.DataFrame(rows)


def summarize_variance(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the long-format variance table into mean/std/min/max/range/cv
    per (label, metric) — same columns as smp/results/seeds_summary.csv.
    """
    grp = df_long.groupby(["label", "metric"])["value"]
    summary = grp.agg(mean="mean", std="std", min="min", max="max").reset_index()
    summary["range"] = summary["max"] - summary["min"]
    summary["cv"] = summary["std"] / summary["mean"]
    return summary[["label", "metric", "mean", "std", "min", "max", "range", "cv"]]



# Table formatting
def format_mean_std_table(
    summary_df: pd.DataFrame,
    order:      list[str] | None = None,
    metrics:    tuple[str, ...] = tuple(METRICS),
    decimals:   int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivot a long summary_df (label, metric, mean, std, ...) into two wide
    tables indexed by label:
      - display_df : "{mean:.{decimals}f} ± {std:.{decimals}f}" strings
      - numeric_df : raw mean values (float) — used to find the per-column
                     best (lowest JSD) entry for bolding

    order: explicit row order (labels); rows not present in summary_df are
    silently skipped. Defaults to summary_df's natural label order.
    """
    mean_wide = summary_df.pivot(index="label", columns="metric", values="mean")
    std_wide  = summary_df.pivot(index="label", columns="metric", values="std")

    if order is not None:
        present = [lbl for lbl in order if lbl in mean_wide.index]
        mean_wide = mean_wide.reindex(present)
        std_wide  = std_wide.reindex(present)

    mean_wide = mean_wide[list(metrics)]
    std_wide  = std_wide[list(metrics)]

    def _fmt(m, s):
        if pd.isna(s):
            return f"{m:.{decimals}f}"
        return f"{m:.{decimals}f} ± {s:.{decimals}f}"

    display_df = pd.DataFrame(
        {col: [_fmt(m, s) for m, s in zip(mean_wide[col], std_wide[col])]
         for col in metrics},
        index=mean_wide.index,
    )
    return display_df, mean_wide


def style_best(display_df: pd.DataFrame, numeric_df: pd.DataFrame, caption: str = ""):
    """
    Style a "mean ± std" display_df, bolding the per-column best (lowest JSD)
    row using numeric_df to determine which row that is. Matches the
    "font-weight: bold; color: green" convention used elsewhere in the
    evaluation notebook for single-run tables.
    """
    def _highlight(col: pd.Series) -> list[str]:
        if col.name not in numeric_df.columns or numeric_df[col.name].isna().all():
            return ["" for _ in col.index]
        best = numeric_df[col.name].idxmin()
        return ["font-weight: bold; color: green" if idx == best else "" for idx in col.index]

    styler = display_df.style.apply(_highlight, axis=0)
    if caption:
        styler = styler.set_caption(caption)
    return styler



#