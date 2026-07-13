"""
fig_gallery.py

One file for every LLM-vs-ODiN(-vs-SMP) figure used in the evaluation notebook,
replacing the previously scattered fig_combined.py / fig_purpose.py /
fig_departure_by_persona.py / fig_dist_heatmap.py + assorted inline notebook cells:

    load_condition_runs        build the per-seed SynTrav DataFrames for one condition
    load_smp_runs               build the per-seed SMP baseline DataFrames (live re-sim)
    plot_combined               5-panel KDE/ECDF headline figure
    plot_purpose_combined       trip-purpose shares + deviation-from-ODiN
    plot_mode_share             overall mode split (Car/Walk/Bicycle/Train/Bus-Tram-Metro/Other)
    plot_mode_distance          mode split by distance band, grouped by occupation/income
    plot_departure_by_persona   departure-time KDE by persona group (Employed / Retired
                                 & Homemaker / Students)
    plot_dist_heatmap           trip distance by purpose x occupation heatmap

Condition selection
--------------------
`load_condition_runs(condition)` pulls straight from
Json_files/variance/{condition}_seed{N}.json for every seed in SEEDS (3, 7, 13,
27, 42) — the same 5 seeds and file layout used by Helpers/variance_evaluation.py
for the JSD tables — and returns one DataFrame per seed. Valid `condition`
values are the keys of Helpers.variance_evaluation.CONDITIONS:

    "full", "openai", "no_plan", "no_patterns_personas",
    "distance_ablation", "mode_ablation", "mode_distance_ablation"

Pass `seeds=[42]` to fall back to a single-seed run if you want a quick preview
instead of the full 5-seed average.

`load_smp_runs(seeds)` gives the SMP baseline the same treatment. The disk-cached
`smp/results/syn_trips.csv` is a single fixed-seed run — the JSD tables already
average SMP over 5 seeds (smp/results/seeds_raw.csv) but never persisted the
per-seed trip records to build distribution figures from. Re-simulating live is
cheap (reuses the cached smp_model.pkl; ~1-2s per extra seed after a ~3s one-time
setup), so load_smp_runs() just re-runs simulate_population() per seed in memory
instead of adding another cached file to keep in sync.

Every figure below therefore shows mean ± 1 std-dev bands for BOTH the SMP
baseline and SynTrav (not just SynTrav), computed across their respective seeds.

Usage from the notebook
------------------------
    from Helpers.visualizations.fig_gallery import (
        FIG_DIR, load_condition_runs, load_smp_runs,
        plot_combined, plot_purpose_combined,
        plot_mode_distance, plot_departure_by_persona, plot_dist_heatmap,
        OCC_LEVELS, INCOME_LEVELS,
    )
    from Helpers.visualizations.viz_utils import prep_odin

    odin_df  = prep_odin(real_weekday)
    smp_runs = load_smp_runs()          # 5 seeds, re-simulated live

    full_runs = load_condition_runs("full")
    mode_runs = load_condition_runs("mode_ablation")

    plot_combined(odin_df, smp_runs, full_runs, save_path=str(FIG_DIR / "fig_combined_full"))
    plot_combined(odin_df, smp_runs, mode_runs, save_path=str(FIG_DIR / "fig_combined_mode_ablation"))

    plot_mode_distance(full_runs, "occ", OCC_LEVELS, "Mode split by distance & occupation",
                        save_path=str(FIG_DIR / "fig_mode_distance_occ_full"))

    plot_dist_heatmap(full_runs, save_path=str(FIG_DIR / "fig_dist_heatmap_full"))

    plot_departure_by_persona(real_weekday, full_runs,
                               save_path=str(FIG_DIR / "fig_departure_by_persona_full"))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties
from scipy.stats import gaussian_kde
import contextily as ctx
from pyproj import Transformer

from Helpers.evaluation import load_syn_records, DEP_TIME_CLASS_RANGE
from Helpers.variance_evaluation import SEEDS, CONDITIONS
from .viz_utils import (
    set_style, strip_axes, add_caption,
    COLOR_ODIN, COLOR_SMP, COLOR_SYNTRAV,
    LS_ODIN, LS_SMP, LS_SYNTRAV, LW, ALPHA_BAND,
    SD_BINS, SI_BINS,
    kde_curve, normalise, extract_si_gaps,
    CANONICAL_PURPOSE_ORDER,
    mean_std_runs, prep_odin,
)


#  Output location 
FIG_DIR = Path("figures/04_evaluation")
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANCE_DIR = Path("Json_files/variance")


#  Shared bins / groupings 
DIST_BINS_KM   = [0, 1, 5, 10, 25, 50, np.inf]
DIST_LABELS_KM = ["<1 km", "1-5 km", "5-10 km", "10-25 km", "25-50 km", ">50 km"]

OCC_LEVELS    = ["employed", "homemaker", "retired", "student", "inactive"]
INCOME_LEVELS = ["Low income", "Below median", "Median", "Above median", "High income"]

TOP_PURPOSES = [
    "To and from work", "Shopping/grocery shopping", "Visitors/staying over",
    "Other leisure activities", "Sports/hobbies", "Services/personal care",
    "Pick up/drop off people", "Taking education/course",
]
PURPOSE_SHORT = {
    "To and from work":          "Work",
    "Shopping/grocery shopping": "Shopping",
    "Visitors/staying over":     "Visitors",
    "Other leisure activities":  "Leisure",
    "Sports/hobbies":            "Sports",
    "Services/personal care":    "Services",
    "Pick up/drop off people":   "Pick up / Drop off",
    "Taking education/course":   "Education",
}

MODES       = ["Bicycle", "Walk", "Car", "Transit", "Other"]
MODE_COLORS = {
    "Car":     "#7B2500",
    "Bicycle": "#B87333",
    "Walk":    "#D4A55A",
    "Transit": "#FF9800",
    "Other":   "#E8CC96",
}


def _mode_cat(mode: str) -> str:
    m = str(mode).lower()
    if any(x in m for x in ["car", "van", "truck", "camper"]):
        return "Car"
    if any(x in m for x in ["bicycle", "bike", "pedelec"]):
        return "Bicycle"
    if "foot" in m:
        return "Walk"
    if any(x in m for x in ["train", "bus", "tram", "subway", "metro", "coach"]):
        return "Transit"
    return "Other"


#  Data loading: one common per-trip schema for every panel 
def _flatten_run(records: list[dict], seed: int = 42) -> pd.DataFrame:
    """
    Flatten one SynTrav run into a per-trip DataFrame with columns:
        person_id, purpose_state, departure_min, arrival_min, distance_km,
        mode, mode_raw, plausible, occ, income, dist_group
    `occ`/`income` are parsed from each person's `group_key`
    (e.g. "employed | 26-30 | weekday | Income=Median"). Home-return trips
    are excluded. Departure times use departure_time_class with uniform
    jitter within the class range, matching evaluation.py / viz_utils.py.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for rec in records:
        pid = str(rec.get("person_id", ""))
        gk_parts = [p.strip() for p in (rec.get("group_key", "") or "").split("|")]
        occ    = gk_parts[0] if gk_parts else None
        income = next((p.replace("Income=", "").strip()
                       for p in gk_parts if p.startswith("Income=")), None)

        trips = [t for t in (rec.get("trips") or [])
                 if (t.get("destination") or "").lower() != "home"]

        deps = []
        for t in trips:
            dep_class = t.get("departure_time_class", "")
            if dep_class in DEP_TIME_CLASS_RANGE:
                lo, hi = DEP_TIME_CLASS_RANGE[dep_class]
                dep = int(rng.integers(lo, hi))
            else:
                try:
                    h, m = t["time"].split(":")
                    dep = int(h) * 60 + int(m)
                except Exception:
                    deps.append(None)
                    continue
            deps.append(dep)

        arrs = deps[1:] + deps[-1:] if deps else []
        for t, dep, arr in zip(trips, deps, arrs):
            if dep is None:
                continue
            rows.append({
                "person_id":     pid,
                "purpose_state": t.get("purpose", ""),
                "departure_min": float(dep),
                "arrival_min":   float(arr) if arr is not None else float(dep),
                "distance_km":   float(t.get("distance_km") or 0.0),
                "mode":          _mode_cat(t.get("mode", "")),
                "mode_raw":      t.get("mode", ""),
                "plausible":     bool(t.get("plausible", True)),
                "occ":           occ,
                "income":        income,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["dist_group"] = pd.cut(
            df["distance_km"], bins=DIST_BINS_KM, labels=DIST_LABELS_KM, right=False,
        )
    return df.reset_index(drop=True)


def load_condition_runs(
    condition: str,
    seeds: list[int] = SEEDS,
    variance_dir: str | Path = VARIANCE_DIR,
) -> list[pd.DataFrame]:
    """
    Build the list of per-seed DataFrames for one condition (e.g. "full",
    "mode_ablation"), reading Json_files/variance/{condition}_seed{N}.json for
    every seed. Missing seed files are skipped with a warning rather than
    raising, so a partially-finished seed sweep can still be plotted.
    """
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. Valid options: {list(CONDITIONS)}"
        )

    variance_dir = Path(variance_dir)
    runs = []
    for seed in seeds:
        path = variance_dir / f"{condition}_seed{seed}.json"
        if not path.exists():
            print(f"[fig_gallery] missing {path}, skipping seed {seed}")
            continue
        runs.append(_flatten_run(load_syn_records(str(path)), seed=seed))

    if not runs:
        raise FileNotFoundError(
            f"No seed files found for condition '{condition}' in {variance_dir}"
        )
    return runs


#  SMP baseline: live re-simulation across seeds 
_SMP_STATE: dict | None = None   # cached (model, persona_list) so repeat calls don't re-fit


def _smp_setup():
    """Load the SMP config/data/cached-model/persona-list once per process."""
    global _SMP_STATE
    if _SMP_STATE is not None:
        return _SMP_STATE

    from smp.episode_extractor import load_config
    from smp.estimator import load_model
    from smp.run_smp import load_data, build_persona_list, CONFIG_PATH, MODEL_PATH, N_SYNTHETIC

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No cached SMP model at {MODEL_PATH}. Run `python smp/run_smp.py` first."
        )

    cfg = load_config(CONFIG_PATH)
    train_df, _holdout_df = load_data(cfg)
    persona_list = build_persona_list(train_df, cfg, N_SYNTHETIC)
    model = load_model(MODEL_PATH)

    _SMP_STATE = {"model": model, "persona_list": persona_list}
    return _SMP_STATE


def load_smp_runs(seeds: list[int] = SEEDS) -> list[pd.DataFrame]:
    """
    Re-simulate the SMP baseline once per seed (reusing the cached smp_model.pkl
    — no re-fitting) and return one trip-level DataFrame per seed, in the same
    common schema as load_condition_runs(). Nothing is written to disk; this is
    the SMP equivalent of load_condition_runs() for the seed-variance figures.
    """
    from smp.simulator import simulate_population

    state = _smp_setup()
    return [
        pd.DataFrame(simulate_population(state["model"], state["persona_list"], seed=seed))
        for seed in seeds
    ]


#  Panel (d)/(e) helpers: ECDF (trips/person, purpose diversity) 
def _trips_per_person(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("person_id").size().values


def _unique_purposes_per_person(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("person_id")["purpose_state"].nunique().values


def _ecdf(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = np.sort(np.asarray(vals, dtype=float))
    if len(vals) == 0:
        return np.zeros_like(grid, dtype=float)
    return np.searchsorted(vals, grid, side="right") / len(vals)


def _draw_ecdf_panel(ax, odin_vals, smp_runs_vals, syntrav_runs_vals, xlabel, x_max):
    grid = np.arange(0, x_max + 1)

    y_odin = _ecdf(odin_vals, grid)
    y_smp, y_smp_std = mean_std_runs([_ecdf(v, grid) for v in smp_runs_vals])
    y_syn, y_syn_std = mean_std_runs([_ecdf(v, grid) for v in syntrav_runs_vals])

    ax.step(grid, y_odin, where="post", color=COLOR_ODIN,    lw=LW, ls=LS_ODIN)
    ax.step(grid, y_smp,  where="post", color=COLOR_SMP,     lw=LW, ls=LS_SMP)
    ax.step(grid, y_syn,  where="post", color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV)
    ax.fill_between(
        grid, np.clip(y_smp - y_smp_std, 0, 1), np.clip(y_smp + y_smp_std, 0, 1),
        step="post", color=COLOR_SMP, alpha=ALPHA_BAND,
    )
    ax.fill_between(
        grid, np.clip(y_syn - y_syn_std, 0, 1), np.clip(y_syn + y_syn_std, 0, 1),
        step="post", color=COLOR_SYNTRAV, alpha=ALPHA_BAND,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative share")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)


#  Panels (a)/(b)/(c): distance / departure-time / step-interval KDEs 
# Shared by plot_combined below. smp_runs / syntrav_runs are lists of DataFrames
# (mean ± 1 std-dev band drawn from each list independently).
def _band(ax, x, y, y_std, color, where=None):
    kwargs = {"step": where} if where else {}
    ax.fill_between(x, np.clip(y - y_std, 0, None), y + y_std, color=color, alpha=ALPHA_BAND, **kwargs)


def _panel_distance(ax, odin_df, smp_runs, syntrav_runs):
    x_max = 60
    x = np.linspace(0.05, x_max, 400)

    y_odin = kde_curve(odin_df["distance_km"].dropna().clip(upper=x_max), x)
    y_smp, y_smp_std = mean_std_runs(
        [kde_curve(r["distance_km"].dropna().clip(upper=x_max), x) for r in smp_runs]
    )
    y_syn, y_syn_std = mean_std_runs(
        [kde_curve(r["distance_km"].dropna().clip(upper=x_max), x) for r in syntrav_runs]
    )

    ax.plot(x, y_odin, color=COLOR_ODIN,    lw=LW, ls=LS_ODIN)
    ax.plot(x, y_smp,  color=COLOR_SMP,     lw=LW, ls=LS_SMP)
    ax.plot(x, y_syn,  color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV)
    _band(ax, x, y_smp, y_smp_std, COLOR_SMP)
    _band(ax, x, y_syn, y_syn_std, COLOR_SYNTRAV)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 60)

    def _sd_hist(df):
        counts, _ = np.histogram(df["distance_km"].dropna(), bins=SD_BINS)
        return normalise(counts.astype(float))

    h_smp_mean, _ = mean_std_runs([_sd_hist(r) for r in smp_runs])
    h_syn_mean, _ = mean_std_runs([_sd_hist(r) for r in syntrav_runs])
    return {"odin": _sd_hist(odin_df), "smp": h_smp_mean, "syn": h_syn_mean}


def _panel_departure(ax, odin_df, smp_runs, syntrav_runs):
    x = np.linspace(0, 24, 400)

    y_odin = kde_curve(odin_df["departure_min"] / 60, x)
    y_smp, y_smp_std = mean_std_runs([kde_curve(r["departure_min"] / 60, x) for r in smp_runs])
    y_syn, y_syn_std = mean_std_runs([kde_curve(r["departure_min"] / 60, x) for r in syntrav_runs])

    ax.plot(x, y_odin, color=COLOR_ODIN,    lw=LW, ls=LS_ODIN)
    ax.plot(x, y_smp,  color=COLOR_SMP,     lw=LW, ls=LS_SMP)
    ax.plot(x, y_syn,  color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV)
    _band(ax, x, y_smp, y_smp_std, COLOR_SMP)
    _band(ax, x, y_syn, y_syn_std, COLOR_SYNTRAV)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])

    return {"odin": normalise(y_odin), "smp": normalise(y_smp), "syn": normalise(y_syn)}


def _panel_si(ax, odin_df, smp_runs, syntrav_runs):
    """SI gap = departure_min[i+1] – departure_min[i] per person, capped at 180 min."""
    x = np.linspace(0, 180, 300)

    y_odin = kde_curve(extract_si_gaps(odin_df), x)
    y_smp, y_smp_std = mean_std_runs([kde_curve(extract_si_gaps(r), x) for r in smp_runs])
    y_syn, y_syn_std = mean_std_runs([kde_curve(extract_si_gaps(r), x) for r in syntrav_runs])

    ax.plot(x, y_odin, color=COLOR_ODIN,    lw=LW, ls=LS_ODIN)
    ax.plot(x, y_smp,  color=COLOR_SMP,     lw=LW, ls=LS_SMP)
    ax.plot(x, y_syn,  color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV)
    _band(ax, x, y_smp, y_smp_std, COLOR_SMP)
    _band(ax, x, y_syn, y_syn_std, COLOR_SYNTRAV)

    ax.set_xlabel("Gap (min)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 60, 120, 180])

    def _si_hist(df):
        counts, _ = np.histogram(extract_si_gaps(df), bins=SI_BINS)
        return normalise(counts.astype(float))

    h_smp_mean, _ = mean_std_runs([_si_hist(r) for r in smp_runs])
    h_syn_mean, _ = mean_std_runs([_si_hist(r) for r in syntrav_runs])
    return {"odin": _si_hist(odin_df), "smp": h_smp_mean, "syn": h_syn_mean}


#  plot_combined: 5-panel headline figure 
def plot_combined(
    odin_df: pd.DataFrame,
    smp_runs: list[pd.DataFrame] | pd.DataFrame,
    syntrav_runs: list[pd.DataFrame] | pd.DataFrame,
    save_path: str | None = None,
    max_trips: int = 12,
    max_purposes: int = 8,
) -> plt.Figure:
    """
    Top row:    (a) Trip Distance KDE | (b) Departure Time KDE | (c) Step Interval KDE
    Bottom row:     (d) Trips per Person ECDF     |     (e) Purpose Diversity ECDF

    smp_runs and syntrav_runs each accept a single DataFrame or a list of
    per-seed DataFrames — every panel shows a mean ± 1 std-dev band for
    whichever side has more than one run.
    """
    if isinstance(smp_runs, pd.DataFrame):
        smp_runs = [smp_runs]
    if isinstance(syntrav_runs, pd.DataFrame):
        syntrav_runs = [syntrav_runs]

    set_style()

    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(2, 6, hspace=0.55, wspace=0.45)

    ax_dist = fig.add_subplot(gs[0, 0:2])
    ax_dep  = fig.add_subplot(gs[0, 2:4])
    ax_si   = fig.add_subplot(gs[0, 4:6])
    ax_tpp  = fig.add_subplot(gs[1, 0:2])
    ax_purp = fig.add_subplot(gs[1, 2:4])

    _panel_distance(ax_dist, odin_df, smp_runs, syntrav_runs)
    strip_axes(ax_dist)
    add_caption(ax_dist, "a", "Trip Distance", fontsize=11)

    _panel_departure(ax_dep, odin_df, smp_runs, syntrav_runs)
    strip_axes(ax_dep)
    ax_dep.set_ylabel("")
    add_caption(ax_dep, "b", "Departure Time", fontsize=11)

    _panel_si(ax_si, odin_df, smp_runs, syntrav_runs)
    strip_axes(ax_si)
    ax_si.set_ylabel("")
    add_caption(ax_si, "c", "Step Interval", fontsize=11)

    _draw_ecdf_panel(
        ax=ax_tpp,
        odin_vals=_trips_per_person(odin_df),
        smp_runs_vals=[_trips_per_person(r) for r in smp_runs],
        syntrav_runs_vals=[_trips_per_person(r) for r in syntrav_runs],
        xlabel="Trips per person",
        x_max=max_trips,
    )
    strip_axes(ax_tpp)
    add_caption(ax_tpp, "d", "Trips per Person", fontsize=11)

    _draw_ecdf_panel(
        ax=ax_purp,
        odin_vals=_unique_purposes_per_person(odin_df),
        smp_runs_vals=[_unique_purposes_per_person(r) for r in smp_runs],
        syntrav_runs_vals=[_unique_purposes_per_person(r) for r in syntrav_runs],
        xlabel="Unique purposes per person",
        x_max=max_purposes,
    )
    strip_axes(ax_purp)
    ax_purp.set_ylabel("")
    add_caption(ax_purp, "e", "Purpose Diversity", fontsize=11)

    smp_label = r"SMP (baseline, $\pm$1$\sigma$ over seeds)" if len(smp_runs) > 1 else "SMP (baseline)"
    syn_label = r"SynTrav (LLM, $\pm$1$\sigma$ over seeds)" if len(syntrav_runs) > 1 else "SynTrav (LLM)"
    handles = [
        mlines.Line2D([], [], color=COLOR_ODIN,    lw=LW, ls=LS_ODIN,    label="ODiN (real data)"),
        mlines.Line2D([], [], color=COLOR_SMP,     lw=LW, ls=LS_SMP,     label=smp_label),
        mlines.Line2D([], [], color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV, label=syn_label),
    ]
    ax_si.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.0),
        frameon=False, fontsize=8, borderaxespad=0,
    )

    if save_path:
  
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_purpose_combined: shares + deviation from ODiN 
def plot_purpose_combined(
    odin_df: pd.DataFrame,
    smp_runs: list[pd.DataFrame] | pd.DataFrame,
    syntrav_runs: list[pd.DataFrame] | pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    if isinstance(smp_runs, pd.DataFrame):
        smp_runs = [smp_runs]
    if isinstance(syntrav_runs, pd.DataFrame):
        syntrav_runs = [syntrav_runs]

    def _shares(df):
        return df["purpose_state"].value_counts(normalize=True).mul(100)

    odin_shares = _shares(odin_df)
    smp_shares  = pd.DataFrame([_shares(r) for r in smp_runs]).fillna(0).mean()
    llm_shares  = pd.DataFrame([_shares(r) for r in syntrav_runs]).fillna(0).mean()

    purposes = [p for p in CANONICAL_PURPOSE_ORDER
                if p in odin_shares.index or p in llm_shares.index or p in smp_shares.index]

    odin_vals = odin_shares.reindex(purposes).fillna(0)
    smp_vals  = smp_shares.reindex(purposes).fillna(0)
    llm_vals  = llm_shares.reindex(purposes).fillna(0)

    order     = odin_vals.sort_values(ascending=True).index
    odin_vals = odin_vals[order]
    smp_vals  = smp_vals[order]
    llm_vals  = llm_vals[order]
    llm_delta = llm_vals - odin_vals
    smp_delta = smp_vals - odin_vals

    set_style()
    plt.rcParams.update({"axes.spines.left": False, "xtick.direction": "out"})

    fig, (ax_bars, ax_div) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    y      = np.arange(len(order))
    height = 0.25

    ax_bars.barh(y + height, odin_vals.values, height, color=COLOR_ODIN,    alpha=0.90)
    ax_bars.barh(y,          smp_vals.values,  height, color=COLOR_SMP,     alpha=0.85)
    ax_bars.barh(y - height, llm_vals.values,  height, color=COLOR_SYNTRAV, alpha=0.85)

    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels(order, fontsize=8.5)
    ax_bars.set_xlabel("Share of trips (%)", fontsize=9)
    ax_bars.set_xlim(0, max(odin_vals.max(), smp_vals.max(), llm_vals.max()) + 5)
    ax_bars.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_bars.tick_params(left=False)
    strip_axes(ax_bars)
    add_caption(ax_bars, "a", "Trip Purpose Distribution", fontsize=11)

    x_max = max(llm_delta.abs().max(), smp_delta.abs().max()) + 4

    ax_div.barh(y + height / 2, llm_delta.values, height, color=COLOR_SYNTRAV, alpha=0.85)
    ax_div.barh(y - height / 2, smp_delta.values, height, color=COLOR_SMP,     alpha=0.85)
    ax_div.axvline(0, color=COLOR_ODIN, linewidth=1.6, linestyle="-", zorder=5)

    for container, delta, color in [
        (ax_div.containers[0], llm_delta.values, COLOR_SYNTRAV),
        (ax_div.containers[1], smp_delta.values, COLOR_SMP),
    ]:
        for bar, val in zip(container, delta):
            if abs(val) <= 0.3:
                continue
            xpos = bar.get_width()
            ax_div.text(
                xpos + (0.2 if xpos >= 0 else -0.2),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}pp",
                va="center", ha="left" if xpos >= 0 else "right",
                fontsize=6.5, color=color, fontweight="bold",
            )

    ax_div.set_xlabel("Deviation from ODiN (pp)", fontsize=9)
    ax_div.set_xlim(-x_max, x_max)
    ax_div.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}pp"))
    ax_div.tick_params(left=False)
    ax_div.axvspan( 0,  x_max, alpha=0.03, color=COLOR_SYNTRAV, zorder=0)
    ax_div.axvspan(-x_max, 0,  alpha=0.03, color="#444444",     zorder=0)
    ax_div.text( x_max * 0.55, len(y) - 0.3, "Over",  fontsize=7, color="#aaaaaa", ha="center")
    ax_div.text(-x_max * 0.55, len(y) - 0.3, "Under", fontsize=7, color="#aaaaaa", ha="center")
    strip_axes(ax_div)
    add_caption(ax_div, "b", "Trip Purpose: Deviation from ODiN", fontsize=11)

    handles = [
        mpatches.Patch(color=COLOR_ODIN,    alpha=0.90, label="ODiN"),
        mpatches.Patch(color=COLOR_SMP,     alpha=0.85, label="SMP"),
        mpatches.Patch(color=COLOR_SYNTRAV, alpha=0.85, label="SynTrav (LLM)"),
    ]
    ax_div.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    plt.tight_layout()

    if save_path:      
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_mode_distance: mode split by distance band, grouped by occ/income 
def _mode_pct_table(df: pd.DataFrame, group_col: str, group_levels: list[str]) -> pd.DataFrame:
    """% share of each MODE per (group, distance-band) cell for one run."""
    rows = []
    for grp in group_levels:
        sub = df[df[group_col] == grp]
        for d in DIST_LABELS_KM:
            band  = sub[sub["dist_group"] == d]
            total = len(band)
            row = {"group": grp, "dist_group": d}
            for mode in MODES:
                row[mode] = (band["mode"] == mode).sum() / total * 100 if total else 0.0
            rows.append(row)
    return pd.DataFrame(rows).set_index(["group", "dist_group"])[MODES]


def plot_mode_distance(
    runs: list[pd.DataFrame] | pd.DataFrame,
    group_col: str,
    group_levels: list[str],
    title: str,
    save_path: str | None = None,
) -> plt.Figure:
    """
    One stacked bar panel per `group_levels` entry (e.g. one per occupation, or
    one per income band); x-axis = distance band, stacked bars = mode share.
    `runs`: single DataFrame or list of per-seed DataFrames (bars show the
    mean share across seeds).
    """
    if isinstance(runs, pd.DataFrame):
        runs = [runs]

    tables = [_mode_pct_table(r.dropna(subset=[group_col]), group_col, group_levels)
              for r in runs]
    mean_table = sum(tables) / len(tables)

    n     = len(group_levels)
    ncols = 3
    nrows = (n + ncols) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()
    x = np.arange(len(DIST_LABELS_KM))

    for ax, grp in zip(axes_flat, group_levels):
        bottoms = np.zeros(len(DIST_LABELS_KM))
        for mode in MODES:
            shares = mean_table.loc[grp, mode].reindex(DIST_LABELS_KM).values
            ax.bar(x, shares, bottom=bottoms, color=MODE_COLORS[mode], label=mode, width=0.65)
            bottoms += shares
        ax.set_title(grp, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(DIST_LABELS_KM, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("% of trips")
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)

    legend_ax = axes_flat[len(group_levels)]
    legend_ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m]) for m in MODES]
    legend_ax.legend(handles, MODES, loc="center", fontsize=11,
                      title="Mode", title_fontsize=12, borderpad=1.2, labelspacing=0.9)

    for ax in axes_flat[len(group_levels) + 1:]:
        ax.axis("off")

    n_runs = len(runs)
    suffix = f" (mean over {n_runs} seeds)" if n_runs > 1 else ""
    plt.suptitle(title + suffix, fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_mode_share: overall mode split, ODiN vs SynTrav 
# Finer-grained than MODES/_mode_cat (which collapses Train + Bus/Tram/Metro into
# "Transit" for the mode_distance panels) — kept separate on purpose so this
# overview chart matches ODiN's own mode vocabulary one level down.
MODE_GROUPS = {
    "Car":            ["Passenger car", "Delivery van", "Truck", "Camper"],
    "Walking":        ["On foot"],
    "Bicycle":        ["Non-electric bicycle", "Electric bike", "Speed pedelec"],
    "Train":          ["Train"],
    "Bus/Tram/Metro": ["Bus", "Tram", "Subway", "Coach"],
    "Other":          ["Moped", "Engine", "Taxi/Taxi van", "Skates/inline skates/step",
                        "Disabled transport vehicle with motor",
                        "Disabled transport vehicle without engine",
                        "Otherwise without engine", "Different with engine",
                        "Agricultural vehicle", "Boat"],
}


def _group_mode_shares(mode_series, groups: dict[str, list[str]] = MODE_GROUPS) -> dict[str, float]:
    """% share of each group in `groups` for one list/Series of raw mode strings."""
    total = len(mode_series)
    if total == 0:
        return {grp: 0.0 for grp in groups}
    return {
        grp: sum(1 for m in mode_series if m in modes) / total * 100
        for grp, modes in groups.items()
    }


def plot_mode_share(
    odin_raw_df: pd.DataFrame,
    runs: list[pd.DataFrame] | pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overall mode-share bar chart: ODiN (real, weekday) vs SynTrav (mean over
    seeds, with error bars = std across seeds when len(runs) > 1).

    odin_raw_df : raw ODiN dataframe with 'Main mode of transport travel',
                  'Destination/Purpose', and 'DayType' columns (e.g. odin_train /
                  real, unfiltered).
    runs        : single DataFrame or list of per-seed DataFrames from
                  load_condition_runs() (uses the 'mode_raw' + 'plausible' columns).

    Home-return legs (Destination/Purpose == "Home") are excluded from the ODiN
    side, matching prep_odin/prep_syntrav_run and the fact that SynTrav's return
    legs just copy the preceding outbound leg's mode rather than being
    independently reasoned — counting them on the ODiN side only would compare
    mismatched populations.
    """
    if isinstance(runs, pd.DataFrame):
        runs = [runs]

    if "DayType" in odin_raw_df.columns:
        odin_raw_df = odin_raw_df[odin_raw_df["DayType"] == "weekday"]
    if "Destination/Purpose" in odin_raw_df.columns:
        odin_raw_df = odin_raw_df[odin_raw_df["Destination/Purpose"].astype(str).str.strip() != "Home"]
    real_shares = _group_mode_shares(odin_raw_df["Main mode of transport travel"].dropna().tolist())

    labels = list(MODE_GROUPS.keys())
    per_run_shares = []
    for r in runs:
        plausible = r[r["plausible"]] if "plausible" in r.columns else r
        per_run_shares.append(_group_mode_shares(plausible["mode_raw"].tolist()))

    syn_mean = {l: float(np.mean([s[l] for s in per_run_shares])) for l in labels}
    syn_std  = {l: float(np.std([s[l] for s in per_run_shares]))  for l in labels}

    set_style()
    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_real = ax.bar(x - width / 2, [real_shares[l] for l in labels], width,
                        label="ODiN (real)", color=COLOR_ODIN)
    n_runs    = len(runs)
    syn_label = "SynTrav (mean over seeds)" if n_runs > 1 else "SynTrav"
    bars_syn  = ax.bar(x + width / 2, [syn_mean[l] for l in labels], width,
                        yerr=[syn_std[l] for l in labels] if n_runs > 1 else None,
                        capsize=4, label=syn_label, color=COLOR_SYNTRAV)

    ax.bar_label(bars_real, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars_syn,  fmt="%.1f%%", padding=(12 if n_runs > 1 else 3), fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Share of trips (%)")
    ax.set_title("Mode of Transportation")
    ax.legend()
    ax.set_ylim(0, max(max(real_shares.values()), max(syn_mean.values())) * 1.25)
    strip_axes(ax)

    plt.tight_layout()

    if save_path:
  
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_departure_by_persona: departure-time KDE by persona group 
_PERSONA_GROUPS = [
    {"title": "Employed",             "occ": ["employed"]},
    {"title": "Retired & Homemaker",  "occ": ["retired", "homemaker"]},
    {"title": "Students",             "occ": ["student"]},
]


def _kde_hours(departure_mins: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    hours = np.asarray(departure_mins, dtype=float) / 60.0
    hours = hours[np.isfinite(hours) & (hours >= 0) & (hours <= 24)]
    if len(hours) < 5:
        return np.zeros_like(x_grid)
    return gaussian_kde(hours, bw_method="scott")(x_grid)


def plot_departure_by_persona(
    odin_raw_df: pd.DataFrame,
    runs: list[pd.DataFrame] | pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    odin_raw_df : raw ODiN trip dataframe (e.g. `real_weekday`) with an
                  'Activity_status' column — NOT the prep_odin() output.
    runs        : single DataFrame or list of per-seed DataFrames from
                  load_condition_runs() (must carry the 'occ' column).
    """
    if isinstance(runs, pd.DataFrame):
        runs = [runs]

    x = np.linspace(0, 24, 500)

    panels = []
    for g in _PERSONA_GROUPS:
        odin_sub = prep_odin(odin_raw_df[odin_raw_df["Activity_status"].isin(g["occ"])])
        y_odin   = _kde_hours(odin_sub["departure_min"].values, x)

        run_kdes = [
            _kde_hours(r[r["occ"].isin(g["occ"])]["departure_min"].values, x)
            for r in runs
        ]
        y_llm, y_llm_std = mean_std_runs(run_kdes)
        panels.append((y_odin, y_llm, y_llm_std))

    y_max = max(max(yo.max(), yl.max()) for yo, yl, _ in panels) * 1.15

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, g, (y_odin, y_llm, y_llm_std) in zip(axes, _PERSONA_GROUPS, panels):
        ax.plot(x, y_odin, color=COLOR_ODIN,    lw=LW, ls=LS_ODIN)
        ax.plot(x, y_llm,  color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV)
        ax.fill_between(
            x, np.clip(y_llm - y_llm_std, 0, None), y_llm + y_llm_std,
            color=COLOR_SYNTRAV, alpha=ALPHA_BAND,
        )
        ax.set_xlim(0, 24)
        ax.set_ylim(0, y_max)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["0h", "6h", "12h", "18h", "24h"])
        ax.set_title(g["title"], fontsize=10, pad=6)
        ax.set_xlabel("Hour of day", fontsize=9)
        strip_axes(ax)

    axes[0].set_ylabel("Density", fontsize=9)

    n_runs    = len(runs)
    syn_label = r"SynTrav (LLM, $\pm$1$\sigma$ over seeds)" if n_runs > 1 else "SynTrav (LLM)"
    handles = [
        mlines.Line2D([], [], color=COLOR_ODIN,    lw=LW, ls=LS_ODIN,    label="ODiN (real data)"),
        mlines.Line2D([], [], color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV, label=syn_label),
    ]
    axes[2].legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    if save_path:
  
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_dist_heatmap: trip distance by purpose x occupation 
_WARM_COLORS = ["#FAF0DC", "#D4A96A", "#A06830", "#6B3818", "#2D1000"]
_CMAP = mcolors.LinearSegmentedColormap.from_list("warm_copper", _WARM_COLORS)
_NORM = mcolors.Normalize(vmin=0.0, vmax=0.75)


def _row_norm_matrix(df: pd.DataFrame, purpose: str) -> np.ndarray:
    sub = df[df["purpose_state"] == purpose]
    mat = np.zeros((len(OCC_LEVELS), len(DIST_LABELS_KM)))
    for i, occ in enumerate(OCC_LEVELS):
        occ_sub = sub[sub["occ"] == occ]
        total   = len(occ_sub)
        if total == 0:
            continue
        for j, dl in enumerate(DIST_LABELS_KM):
            mat[i, j] = len(occ_sub[occ_sub["dist_group"] == dl]) / total
    return mat


def _draw_heatmap_panel(ax: plt.Axes, mat: np.ndarray, title: str) -> None:
    n_occ, n_dist = len(OCC_LEVELS), len(DIST_LABELS_KM)
    ax.imshow(mat, cmap=_CMAP, norm=_NORM, aspect="auto", interpolation="none")

    for i in range(n_occ):
        for j in range(n_dist):
            val = mat[i, j]
            if val >= 0.05:
                rgba       = _CMAP(_NORM(val))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color  = "white" if brightness < 0.50 else "#2a1000"
                ax.text(j, i, f"{val * 100:.0f}%", ha="center", va="center",
                        fontsize=5.8, color=txt_color, fontweight="bold")

    ax.set_xticks(range(n_dist))
    ax.set_xticklabels(DIST_LABELS_KM, fontsize=6.5, rotation=30, ha="right")
    ax.set_yticks(range(n_occ))
    ax.set_yticklabels([o.capitalize() for o in OCC_LEVELS], fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _draw_legend_panel(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(0.50, 0.92, "Share within occupation", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, fontweight="bold")

    cbar_ax = ax.inset_axes([0.08, 0.58, 0.84, 0.22])
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=_NORM)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75])
    cbar.set_ticklabels(["0%", "25%", "50%", ">=75%"])
    cbar.ax.tick_params(labelsize=8, length=3)
    cbar.outline.set_linewidth(0.6)

    ax.text(0.50, 0.10,
            "- Each row sums to 100% within occupation x purpose.\n"
            "- Services reflects smaller trip samples.",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color="#666666", style="italic")


def plot_dist_heatmap(
    runs: list[pd.DataFrame] | pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """3x3 grid (8 purpose panels + legend); cells = mean row-normalised share across seeds."""
    if isinstance(runs, pd.DataFrame):
        runs = [runs]

    matrices = []
    for purpose in TOP_PURPOSES:
        per_run = [_row_norm_matrix(r, purpose) for r in runs]
        matrices.append(np.mean(per_run, axis=0))

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                          "figure.dpi": 150, "pdf.fonttype": 42})

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes_flat = axes.flatten()

    for ax, purpose, mat in zip(axes_flat[:8], TOP_PURPOSES, matrices):
        _draw_heatmap_panel(ax, mat, PURPOSE_SHORT.get(purpose, purpose))
    _draw_legend_panel(axes_flat[8])

    n_runs = len(runs)
    suffix = f" (mean over {n_runs} seeds)" if n_runs > 1 else ""
    fig.suptitle("Trip distance distribution by purpose & occupation" + suffix,
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=2.5, w_pad=1.8)

    if save_path:

        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── plot_activity_intention: spatial destination map, per persona group ───────
# Needs geo-allocated trips (destination_lat/lon/poi_label) — only the Rotterdam
# POI-allocation output has these, not the plain Json_files/variance/*.json runs.
ROTTERDAM_GEO_DIR = Path("Portugal_POI_data/NT_poi")
ROTTERDAM_GEO_TEMPLATE = "trajectories_weekday_rotterdam_geo_pois_big_seed{seed}.json"

_EMOJI_FONT = FontProperties(fname=r"C:\Windows\Fonts\seguiemj.ttf")

_ROTTERDAM_LAT_MIN, _ROTTERDAM_LAT_MAX = 51.86, 51.97
_ROTTERDAM_LON_MIN, _ROTTERDAM_LON_MAX = 4.35, 4.60

# OSM tag type -> (emoji, category) — covers all values in nl_rotterdam_pois
_POI_EXPLICIT: dict[str, tuple[str, str]] = {
    "school": ("\U0001F393", "School"), "university": ("\U0001F393", "School"),
    "college": ("\U0001F393", "School"), "kindergarten": ("\U0001F393", "School"),
    "research_institute": ("\U0001F393", "School"), "educational_institution": ("\U0001F393", "School"),
    "cafe": ("☕", "Café"), "restaurant": ("☕", "Café"),
    "bar": ("☕", "Café"), "fast_food": ("☕", "Café"),
    "pharmacy": ("\U0001F3E5", "Health"), "hospital": ("\U0001F3E5", "Health"),
    "clinic": ("\U0001F3E5", "Health"), "social_facility": ("\U0001F3E5", "Health"),
    "social_centre": ("\U0001F3E5", "Health"),
    "bank": ("\U0001F3E2", "Office"), "office": ("\U0001F3E2", "Office"),
    "post_office": ("\U0001F3E2", "Office"), "archive": ("\U0001F3E2", "Office"),
    "place_of_worship": ("\U0001F4CD", "Other"), "fuel": ("\U0001F4CD", "Other"),
    "bus_station": ("\U0001F4CD", "Other"), "station": ("\U0001F4CD", "Other"),
    "ferry_terminal": ("\U0001F4CD", "Other"),
    "supermarket": ("\U0001F6D2", "Market"), "convenience": ("\U0001F6D2", "Market"),
    "mall": ("\U0001F6D2", "Market"), "department_store": ("\U0001F6D2", "Market"),
    "bakery": ("\U0001F6D2", "Market"), "clothes": ("\U0001F6D2", "Market"),
    "deli": ("\U0001F6D2", "Market"), "chocolate": ("\U0001F6D2", "Market"),
    "confectionery": ("\U0001F6D2", "Market"), "florist": ("\U0001F6D2", "Market"),
    "jewelry": ("\U0001F6D2", "Market"), "newsagent": ("\U0001F6D2", "Market"),
    "tea": ("\U0001F6D2", "Market"),
    "park": ("\U0001F333", "Park"), "garden": ("\U0001F333", "Park"),
    "pitch": ("⚽", "Sports"), "sports_centre": ("⚽", "Sports"),
    "fitness_centre": ("⚽", "Sports"), "stadium": ("⚽", "Sports"),
    "company": ("\U0001F3E2", "Office"), "consulting": ("\U0001F3E2", "Office"),
    "lawyer": ("\U0001F3E2", "Office"), "accountant": ("\U0001F3E2", "Office"),
    "administration": ("\U0001F3E2", "Office"), "advertising_agency": ("\U0001F3E2", "Office"),
    "architect": ("\U0001F3E2", "Office"), "association": ("\U0001F3E2", "Office"),
    "cooperative": ("\U0001F3E2", "Office"), "coworking": ("\U0001F3E2", "Office"),
    "design": ("\U0001F3E2", "Office"), "diplomatic": ("\U0001F3E2", "Office"),
    "employment_agency": ("\U0001F3E2", "Office"), "engineer": ("\U0001F3E2", "Office"),
    "engineering": ("\U0001F3E2", "Office"), "estate_agent": ("\U0001F3E2", "Office"),
    "financial": ("\U0001F3E2", "Office"), "financial_advisor": ("\U0001F3E2", "Office"),
    "foundation": ("\U0001F3E2", "Office"), "government": ("\U0001F3E2", "Office"),
    "graphic_design": ("\U0001F3E2", "Office"), "insurance": ("\U0001F3E2", "Office"),
    "it": ("\U0001F3E2", "Office"),
    "attraction": ("\U0001F333", "Park"), "viewpoint": ("\U0001F333", "Park"),
    "commercial": ("\U0001F3E2", "Office"), "industrial": ("\U0001F3E2", "Office"),
    "education": ("\U0001F393", "School"), "forest": ("\U0001F333", "Park"),
    "grass": ("\U0001F333", "Park"), "recreation_ground": ("\U0001F333", "Park"),
    "retail": ("\U0001F6D2", "Market"),
}

_POI_KEYWORD_RULES: list[tuple[list[str], tuple[str, str]]] = [
    (["school", "college", "universit", "hogeschool", "lyceum", "mavo",
      "waldorf", "montessori", "jenaplan", "basisschool", "zadkine", "albeda"],
     ("\U0001F393", "School")),
    (["albert heijn", "jumbo", "lidl", "aldi", "spar", "hoogvliet", "dirk",
      "plus ", "supermarkt", "markt", "winkel", "toko", "buurtwinkel",
      "minimarket", "ekoplaza", "hema"],
     ("\U0001F6D2", "Market")),
    (["park", "bos ", "rosarium", "kralingse bos"],
     ("\U0001F333", "Park")),
    (["cafe", "café", "restaurant", "bar ", "coffee", "starbucks", "verhage",
      "eetcaf", "lunch", "café"],
     ("☕", "Café")),
    (["sport", "gym", "hockey", "fitness", "tennis", "judo", "sportcity",
      "sportfonds", "cruyff"],
     ("⚽", "Sports")),
    (["b.v.", " bv", "group", "services", "cargo", "shipping", "logistics",
      "consulting", "advocaten", "consulaat", "makelaardij", "bedrijven"],
     ("\U0001F3E2", "Office")),
]

_POI_SKIP_LABELS = {
    "home", "residential", "residential_centroid",
    "zone_centroid_fallback", "unknown",
}


def _classify_poi(label: str | None) -> tuple[str, str] | None:
    if not label or label in _POI_SKIP_LABELS:
        return None
    if label in _POI_EXPLICIT:
        return _POI_EXPLICIT[label]
    ll = label.lower()
    for keywords, cat in _POI_KEYWORD_RULES:
        if any(k in ll for k in keywords):
            return cat
    return ("\U0001F4CD", "Other")


def _to_merc(lons, lats):
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return t.transform(np.asarray(lons, float), np.asarray(lats, float))


def _rotterdam_bbox_merc():
    (xmin,), (ymin,) = _to_merc([_ROTTERDAM_LON_MIN], [_ROTTERDAM_LAT_MIN])
    (xmax,), (ymax,) = _to_merc([_ROTTERDAM_LON_MAX], [_ROTTERDAM_LAT_MAX])
    return xmin, xmax, ymin, ymax


def _kde_grid(x, y, xmin, xmax, ymin, ymax, res=300, bw=0.25):
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    xx, yy = np.meshgrid(xi, yi)
    if len(x) < 3:
        return np.zeros((res, res))
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(res, res)
    lo, hi = z.min(), z.max()
    return (z - lo) / (hi - lo + 1e-12)


def load_rotterdam_geo_records(seeds: list[int] = SEEDS) -> list[dict]:
    """
    Pool geo-allocated Rotterdam trip records across every available seed
    (concatenation, not mean±std — more points just makes the spatial KDE/
    density map denser and more robust, unlike the JSD-style panels elsewhere
    in this module). Missing seeds are skipped with a warning.
    """
    records: list[dict] = []
    for seed in seeds:
        path = ROTTERDAM_GEO_DIR / ROTTERDAM_GEO_TEMPLATE.format(seed=seed)
        if not path.exists():
            print(f"[fig_gallery] missing {path}, skipping seed {seed}")
            continue
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))

    if not records:
        raise FileNotFoundError(
            f"No Rotterdam geo-allocation files found for seeds {seeds} in {ROTTERDAM_GEO_DIR}"
        )
    return records


def _load_activity_group(records: list[dict], group_key) -> pd.DataFrame:
    keys = {group_key} if isinstance(group_key, str) else set(group_key)
    rows = []
    for rec in records:
        if rec.get("group_key") not in keys:
            continue
        for t in rec.get("trips", []):
            lat = t.get("destination_lat")
            lon = t.get("destination_lon")
            lbl = t.get("destination_poi_label", "")
            if lat is None or lon is None:
                continue
            cat = _classify_poi(lbl)
            if cat is None:
                continue
            rows.append(dict(lat=float(lat), lon=float(lon),
                              poi_label=lbl, emoji=cat[0], cat_name=cat[1]))
    if not rows:
        return pd.DataFrame(columns=["lat", "lon", "poi_label", "emoji", "cat_name",
                                      "x_merc", "y_merc"])
    df = pd.DataFrame(rows)
    df["x_merc"], df["y_merc"] = _to_merc(df["lon"].values, df["lat"].values)
    return df


def _render_activity_panel(ax, df, title, letter, xmin, xmax, ymin, ymax, kde_res=300):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom="auto", attribution=False)
    except Exception:
        ax.set_facecolor("#dce8d4")

    if len(df) >= 3:
        z = _kde_grid(df["x_merc"].values, df["y_merc"].values,
                      xmin, xmax, ymin, ymax, kde_res, bw=0.25)
        ax.imshow(z, extent=[xmin, xmax, ymin, ymax], origin="lower",
                  cmap="RdYlGn_r", alpha=0.30, vmin=0, vmax=1,
                  aspect="auto", zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    total = max(len(df), 1)
    cat_groups = (
        df.groupby(["emoji", "cat_name"])
        .agg(count=("lat", "size"), mean_x=("x_merc", "mean"), mean_y=("y_merc", "mean"))
        .reset_index().sort_values("count", ascending=False).head(4)
    )

    legend_lines = []
    for _, row in cat_groups.iterrows():
        pct = row["count"] / total * 100
        emoji, cat = row["emoji"], row["cat_name"]
        ax.text(row["mean_x"] + 200, row["mean_y"] - 200, emoji,
                fontsize=22, ha="center", va="center",
                color="black", alpha=0.2, zorder=5, fontproperties=_EMOJI_FONT)
        ax.text(row["mean_x"], row["mean_y"], emoji,
                fontsize=22, ha="center", va="center",
                zorder=6, fontproperties=_EMOJI_FONT)
        legend_lines.append((emoji, cat, pct))

    n_lines = len(legend_lines)
    line_h  = 0.068
    pad     = 0.018
    box_w   = 0.30
    box_h   = n_lines * line_h + 2 * pad
    lx = 0.97
    ly = 0.02

    ax.add_patch(mpatches.FancyBboxPatch(
        (lx - box_w, ly), box_w, box_h,
        transform=ax.transAxes,
        boxstyle="square,pad=0.0",
        fc="white", ec="#999999", lw=0.7, alpha=0.92, zorder=7,
    ))
    for i, (emoji, cat, pct) in enumerate(reversed(legend_lines)):
        y_pos = ly + pad + i * line_h + line_h * 0.35
        ax.text(lx - box_w + 0.012, y_pos,
                f"{emoji} {cat}  {pct:.0f}%",
                transform=ax.transAxes, fontsize=7.8,
                va="center", ha="left", zorder=9,
                fontproperties=_EMOJI_FONT)

    ax.text(0.015, 0.975, f"({letter})",
            transform=ax.transAxes, fontsize=12, fontfamily="serif",
            va="top", ha="left", color="black", zorder=10,
            bbox=dict(fc="white", alpha=0.6, ec="none", pad=1))
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_activity_intention(
    records: list[dict],
    group_keys: list,
    panel_titles: list[str] | None = None,
    fig_title: str = "Activity intention",
    save_path: str | None = None,
    kde_res: int = 300,
) -> plt.Figure:
    """
    1xN spatial comparison figure: OSM basemap + KDE density + emoji POI-category
    markers, one panel per entry in `group_keys`.

    Parameters
    ----------
    records      : pooled geo-allocated records, e.g. from load_rotterdam_geo_records()
    group_keys   : one entry per panel; each entry is a group_key string, or a
                   list[str] of group_keys pooled together (e.g. all income bands)
    panel_titles : display title per panel (defaults to the group_key itself)
    fig_title    : suptitle
    save_path    : saves <path>.png at 300 dpi
    kde_res      : KDE grid resolution (default 300)
    """
    n = len(group_keys)
    if panel_titles is None:
        panel_titles = [(gk if isinstance(gk, str) else gk[0]) for gk in group_keys]
    letters = "abcde"[:n]

    xmin, xmax, ymin, ymax = _rotterdam_bbox_merc()

    fig, axes = plt.subplots(
        1, n,
        figsize=(7.2 * n + 1.0, 7.4),
        gridspec_kw={"wspace": 0.05},
    )
    axes_flat = list(axes) if n > 1 else [axes]

    for ax, gkey, title, letter in zip(axes_flat, group_keys, panel_titles, letters):
        df = _load_activity_group(records, gkey)
        _render_activity_panel(ax, df, title, letter, xmin, xmax, ymin, ymax, kde_res)

    fig.subplots_adjust(left=0.01, right=0.90, top=0.91, bottom=0.04)
    cbar_ax = fig.add_axes([0.92, 0.08, 0.013, 0.78])
    sm = ScalarMappable(cmap="RdYlGn_r", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=8.5)
    cbar.set_label("Visit density (normalised)", fontsize=9, labelpad=7)
    cbar.outline.set_visible(False)

    fig.suptitle(fig_title, fontsize=11.5, y=0.97, fontweight="bold")
    fig.text(0.46, 0.005,
             "Weekday activity destinations · Rotterdam · all day",
             ha="center", va="bottom", fontsize=8.5, style="italic")

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig

def plot_distance_pair(odin_df, *run_label_pairs, save_path=None):
    """Distance KDE for two or more SynTrav conditions side by side, each vs ODiN.

    Accepts either the legacy call signature (runs_a, runs_b, label_a, label_b)
    or any number of (runs, label) pairs, e.g.
    plot_distance_pair(odin_df, runs_full, "Full", runs_mode, "Mode",
                        runs_distance, "Distance").
    """
    set_style()

    # Back-compat: (runs_a, runs_b, label_a, label_b) positional form.
    if len(run_label_pairs) == 4 and isinstance(run_label_pairs[2], str):
        runs_a, runs_b, label_a, label_b = run_label_pairs
        conditions = [(runs_a, label_a), (runs_b, label_b)]
    else:
        if len(run_label_pairs) % 2 != 0:
            raise ValueError("Expected pairs of (runs, label) arguments.")
        conditions = list(zip(run_label_pairs[0::2], run_label_pairs[1::2]))

    n = len(conditions)
    x_max = 60
    x = np.linspace(0.05, x_max, 400)
    y_odin = kde_curve(odin_df["distance_km"].dropna().clip(upper=x_max), x)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    axes = list(axes) if n > 1 else [axes]

    for ax, (runs, label) in zip(axes, conditions):
        y, y_std = mean_std_runs(
            [kde_curve(r["distance_km"].dropna().clip(upper=x_max), x) for r in runs]
        )
        ax.plot(x, y_odin, color=COLOR_ODIN, lw=LW, ls=LS_ODIN, label="ODiN")
        ax.plot(x, y, color=COLOR_SYNTRAV, label=label)
        ax.fill_between(x, np.clip(y - y_std, 0, None), y + y_std,
                         color=COLOR_SYNTRAV, alpha=ALPHA_BAND)
        ax.set_xlabel("Distance (km)")
        ax.set_xlim(0, x_max)
        ax.set_title(label)
        strip_axes(ax)

    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=9)

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")


