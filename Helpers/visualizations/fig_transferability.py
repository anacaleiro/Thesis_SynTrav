"""
fig_transferability.py

Figures for the internal/external transferability analysis (Notebook 6),
following the same one-function-per-figure convention as fig_gallery.py.

    compute_province_ranking     results_all (SD/SI/DARD/DailyLoc per province) -> composite score
    plot_province_ranking_map    choropleth of that composite score

    compute_mode_shift_data      nl_records/pt_records -> modes/nl/pt/imob lists + nl_ms/pt_ms dicts
    plot_mode_shift              NL vs PT vs IMOB 2017 mode-share dumbbell chart
    build_mode_shift_table       NL/PT/IMOB per-mode shift table (not saved as a figure)

    load_pt_ablation_records     Json_files/pt_ablations/*.json -> {condition: records}
    compute_ablation_mode_shares {condition: records} -> mode-share (%) DataFrame
    rank_ablation_drivers        mode-share DataFrame -> |shift| vs full, ranked (which context element matters most)
    plot_ablation_mode_shares    grouped bar chart, mode share by ablation condition
    plot_combined_nl_pt          5-panel KDE/ECDF headline figure, NL synthetic vs PT synthetic
    plot_departure_by_persona    re-exported from fig_gallery (ODiN vs SynTrav, by persona group)
    plot_mode_share_by_purpose   mode-share-by-purpose comparison, NL vs PT synthetic
    plot_activity_intention_pt   spatial destination map, Oeiras (PT) instead of Rotterdam

    build_pt_target_table        nl_records/pt_records -> 12-row IMOB 2017 fit table (mode/purpose share, mean distance by purpose, trip volume)
    compute_pt_fit_score         build_pt_target_table() -> single mean |%error| number
    plot_pt_target_fit           horizontal divergence chart of that table (%error per target, 0 = perfect fit)

Every figure below is saved as a PNG only (no PDF), under figures/05_transferability.

Usage from the notebook
------------------------
    from Helpers.visualizations.fig_transferability import (
        FIG_DIR, compute_province_ranking, plot_province_ranking_map,
        IMOB_2017, compute_mode_shift_data, plot_mode_shift, build_mode_shift_table,
        plot_combined_nl_pt, plot_departure_by_persona, plot_mode_share_by_purpose,
        load_oeiras_geo_records, plot_activity_intention_pt,
        load_pt_ablation_records, compute_ablation_mode_shares,
        rank_ablation_drivers, plot_ablation_mode_shares,
        build_pt_target_table, compute_pt_fit_score, plot_pt_target_fit,
    )

    province_rank = compute_province_ranking(results_all)
    plot_province_ranking_map(province_rank, save_path=str(FIG_DIR / "fig_province_ranking_map"))

    modes, nl_vals, pt_vals, imob_vals, nl_ms, pt_ms = compute_mode_shift_data(nl_records, pt_records)
    plot_mode_shift(modes, nl_vals, pt_vals, imob_vals,
                     save_path=str(FIG_DIR / "fig_mode_shift_nl_pt_imob"))
    df_shift = build_mode_shift_table(nl_ms, pt_ms, IMOB_2017)

    plot_combined_nl_pt(nl_runs, pt_runs, save_path=str(FIG_DIR / "fig_combined_nl_pt"))
    plot_departure_by_persona(real_weekday, runs_mode, save_path=str(FIG_DIR / "fig_departure_mode"))
    plot_mode_share_by_purpose(nl_records, pt_records, save_path=str(FIG_DIR / "fig_mode_share_by_purpose"))

    oeiras_records = load_oeiras_geo_records()
    plot_activity_intention_pt(
        oeiras_records,
        group_keys=[[f"student | 18-25 | weekday | Income={inc}" for inc in INCOMES]],
        panel_titles=["Student (18-25)"],
        fig_title="Activity destinations on a weekday by demographic profile - Oeiras, PT",
        save_path=str(FIG_DIR / "fig_activity_groups_pt"),
    )

    ablation_records = load_pt_ablation_records()
    ablation_shares = compute_ablation_mode_shares(ablation_records)
    print(rank_ablation_drivers(ablation_shares))   # which context element matters most
    plot_ablation_mode_shares(ablation_shares,
                               save_path=str(FIG_DIR / "fig_pt_context_ablation"))

    target_table = build_pt_target_table(nl_records, pt_records)
    print(f"PT fit score (mean |%error| across 12 IMOB targets): {compute_pt_fit_score(target_table):.1f}%")
    plot_pt_target_fit(target_table, save_path=str(FIG_DIR / "fig_pt_target_fit"))
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import contextily as ctx

from Helpers.evaluation import load_syn_records

from . import fig_gallery as _fig_gallery
from .fig_gallery import (
    _ecdf, _trips_per_person, _unique_purposes_per_person,
    plot_departure_by_persona,
)
from .viz_utils import (
    set_style, strip_axes, add_caption,
    kde_curve, mean_std_runs, extract_si_gaps,
)


#  Output location
FIG_DIR = Path("figures/05_transferability")
FIG_DIR.mkdir(parents=True, exist_ok=True)


#  Shared color palette
ORANGE     = "#E8772E"   # NL synthetic
GREEN      = "#2D6A4F"   # PT synthetic
GREY       = "#000000"   # IMOB 2017 target
LIGHT_GREY = "#D3D1C7"
TEXT       = "#2C2C2A"
MUTED      = "#5F5E5A"


# ── MAP FIGURE: Province ranking choropleth ───────────────────────────────────
#  Province geometry source
PROVINCES_GEOJSON_URL = "https://cartomap.github.io/nl/wgs84/provincie_2022.geojson"

# CBS/cartomap Dutch names -> the English names used in results_all's "label" index
NAME_MAP = {
    "Noord-Holland": "North Holland",
    "Zuid-Holland":  "South-Holland",
    "Noord-Brabant": "North Brabant",
    "Zeeland":       "Zealand",
    "Fryslân":       "Friesland",
}

# Non-province rows that can show up in results_all (national aggregate)
_NATIONAL_LABELS = {"Full NL", "Full NL holdout"}

_PROVINCES_GDF_CACHE: gpd.GeoDataFrame | None = None


def _load_provinces_gdf(geojson_url: str = PROVINCES_GEOJSON_URL,
                         name_map: dict[str, str] = NAME_MAP) -> gpd.GeoDataFrame:
    """Fetch + cache the province geometries so repeat plot calls don't re-download."""
    global _PROVINCES_GDF_CACHE
    if _PROVINCES_GDF_CACHE is not None:
        return _PROVINCES_GDF_CACHE

    gdf = gpd.read_file(geojson_url)[["statnaam", "geometry"]].copy()
    gdf["province"] = gdf["statnaam"].map(name_map).fillna(gdf["statnaam"])

    _PROVINCES_GDF_CACHE = gdf
    return gdf


def compute_province_ranking(
    results_all: pd.DataFrame,
    metrics: list[str] = ["SD", "SI", "DARD", "DailyLoc"],
    exclude_labels: set[str] = _NATIONAL_LABELS,
) -> pd.DataFrame:
    """
    Composite province ranking from results_all (indexed by "label", one row
    per province + a national aggregate row).

    All four metrics are divergence/error scores (lower = closer match to real
    ODiN data), so per metric each province is ranked ascending (rank 1 = best).
    The 4 ranks are averaged per province, then min-max scaled across provinces
    so the best-performing province scores 1.0 and the worst scores 0.0.

    Returns a DataFrame with columns ["province", "score"], ready to merge onto
    the province geometries in plot_province_ranking_map().
    """
    provinces = results_all[~results_all.index.isin(exclude_labels)][metrics]

    ranks     = provinces.rank(method="average", ascending=True)
    avg_rank  = ranks.mean(axis=1)

    lo, hi = avg_rank.min(), avg_rank.max()
    score  = (hi - avg_rank) / (hi - lo) if hi > lo else avg_rank * 0.0 + 1.0

    return pd.DataFrame({
        "province": provinces.index,
        "score":    score.values,
    })


def plot_province_ranking_map(
    province_rank: pd.DataFrame,
    score_col: str = "score",
    title: str = "Overall Province Ranking Score",
    cmap: str = "YlOrRd",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Choropleth of `province_rank` (columns: "province", `score_col`) over the
    Dutch provinces, with a CartoDB basemap and per-province score labels.

    province_rank : DataFrame with one row per province and a numeric score
                    column, e.g. from compute_province_ranking(results_all).
    save_path     : if given, saves "<save_path>.png" at 300 dpi (PNG only).
    """
    set_style()

    provinces_gdf = _load_provinces_gdf()
    merged = provinces_gdf.merge(province_rank, on="province", how="left").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(7, 9))
    fig.patch.set_facecolor("white")

    merged.plot(
        column=score_col,
        ax=ax,
        cmap=cmap,
        alpha=0.75,
        legend=True,
        legend_kwds={
            "orientation": "vertical",
            "shrink": 0.5,
            "pad": 0.02,
            "fraction": 0.04,
            "label": "Composite score",
        },
        edgecolor="white",
        linewidth=0.8,
        missing_kwds={"color": "lightgrey"},
        vmin=0, vmax=1,
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    for _, row in merged.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.text(cx, cy, row["province"], ha="center", va="center",
                fontsize=7.5, color="black", weight="bold")

    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=12, color="black")
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── MODE SHIFT: NL vs PT vs IMOB 2017 (dumbbell chart + table) ────────────────
# IMOB 2017 is an external Portuguese mobility-survey benchmark, not derived
# from any SynTrav output — kept as the one hardcoded constant here.
IMOB_2017 = {
    "Passenger car":        58.9,
    "On foot":              23.0,
    "Public transport":     15.8,   # Train + Bus combined
    "Non-electric bicycle": 0.5,
}

DEFAULT_MODE_SHIFT_MODES = ["Passenger car", "On foot", "Public transport", "Non-electric bicycle"]

# Display labels for plot_mode_shift's y-axis (internal keys above must stay
# unchanged since they're used to look up nl_ms/pt_ms/IMOB_2017 values).
MODE_DISPLAY_LABELS = {
    "Passenger car":        "Car",
    "On foot":              "Walking",
    "Public transport":     "Public transport",
    "Non-electric bicycle": "Bicycle",
}


def compute_grouped_mode_share(records: list[dict]) -> dict[str, float]:
    """
    Fraction of trips (0-1) per mode bucket, for one SynTrav record list.
    Train + Bus are grouped into "Public transport" to match IMOB_2017's
    categories; all other raw mode labels are dropped except the ones below.
    """
    counts = collections.Counter()
    for r in records:
        for t in r.get("trips", []):
            m = t.get("mode")
            if m:
                counts[m] += 1

    total = sum(counts.values())
    raw = {k: v / total for k, v in counts.items()} if total else {}

    return {
        "Passenger car":        raw.get("Passenger car", 0.0),
        "On foot":              raw.get("On foot", 0.0),
        "Public transport":     raw.get("Train", 0.0) + raw.get("Bus", 0.0),
        "Non-electric bicycle": raw.get("Non-electric bicycle", 0.0),
    }


def compute_mode_shift_data(
    nl_records: list[dict],
    pt_records: list[dict],
    imob_targets: dict[str, float] = IMOB_2017,
    modes: list[str] = DEFAULT_MODE_SHIFT_MODES,
) -> tuple[list[str], list[float], list[float], list[float], dict[str, float], dict[str, float]]:
    """
    Build every input plot_mode_shift() / build_mode_shift_table() need,
    straight from the raw NL/PT record lists (load_syn_records() output).

    Returns (modes, nl_vals, pt_vals, imob_vals, nl_ms, pt_ms):
      - modes/nl_vals/pt_vals/imob_vals : parallel lists, percentages 0-100,
                                          ready for plot_mode_shift().
      - nl_ms/pt_ms                     : mode -> fraction (0-1) dicts, ready
                                          for build_mode_shift_table().
    """
    nl_ms = compute_grouped_mode_share(nl_records)
    pt_ms = compute_grouped_mode_share(pt_records)

    nl_vals   = [round(nl_ms.get(m, 0.0) * 100, 1) for m in modes]
    pt_vals   = [round(pt_ms.get(m, 0.0) * 100, 1) for m in modes]
    imob_vals = [imob_targets.get(m, 0.0) for m in modes]

    return modes, nl_vals, pt_vals, imob_vals, nl_ms, pt_ms


def plot_mode_shift(
    modes: list[str],
    nl: list[float],
    pt: list[float],
    imob: list[float],
    title: str = "PT vs NL Mode Behaviour Shift",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Dumbbell chart: NL synthetic (orange circle) -> PT synthetic (green
    triangle) per mode, connected by a grey line, with the IMOB 2017 target
    marked as a black dashed tick.

    modes, nl, pt, imob : parallel sequences, one entry per travel mode.
                          nl / pt / imob are all percentages (0-100), not
                          fractions.
    save_path           : if given, saves "<save_path>.png" at 300 dpi.
    """
    set_style()
    y_pos = np.arange(len(modes))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Connector lines NL -> PT
    for i in range(len(modes)):
        ax.plot([nl[i], pt[i]], [y_pos[i], y_pos[i]],
                color=LIGHT_GREY, linewidth=2, zorder=1, solid_capstyle="round")

    # IMOB target - vertical tick
    for i in range(len(modes)):
        ax.plot([imob[i], imob[i]], [y_pos[i] - 0.22, y_pos[i] + 0.22],
                color=GREY, linewidth=2, linestyle="--", zorder=2)

    # NL circles
    ax.scatter(nl, y_pos, s=160, color=ORANGE, zorder=3, marker="o", linewidths=0)

    # PT triangles
    ax.scatter(pt, y_pos, s=280, color=GREEN, zorder=3, marker="^", linewidths=0)

    # Annotations: NL values
    for i, v in enumerate(nl):
        offset = -1.8 if nl[i] < pt[i] else 1.8
        ax.text(v + offset, y_pos[i] + 0.28, f"{v}%",
                ha="center", va="bottom", fontsize=9, color=ORANGE, fontweight="bold")

    # Annotations: PT values
    for i, v in enumerate(pt):
        offset = 1.8 if pt[i] > nl[i] else -1.8
        ax.text(v + offset, y_pos[i] + 0.28, f"{v}%",
                ha="center", va="bottom", fontsize=9, color=GREEN, fontweight="bold")

    # Axes formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODE_DISPLAY_LABELS.get(m, m) for m in modes], fontsize=11, color=TEXT)
    ax.set_xlim(-7, 75)
    ax.set_ylim(-0.9, len(modes) - 0.1)
    ax.set_xlabel("Behaviour shift (%)", fontsize=11, color=MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.tick_params(axis="x", colors=MUTED, labelsize=10)
    ax.tick_params(axis="y", length=0, pad=10)

    # Grid
    ax.set_xticks(np.arange(0, 75, 10))
    ax.xaxis.grid(True, color=LIGHT_GREY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(GREY)
        spine.set_linewidth(1.2)

    # Legend - top right outside, with correct markers
    nl_handle   = plt.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=ORANGE, markersize=9,
                             label="NL synthetic")
    pt_handle   = plt.Line2D([0], [0], marker="^", color="w",
                             markerfacecolor=GREEN, markersize=11,
                             label="PT synthetic")
    imob_handle = plt.Line2D([0], [0], color=GREY, linewidth=2,
                             linestyle="--", label="IMOB 2017 target")
    ax.legend(handles=[nl_handle, pt_handle, imob_handle],
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=10, frameon=False, labelcolor=MUTED)

    ax.set_title(title, fontsize=13, color=TEXT, pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def build_mode_shift_table(
    nl_ms: dict[str, float],
    pt_ms: dict[str, float],
    imob: dict[str, float],
    modes: list[str] = ["Passenger car", "On foot", "Public transport", "Non-electric bicycle"],
) -> pd.DataFrame:
    """
    Per-mode shift table: NL synthetic vs PT synthetic (both mode-share
    fractions in [0, 1]) against the IMOB 2017 target (already in percent).

    "Direction" is a check mark when the NL -> PT shift moves the same way as
    the gap between NL and the IMOB target (i.e. PT moved towards IMOB), a
    cross when it overshoots or moves the wrong way.
    """
    rows = []
    for m in modes:
        shift = pt_ms[m] * 100 - nl_ms[m] * 100
        rows.append({
            "Mode":         m,
            "NL synthetic": f"{nl_ms[m]*100:.1f}%",
            "PT synthetic": f"{pt_ms[m]*100:.1f}%",
            "Shift":        f"{shift:+.1f}pp",
            "IMOB target":  f"{imob[m]:.1f}%",
            "Direction":    "✓" if (shift > 0) == (imob[m] - nl_ms[m] * 100 > 0) else "✗",
        })
    return pd.DataFrame(rows).set_index("Mode")


# ── PT CONTEXT ABLATION: which injected prompt element drives the shift? ──────
# The PT "full" prompt injects Oeiras-specific commuting statistics (a fact
# block) plus a cycling treatment that varies in strength — see
# prompt_template/portugal_context.py::_CONDITION_SPECS for the source of truth:
#
#   condition            commuting stats   cycling treatment
#   full                      yes          soft    (descriptive: "not a viable mode")
#   no_commuting                no          soft
#   no_cycling                 yes          omit    (no mention at all)
#   minimal                     no          omit
#   strict_no_cycling          yes          strict  (soft line + explicit directive:
#                                                     "Do not generate any cycling trips")
#
# Two things worth not getting wrong: "minimal" still carries the modal
# ground truth, distance tables, and mobility narrative (Taguspark/Linha de
# Cascais/A5, walking-threshold guidance) — it is "full minus commuting minus
# cycling mention", not a context-free prompt. And "strict_no_cycling" does
# NOT drop the cycling note; it ADDS an explicit behavioural command on top of
# the same descriptive line "full" already has, so the cycling dimension is
# really a 3-level dose (omit < soft < strict), not a binary present/absent.
PT_ABLATION_DIR = Path("Json_files/pt_ablations")

# "full_small" (226 records), not "full" (564, a separate larger/pooled run),
# is the size-matched full-context baseline for the other ablations.
# Ordered as the "specificity staircase" (bare -> minimal -> full -> strict,
# each step adding more/stronger context) followed by the single-factor
# branch ablations off "full" — plot_ablation_mode_shares() groups the bars
# to match, via _ABLATION_STAIRCASE below.
# The four-step specificity staircase — the default, headline comparison.
# "bare_location"/"minimal" (see prompt_template/portugal_context.py) were
# renamed here to "No context"/"Base context": "minimal" was never actually
# minimal (it still carries modal ground truth, distance tables, the mobility
# narrative, and walking constraints — see load_pt_ablation_records()
# docstring), which read as backwards next to the true no-information
# "bare_location" condition. These display labels are presentation-only; the
# underlying context_condition strings baked into the JSON files are unchanged.
PT_ABLATION_FILES = {
    "No context":     "trajectories_portugal_weekday_bare_location.json",
    "Base context":   "trajectories_portugal_weekday_minimal.json",
    "Full context":   "trajectories_portugal_weekday_full_small.json",
    "Strict context": "trajectories_portugal_weekday_strict.json",
}

# The three single-factor branch ablations off "Full context" — not loaded by
# load_pt_ablation_records() by default; merge these in explicitly (see its
# docstring) when you want the fuller which-element-matters-most comparison.
PT_ABLATION_BRANCH_FILES = {
    "No commuting stats":    "trajectories_portugal_weekday_no_commuting.json",
    "No cycling mention":    "trajectories_portugal_weekday_no_cycling.json",
    "No modal ground truth": "trajectories_portugal_weekday_no_ground_truth.json",
}

# The staircase group, in reading order — any condition in a records dict
# passed to plot_ablation_mode_shares() that ISN'T listed here is treated as
# a branch ablation and grouped separately in the plot (see PT_ABLATION_BRANCH_FILES).
_ABLATION_STAIRCASE = (
    "No context",
    "Base context",
    "Full context",
    "Strict context",
)


def load_pt_ablation_records(
    files: dict[str, str] = PT_ABLATION_FILES,
    base_dir: str | Path = PT_ABLATION_DIR,
) -> dict[str, list[dict]]:
    """
    condition label -> flattened SynTrav records (load_syn_records() output).

    Defaults to the four-step specificity staircase (PT_ABLATION_FILES):
    "No context" (just "this person lives in Oeiras, Portugal", nothing
    else), "Base context" (adds modal ground truth, distance tables, the
    mobility narrative, and walking constraints — but no commuting stats and
    no cycling mention at all), "Full context" (adds commuting stats + a soft
    cycling note), "Strict context" (upgrades the cycling note to an explicit
    "do not generate cycling trips" directive).

    To also see the three single-factor branch ablations off "Full context"
    (which of commuting stats / cycling note / modal ground truth matters
    most), pass files={**PT_ABLATION_FILES, **PT_ABLATION_BRANCH_FILES}.
    """
    return {
        label: load_syn_records(str(Path(base_dir) / fname))
        for label, fname in files.items()
    }


def compute_ablation_mode_shares(
    records_by_condition: dict[str, list[dict]],
    modes: list[str] = DEFAULT_MODE_SHIFT_MODES,
) -> pd.DataFrame:
    """
    One row per condition, one column per mode (display-labelled), values are
    mode share in %. Row order follows `records_by_condition`'s insertion order.
    """
    shares = {
        label: compute_grouped_mode_share(recs)
        for label, recs in records_by_condition.items()
    }
    df = pd.DataFrame(shares).T[modes] * 100
    df.columns = [MODE_DISPLAY_LABELS.get(m, m) for m in modes]
    return df.round(1)


def rank_ablation_drivers(
    mode_share_df: pd.DataFrame,
    full_label: str = "Full context",
) -> pd.Series:
    """
    Sum of |mode-share deviation| (pp) from `full_label`, one value per other
    condition, sorted descending (biggest driver first).

    A large value means dropping that piece of context moved mode share far
    from the full-context run -> that element is doing the most work. A value
    near 0 means removing it barely changed behaviour at all.
    """
    deltas = (mode_share_df.drop(index=full_label) - mode_share_df.loc[full_label]).abs()
    return deltas.sum(axis=1).sort_values(ascending=False).rename("Total |shift| vs full (pp)")


def rank_ablation_realism(
    mode_share_df: pd.DataFrame,
    imob_targets: dict[str, float] = IMOB_2017,
) -> pd.Series:
    """
    Sum of |mode-share deviation| (pp) from the real IMOB 2017 benchmark, one
    value per condition (including "Full context" itself), sorted ascending
    (most realistic condition first).

    Answers a different question than rank_ablation_drivers(): not "which
    context element moves behaviour the most" but "which condition actually
    matches real Portuguese mode share the closest" - the two don't have to
    agree, and when they don't it means the injected context isn't purely
    helping realism.
    """
    imob_by_label = {MODE_DISPLAY_LABELS.get(m, m): v for m, v in imob_targets.items()}
    imob_row = pd.Series({m: imob_by_label[m] for m in mode_share_df.columns})
    deltas = (mode_share_df - imob_row).abs()
    return deltas.sum(axis=1).sort_values(ascending=True).rename("Total |shift| vs IMOB target (pp)")


def plot_ablation_mode_shares(
    mode_share_df: pd.DataFrame,
    imob_targets: dict[str, float] = IMOB_2017,
    title: str = "PT context ablation: mode share by condition",
    save_path: str | None = None,
    staircase: tuple[str, ...] = _ABLATION_STAIRCASE,
) -> plt.Figure:
    """
    Grouped bar chart: one cluster of bars per mode, one bar per ablation
    condition (row of `mode_share_df`), with the IMOB 2017 target marked as a
    black dashed tick per mode.

    Conditions listed in `staircase` are drawn first, in that order, then a
    visual gap, then every other condition (the single-factor branch
    ablations off "full") — so the bare -> minimal -> full -> strict
    specificity progression reads as one group, separate from the branches.
    """
    set_style()

    all_conditions = list(mode_share_df.index)
    conditions = [c for c in staircase if c in all_conditions] + \
                 [c for c in all_conditions if c not in staircase]
    modes = list(mode_share_df.columns)
    imob_by_label = {MODE_DISPLAY_LABELS.get(m, m): v for m, v in imob_targets.items()}

    n_cond = len(conditions)
    x = np.arange(len(modes))
    bar_width = 0.8 / n_cond
    # Same palette as Helpers/persona_scoring_validation.py::_plot_comparison
    # ("LLM Persona-Scoring Validation" chart) — copper colormap sampled
    # evenly across the bars, so the two figures read as one visual family.
    cmap = plt.cm.copper
    colors = [cmap(i / max(n_cond - 1, 1)) for i in range(n_cond)]

    # Slot position per bar: 1 unit apart, with an extra gap where the
    # staircase group ends and the branch-ablation group begins.
    GAP = 0.65
    n_staircase = sum(1 for c in conditions if c in staircase)
    slots, cursor = [], 0.0
    for i in range(n_cond):
        slots.append(cursor)
        cursor += 1.0
        if n_staircase and i == n_staircase - 1:
            cursor += GAP
    slots = np.array(slots)
    slots -= slots.mean()
    cluster_half_width = (slots.max() - slots.min()) / 2 + 0.5
    offsets = slots * bar_width

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, cond in enumerate(conditions):
        vals = mode_share_df.loc[cond].values
        ax.bar(x + offsets[i], vals, width=bar_width * 0.92, color=colors[i], label=cond,
               alpha=0.88, zorder=3)

    for i, m in enumerate(modes):
        target = imob_by_label.get(m)
        if target is not None:
            span = cluster_half_width * bar_width
            ax.plot([x[i] - span, x[i] + span], [target, target],
                    color=GREY, linewidth=2, linestyle="--", zorder=4)

    # Divider between the staircase group and the branch-ablation group, one
    # per mode cluster (thin dotted line spanning the full plot height).
    if 0 < n_staircase < n_cond:
        divider_offset = (offsets[n_staircase - 1] + offsets[n_staircase]) / 2
        for xi in x:
            ax.axvline(xi + divider_offset, color=LIGHT_GREY, linewidth=1,
                       linestyle=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11, color=TEXT)
    ax.set_ylabel("Mode share (%)", fontsize=11, color=MUTED)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
    ax.tick_params(axis="y", colors=MUTED, labelsize=10)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.grid(True, color=LIGHT_GREY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    bar_handles, bar_labels = ax.get_legend_handles_labels()
    imob_handle = plt.Line2D([0], [0], color=GREY, linewidth=2, linestyle="--", label="IMOB 2017 target")
    ax.legend(handles=bar_handles + [imob_handle], labels=bar_labels + ["IMOB 2017 target"],
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9.5, frameon=False, labelcolor=MUTED)

    ax.set_title(title, fontsize=13, color=TEXT, pad=12)
    if 0 < n_staircase < n_cond:
        fig.suptitle(
            "left of divider: specificity staircase (bare → minimal → full → strict)"
            "   |   right of divider: single-factor removals from full",
            fontsize=8.5, color=MUTED, style="italic", y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── plot_combined_nl_pt: 5-panel headline figure, NL vs PT ────────────────────
def _panel_distance_nl_pt(ax, nl_runs, pt_runs) -> None:
    x_max = 60
    x = np.linspace(0.05, x_max, 400)

    y_nl, y_nl_std = mean_std_runs(
        [kde_curve(r["distance_km"].dropna().clip(upper=x_max), x) for r in nl_runs]
    )
    y_pt, y_pt_std = mean_std_runs(
        [kde_curve(r["distance_km"].dropna().clip(upper=x_max), x) for r in pt_runs]
    )

    ax.plot(x, y_nl, color=ORANGE, lw=2)
    ax.plot(x, y_pt, color=GREEN,  lw=2)
    ax.fill_between(x, np.clip(y_nl - y_nl_std, 0, None), y_nl + y_nl_std, color=ORANGE, alpha=0.18)
    ax.fill_between(x, np.clip(y_pt - y_pt_std, 0, None), y_pt + y_pt_std, color=GREEN,  alpha=0.18)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 60)


def _panel_departure_nl_pt(ax, nl_runs, pt_runs) -> None:
    x = np.linspace(0, 24, 400)

    y_nl, y_nl_std = mean_std_runs([kde_curve(r["departure_min"] / 60, x) for r in nl_runs])
    y_pt, y_pt_std = mean_std_runs([kde_curve(r["departure_min"] / 60, x) for r in pt_runs])

    ax.plot(x, y_nl, color=ORANGE, lw=2)
    ax.plot(x, y_pt, color=GREEN,  lw=2)
    ax.fill_between(x, np.clip(y_nl - y_nl_std, 0, None), y_nl + y_nl_std, color=ORANGE, alpha=0.18)
    ax.fill_between(x, np.clip(y_pt - y_pt_std, 0, None), y_pt + y_pt_std, color=GREEN,  alpha=0.18)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])


def _panel_si_nl_pt(ax, nl_runs, pt_runs) -> None:
    x = np.linspace(0, 180, 300)

    y_nl, y_nl_std = mean_std_runs([kde_curve(extract_si_gaps(r), x) for r in nl_runs])
    y_pt, y_pt_std = mean_std_runs([kde_curve(extract_si_gaps(r), x) for r in pt_runs])

    ax.plot(x, y_nl, color=ORANGE, lw=2)
    ax.plot(x, y_pt, color=GREEN,  lw=2)
    ax.fill_between(x, np.clip(y_nl - y_nl_std, 0, None), y_nl + y_nl_std, color=ORANGE, alpha=0.18)
    ax.fill_between(x, np.clip(y_pt - y_pt_std, 0, None), y_pt + y_pt_std, color=GREEN,  alpha=0.18)

    ax.set_xlabel("Gap (min)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 60, 120, 180])


def _draw_ecdf_panel_nl_pt(ax, nl_vals_runs, pt_vals_runs, xlabel, x_max) -> None:
    grid = np.arange(0, x_max + 1)

    y_nl, y_nl_std = mean_std_runs([_ecdf(v, grid) for v in nl_vals_runs])
    y_pt, y_pt_std = mean_std_runs([_ecdf(v, grid) for v in pt_vals_runs])

    ax.step(grid, y_nl, where="post", color=ORANGE, lw=2)
    ax.step(grid, y_pt, where="post", color=GREEN,  lw=2)
    ax.fill_between(
        grid, np.clip(y_nl - y_nl_std, 0, 1), np.clip(y_nl + y_nl_std, 0, 1),
        step="post", color=ORANGE, alpha=0.18,
    )
    ax.fill_between(
        grid, np.clip(y_pt - y_pt_std, 0, 1), np.clip(y_pt + y_pt_std, 0, 1),
        step="post", color=GREEN, alpha=0.18,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative share")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)


def plot_combined_nl_pt(
    nl_runs: list[pd.DataFrame] | pd.DataFrame,
    pt_runs: list[pd.DataFrame] | pd.DataFrame,
    save_path: str | None = None,
    max_trips: int = 12,
    max_purposes: int = 8,
) -> plt.Figure:
    """
    5-panel headline figure comparing NL synthetic vs PT synthetic directly
    (no ODiN/SMP reference line) — same layout as fig_gallery.plot_combined:
    (a) Trip Distance | (b) Departure Time | (c) Step Interval KDEs,
    (d) Trips per Person | (e) Purpose Diversity ECDFs.

    nl_runs / pt_runs accept a single DataFrame or a list of per-seed
    DataFrames in the common per-trip schema (see fig_gallery._flatten_run) —
    every panel shows a mean +/- 1 std-dev band when more than one run is given.
    """
    if isinstance(nl_runs, pd.DataFrame):
        nl_runs = [nl_runs]
    if isinstance(pt_runs, pd.DataFrame):
        pt_runs = [pt_runs]

    set_style()

    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(2, 6, hspace=0.55, wspace=0.45)

    ax_dist = fig.add_subplot(gs[0, 0:2])
    ax_dep  = fig.add_subplot(gs[0, 2:4])
    ax_si   = fig.add_subplot(gs[0, 4:6])
    ax_tpp  = fig.add_subplot(gs[1, 0:2])
    ax_purp = fig.add_subplot(gs[1, 2:4])

    _panel_distance_nl_pt(ax_dist, nl_runs, pt_runs)
    strip_axes(ax_dist)
    add_caption(ax_dist, "a", "Trip Distance", fontsize=11)

    _panel_departure_nl_pt(ax_dep, nl_runs, pt_runs)
    strip_axes(ax_dep)
    ax_dep.set_ylabel("")
    add_caption(ax_dep, "b", "Departure Time", fontsize=11)

    _panel_si_nl_pt(ax_si, nl_runs, pt_runs)
    strip_axes(ax_si)
    ax_si.set_ylabel("")
    add_caption(ax_si, "c", "Step Interval", fontsize=11)

    _draw_ecdf_panel_nl_pt(
        ax_tpp,
        [_trips_per_person(r) for r in nl_runs],
        [_trips_per_person(r) for r in pt_runs],
        xlabel="Trips per person", x_max=max_trips,
    )
    strip_axes(ax_tpp)
    add_caption(ax_tpp, "d", "Trips per Person", fontsize=11)

    _draw_ecdf_panel_nl_pt(
        ax_purp,
        [_unique_purposes_per_person(r) for r in nl_runs],
        [_unique_purposes_per_person(r) for r in pt_runs],
        xlabel="Unique purposes per person", x_max=max_purposes,
    )
    strip_axes(ax_purp)
    ax_purp.set_ylabel("")
    add_caption(ax_purp, "e", "Purpose Diversity", fontsize=11)

    nl_label = r"NL synthetic ($\pm$1$\sigma$ over seeds)" if len(nl_runs) > 1 else "NL synthetic"
    pt_label = r"PT synthetic ($\pm$1$\sigma$ over seeds)" if len(pt_runs) > 1 else "PT synthetic"
    handles = [
        mlines.Line2D([], [], color=ORANGE, lw=2, label=nl_label),
        mlines.Line2D([], [], color=GREEN,  lw=2, label=pt_label),
    ]
    ax_si.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.0),
        frameon=False, fontsize=8, borderaxespad=0,
    )

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── plot_mode_share_by_purpose: mode-share lines, NL vs PT, per top purpose ───
_MODE_MAP_PURPOSE = {
    "Non-electric bicycle": "Bicycle", "Electric bike": "Bicycle", "Speed pedelec": "Bicycle",
    "On foot":              "Walk",
    "Passenger car":        "Car", "Delivery van": "Car", "Truck": "Car", "Camper": "Car",
    "Train":                "Transit", "Bus": "Transit", "Tram": "Transit",
    "Subway":               "Transit", "Coach": "Transit",
}
MODE_ORDER_PURPOSE = ["Bicycle", "Walk", "Transit", "Car", "Other"]

PURPOSE_DISPLAY = {
    "To and from work":          "Work",
    "Taking education/course":   "Education",
    "Shopping/grocery shopping": "Shopping",
    "Other leisure activities":  "Leisure",
    "Sports/hobbies":            "Sports",
}


def _group_mode_purpose(raw: str) -> str:
    return _MODE_MAP_PURPOSE.get(raw, "Other")


def _mode_by_purpose(records: list[dict]) -> dict[str, "collections.Counter"]:
    data = collections.defaultdict(collections.Counter)
    for rec in records:
        for t in rec.get("trips", []):
            p = t.get("purpose", "")
            m = t.get("mode", "")
            if p and m:
                data[p][_group_mode_purpose(m)] += 1
    return data


def _to_pct(counter: "collections.Counter") -> dict[str, float]:
    total = sum(counter.values())
    return {m: counter.get(m, 0) / total * 100 if total else 0.0 for m in MODE_ORDER_PURPOSE}


def plot_mode_share_by_purpose(
    nl_records: list[dict],
    pt_records: list[dict],
    n_purposes: int = 4,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grid of mode-share-by-purpose panels (NL orange circles/line, PT green
    triangles/line) for the `n_purposes` trip purposes with the largest
    combined NL+PT trip volume.

    nl_records / pt_records : raw SynTrav record lists (output of
                              load_syn_records()), one set per country.
    """
    nl_by_p = _mode_by_purpose(nl_records)
    pt_by_p = _mode_by_purpose(pt_records)

    all_purposes = set(nl_by_p) | set(pt_by_p)
    top_purposes = sorted(
        all_purposes,
        key=lambda p: sum(nl_by_p.get(p, {}).values()) + sum(pt_by_p.get(p, {}).values()),
        reverse=True,
    )[:n_purposes]

    nl_shares = {p: _to_pct(nl_by_p.get(p, {})) for p in top_purposes}
    pt_shares = {p: _to_pct(pt_by_p.get(p, {})) for p in top_purposes}

    x = np.arange(len(MODE_ORDER_PURPOSE))

    set_style()
    ncols = 2
    nrows = (n_purposes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), sharey=True)
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, purpose in zip(axes_flat, top_purposes):
        ax.set_facecolor("white")

        nl = [nl_shares[purpose][m] for m in MODE_ORDER_PURPOSE]
        pt = [pt_shares[purpose][m] for m in MODE_ORDER_PURPOSE]

        # Lines
        ax.plot(x, nl, color=ORANGE, linewidth=2, zorder=2)
        ax.plot(x, pt, color=GREEN,  linewidth=2, zorder=2)

        # Markers
        ax.scatter(x, nl, s=120, color=ORANGE, zorder=3, marker="o", linewidths=0)
        ax.scatter(x, pt, s=220, color=GREEN,  zorder=3, marker="^", linewidths=0)

        # Value labels - PT's triangle marker (s=220) is visually larger than
        # NL's circle (s=120), so it needs more clearance to keep the label
        # from sitting on top of the marker.
        NL_LABEL_OFFSET = 5.0
        PT_LABEL_OFFSET = 8.0
        for i, (nv, pv) in enumerate(zip(nl, pt)):
            nl_va = "bottom" if nv >= pv else "top"
            pt_va = "bottom" if pv > nv else "top"
            nl_y  = nv + NL_LABEL_OFFSET if nl_va == "bottom" else nv - NL_LABEL_OFFSET
            pt_y  = pv + PT_LABEL_OFFSET if pt_va == "bottom" else pv - PT_LABEL_OFFSET
            ax.text(x[i], nl_y, f"{nv:.1f}%",
                    ha="center", va=nl_va, fontsize=8, color=ORANGE, fontweight="bold")
            ax.text(x[i], pt_y, f"{pv:.1f}%",
                    ha="center", va=pt_va, fontsize=8, color=GREEN, fontweight="bold")

        ax.set_title(PURPOSE_DISPLAY.get(purpose, purpose),
                     fontsize=11, color=TEXT, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODE_ORDER_PURPOSE, fontsize=10, color=TEXT)
        ax.set_ylim(-14, 114)
        ax.set_yticks(np.arange(0, 101, 25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
        ax.set_ylabel("% of trips", fontsize=9, color=MUTED)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", colors=MUTED, labelsize=9)
        ax.yaxis.grid(True, color=LIGHT_GREY, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_edgecolor(LIGHT_GREY)
        ax.spines["left"].set_linewidth(1.2)

    for ax in axes_flat[len(top_purposes):]:
        ax.axis("off")

    nl_handle = plt.Line2D([0], [0], marker="o", color=ORANGE,
                           markerfacecolor=ORANGE, markersize=8, label="NL synthetic")
    pt_handle = plt.Line2D([0], [0], marker="^", color=GREEN,
                           markerfacecolor=GREEN,  markersize=10, label="PT synthetic")
    fig.legend(
        handles=[nl_handle, pt_handle],
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=2, fontsize=10, frameon=False, labelcolor=MUTED,
    )

    fig.suptitle("Mode share shift by trip purpose: NL vs PT synthetic",
                 fontsize=13, color=TEXT, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── PT TARGET FIT: mode share + purpose share + distance-by-purpose + volume ──
# IMOB 2017 (AML) publishes more than modal split: trip purpose shares
# (Figura 120, p.111) and mean trip distance by purpose (Figura 125, p.113),
# plus the weekday mobile-population share and trips/day among mobile persons
# (Executive Summary). Verified directly against the IMOB_2017A.pdf text.
#
# Duration by purpose is also published (Figura 124) but SynTrav trip records
# carry no duration field (only a departure_time_class bucket), so there is
# nothing to compare it against — intentionally left out rather than proxied.
IMOB_PURPOSE_SHARE = {
    "To and from work":          30.8,
    "Taking education/course":   10.5,
    "Shopping/grocery shopping": 19.8,
}

IMOB_DISTANCE_BY_PURPOSE_KM = {
    "To and from work":          14.8,
    "Taking education/course":    6.9,
    "Shopping/grocery shopping":  5.7,
}

IMOB_MOBILE_SHARE = 85.1   # % of AML residents making >=1 trip on a weekday
IMOB_TRIP_RATE    = 2.60   # trips/day, among mobile persons


def _pt_valid_trips(records: list[dict]) -> list[dict]:
    """
    Real (non-return-home) trips with a usable purpose label, pooled across
    `records`. Mirrors the destination == "home" exclusion used by
    prep_syntrav_run()/_flatten_run() and the len(purpose) < 60 filter used
    elsewhere in this module to drop garbled LLM purpose strings.
    """
    return [
        t for r in records for t in r.get("trips", [])
        if (t.get("destination") or "").lower() != "home"
        and t.get("purpose") and len(t["purpose"]) < 60
    ]


def compute_purpose_share(
    records: list[dict],
    purposes: dict[str, float] = IMOB_PURPOSE_SHARE,
) -> dict[str, float]:
    """Fraction (0-1) of trips per purpose, pooled over trips of any purpose."""
    trip_purposes = [t["purpose"] for t in _pt_valid_trips(records)]
    total = len(trip_purposes)
    if not total:
        return {p: 0.0 for p in purposes}
    counts = collections.Counter(trip_purposes)
    return {p: counts.get(p, 0) / total for p in purposes}


def compute_mean_distance_by_purpose(
    records: list[dict],
    purposes: dict[str, float] = IMOB_DISTANCE_BY_PURPOSE_KM,
) -> dict[str, float | None]:
    """
    Mean trip distance (km) per purpose. distance_km falls back to the
    DISTANCE_CLASS_KM midpoint (same convention as prep_syntrav_run()) since
    some PT runs only carry distance_class, not a numeric distance_km.
    """
    from Helpers.evaluation import DISTANCE_CLASS_KM

    by_purpose = collections.defaultdict(list)
    for t in _pt_valid_trips(records):
        p = t["purpose"]
        if p not in purposes:
            continue
        km = t.get("distance_km")
        if km is None:
            km = DISTANCE_CLASS_KM.get(t.get("distance_class", ""))
        if km is not None:
            by_purpose[p].append(float(km))
    return {p: (float(np.mean(by_purpose[p])) if by_purpose.get(p) else None) for p in purposes}


def compute_mobile_share(records: list[dict]) -> float:
    """% of persons in `records` with at least one real (non-return-home) trip."""
    if not records:
        return 0.0
    mobile = sum(
        1 for r in records
        if any((t.get("destination") or "").lower() != "home" for t in r.get("trips", []))
    )
    return mobile / len(records) * 100


def compute_trip_rate(records: list[dict]) -> float:
    """
    Mean trips/day among persons with >=1 real trip — matches IMOB's "mobile
    person" denominator for its 2.60 trips/day figure (rather than averaging
    over the whole population, which would understate the rate).
    """
    counts = [
        sum(1 for t in r.get("trips", []) if (t.get("destination") or "").lower() != "home")
        for r in records
    ]
    mobile_counts = [c for c in counts if c > 0]
    return float(np.mean(mobile_counts)) if mobile_counts else 0.0


def _pt_target_row(label: str, category: str, nl_val: float | None,
                    pt_val: float | None, target: float) -> dict:
    """
    One build_pt_target_table() row, incl. the Gap (natural units: pp / km /
    trips-per-day depending on `category`) and a bounded signed %-error.

    %-error uses symmetric relative error (2*gap / (|PT|+|target|) * 100,
    bounded to +/-200) rather than plain (PT-target)/target*100: several
    targets here (e.g. IMOB's 0.5% cycling share) have a near-zero
    denominator, where plain relative error explodes into meaningless
    four-digit numbers for what is a small absolute gap. The bounded version
    stays interpretable and is what compute_pt_fit_score() aggregates.
    """
    if pt_val is None:
        return {"Target": label, "Category": category, "NL synthetic": nl_val,
                "PT synthetic": None, "IMOB target": target, "Gap": None, "% error": None}
    gap = pt_val - target
    denom = abs(pt_val) + abs(target)
    pct_error = (2 * gap / denom * 100) if denom else 0.0
    return {
        "Target": label, "Category": category,
        "NL synthetic": nl_val, "PT synthetic": pt_val, "IMOB target": target,
        "Gap": gap, "% error": pct_error,
    }


def build_pt_target_table(
    nl_records: list[dict],
    pt_records: list[dict],
) -> pd.DataFrame:
    """
    Combined IMOB 2017 (AML) fit table for the PT synthetic run: 4 mode-share
    targets + 3 purpose-share targets + 3 mean-distance-by-purpose targets +
    2 trip-volume targets (mobile share, trip rate) = 12 rows total.

    Each row reports PT synthetic vs IMOB target as both a raw Gap (pp for
    share/volume rows, km for distance rows) and a bounded signed %-error —
    see _pt_target_row(). NL synthetic is included for reference only; it is
    not itself scored against IMOB, which is a PT-specific survey.
    """
    rows = []

    _, _, _, _, nl_ms, pt_ms = compute_mode_shift_data(nl_records, pt_records)
    for m in DEFAULT_MODE_SHIFT_MODES:
        rows.append(_pt_target_row(
            MODE_DISPLAY_LABELS.get(m, m), "Mode share",
            nl_ms[m] * 100, pt_ms[m] * 100, IMOB_2017[m],
        ))

    nl_purp, pt_purp = compute_purpose_share(nl_records), compute_purpose_share(pt_records)
    for p, target in IMOB_PURPOSE_SHARE.items():
        rows.append(_pt_target_row(
            PURPOSE_DISPLAY.get(p, p), "Purpose share",
            nl_purp[p] * 100, pt_purp[p] * 100, target,
        ))

    nl_dist = compute_mean_distance_by_purpose(nl_records)
    pt_dist = compute_mean_distance_by_purpose(pt_records)
    for p, target in IMOB_DISTANCE_BY_PURPOSE_KM.items():
        rows.append(_pt_target_row(
            f"{PURPOSE_DISPLAY.get(p, p)} distance (km)", "Mean distance",
            nl_dist[p], pt_dist[p], target,
        ))

    nl_mob, pt_mob = compute_mobile_share(nl_records), compute_mobile_share(pt_records)
    rows.append(_pt_target_row("Mobile share", "Trip volume", nl_mob, pt_mob, IMOB_MOBILE_SHARE))

    nl_rate, pt_rate = compute_trip_rate(nl_records), compute_trip_rate(pt_records)
    rows.append(_pt_target_row("Trips/day (mobile)", "Trip volume", nl_rate, pt_rate, IMOB_TRIP_RATE))

    df = pd.DataFrame(rows).set_index("Target")
    for col in ["NL synthetic", "PT synthetic", "IMOB target", "Gap", "% error"]:
        df[col] = df[col].astype(float).round(1)
    return df


def compute_pt_fit_score(target_table: pd.DataFrame) -> float:
    """Mean absolute bounded %-error of PT synthetic vs IMOB target, pooled across all rows."""
    return float(target_table["% error"].dropna().abs().mean())


PT_TARGET_CATEGORY_COLORS = {
    "Mode share":    GREEN,
    "Purpose share": "#4C956C",
    "Mean distance": "#7A9E7E",
    "Trip volume":   "#1B4332",
}


def plot_pt_target_fit(
    target_table: pd.DataFrame,
    title: str = "PT synthetic vs IMOB 2017: fit across all targets",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal divergence chart: one bar per build_pt_target_table() row,
    x-axis is the signed %-error of PT synthetic vs the IMOB value (0 =
    perfect fit). Rows are grouped and colored by category, ordered by
    |%-error| within each category block (best-fitting at top).
    """
    set_style()

    df = target_table.copy()
    df["abs_err"] = df["% error"].abs()
    df = df.sort_values(["Category", "abs_err"])

    y = np.arange(len(df))
    colors = [PT_TARGET_CATEGORY_COLORS.get(c, MUTED) for c in df["Category"]]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(y, df["% error"], color=colors, zorder=3, height=0.62)
    ax.axvline(0, color=GREY, linewidth=1.4, zorder=4)

    for yi, val in zip(y, df["% error"]):
        if pd.isna(val):
            continue
        ax.text(val + (2 if val >= 0 else -2), yi, f"{val:+.0f}%",
                 ha="left" if val >= 0 else "right", va="center",
                 fontsize=8, color=TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=9, color=TEXT)
    ax.set_xlabel("PT synthetic vs IMOB 2017 target (% error)", fontsize=10, color=MUTED)
    ax.xaxis.grid(True, color=LIGHT_GREY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [mlines.Line2D([0], [0], color=c, lw=6, label=cat)
               for cat, c in PT_TARGET_CATEGORY_COLORS.items() if cat in df["Category"].values]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, frameon=False, labelcolor=MUTED)

    ax.set_title(title, fontsize=13, color=TEXT, pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── plot_activity_intention_pt: spatial destination map, Oeiras (PT) ──────────
# Reuses fig_gallery.plot_activity_intention (basemap + KDE density + emoji POI
# markers) but retargeted at Oeiras instead of Rotterdam: swaps in the Oeiras
# bounding box for the duration of the call (same module-attribute-override
# trick as the old standalone fig_activity_intention.py script), then rewrites
# the hardcoded "Rotterdam" footer caption.
OEIRAS_LAT_MIN, OEIRAS_LAT_MAX = 38.65, 38.76
OEIRAS_LON_MIN, OEIRAS_LON_MAX = -9.38, -9.20

OEIRAS_GEO_PATH = Path("Portugal_POI_data/PT_poi/trajectories_portugal_weekday_big_geo.json")


def load_oeiras_geo_records(json_path: str | Path = OEIRAS_GEO_PATH) -> list[dict]:
    """Load the geo-allocated Oeiras (PT) trip records used by plot_activity_intention_pt()."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def plot_activity_intention_pt(
    records: list[dict],
    group_keys: list,
    panel_titles: list[str] | None = None,
    fig_title: str = "Activity destinations - Oeiras, PT",
    save_path: str | None = None,
    kde_res: int = 300,
) -> plt.Figure:
    """
    Same panels as fig_gallery.plot_activity_intention (OSM basemap + KDE
    density + emoji POI-category markers, one panel per group_keys entry),
    rendered over Oeiras instead of Rotterdam.
    """
    orig_bbox = (
        _fig_gallery._ROTTERDAM_LAT_MIN, _fig_gallery._ROTTERDAM_LAT_MAX,
        _fig_gallery._ROTTERDAM_LON_MIN, _fig_gallery._ROTTERDAM_LON_MAX,
    )
    _fig_gallery._ROTTERDAM_LAT_MIN, _fig_gallery._ROTTERDAM_LAT_MAX = OEIRAS_LAT_MIN, OEIRAS_LAT_MAX
    _fig_gallery._ROTTERDAM_LON_MIN, _fig_gallery._ROTTERDAM_LON_MAX = OEIRAS_LON_MIN, OEIRAS_LON_MAX
    try:
        fig = _fig_gallery.plot_activity_intention(
            records, group_keys, panel_titles=panel_titles,
            fig_title=fig_title, save_path=None, kde_res=kde_res,
        )
    finally:
        (_fig_gallery._ROTTERDAM_LAT_MIN, _fig_gallery._ROTTERDAM_LAT_MAX,
         _fig_gallery._ROTTERDAM_LON_MIN, _fig_gallery._ROTTERDAM_LON_MAX) = orig_bbox

    for txt in fig.texts:
        if "Rotterdam" in txt.get_text():
            txt.set_text("Weekday activity destinations · Oeiras · all day")

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    return fig
