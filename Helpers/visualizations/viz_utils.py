"""
viz_utils.py 

Shared style, colors, and distribution helpers for SynTrav figures.

Imports expected by fig_distributions.py:
    from viz_utils import (
        set_style, strip_axes, add_caption,
        COLOR_ODIN, COLOR_SMP, COLOR_SYNTRAV,
        LS_ODIN, LS_SMP, LS_SYNTRAV, LW, ALPHA_BAND,
        PANEL_CAPTIONS, PURPOSE_DISPLAY_LABELS, CANONICAL_PURPOSE_ORDER,
        kde_curve, normalise, mean_std_runs,
        extract_departure_hour_props, extract_si_gaps,
        extract_trips_per_person_pmf, extract_purpose_props,
        extract_purpose_diversity_pmf, get_purpose_labels,
        compute_jsd,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon


#Colors & line styles
COLOR_ODIN    = "#9E9E9E"   # muted red
COLOR_SMP     = "#0A030E"   # purple
COLOR_SYNTRAV = "#FF9800"   

LS_ODIN    = "-"
LS_SMP     = "--"
LS_SYNTRAV = "-"
LW         = 2
ALPHA_BAND = 0.18


#Histogram bins — must match evaluation.py exactly so JSD table numbers align 
SD_BINS = np.array(
    [0, 0.5, 1.0, 2.5, 3.7, 5.0, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 250],
    dtype=float,
)
SI_BINS = np.array(
    [0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440],
    dtype=float,
)


#Panel metadata 
# (letter, caption) pairs in display order
PANEL_CAPTIONS = [
    ("a", "Trip Distance"),
    ("b", "Departure Time"),
    ("c", "Step Interval"),
    ("d", "Trips per Person"),
    ("e", "Purpose Diversity"),
]


#Purpose label utilities 
# Canonical order matches config.yaml state_to_odin_label; HOME is excluded.
CANONICAL_PURPOSE_ORDER = [
    "To and from work",
    "Taking education/course",
    "Shopping/grocery shopping",
    "Collect/deliver goods",
    "Services/personal care",
    "Pick up/drop off people",
    "Other leisure activities",
    "Sports/hobbies",
    "Visitors/staying over",
    "Touring/hiking",
    "Different motive",
]

PURPOSE_DISPLAY_LABELS = {
    "To and from work":                       "Wk",
    "Taking education/course":                "Ed",
    "Shopping/grocery shopping":              "Sh",
    "Collect/deliver goods":                  "Cd",
    "Services/personal care":                 "Sv",
    "Pick up/drop off people":                "Pu",
    "Other leisure activities":               "Le",
    "Sports/hobbies":                         "Sp",
    "Visitors/staying over":                  "Vi",
    "Touring/hiking":                         "To",
    "Different motive":                       "Ot",
}


#Global rcParams 
def set_style() -> None:
    """Apply global serif rcParams for publication figures."""
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       9,
        "axes.titlesize":  9,
        "axes.labelsize":  9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "legend.frameon":  False,
        "figure.dpi":      150,
        "pdf.fonttype":    42,   # embed fonts in PDF/EPS
        "ps.fonttype":     42,
    })


#Axis helpers 
def strip_axes(ax: plt.Axes) -> None:
    """Remove top/right spines and disable gridlines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="out", length=3)


def add_caption(ax: plt.Axes, letter: str, caption: str,
                y_offset: float = -0.22, fontsize: int = 8) -> None:
    """Place centred italic caption below the x-axis of `ax`."""
    ax.text(
        0.5, y_offset,
        caption,
        transform=ax.transAxes,
        ha="center", va="top",
        style="italic", fontsize=fontsize,
    )


#Statistical helpers 
def kde_curve(data: np.ndarray, x_grid: np.ndarray,
              bw: str = "scott") -> np.ndarray:
    """
    KDE of positive finite values in `data`, evaluated on `x_grid`.
    Returns a zero array when data is insufficient (< 5 points).
    """
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 5:
        return np.zeros_like(x_grid, dtype=float)
    return gaussian_kde(d, bw_method=bw)(x_grid)


def normalise(arr: np.ndarray) -> np.ndarray:
    """Scale array to sum to 1; return a zero array if the total is zero."""
    a = np.asarray(arr, dtype=float)
    s = a.sum()
    return a / s if s > 0 else a


def mean_std_runs(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Element-wise mean and std over a list of same-shape 1-D arrays.
    Used to summarise multiple SynTrav generation runs.
    """
    mat = np.array(arrays, dtype=float)
    return mat.mean(axis=0), mat.std(axis=0)


#Distribution extractors 
def extract_departure_hour_props(df) -> np.ndarray:
    """24-element array of proportions: trips departing in each hour 0–23."""
    hours = (df["departure_min"] // 60).clip(0, 23).astype(int)
    counts = np.bincount(hours, minlength=24).astype(float)
    return normalise(counts)


def extract_si_gaps(df) -> np.ndarray:
    """
    Step-interval gaps per person: departure_min[i+1] – departure_min[i].

    Uses departure-to-departure diffs (consistent with evaluation.py real_si/syn_si),
    which works for both ODiN and SynTrav (SynTrav has no real arrival times).
    Negative values and gaps > 180 min are dropped.
    Persons with fewer than 2 trips contribute no gaps.
    """
    gaps: list[float] = []
    for _, grp in df.sort_values("departure_min").groupby("person_id"):
        dep = grp["departure_min"].values
        if len(dep) < 2:
            continue
        g = np.diff(dep)
        gaps.extend(g[(g >= 0) & (g <= 180)].tolist())
    return np.array(gaps, dtype=float)


def extract_trips_per_person_pmf(
    df, max_trips: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (centers, pmf) where centers = 0..max_trips (integers) and
    pmf is the normalised frequency of that trip count across persons.
    Person-days with more than max_trips trips are clipped to max_trips.
    """
    counts = df.groupby("person_id").size().clip(upper=max_trips).values
    edges = np.arange(0, max_trips + 2)
    hist, _ = np.histogram(counts, bins=edges)
    return edges[:-1], normalise(hist.astype(float))


def extract_purpose_props(df, labels: list[str]) -> np.ndarray:
    """
    Normalised trip-count proportions over `labels` (in the given order).
    Labels absent from the data receive proportion 0.
    """
    vc = df["purpose_state"].value_counts()
    counts = np.array([vc.get(lbl, 0) for lbl in labels], dtype=float)
    return normalise(counts)


def extract_purpose_diversity_pmf(
    df, max_unique: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (centers, pmf) for unique-purpose count per person.
    Centers = 0..max_unique; counts above max_unique are clipped.
    """
    uniq = (
        df.groupby("person_id")["purpose_state"]
        .nunique()
        .clip(upper=max_unique)
        .values
    )
    edges = np.arange(0, max_unique + 2)
    hist, _ = np.histogram(uniq, bins=edges)
    return edges[:-1], normalise(hist.astype(float))


def get_purpose_labels(*dfs, use_canonical: bool = True) -> list[str]:
    """
    Return the ordered purpose label list to use for panel (e).

    If use_canonical=True (default), returns CANONICAL_PURPOSE_ORDER filtered
    to labels that actually appear in at least one DataFrame, then appends
    any remaining labels in alphabetical order.

    If use_canonical=False, orders by frequency in the first (ODiN) DataFrame.
    """
    present: set[str] = set()
    for df in dfs:
        present |= set(df["purpose_state"].dropna().unique())

    if use_canonical:
        ordered = [lbl for lbl in CANONICAL_PURPOSE_ORDER if lbl in present]
        remaining = sorted(present - set(ordered))
        return ordered + remaining

    ref_freq = dfs[0]["purpose_state"].value_counts()
    ordered = [lbl for lbl in ref_freq.index if lbl in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


#Jensen-Shannon Divergence 
def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence in [0, 1] (base-2 log).
    Both arrays are normalised; the shorter is zero-padded to match lengths.

    JSD = jensenshannon(p, q, base=2)^2  (scipy returns the JS *distance*,
    which is the square root of the divergence).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    n = max(len(p), len(q))
    p = normalise(np.pad(p, (0, n - len(p))))
    q = normalise(np.pad(q, (0, n - len(q))))
    return float(jensenshannon(p, q, base=2) ** 2)


#Data-prep helpers 

def prep_odin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the ODiN trip DataFrame (e.g. real_weekday) into the common schema.

    Input columns used:
        Person_index, Motive, Destination/Purpose,
        Departure time transfer, Departure minute displacement,
        Arrival time transfer,   Arrival minute displacement,
        Travel distance in the Netherlands (in hectometers)

    Home-return trips (Destination/Purpose == "Home") are excluded so that
    only activity trips are counted, matching prep_syntrav_run and the SMP output.
    """
    dest_col = "Destination/Purpose"
    if dest_col in df.columns:
        df = df[df[dest_col].astype(str).str.strip() != "Home"]

    dep = (pd.to_numeric(df["Departure time transfer"],       errors="coerce") * 60
         + pd.to_numeric(df["Departure minute displacement"], errors="coerce"))
    arr = (pd.to_numeric(df["Arrival time transfer"],         errors="coerce") * 60
         + pd.to_numeric(df["Arrival minute displacement"],   errors="coerce"))
    dist = pd.to_numeric(
        df["Travel distance in the Netherlands (in hectometers)"], errors="coerce"
    ) * 0.1

    return pd.DataFrame({
        "person_id":     df["Person_index"].astype(str).values,
        "purpose_state": df["Motive"].values,
        "departure_min": dep.values,
        "arrival_min":   arr.values,
        "distance_km":   dist.values,
    }).dropna(subset=["departure_min", "purpose_state"]).reset_index(drop=True)


def prep_syntrav_run(records: list[dict], seed: int = 42) -> pd.DataFrame:
    """
    Flatten one SynTrav run (output of load_syn_records()) into the common schema.

    Departure times: uses departure_time_class with uniform jitter within the class
    range (same as evaluation.py syn_si/syn_dard) so round LLM times like "08:00"
    don't spike a single hour bin. Falls back to the raw time field if the class
    label is absent or unrecognised.
    Home-return trips (destination == "home") are excluded.
    """
    from Helpers.evaluation import DEP_TIME_CLASS_RANGE, DISTANCE_CLASS_KM
    rng = np.random.default_rng(seed)

    rows = []
    for rec in records:
        pid   = str(rec.get("person_id", ""))
        trips = [t for t in (rec.get("trips") or [])
                 if (t.get("destination") or "").lower() != "home"]

        deps, purposes, distances = [], [], []
        for t in trips:
            dep_class = t.get("departure_time_class", "")
            if dep_class in DEP_TIME_CLASS_RANGE:
                lo, hi = DEP_TIME_CLASS_RANGE[dep_class]
                dep = int(rng.integers(lo, hi))
            else:
                try:
                    h, m = t["time"].split(":")
                    dep  = int(h) * 60 + int(m)
                except Exception:
                    continue
            deps.append(dep)
            purposes.append(t.get("purpose", ""))
            km = t.get("distance_km")
            if km is None:
                km = DISTANCE_CLASS_KM.get(t.get("distance_class", ""))
            distances.append(float(km or 0.0))

        arrs = deps[1:] + deps[-1:]  # arrival[i] = departure[i+1]
        for dep, arr, purpose, dist in zip(deps, arrs, purposes, distances):
            rows.append({
                "person_id":     pid,
                "purpose_state": purpose,
                "departure_min": float(dep),
                "arrival_min":   float(arr),
                "distance_km":   dist,
            })

    return pd.DataFrame(rows).reset_index(drop=True)


#Purpose divergence chart 
def plot_purpose_divergence(
    odin_df: pd.DataFrame,
    smp_df: pd.DataFrame,
    syntrav_runs: "list[pd.DataFrame] | pd.DataFrame",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal divergence bar chart: LLM and SMP deviations from ODiN (reference = 0).

    Parameters
    ----------
    odin_df, smp_df   : DataFrames in the common schema (must have 'purpose_state')
    syntrav_runs      : one DataFrame or list of DataFrames (multiple runs averaged)
    save_path         : if given, saves as both .pdf and .png
    """
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches

    if isinstance(syntrav_runs, pd.DataFrame):
        syntrav_runs = [syntrav_runs]

    def _shares(df):
        return df["purpose_state"].value_counts(normalize=True).mul(100)

    odin_shares = _shares(odin_df)
    smp_shares  = _shares(smp_df)

    # Average purpose shares across runs
    run_share_df = pd.DataFrame(
        [_shares(r) for r in syntrav_runs]
    ).fillna(0)
    llm_shares = run_share_df.mean()

    purposes = [p for p in CANONICAL_PURPOSE_ORDER
                if p in odin_shares.index or p in llm_shares.index or p in smp_shares.index]

    odin_vals = odin_shares.reindex(purposes).fillna(0)
    llm_delta = llm_shares.reindex(purposes).fillna(0) - odin_vals
    smp_delta = smp_shares.reindex(purposes).fillna(0)  - odin_vals

    # Sort by absolute LLM deviation (most divergent at top)
    order     = llm_delta.abs().sort_values(ascending=True).index
    llm_delta = llm_delta[order]
    smp_delta = smp_delta[order]
    odin_vals = odin_vals[order]

    set_style()
    plt.rcParams.update({
        "axes.spines.left": False,
        "xtick.direction":  "out",
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    y      = np.arange(len(order))
    height = 0.35

    ax.barh(y + height / 2, llm_delta.values, height,
            color=COLOR_SYNTRAV, alpha=0.85, label="LLM − ODiN")
    bars_smp = ax.barh(y - height / 2, smp_delta.values, height,
                       color=COLOR_SMP, alpha=0.85, label="SMP − ODiN")

    ax.axvline(0, color=COLOR_ODIN, linewidth=1.6, linestyle="-", zorder=5)

    x_max = max(llm_delta.abs().max(), smp_delta.abs().max()) + 1

    # Delta labels on bars (skip tiny bars)
    for bars, delta, color in [
        (ax.containers[0], llm_delta.values, COLOR_SYNTRAV),
        (ax.containers[1], smp_delta.values, COLOR_SMP),
    ]:
        for bar, val in zip(bars, delta):
            if abs(val) <= 0.3:
                continue
            xpos = bar.get_width()
            ax.text(
                xpos + (0.2 if xpos >= 0 else -0.2),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}pp",
                va="center", ha="left" if xpos >= 0 else "right",
                fontsize=6.5, color=color, fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("Deviation from ODiN (percentage points)", fontsize=9)
    ax.set_xlim(-x_max, x_max)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}pp"))
    ax.tick_params(left=False)
    strip_axes(ax)

    ax.axvspan( 0,  x_max, alpha=0.03, color=COLOR_SYNTRAV, zorder=0)
    ax.axvspan(-x_max, 0,  alpha=0.03, color="#444444",     zorder=0)
    ax.text( x_max * 0.55, len(y) - 0.3, "Overestimated",  fontsize=7, color="#aaaaaa", ha="center")
    ax.text(-x_max * 0.55, len(y) - 0.3, "Underestimated", fontsize=7, color="#aaaaaa", ha="center")

    handles = [
        mpatches.Patch(color=COLOR_SYNTRAV, alpha=0.85, label="LLM deviation"),
        mpatches.Patch(color=COLOR_SMP,     alpha=0.85, label="SMP deviation"),
        plt.Line2D([0], [0], color=COLOR_ODIN, lw=1.6, label="ODiN (reference = 0)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    add_caption(ax, "f", "Trip Purpose: Deviation from ODiN", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    return fig


#Purpose grouped bar chart (absolute shares) 
def plot_purpose_bars(
    odin_df: pd.DataFrame,
    smp_df: pd.DataFrame,
    syntrav_runs: "list[pd.DataFrame] | pd.DataFrame",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal grouped bar chart of absolute trip-purpose shares (%) for
    ODiN, SMP, and SynTrav (multiple runs averaged).

    Parameters
    ----------
    odin_df, smp_df   : DataFrames in the common schema (must have 'purpose_state')
    syntrav_runs      : one DataFrame or list of DataFrames (multiple runs averaged)
    save_path         : if given, saves as both .pdf and .png
    """
    import matplotlib.patches as mpatches

    if isinstance(syntrav_runs, pd.DataFrame):
        syntrav_runs = [syntrav_runs]

    def _shares(df):
        return df["purpose_state"].value_counts(normalize=True).mul(100)

    odin_shares = _shares(odin_df)
    smp_shares  = _shares(smp_df)
    llm_shares  = pd.DataFrame(
        [_shares(r) for r in syntrav_runs]
    ).fillna(0).mean()

    purposes = [p for p in CANONICAL_PURPOSE_ORDER
                if p in odin_shares.index or p in llm_shares.index or p in smp_shares.index]

    odin_vals = odin_shares.reindex(purposes).fillna(0)
    smp_vals  = smp_shares.reindex(purposes).fillna(0)
    llm_vals  = llm_shares.reindex(purposes).fillna(0)

    # Sort by ODiN share ascending so most common purpose is at top
    order     = odin_vals.sort_values(ascending=True).index
    odin_vals = odin_vals[order]
    smp_vals  = smp_vals[order]
    llm_vals  = llm_vals[order]

    set_style()
    plt.rcParams.update({"axes.spines.left": False, "xtick.direction": "out"})

    fig, ax = plt.subplots(figsize=(10, 6))

    y      = np.arange(len(order))
    height = 0.25

    ax.barh(y + height,     odin_vals.values, height, color=COLOR_ODIN,    alpha=0.90, label="ODiN")
    ax.barh(y,              smp_vals.values,  height, color=COLOR_SMP,     alpha=0.85, label="SMP")
    ax.barh(y - height,     llm_vals.values,  height, color=COLOR_SYNTRAV, alpha=0.85, label="SynTrav (LLM)")

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("Share of trips (%)", fontsize=9)
    ax.set_xlim(0, max(odin_vals.max(), smp_vals.max(), llm_vals.max()) + 5)
    ax.xaxis.set_major_formatter(
        __import__("matplotlib.ticker", fromlist=["FuncFormatter"])
        .FuncFormatter(lambda x, _: f"{x:.0f}%")
    )
    ax.tick_params(left=False)
    strip_axes(ax)

    handles = [
        mpatches.Patch(color=COLOR_ODIN,    alpha=0.90, label="ODiN"),
        mpatches.Patch(color=COLOR_SMP,     alpha=0.85, label="SMP"),
        mpatches.Patch(color=COLOR_SYNTRAV, alpha=0.85, label="SynTrav (LLM)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    add_caption(ax, "g", "Trip Purpose Distribution", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    return fig


#Step Interval KDE 
def plot_si_kde(
    odin_df: pd.DataFrame,
    smp_df: pd.DataFrame,
    syntrav_runs: "list[pd.DataFrame] | pd.DataFrame",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlapping KDE of inter-trip gaps (step interval) for ODiN, SMP, SynTrav.

    Gaps = departure_min[i+1] − departure_min[i] per person, negatives dropped,
    capped at 180 min. Multiple syntrav_runs produce a mean ± 1σ band.
    """
    import matplotlib.lines as mlines

    if isinstance(syntrav_runs, pd.DataFrame):
        syntrav_runs = [syntrav_runs]

    x = np.linspace(0, 180, 400)

    y_odin = kde_curve(extract_si_gaps(odin_df), x)
    y_smp  = kde_curve(extract_si_gaps(smp_df),  x)

    run_kdes = [kde_curve(extract_si_gaps(r), x) for r in syntrav_runs]
    y_syn, y_syn_std = mean_std_runs(run_kdes)

    set_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, y_odin, color=COLOR_ODIN,    lw=LW, ls=LS_ODIN)
    ax.plot(x, y_smp,  color=COLOR_SMP,     lw=LW, ls=LS_SMP)
    ax.plot(x, y_syn,  color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV)
    ax.fill_between(
        x,
        np.clip(y_syn - y_syn_std, 0, None),
        y_syn + y_syn_std,
        color=COLOR_SYNTRAV, alpha=ALPHA_BAND,
    )

    ax.set_xlabel("Gap (min)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 60, 120, 180])
    strip_axes(ax)

    syn_label = r"SynTrav (LLM, $\pm$1$\sigma$)" if len(syntrav_runs) > 1 else "SynTrav (LLM)"
    handles = [
        mlines.Line2D([], [], color=COLOR_ODIN,    lw=LW, ls=LS_ODIN,    label="ODiN"),
        mlines.Line2D([], [], color=COLOR_SMP,     lw=LW, ls=LS_SMP,     label="SMP"),
        mlines.Line2D([], [], color=COLOR_SYNTRAV, lw=LW, ls=LS_SYNTRAV, label=syn_label),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    add_caption(ax, "c", "Step Interval")
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.show()
    return fig
