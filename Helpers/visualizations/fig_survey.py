"""
fig_survey.py

Figures for the human-evaluation survey EDA (human_evaluation_survey/answers.csv),
following the same one-function-per-figure convention as fig_gallery.py /
fig_transferability.py.

    load_survey_long              answers.csv + survey_diaries_key.csv -> tidy
                                   one-row-per-(respondent, diary) DataFrame
    compute_survey_summary        long_df -> headline accuracy / plausibility stats

    plot_respondent_demographics  familiarity + age-group counts among respondents
    plot_confusion_matrix         true label vs guessed label heatmap
    plot_plausibility_distribution plausibility-rating distribution, real vs synthetic
    plot_accuracy_by_familiarity  guess accuracy rate by self-reported familiarity
    plot_per_diary_accuracy       per-diary correct-guess rate, coloured by true label
    plot_implausibility_reasons   frequency of Q2 implausibility reasons, by true label

Every figure below is saved as a PNG only (no PDF), under figures/06_human_survey.

Usage from the notebook
------------------------
    from Helpers.visualizations.fig_survey import (
        FIG_DIR, load_survey_long, compute_survey_summary,
        plot_respondent_demographics, plot_confusion_matrix,
        plot_plausibility_distribution, plot_accuracy_by_familiarity,
        plot_per_diary_accuracy, plot_implausibility_reasons,
    )

    long_df = load_survey_long()
    summary = compute_survey_summary(long_df)

    plot_respondent_demographics(long_df, save_path=str(FIG_DIR / "fig_respondent_demographics"))
    plot_confusion_matrix(long_df, save_path=str(FIG_DIR / "fig_confusion_matrix"))
    plot_plausibility_distribution(long_df, save_path=str(FIG_DIR / "fig_plausibility_distribution"))
    plot_accuracy_by_familiarity(long_df, save_path=str(FIG_DIR / "fig_accuracy_by_familiarity"))
    plot_per_diary_accuracy(long_df, save_path=str(FIG_DIR / "fig_per_diary_accuracy"))
    plot_implausibility_reasons(long_df, save_path=str(FIG_DIR / "fig_implausibility_reasons"))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .viz_utils import set_style, strip_axes, add_caption, COLOR_ODIN, COLOR_SYNTRAV


#  Output location
FIG_DIR = Path("figures/06_human_survey")
FIG_DIR.mkdir(parents=True, exist_ok=True)


#  Input files
ANSWERS_CSV = Path("human_evaluation_survey/answers.csv")
KEY_CSV     = Path("human_evaluation_survey/survey_diaries_key.csv")


#  Shared color palette (real = ODiN grey, synthetic = SynTrav orange,
#  matching every other real-vs-synthetic figure in the thesis)
REAL      = COLOR_ODIN
SYNTHETIC = COLOR_SYNTRAV
GREY      = "#000000"
LIGHT_GREY = "#D3D1C7"
TEXT      = "#2C2C2A"
MUTED     = "#5F5E5A"
NEUTRAL   = "#5F5E5A"


#  Category ordering (present-only subsets are used at plot time).
#  FAMILIARITY_ORDER must match the raw survey values for correct grouping;
#  FAMILIARITY_DISPLAY only shortens the text shown on chart tick labels.
FAMILIARITY_ORDER = [
    "Not familiar",
    "Somewhat familiar",
    "Familiar",
    "Very familiar / I work with ODiN data",
]
FAMILIARITY_DISPLAY = {
    "Very familiar / I work with ODiN data": "Very familiar",
}
AGE_ORDER = ["18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 to 64", "65 to 74", "75+"]

# Warm-copper heatmap palette, matching fig_gallery.py's plot_dist_heatmap
_WARM_COLORS = ["#FAF0DC", "#D4A96A", "#A06830", "#6B3818", "#2D1000"]
_CMAP = mcolors.LinearSegmentedColormap.from_list("warm_copper", _WARM_COLORS)
_NORM = mcolors.Normalize(vmin=0.0, vmax=0.75)

GUESS_MAP = {
    "Synthetic (model-generated)": "synthetic",
    "Real (ODiN survey)":          "real",
}

# Google Forms checkbox options for Q2. One option itself contains a comma
# ("Nothing specific, just a gut feeling"), so a naive ", ".split() would
# shred it — _split_reasons() below protects it before splitting.
REASON_OPTIONS = [
    "Travel mode is unrealistic for this person",
    "Trip timing is implausible",
    "Trip sequence doesn't make sense",
    "Distances are unrealistic",
    "Too many or too few trips",
    "Purposes don't fit the profile",
    "Diary is too clean",
    "Nothing specific, just a gut feeling",
]
_PROTECTED_PHRASE = "Nothing specific, just a gut feeling"
_PLACEHOLDER = "Nothing specific\x00just a gut feeling"


def _split_reasons(raw) -> list[str]:
    """Split a Q2 checkbox answer into its individual reason strings."""
    if pd.isna(raw):
        return []
    s = str(raw).replace(_PROTECTED_PHRASE, _PLACEHOLDER)
    parts = [p.strip().replace("\x00", ", ") for p in s.split(",")]
    return [p for p in parts if p]


#  Data loading
def load_survey_long(
    answers_path: str | Path = ANSWERS_CSV,
    key_path: str | Path = KEY_CSV,
) -> pd.DataFrame:
    """
    Reshape the wide Google Forms export (one row per respondent, 4 repeated
    Q1-Q4 columns per diary) into a tidy one-row-per-(respondent, diary) table.

    Assumes every respondent was shown the diaries in the same fixed order as
    `key_path` (survey_diaries_key.csv), i.e. block i of 4 columns -> diary i.

    Returns columns:
        respondent_id, timestamp, familiarity, age_group,
        diary_id, true_label, plausibility, reasons (list[str]),
        guess_label, correct (bool), comment
    """
    raw = pd.read_csv(answers_path)
    raw = raw[raw.iloc[:, 1].astype(str).str.strip().eq("Agree")].reset_index(drop=True)

    key = pd.read_csv(key_path)
    diary_ids   = key["diary_id"].tolist()
    true_labels = dict(zip(key["diary_id"], key["label"]))
    n_diaries   = len(diary_ids)

    expected_cols = 4 + n_diaries * 4
    if raw.shape[1] != expected_cols:
        raise ValueError(
            f"{answers_path} has {raw.shape[1]} columns, expected {expected_cols} "
            f"(4 metadata + 4 x {n_diaries} diaries from {key_path}). "
            "Did the form change or the diary count differ?"
        )

    rows = []
    for ridx in range(len(raw)):
        respondent_id = ridx + 1
        timestamp   = raw.iat[ridx, 0]
        familiarity = raw.iat[ridx, 2]
        age_group   = raw.iat[ridx, 3]

        for i, diary_id in enumerate(diary_ids):
            base = 4 + i * 4
            plausibility = pd.to_numeric(raw.iat[ridx, base], errors="coerce")
            reasons      = _split_reasons(raw.iat[ridx, base + 1])
            reasons      = [r for r in reasons if r in REASON_OPTIONS]
            guess_raw    = raw.iat[ridx, base + 2]
            guess_label  = GUESS_MAP.get(str(guess_raw).strip())
            comment_raw  = raw.iat[ridx, base + 3]

            true_label = true_labels.get(diary_id)
            rows.append({
                "respondent_id": respondent_id,
                "timestamp":     timestamp,
                "familiarity":   familiarity,
                "age_group":     age_group,
                "diary_id":      diary_id,
                "true_label":    true_label,
                "plausibility":  plausibility,
                "reasons":       reasons,
                "guess_label":   guess_label,
                "correct": (
                    guess_label == true_label
                    if guess_label is not None and true_label is not None
                    else np.nan
                ),
                "comment": "" if pd.isna(comment_raw) else str(comment_raw),
            })

    return pd.DataFrame(rows)


def compute_survey_summary(long_df: pd.DataFrame) -> pd.Series:
    """
    Headline stats for the notebook narrative: overall + per-true-label
    accuracy (guess == true label), and mean plausibility rating per
    true label. NaN rows (unanswered items) are dropped before aggregating.
    """
    sub = long_df.dropna(subset=["correct", "true_label"])

    out = {
        "n_respondents":       long_df["respondent_id"].nunique(),
        "n_ratings":           len(long_df.dropna(subset=["plausibility"])),
        "overall_accuracy":    sub["correct"].mean(),
        "accuracy_on_real":    sub.loc[sub["true_label"] == "real", "correct"].mean(),
        "accuracy_on_synthetic": sub.loc[sub["true_label"] == "synthetic", "correct"].mean(),
        "mean_plausibility_real":      long_df.loc[long_df["true_label"] == "real", "plausibility"].mean(),
        "mean_plausibility_synthetic": long_df.loc[long_df["true_label"] == "synthetic", "plausibility"].mean(),
    }
    return pd.Series(out)


#  plot_respondent_demographics: familiarity + age-group counts
def plot_respondent_demographics(
    long_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel bar chart: respondent familiarity with ODiN, and age group."""
    set_style()
    resp = long_df.drop_duplicates("respondent_id")[["respondent_id", "familiarity", "age_group"]]

    fam_counts = resp["familiarity"].value_counts()
    fam_order  = [f for f in FAMILIARITY_ORDER if f in fam_counts.index] + \
                 [f for f in fam_counts.index if f not in FAMILIARITY_ORDER]
    fam_counts = fam_counts.reindex(fam_order)

    age_counts = resp["age_group"].value_counts()
    age_order  = [a for a in AGE_ORDER if a in age_counts.index] + \
                 [a for a in age_counts.index if a not in AGE_ORDER]
    age_counts = age_counts.reindex(age_order)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    for ax, counts, letter, caption in [
        (axes[0], fam_counts, "a", "Familiarity with ODiN"),
        (axes[1], age_counts, "b", "Age group"),
    ]:
        ax.set_facecolor("white")
        x = np.arange(len(counts))
        ax.bar(x, counts.values, color=NEUTRAL, alpha=0.85)
        for xi, v in zip(x, counts.values):
            ax.text(xi, v + 0.3, str(int(v)), ha="center", va="bottom",
                    fontsize=9, color=TEXT)
        labels = [FAMILIARITY_DISPLAY.get(c, c) for c in counts.index]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=25, ha="right")
        ax.set_ylabel("Respondents", fontsize=9, color=MUTED)
        strip_axes(ax)
        add_caption(ax, letter, caption, fontsize=10)

    fig.suptitle("Survey respondent profile", fontsize=13, color=TEXT, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_confusion_matrix: true label vs guessed label
def plot_confusion_matrix(
    long_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap of true diary label (rows) vs evaluator guess (columns), with
    row-normalised percentages and raw counts annotated in each cell.
    """
    set_style()
    sub = long_df.dropna(subset=["guess_label", "true_label"])

    labels = ["real", "synthetic"]
    counts = pd.crosstab(sub["true_label"], sub["guess_label"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("white")

    frac = pct.values / 100.0
    im = ax.imshow(frac, cmap=_CMAP, norm=_NORM)

    for i in range(len(labels)):
        for j in range(len(labels)):
            rgba       = _CMAP(_NORM(frac[i, j]))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color  = "white" if brightness < 0.50 else "#2a1000"
            ax.text(
                j, i, f"{pct.values[i, j]:.0f}%\n(n={counts.values[i, j]})",
                ha="center", va="center", fontsize=10,
                color=txt_color, fontweight="bold",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["Guessed real", "Guessed synthetic"], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(["True real", "True synthetic"], fontsize=9)
    ax.set_title("Evaluator judgement accuracy", fontsize=12, color=TEXT, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("Row %", fontsize=9, color=MUTED)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t * 100:.0f}"))

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_plausibility_distribution: rating histogram, real vs synthetic
def plot_plausibility_distribution(
    long_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart of the 1-5 plausibility rating share, real vs synthetic diaries."""
    set_style()
    sub = long_df.dropna(subset=["plausibility", "true_label"])
    ratings = [1, 2, 3, 4, 5]

    real_pct = (
        sub.loc[sub["true_label"] == "real", "plausibility"]
        .value_counts(normalize=True).reindex(ratings, fill_value=0) * 100
    )
    syn_pct = (
        sub.loc[sub["true_label"] == "synthetic", "plausibility"]
        .value_counts(normalize=True).reindex(ratings, fill_value=0) * 100
    )

    x = np.arange(len(ratings))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor("white")

    bars_real = ax.bar(x - width / 2, real_pct.values, width, color=REAL, alpha=0.90, label="Real diaries")
    bars_syn  = ax.bar(x + width / 2, syn_pct.values,  width, color=SYNTHETIC, alpha=0.85, label="Synthetic diaries")
    ax.bar_label(bars_real, fmt="%.0f%%", padding=2, fontsize=8, color=MUTED)
    ax.bar_label(bars_syn,  fmt="%.0f%%", padding=2, fontsize=8, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(ratings)
    ax.set_xlabel("Plausibility rating (1 = implausible, 5 = very plausible)", fontsize=9, color=MUTED)
    ax.set_ylabel("Share of ratings (%)", fontsize=9, color=MUTED)
    ax.set_ylim(0, max(real_pct.max(), syn_pct.max()) * 1.25)
    strip_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax.set_title("Plausibility ratings: real vs synthetic diaries", fontsize=12, color=TEXT, pad=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_accuracy_by_familiarity: guess accuracy vs self-reported ODiN familiarity
def plot_accuracy_by_familiarity(
    long_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of guess accuracy rate by respondent's self-reported ODiN familiarity."""
    set_style()
    sub = long_df.dropna(subset=["correct", "familiarity"])

    acc = sub.groupby("familiarity")["correct"].mean() * 100
    n   = sub.groupby("familiarity").size()
    order = [f for f in FAMILIARITY_ORDER if f in acc.index] + \
            [f for f in acc.index if f not in FAMILIARITY_ORDER]
    acc, n = acc.reindex(order), n.reindex(order)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(order))
    bars = ax.bar(x, acc.values, color=NEUTRAL, alpha=0.85)
    for xi, v, ni in zip(x, acc.values, n.values):
        ax.text(xi, v + 1.5, f"{v:.0f}%\n(n={ni})", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)

    ax.axhline(50, color=GREY, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(len(order) - 0.4, 51.5, "Chance level", fontsize=8, color=MUTED, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILIARITY_DISPLAY.get(f, f) for f in order],
                       fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("Guess accuracy (%)", fontsize=9, color=MUTED)
    ax.set_ylim(0, max(acc.max() + 15, 60))
    strip_axes(ax)

    ax.set_title("Guess accuracy by self-reported ODiN familiarity", fontsize=12, color=TEXT, pad=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_per_diary_accuracy: which diaries fooled evaluators most
def plot_per_diary_accuracy(
    long_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of the correct-guess rate per diary, sorted ascending
    (most-often-misjudged diaries at top), coloured by the diary's true label.
    """
    set_style()
    sub = long_df.dropna(subset=["correct"])

    stats = (
        sub.groupby(["diary_id", "true_label"])["correct"]
        .mean().mul(100).reset_index()
        .sort_values("correct")
        .reset_index(drop=True)
    )
    colors = [REAL if lbl == "real" else SYNTHETIC for lbl in stats["true_label"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    y = np.arange(len(stats))
    ax.barh(y, stats["correct"].values, color=colors, alpha=0.88)
    for yi, v in zip(y, stats["correct"].values):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=8, color=TEXT)

    ax.axvline(50, color=GREY, linewidth=1.2, linestyle="--", zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(stats["diary_id"], fontsize=8.5)
    ax.set_xlabel("Evaluators correctly identified this diary (%)", fontsize=9, color=MUTED)
    ax.set_xlim(0, 100)
    strip_axes(ax)

    handles = [
        mpatches.Patch(color=REAL, alpha=0.88, label="True label: real"),
        mpatches.Patch(color=SYNTHETIC, alpha=0.88, label="True label: synthetic"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    ax.set_title("Per-diary judgement accuracy", fontsize=12, color=TEXT, pad=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


#  plot_implausibility_reasons: Q2 reason frequency, split by true label
def plot_implausibility_reasons(
    long_df: pd.DataFrame,
    top_n: int = 8,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Stacked horizontal bar chart of how often each Q2 implausibility reason
    was cited, split by whether the diary being judged was actually real or
    synthetic (reveals which "tells" are genuine artifacts vs false alarms).
    """
    set_style()
    exploded = long_df.explode("reasons")
    exploded = exploded[exploded["reasons"].isin(REASON_OPTIONS)]

    counts = (
        exploded.groupby(["reasons", "true_label"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["real", "synthetic"], fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total").tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    y = np.arange(len(counts))
    ax.barh(y, counts["real"].values, color=REAL, alpha=0.88, label="Cited on real diaries")
    ax.barh(y, counts["synthetic"].values, left=counts["real"].values,
            color=SYNTHETIC, alpha=0.85, label="Cited on synthetic diaries")

    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel("Times cited", fontsize=9, color=MUTED)
    strip_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax.set_title("Reasons cited for a low plausibility rating", fontsize=12, color=TEXT, pad=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig
