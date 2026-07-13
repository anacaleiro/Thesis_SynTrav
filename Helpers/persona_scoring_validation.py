import json, time, os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from groq import RateLimitError
from llm_config.llm_config import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_split_prompt(base_group, summary, dim):
    return f"""You are a travel behaviour analyst designing synthetic traveller
   personas. You have a travel survey with a preliminary data analysis that has identified a base group of travellers which revealed key insights:

  <INPUT 1> Group: {base_group}
  - Persons: {summary['n_persons']} | Trips: {summary['n_trips']} | Trips/person:
   {summary['trips_per_person']}
  - Mode share: {json.dumps(summary['mode_share'])}
  - Trip purpose: {json.dumps(summary['purpose_share'])}
  - Departure time: {json.dumps(summary['dep_time_class_share'])}
  - Distance class: {json.dumps(summary['distance_class_share'])}

  Your goal is to capture the complex diversity in human mobility behaviors present in these 
  data across the base groups, aiming to uncover the subtle differences and unique aspects of travel 
  patterns among various subgroups.

  <INPUT 2> Currently classified on: Activity status × Age group × Day type

  <INPUT 3> Candidate split dimension: {dim['name']}
  Distribution within group:
  {json.dumps(summary['candidate_distributions'][dim['name']])}

  To gain a more refined understanding of the diversity in travel patterns, please evaluate 
  whether it is necessary to further segment the groups based on another specific dimension.  
  Essentially, does splitting this group by {dim['name']} capture meaningfully different
  travel behaviours?
  Score 1-10 (1=no difference expected, 10=strong recommendation for additional segmentation).
  Don't be hesitant  to give a low score if the dimension doesn't add value, or a high score if otherwise.

  Reply ONLY with valid JSON: {{"score": <int 1-10>, "reason": "<one
  sentence>"}}"""


def parse_score(raw):
      try:
          result = json.loads(raw)
          return int(result['score']), result.get('reason', '')
      except (json.JSONDecodeError, KeyError):
          match = re.search(r'"score"\s*:\s*(\d+)', raw)
          return (int(match.group(1)), raw) if match else (None, raw)


def score_dim(base_group, summary, dim, provider='groq', delay=2.1, retries=5):
      prompt = build_split_prompt(base_group, summary, dim)
      for attempt in range(retries):
          try:
              raw = call_llm(prompt, provider=provider, max_tokens=80)
              score, reason = parse_score(raw)
              time.sleep(delay)
              return score, reason
          except RateLimitError:
              wait = 15 * (attempt + 1)
              print(f"  Rate limit — waiting {wait}s...")
              time.sleep(wait)
      return None, 'max retries exceeded'


def run_scoring_pass(base_groups, group_summaries, candidate_dims,scores_file, provider='groq', delay=2.1):
    scores = json.load(open(scores_file)) if os.path.exists(scores_file) else {}
    total = len(base_groups) * len(candidate_dims)
    done = sum(len(v) for v in scores.values())

    for base_group in base_groups:
        scores.setdefault(base_group, {})
        for dim in candidate_dims:
            if dim['name'] in scores[base_group]:
                continue
            score, reason = score_dim(base_group, group_summaries[base_group],
                                        dim, provider, delay)
            scores[base_group][dim['name']] = {'score': score, 'reason': reason}
            done += 1
            print(f"[{done}/{total}] {base_group} | {dim['name']} → {score}")
            with open(scores_file, 'w') as f:
                json.dump(scores, f)
                  
    return scores

### Scoring Evaluation

OUTCOME_COLS = {
    "mode":     "Main mode of transport travel",
    "distance": "Travel distance class in the Netherlands",
}



# Build the ranking + split-variable tables from the real scoring run

def compute_dim_ranking(scores):
    """Mean LLM score per dimension across all scored groups."""
    dim_scores = {}
    for group, dims in scores.items():
        for dim, info in dims.items():
            if info.get("score") is not None:
                dim_scores.setdefault(dim, []).append(info["score"])
    return {dim: float(np.mean(vals)) for dim, vals in dim_scores.items()}


def build_split_vars(scores, dim_col, used_label, used_col, top_n=3):
    """
    Rank every dimension the LLM actually scored (from `scores`) and map it
    to its ODiN column (via `dim_col`), so the comparison always reflects
    the current candidate_dims / scores.json rather than a stale snapshot.

    Parameters
    ----------
    used_label : the candidate_dims name of the variable actually applied
                 to split personas (e.g. 'Income').
    used_col   : the column actually used for the split (may differ from
                 dim_col[used_label], e.g. 'income_level_split' vs the raw
                 'income_level').

    Returns
    -------
    all_vars : dict  label -> (col, llm_score) for every scored dim, ranked
    top_vars : dict  label -> (col, llm_score) for the top `top_n` dims the
               LLM ranked above the used dim, plus the used dim itself
    """
    ranking = compute_dim_ranking(scores)
    ranked = sorted(ranking.items(), key=lambda kv: kv[1], reverse=True)

    all_vars = {}
    for dim, score in ranked:
        col = used_col if dim == used_label else dim_col.get(dim)
        if col is None:
            continue
        label = f"{dim} (used)" if dim == used_label else dim
        all_vars[label] = (col, score)

    used_key = f"{used_label} (used)"
    top_unselected = [lbl for lbl in all_vars if lbl != used_key][:top_n]
    top_vars = {lbl: all_vars[lbl] for lbl in top_unselected}
    top_vars[used_key] = all_vars[used_key]
    return all_vars, top_vars



# Internal helpers
def _pairwise_jsd(vecs):
    """Mean pairwise JSD (base-2, squared) across a list of probability vectors."""
    if len(vecs) < 2:
        return 0.0
    return np.mean([
        jensenshannon(a, b, base=2) ** 2
        for a, b in combinations(vecs, 2)
    ])


def _distribution_vecs(sub_dfs, col, vocabulary):
    """Normalised count vector per sub-DataFrame."""
    vecs = []
    for sdf in sub_dfs:
        counts = sdf[col].value_counts()
        vec = np.array([counts.get(c, 0) for c in vocabulary], dtype=float)
        if vec.sum() > 0:
            vecs.append(vec / vec.sum())
    return vecs


def _attempt_split(base_df, split_col, min_persons, person_col):
    """
    Same split logic as original persona construction:
    if ANY value produces < min_persons unique persons, return the group unsplit.
    Returns a list of sub-DataFrames (length 1 = unsplit).
    """
    valid = base_df[base_df[split_col].notna()]
    cells = [(val, sdf) for val, sdf in valid.groupby(split_col)]
    if not cells:
        return [base_df]
    if any(sdf[person_col].nunique() < min_persons for _, sdf in cells):
        return [base_df]
    return [sdf for _, sdf in cells]


# Core metric: JSD + trip-frequency variance
def compute_split_metrics(
    odin_train,
    split_col,
    min_persons=15,
    person_col="Person_index",
    base_group_col="base_group",
):
    """
    Per base group: split by split_col and compute three differentiation metrics.
    Returns a DataFrame with one row per base group.
    """
    rows = []
    for base_group, base_df in odin_train.groupby(base_group_col):
        sub_dfs  = _attempt_split(base_df, split_col, min_persons, person_col)
        was_split = len(sub_dfs) > 1

        if not was_split:
            rows.append({
                "base_group":    base_group,
                "n_splits":      1,
                "trip_freq_var": 0.0,
                "modal_jsd":     0.0,
                "distance_jsd":  0.0,
                "was_split":     False,
            })
            continue

        # weighted variance of per-subgroup mean trips/person
        means, weights = [], []
        for sdf in sub_dfs:
            n_p = sdf[person_col].nunique()
            means.append(len(sdf) / n_p)
            weights.append(n_p)
        w = np.array(weights, dtype=float)
        m = np.array(means,   dtype=float)
        w_mean        = np.average(m, weights=w)
        trip_freq_var = float(np.average((m - w_mean) ** 2, weights=w))

        mode_vocab = sorted(base_df[OUTCOME_COLS["mode"]].dropna().unique())
        dist_vocab = sorted(base_df[OUTCOME_COLS["distance"]].dropna().unique())

        rows.append({
            "base_group":    base_group,
            "n_splits":      len(sub_dfs),
            "trip_freq_var": trip_freq_var,
            "modal_jsd":     _pairwise_jsd(
                                 _distribution_vecs(sub_dfs, OUTCOME_COLS["mode"], mode_vocab)),
            "distance_jsd":  _pairwise_jsd(
                                 _distribution_vecs(sub_dfs, OUTCOME_COLS["distance"], dist_vocab)),
            "was_split":     True,
        })
    return pd.DataFrame(rows)


# Main validation entry point

def validate_llm_persona_scoring(odin_train, scores, dim_col, used_label="Income",
                                  used_col="income_level_split", top_n=3,
                                  min_persons=15, plot=True, save_path=None):
    """
    Compare the used split variable against the top `top_n` LLM-ranked
    dimensions it was NOT used in favour of, using JSD and trip-frequency
    variance — each variable evaluated in its real, actually-usable form
    (no artificial binarisation).

    A "Base groups split" count of 0/N means the variable's categories are
    too sparse to clear min_persons in any base group, so it was never
    actually applied — that is itself part of the answer (a top-ranked
    variable can be practically unusable), not a sign of "no signal".

    Returns
    -------
    summary : pd.DataFrame  — one row per split variable, mean metrics
    details : dict[str, pd.DataFrame]  — per-base-group results per variable
    """
    _, split_vars = build_split_vars(scores, dim_col, used_label, used_col, top_n=top_n)

    missing = [col for col, _ in split_vars.values() if col not in odin_train.columns]
    if missing:
        raise ValueError(f"Missing columns in odin_train: {missing}")

    details, summary_rows = {}, []

    for label, (col, llm_score) in split_vars.items():
        df_m   = compute_split_metrics(odin_train, split_col=col, min_persons=min_persons)
        details[label] = df_m
        n_split = int(df_m["was_split"].sum())
        summary_rows.append({
            "Split variable":     label,
            "LLM score":          round(llm_score, 2),
            "Base groups split":  f"{n_split} / {len(df_m)}",
            "Trip freq variance": round(df_m["trip_freq_var"].mean(), 5),
            "Modal split JSD":    round(df_m["modal_jsd"].mean(), 5),
            "Distance JSD":       round(df_m["distance_jsd"].mean(), 5),
        })

    summary = pd.DataFrame(summary_rows)

    print("=" * 72)
    print("LLM Persona-Scoring Validation")
    print("(higher = more travel-behaviour differentiation between personas)")
    print("=" * 72)
    print(summary.to_string(index=False))
    print()

    _interpret(summary, used_label)

    if plot:
        _plot_comparison(summary, save_path=save_path)

    return summary, details


# Reporting helpers

def _interpret(summary, used_label):
    used_key = f"{used_label} (used)"
    metrics    = ["Trip freq variance", "Modal split JSD", "Distance JSD"]
    used_row   = summary[summary["Split variable"] == used_key].iloc[0]
    other_vars = [v for v in summary["Split variable"] if v != used_key]

    zero_split = summary.loc[summary["Base groups split"].str.startswith("0 /"), "Split variable"].tolist()
    if zero_split:
        print("Note: the following variables never actually split any base group")
        print("(too many sparse categories to clear min_persons) — their 0.00000")
        print("metrics mean 'could not be applied', not 'no behavioural difference':")
        print(f"  {', '.join(zero_split)}")
        print()

    print("Interpretation:")
    for m in metrics:
        ranked    = summary.sort_values(m, ascending=False)["Split variable"].tolist()
        used_val  = used_row[m]
        used_rank = ranked.index(used_key) + 1
        better = [v for v in other_vars if ranked.index(v) < used_rank]
        worse  = [v for v in other_vars if ranked.index(v) > used_rank]

        if not better:
            print(f"  [{m}] {used_key} ranks 1st ({used_val:.5f}) — "
                  "outperforms all LLM top picks.")
        else:
            beaten_str = ", ".join(
                f"{v} ({summary.loc[summary['Split variable']==v, m].values[0]:.5f})"
                for v in better
            )
            print(f"  [{m}] {used_key} ranks {used_rank}/{len(ranked)} ({used_val:.5f}). "
                  f"Outperformed by: {beaten_str}.")
        if worse:
            worse_str = ", ".join(
                f"{v} ({summary.loc[summary['Split variable']==v, m].values[0]:.5f})"
                for v in worse
            )
            print(f"           {used_key} beats: {worse_str}.")
    print()


def _plot_comparison(summary, save_path=None):
    metrics = ["Trip freq variance", "Modal split JSD", "Distance JSD"]
    labels  = summary["Split variable"].tolist()
    n       = len(labels)
    x       = np.arange(len(metrics))
    width   = 0.8 / n
    cmap    = plt.cm.copper
    colors  = [cmap(i / max(n - 1, 1)) for i in range(n)]

    with plt.rc_context({'font.family': 'serif'}):
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - n / 2 + 0.5) * width
            vals   = summary.loc[summary["Split variable"] == label, metrics].values.flatten()
            bars   = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.88)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0001, f"{v:.4f}",
                        ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Mean between-subgroup metric")
        ax.set_title("LLM Persona-Scoring Validation")
        ax.legend(frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# All-candidate comparison

def validate_all_candidates(odin_train, scores, dim_col, used_label="Income",
                             used_col="income_level_split", min_persons=15, plot=True,
                             save_path=None):
    """
    Run JSD metrics across every LLM-scored candidate variable and display
    them ranked alongside their original LLM scores.

    Useful for checking whether the LLM ranking correlates with the
    empirical differentiation metrics (Spearman ρ), and where the used
    variable sits in the full picture.

    Returns
    -------
    all_df : pd.DataFrame  — one row per variable, all metrics + LLM score
    """
    from scipy.stats import spearmanr

    all_vars, _ = build_split_vars(scores, dim_col, used_label, used_col, top_n=len(dim_col))
    used_key = f"{used_label} (used)"

    missing_cols = [col for col, _ in all_vars.values() if col not in odin_train.columns]
    if missing_cols:
        print(f"Skipping (column missing in odin_train): {missing_cols}")

    rows = []
    for label, (col, llm_score) in all_vars.items():
        if col not in odin_train.columns:
            continue
        jsd_df = compute_split_metrics(odin_train, split_col=col, min_persons=min_persons)
        rows.append({
            "Variable":           label,
            "LLM score":          llm_score,
            "LLM rank":           None,          # filled below
            "Base grps split":    int(jsd_df["was_split"].sum()),
            "Trip freq var":      round(jsd_df["trip_freq_var"].mean(), 5),
            "Modal JSD":          round(jsd_df["modal_jsd"].mean(),     5),
            "Distance JSD":       round(jsd_df["distance_jsd"].mean(),  5),
            "used":               label == used_key,
        })

    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values("LLM score", ascending=False).reset_index(drop=True)
    all_df["LLM rank"] = range(1, len(all_df) + 1)
    all_df["Modal JSD rank"] = all_df["Modal JSD"].rank(ascending=False).astype(int)

    # Spearman correlation between LLM rank and Modal JSD rank
    rho, pval = spearmanr(all_df["LLM rank"], all_df["Modal JSD rank"])

    print("=" * 90)
    print(f"All-candidate validation — LLM score vs empirical metrics (all {len(all_df)} dimensions)")
    print("=" * 90)
    display_cols = ["Variable", "LLM score", "LLM rank", "Modal JSD rank",
                    "Base grps split", "Modal JSD", "Distance JSD"]
    print(all_df[display_cols].to_string(index=False))
    print()
    used_row  = all_df[all_df["used"]]
    used_llmr = used_row["LLM rank"].values[0]
    used_jsdr = used_row["Modal JSD rank"].values[0]
    print(f"  {used_key}: LLM rank {used_llmr}/{len(all_df)}, Modal JSD rank {used_jsdr}/{len(all_df)}")
    print(f"  Spearman ρ between LLM rank and Modal JSD rank: {rho:.3f} (p={pval:.3f})")
    if abs(rho) < 0.3:
        print("  → Weak correlation: LLM scores and empirical JSD rank variables differently.")
    elif abs(rho) < 0.6:
        print("  → Moderate correlation: LLM scoring partially tracks empirical differentiation.")
    else:
        print("  → Strong correlation: LLM scoring aligns well with empirical differentiation.")
    print()

    if plot:
        _plot_all_candidates(all_df, save_path=save_path)

    return all_df


def _plot_all_candidates(all_df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.copper

    for ax, metric, title in [
        (axes[0], "Distance JSD", "Distance split JSD"),
        (axes[1], "Modal JSD",    "Modal split JSD"),
    ]:
        df_sorted = all_df.sort_values(metric, ascending=True)
        n      = len(df_sorted)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        # highlight used bar with an edge
        edge_colors = ["red" if u else "none" for u in df_sorted["used"]]
        edge_widths = [2.0  if u else 0.0  for u in df_sorted["used"]]

        bars = ax.barh(df_sorted["Variable"], df_sorted[metric],
                       color=colors, edgecolor=edge_colors,
                       linewidth=edge_widths, alpha=0.88)
        for bar, v in zip(bars, df_sorted[metric]):
            ax.text(v + max(df_sorted[metric]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.5f}", va="center", fontsize=7)
        ax.set_xlabel(title)
        ax.set_title(f"All {n} candidates — {title}\n(red outline = used variable)")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Per-base-group drill-down
def show_base_group_detail(details, base_group):
    """Print per-split-variable metrics for a single base group."""
    print(f"Base group: {base_group}")
    print(f"{'Variable':<35} {'n_splits':>8} {'trip_freq_var':>14} "
          f"{'modal_jsd':>10} {'distance_jsd':>13}")
    print("-" * 83)
    for label, df in details.items():
        row = df[df["base_group"] == base_group]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"{label:<35} {int(r['n_splits']):>8} {r['trip_freq_var']:>14.5f} "
              f"{r['modal_jsd']:>10.5f} {r['distance_jsd']:>13.5f}")
