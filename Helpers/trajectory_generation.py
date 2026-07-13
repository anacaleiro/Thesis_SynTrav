import json, os, re, random, time, sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import RateLimitError as GroqRateLimitError
from openai import RateLimitError as OpenAIRateLimitError
from llm_config.llm_config import call_llm
from prompt_template.generation_prompt import (
    MOTIVATIONAL_SUMMARY_PROMPT,
    DAILY_PLAN_PROMPT,
    RECURSIVE_REASONING_PROMPT,
    ATYPICAL_ZERO_TRIP_PLAN,
)
from Helpers.atypical_travelers import should_be_atypical, ATYPICAL_DESCRIPTIONS

# In the standard pipeline, departure times come from plan step parsing.
WEEKDAY_CHECKPOINTS = ["07:00", "08:30", "10:00", "11:30", "13:00", "15:00", "17:00", "18:30", "20:00"]
WEEKEND_CHECKPOINTS = ["09:00", "10:30", "12:00", "14:00", "16:00", "18:00", "20:30"]

# Controlled vocabulary for DESTINATION field, aligned to ODiN purpose categories
# and structured for downstream OSM/BGRI spatial assignment.
DESTINATION_TYPES = [
    "workplace",
    "educational institution",
    "supermarket or shop",
    "sports or recreation facility",
    "social or family visit destination",
    "healthcare or personal service",
    "park or nature area",
    "restaurant or café",
    "transit hub",
    "home",
    "other",
]

# Closed set of movement verbs that MUST prefix travel steps in the daily plan.
# The keyword filter is a simple prefix match: no regex needed.
MOVEMENT_VERBS = [
    "Travel to",
    "Walk to",
    "Cycle to",
    "Drive to",
    "Take the train to",
    "Return home by",
]

# Maps checkpoint / plan-step time (HH:MM) → ODiN departure time class label.
# Assigned deterministically in the pipeline rather than asking the LLM to infer it.
_DEP_CLASS_BREAKPOINTS = [
    (360,  "Before 6:00 AM"),
    (420,  "6:00 AM to 7:00 AM"),
    (480,  "7:00 AM to 8:00 AM"),
    (540,  "8:00 AM to 9:00 AM"),
    (720,  "9am to 12pm"),
    (780,  "12 noon to 1 p.m"),
    (840,  "1:00 PM to 2:00 PM"),
    (960,  "2:00 PM to 4:00 PM"),
    (1020, "4:00 PM to 5:00 PM"),
    (1080, "5:00 PM to 6:00 PM"),
    (1140, "6:00 PM to 7:00 PM"),
    (1200, "7:00 PM to 8:00 PM"),
]


def _time_to_dep_class(time_str: str) -> str:
    """Map 'HH:MM' to the ODiN departure time class label."""
    try:
        h, m = map(int, time_str.split(":"))
        mins = h * 60 + m
    except (ValueError, AttributeError):
        return "8 p.m. to midnight"
    for threshold, label in _DEP_CLASS_BREAKPOINTS:
        if mins < threshold:
            return label
    return "8 p.m. to midnight"


def _call_with_retry(prompt, provider, max_tokens, retries=5, delay=2.1):
    for attempt in range(retries):
        try:
            result = call_llm(prompt, provider=provider, max_tokens=max_tokens)
            time.sleep(delay)
            return result
        except (GroqRateLimitError, OpenAIRateLimitError):
            wait = 15 * (attempt + 1)
            print(f"    Rate limit — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {type(e).__name__}: {e}")
            time.sleep(5)
    return None


def _top_shares(share_dict, top_n=6):
    items = sorted(share_dict.items(), key=lambda x: -x[1])[:top_n]
    return ", ".join(f"{k} ({v}%)" for k, v in items)


def _top_keys(share_dict, top_n=6):
    return ", ".join(k for k, _ in sorted(share_dict.items(), key=lambda x: -x[1])[:top_n])


# Maximum plausible distance (km, upper bound of class) per slow mode.
# Car / train / bus are uncapped — all distance classes are physically plausible.
_MODE_MAX_KM = {
    "On foot":             5.0,
    "Non-electric bicycle": 20.0,
    "Electric bike":       30.0,
    "Moped":               50.0,
}

# Midpoint (km) of each ODiN distance class — used to compute median/P80 summary.
_DIST_MIDPOINT_KM = {
    "0.1 to 0.5 km":   0.30,
    "0.5 to 1.0 km":   0.75,
    "1.0 to 2.5 km":   1.75,
    "2.5 to 3.7 km":   3.10,
    "3.7 to 5.0 km":   4.35,
    "5.0 to 7.5 km":   6.25,
    "7.5 to 10 km":    8.75,
    "10 to 15 km":    12.50,
    "15 to 20 km":    17.50,
    "20 to 30 km":    25.00,
    "30 to 40 km":    35.00,
    "40 to 50 km":    45.00,
    "50 to 75 km":    62.50,
    "75 to 100 km":   87.50,
    "100 km or more": 125.00,
}

# Upper bound (km) of each ODiN distance class label.
_DIST_UPPER_KM = {
    "0.1 to 0.5 km":   0.5,
    "0.5 to 1.0 km":   1.0,
    "1.0 to 2.5 km":   2.5,
    "2.5 to 3.7 km":   3.7,
    "3.7 to 5.0 km":   5.0,
    "5.0 to 7.5 km":   7.5,
    "7.5 to 10 km":   10.0,
    "10 to 15 km":    15.0,
    "15 to 20 km":    20.0,
    "20 to 30 km":    30.0,
    "30 to 40 km":    40.0,
    "40 to 50 km":    50.0,
    "50 to 75 km":    75.0,
    "75 to 100 km":  100.0,
    "100 km or more": 999.0,
}


def _sample_mode(persona: dict) -> str | None:
    """Sample a mode from the group's ODiN empirical mode distribution."""
    ms = persona.get("mode_share", {})
    if not ms:
        return None
    keys = list(ms.keys())
    weights = [ms[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def _compute_population_mode_share(persona_objects: list[dict]) -> dict:
    """
    Compute population-weighted marginal mode share across all persona groups.
    Used for the no-persona ablation so mode sampling does not carry hidden
    group-level conditioning. Weights each group by n_persons.
    """
    combined = {}
    total_persons = 0
    for p in persona_objects:
        n = p.get("n_persons", 1)
        total_persons += n
        for mode, share in p.get("mode_share", {}).items():
            combined[mode] = combined.get(mode, 0) + share * n
    if total_persons == 0:
        return combined
    return {k: v / total_persons for k, v in combined.items()}


def _compute_population_distance_share(persona_objects: list[dict]) -> dict:
    """
    Compute population-weighted marginal distance share across all persona groups.
    Used for the no-persona ablation so distance prompting does not carry hidden
    group-level conditioning. Weights each group by n_persons.
    """
    combined = {}
    total_persons = 0
    for p in persona_objects:
        n = p.get("n_persons", 1)
        total_persons += n
        for cls, share in p.get("distance_share", {}).items():
            combined[cls] = combined.get(cls, 0) + share * n
    if total_persons == 0:
        return combined
    return {k: v / total_persons for k, v in combined.items()}


def _compute_population_purpose_share(persona_objects: list[dict]) -> dict:
    """
    Compute population-weighted marginal purpose share across all persona groups.
    Used for the no-persona ablation so purpose prompting does not carry hidden
    group-level conditioning. Weights each group by n_persons.
    """
    combined = {}
    total_persons = 0
    for p in persona_objects:
        n = p.get("n_persons", 1)
        total_persons += n
        for purpose, share in p.get("purpose_share", {}).items():
            combined[purpose] = combined.get(purpose, 0) + share * n
    if total_persons == 0:
        return combined
    return {k: v / total_persons for k, v in combined.items()}


def _sample_distance(persona: dict, mode: str | None = None) -> str | None:
    """
    Sample a distance class from the group's ODiN empirical distribution.
    When mode is provided, distance classes whose upper bound exceeds the
    per-mode physical ceiling are removed before sampling. The remaining
    original ODiN weights are passed as-is to random.choices, which treats
    them as relative — so empirical proportions are preserved without an
    explicit renormalization step.
    Fallback to the full distribution if all classes exceed the cap.
    """
    dist = persona.get("distance_share", {})
    if not dist:
        return None

    max_km = _MODE_MAX_KM.get(mode) if mode else None

    keys, weights = [], []
    for k, w in dist.items():
        if max_km is not None and _DIST_UPPER_KM.get(k, 999.0) > max_km:
            continue
        keys.append(k)
        weights.append(w)

    if not keys:  # entire distribution above cap — fall back to full pool
        keys    = list(dist.keys())
        weights = [dist[k] for k in keys]

    return random.choices(keys, weights=weights, k=1)[0]


def format_persona_profile(persona, use_persona=True):
    if not use_persona:
        return ""
    demos = persona.get("demographics", {})
    return (
        f"Group: {persona['group_key']}\n"
        f"Activity status: {persona['activity_status']} | "
        f"Age class: {persona['age_class']} | "
        f"Income: {persona.get('income_level') or 'not specified'}\n"
        f"\nDemographics:\n"
        f"  Gender:       {_top_shares(demos.get('gender', {}), 2)}\n"
        f"  Household:    {_top_shares(demos.get('household', {}), 3)}\n"
        f"  Urbanisation: {_top_shares(demos.get('urbanisation', {}), 3)}\n"
        f"  Education:    {_top_shares(demos.get('education', {}), 3)}\n"
        f"  Car ownership:{_top_shares(demos.get('car_ownership', {}), 3)}\n"
    )


def _sample_individual_budget(persona) -> int:
    """
    Sample a per-person trip budget from the group mean via Poisson distribution.
    Clipped to [1, 10]; minimum 1 because zero-trip days are handled separately
    by the atypical mechanism.
    """
    avg = persona['n_trips'] / persona['n_persons']
    return max(1, min(10, int(np.random.poisson(max(0.5, avg)))))



# Plan parsing helpers

def _parse_plan_steps(daily_plan: str) -> list[tuple[str, str]]:
    """Extract [(time_str, activity_text), ...] from a [HH:MM] Activity plan."""
    steps = []
    for line in daily_plan.splitlines():
        m = re.match(r'\[(\d{1,2}:\d{2})\]\s+(.+)', line.strip())
        if m:
            steps.append((m.group(1).strip(), m.group(2).strip()))
    return steps


def _is_travel_step(activity: str) -> bool:
    """True iff the activity string starts with one of the closed movement verbs."""
    low = activity.lower()
    return any(low.startswith(v.lower()) for v in MOVEMENT_VERBS)


def _is_return_home_step(activity: str) -> bool:
    """True iff this is the closing leg of a trip chain (Return home by ...)."""
    return activity.lower().startswith("return home by")



# LLM call wrappers

def _generate_motivational_summary(persona, pattern, provider, use_persona=True):
    profile = format_persona_profile(persona, use_persona)
    prompt = MOTIVATIONAL_SUMMARY_PROMPT.format(
        persona_profile=profile,
        mobility_pattern=pattern,
        day_type=persona.get("day_type", "weekday"),
    )
    return _call_with_retry(prompt, provider, max_tokens=350)


def _generate_daily_plan(persona, pattern, provider, motivational_summary="", individual_trip_budget=3, zero_trip=False, use_daily_plan=True, use_persona=True):
    if not use_daily_plan:
        return "No daily plan — reason directly from the persona profile and mobility patterns at each checkpoint."
    profile = format_persona_profile(persona, use_persona)
    if zero_trip:
        prompt = ATYPICAL_ZERO_TRIP_PLAN.format(
            motivational_summary=motivational_summary or "",
            persona_profile=profile,
            mobility_pattern=pattern,
            day_type=persona.get("day_type", "weekday"),
        )
    else:
        prompt = DAILY_PLAN_PROMPT.format(
            motivational_summary=motivational_summary or "",
            individual_trip_budget=individual_trip_budget,
            persona_profile=profile,
            mobility_pattern=pattern,
            day_type=persona.get("day_type", "weekday"),
        )
    return _call_with_retry(prompt, provider, max_tokens=700)


def _run_reasoning_step(
    current_time, plan_step, daily_plan, earlier_steps,
    persona, pattern, provider,
    motivational_summary="", use_persona=True, assigned_mode=None,
    purpose_share=None,
):
    earlier_text = "\n".join(earlier_steps) if earlier_steps else "None yet — day just started."
    profile = format_persona_profile(persona, use_persona)
    purp = purpose_share if purpose_share is not None else persona.get("purpose_share", {})
    prompt = RECURSIVE_REASONING_PROMPT.format(
        current_time=current_time,
        plan_step=plan_step,
        motivational_summary=motivational_summary or "",
        daily_plan=daily_plan,
        earlier_schedule=earlier_text,
        persona_profile=profile,
        mobility_pattern=pattern,
        assigned_mode=assigned_mode or "Not specified",
        available_purposes=_top_keys(purp, 6),
        available_destinations=", ".join(DESTINATION_TYPES),
    )
    return _call_with_retry(prompt, provider, max_tokens=300)


def _parse_step(raw, checkpoint, plan_step="", inherited_distance=None, inherited_purpose=None):
    """
    Parse one RECURSIVE_REASONING_PROMPT response.
    The loop is only called for travel steps, so TRAVEL is always implicitly YES.
    PLAUSIBLE handles the sole exception: within-day logical contradictions.
    DEPARTURE_TIME is assigned deterministically from the checkpoint time.

    inherited_distance: if provided (return trips), overrides the LLM's DISTANCE
    output so that both legs of a chain share the same distance class.
    Return-home steps are unconditionally plausible — the LLM cannot suppress them.
    """
    is_return = _is_return_home_step(plan_step)
    result = {
        "time":                  checkpoint,
        "plan_step":             plan_step,
        "travel":                True,
        "plausible":             True,       # default; overridden below for outbound steps only
        "implausibility_reason": None,
        "departure_time_class":  _time_to_dep_class(checkpoint),
    }
    if not raw:
        # Return trips are always recorded even on LLM failure — use inherited fields.
        if not is_return:
            result["plausible"] = False
            result["implausibility_reason"] = "LLM call failed"
        return result

    def _extract(field):
        m = re.search(rf"^{field}:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    # Return trips are unconditionally plausible — never let the LLM suppress them.
    if not is_return:
        plausible_val = (_extract("PLAUSIBLE") or "YES").strip().upper()
        result["plausible"] = (plausible_val == "YES")
        result["implausibility_reason"] = (
            _extract("IMPLAUSIBILITY_REASON") if not result["plausible"] else None
        )

    result["motivation"]  = _extract("MOTIVATION") or None
    result["mode"]        = _extract("MODE")
    result["purpose"]     = inherited_purpose or _extract("PURPOSE")
    result["destination"] = _extract("DESTINATION")
    return result



# Trajectory generation

def _generate_full_trajectory(
    persona, pattern, provider, checkpoints,
    individual_trip_budget=None,
    use_daily_plan=True,
    use_persona=True,
    population_mode_share=None,
    population_purpose_share=None,
):
    """
    Three-step pipeline: motivational summary → daily plan → plan-driven reasoning.

    Budget enforcement is done HERE in pipeline code, not delegated to the LLM.
    The reasoning loop is called only for travel steps identified by keyword
    prefix matching against MOVEMENT_VERBS. For non-travel plan steps, the step
    text is added directly to the earlier-schedule context without an LLM call.

    checkpoints is used only when use_daily_plan=False (ablation path).
    """
    if individual_trip_budget is None:
        individual_trip_budget = _sample_individual_budget(persona)

    if use_daily_plan:
        motivational_summary = _generate_motivational_summary(persona, pattern, provider, use_persona=use_persona)
        if not motivational_summary:
            return None
        daily_plan = _generate_daily_plan(
            persona, pattern, provider,
            motivational_summary=motivational_summary,
            individual_trip_budget=individual_trip_budget,
            use_daily_plan=True,
            use_persona=use_persona,
        )
    else:
        motivational_summary = ""
        daily_plan = _generate_daily_plan(
            persona, pattern, provider,
            motivational_summary="",
            individual_trip_budget=individual_trip_budget,
            use_daily_plan=False,
            use_persona=use_persona,
        )

    if not daily_plan:
        return None

    trips           = []
    reasoning_steps = []
    earlier_steps   = []

    if use_daily_plan:
        plan_steps = _parse_plan_steps(daily_plan)
    else:
        # Ablation fallback: treat each fixed checkpoint as a synthetic travel step.
        plan_steps = [(cp, f"Travel to destination") for cp in checkpoints]

    for checkpoint, plan_activity in plan_steps:
        if _is_travel_step(plan_activity):
            is_return = _is_return_home_step(plan_activity)

            # ── Budget ceiling: caps outbound trips at B; returns always pass ──
            if len(trips) >= individual_trip_budget and not is_return:
                earlier_steps.append(f"{checkpoint}: {plan_activity} [budget reached — skipped]")
                continue

            if is_return:
                # Return trips: skip LLM call — inherit mode, distance, purpose from last outbound.
                last_outbound = next(
                    (t for t in reversed(trips) if not _is_return_home_step(t.get("plan_step", ""))),
                    None,
                )
                step = {
                    "time":                  checkpoint,
                    "plan_step":             plan_activity,
                    "travel":                True,
                    "plausible":             True,
                    "implausibility_reason": None,
                    "departure_time_class":  _time_to_dep_class(checkpoint),
                    "mode":        last_outbound.get("mode")        if last_outbound else None,
                    "purpose":     last_outbound.get("purpose")     if last_outbound else None,
                    "distance_km": last_outbound.get("distance_km") if last_outbound else None,
                    "destination": "home",
                    "motivation":  None,
                }
            else:
                # Outbound trips: pre-sample mode, then call LLM for purpose/distance/destination.
                # When use_persona=False, use population-level shares to avoid hidden group conditioning.
                if not use_persona and population_mode_share:
                    sampled_mode = random.choices(
                        list(population_mode_share.keys()),
                        weights=list(population_mode_share.values()),
                        k=1,
                    )[0]
                else:
                    sampled_mode = _sample_mode(persona)

                purp_share_for_prompt = (
                    population_purpose_share
                    if (not use_persona and population_purpose_share)
                    else persona.get("purpose_share", {})
                )
                dist_class = _sample_distance(persona, sampled_mode)
                raw  = _run_reasoning_step(
                    checkpoint, plan_activity, daily_plan, earlier_steps,
                    persona, pattern, provider,
                    motivational_summary=motivational_summary,
                    use_persona=use_persona,
                    assigned_mode=sampled_mode,
                    purpose_share=purp_share_for_prompt,
                )
                step = _parse_step(raw, checkpoint, plan_activity)
                step["mode"]        = sampled_mode
                step["distance_km"] = _DIST_MIDPOINT_KM.get(dist_class)

            reasoning_steps.append(step)

            if step["plausible"]:
                trips.append(step.copy())
                desc = f"{plan_activity} → [{step.get('mode')}] to {step.get('destination')}"
            else:
                desc = f"{plan_activity} [IMPLAUSIBLE: {step.get('implausibility_reason', '')}]"
        else:
            # Non-travel step: add plan text to context, no LLM call.
            desc = plan_activity

        earlier_steps.append(f"{checkpoint}: {desc}")

    # ── Guarantee day closure ────────────────────────────────────────────────
    # If the LLM plan omitted the final return-home step, inject one so that
    # every synthetic day ends at home.  Inherits mode/distance from the last
    # outbound trip; flagged with injected_return_home=True for transparency.
    if trips and (trips[-1].get("destination") or "").strip().lower() != "home":
        last = trips[-1]
        try:
            h, m = map(int, last["time"].split(":"))
            ret_mins = min(h * 60 + m + 60, 23 * 60 + 30)
            ret_time = f"{ret_mins // 60:02d}:{ret_mins % 60:02d}"
        except (ValueError, KeyError):
            ret_time = "22:00"
        return_trip = {
            "time":                  ret_time,
            "plan_step":             f"Return home by {last.get('mode', 'unknown')}",
            "travel":                True,
            "plausible":             True,
            "implausibility_reason": None,
            "departure_time_class":  _time_to_dep_class(ret_time),
            "mode":                  last.get("mode"),
            "purpose":               last.get("purpose"),
            "distance_km":           last.get("distance_km"),
            "destination":           "home",
            "motivation":            None,
            "injected_return_home":  True,
        }
        trips.append(return_trip)
        reasoning_steps.append(return_trip.copy())

    return {
        "motivational_summary": motivational_summary,
        "daily_plan":           daily_plan,
        "reasoning_steps":      reasoning_steps,
        "trips":                trips,
    }


def _generate_zero_trip_trajectory(persona, pattern, provider, use_persona=True):
    motivational_summary = _generate_motivational_summary(persona, pattern, provider, use_persona=use_persona)
    daily_plan = _generate_daily_plan(
        persona, pattern, provider,
        motivational_summary=motivational_summary or "",
        zero_trip=True,
        use_persona=use_persona,
    )
    return {
        "motivational_summary": motivational_summary or "",
        "daily_plan":           daily_plan or "[Zero-trip day — stayed at home]",
        "reasoning_steps":      [],
        "trips":                [],
    }



# Mobility pattern augmentation

def _append_distance_table(pattern_text: str, persona: dict) -> str:
    """
    Append a compact distance calibration hint (median + P80) to the CoT pattern text.
    Gives the LLM a quantitative anchor for Dutch context without providing a full
    distribution to sample from mechanically. Framed as Dutch-specific so the LLM
    knows to adapt it when generating for a different geographic context.
    """
    dist = persona.get("distance_share", {})
    if not dist:
        return pattern_text

    pairs = sorted(
        [(km, dist[label]) for label, km in _DIST_MIDPOINT_KM.items() if label in dist],
        key=lambda x: x[0],
    )
    if not pairs:
        return pattern_text

    total = sum(w for _, w in pairs)
    if total == 0:
        return pattern_text

    cumulative = 0.0
    median_km = p80_km = None
    for km, weight in pairs:
        cumulative += weight / total * 100
        if median_km is None and cumulative >= 50:
            median_km = km
        if p80_km is None and cumulative >= 80:
            p80_km = km

    parts = [f"median ~{median_km:.1f} km"]
    if p80_km is not None:
        parts.append(f"80% of trips under {p80_km:.1f} km")

    hint = "; ".join(parts)
    return (
        f"{pattern_text}\n\n"
        f"Distance reference (Dutch ODiN 2022, this group): {hint}. "
        f"Adapt these figures if the geographic or cultural context differs from the Netherlands."
    )



# Public entry point

def run_generation_pipeline(
    persona_objects,
    patterns,
    output_file,
    day_type       = "weekday",
    provider       = "groq_generation",
    n_per_group    = 3,
    atypical_rates = None,
    seed           = 42,
    use_daily_plan = True,
    use_persona    = True,
    use_pattern    = True,
):
    """
    Generate n_per_group synthetic trajectories for every persona whose day_type
    matches. Resumable: already-completed groups are skipped.

    Parameters
    ----------
    persona_objects : list of dicts (from persona_objects.json)
    patterns        : {group_key: pattern_text} (from extract_final_patterns)
    output_file     : path to write/resume the results JSON
    day_type        : 'weekday' | 'weekend'
    provider        : LLM provider key (groq_generation recommended)
    n_per_group     : synthetic persons to generate per persona group
    atypical_rates  : optional override {activity_status: {type: float}}
    seed            : for reproducibility of atypical sampling and Poisson budgets
    use_daily_plan  : ablation flag — if False, skips motivational summary and
                      daily plan; falls back to fixed WEEKDAY/WEEKEND_CHECKPOINTS
    use_persona     : ablation flag — if False, omits demographic profile from prompts
    use_pattern     : ablation flag — if False, replaces mobility pattern with a stub
                      so the LLM must reason from persona profile alone
    """
    random.seed(seed)
    np.random.seed(seed)
    checkpoints = WEEKDAY_CHECKPOINTS if day_type == "weekday" else WEEKEND_CHECKPOINTS
    population_mode_share    = _compute_population_mode_share(persona_objects)
    population_purpose_share = _compute_population_purpose_share(persona_objects)

    results = {}
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)

    target_personas = [p for p in persona_objects if p.get("day_type") == day_type]
    total = len(target_personas)

    existing_ids = [
        r["person_id"]
        for group_list in results.values()
        if isinstance(group_list, list)
        for r in group_list
    ]
    syn_counter = max(
        (int(pid.split("_")[-1]) for pid in existing_ids if pid.split("_")[-1].isdigit()),
        default=0,
    ) + 1

    for idx, persona in enumerate(target_personas):
        group_key = persona["group_key"]

        if use_pattern:
            pattern = patterns.get(group_key)
            if not pattern:
                print(f"[{idx+1}/{total}] SKIP {group_key} — no pattern found")
                continue
            pattern = _append_distance_table(pattern, persona)
        else:
            pattern = "No mobility pattern data available — reason from the persona profile alone."

        group_results = results.get(group_key, [])
        n_needed = n_per_group - len(group_results)

        if n_needed <= 0:
            print(f"[{idx+1}/{total}] SKIP {group_key} (already {len(group_results)})")
            continue

        print(f"[{idx+1}/{total}] {group_key} — generating {n_needed}")

        for _ in range(n_needed):
            person_id = f"syn_{day_type}_{syn_counter:04d}"
            syn_counter += 1

            atypical_type = should_be_atypical(persona, atypical_rates)
            is_atypical   = atypical_type is not None
            tag           = f"[{atypical_type.upper()}]" if is_atypical else ""
            print(f"  {person_id} {tag}")

            if atypical_type == "zero_trip":
                traj = _generate_zero_trip_trajectory(persona, pattern, provider, use_persona=use_persona)
            else:
                individual_budget = _sample_individual_budget(persona)
                traj = _generate_full_trajectory(
                    persona, pattern, provider, checkpoints,
                    individual_trip_budget=individual_budget,
                    use_daily_plan=use_daily_plan,
                    use_persona=use_persona,
                    population_mode_share=population_mode_share,
                    population_purpose_share=population_purpose_share,
                )

            if traj is None:
                print(f"    FAILED — skipping {person_id}")
                continue

            group_results.append({
                "person_id":     person_id,
                "group_key":     group_key,
                "day_type":      day_type,
                "is_atypical":   is_atypical,
                "atypical_type": atypical_type,
                **traj,
            })

        results[group_key] = group_results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  -> saved ({len(group_results)} total for {group_key})")

    _write_summary(results, output_file)
    return results


def run_resampling_baseline(
    odin_train_path,
    persona_objects,
    day_type    = "weekday",
    n_per_group = 5,
    seed        = 42,
):
    """
    Empirical resampling baseline — no LLM.

    For each persona group, sample real ODiN persons from the training set
    and return their actual trips formatted as synthetic records.  This gives
    the lower-bound sanity check: if SynTrav does not beat resampling on
    SD/SI/DARD/DailyLoc, the LLM adds no value over statistical copying.

    Parameters
    ----------
    odin_train_path : str  — path to odin_train.csv (trip-level, one row per trip)
    persona_objects : list[dict]  — from persona_objects.json
    day_type        : 'weekday' | 'weekend'
    n_per_group     : persons to sample per persona group (with replacement if needed)
    seed            : for reproducibility

    Returns
    -------
    list[dict] in the same format as load_syn_records() — compatible with evaluate().
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    df  = pd.read_csv(odin_train_path, low_memory=False)
    df  = df[df["DayType"] == day_type].copy()

    # Build per-person trip lists indexed by group_key
    group_persons = {}
    for group_key, grp in df.groupby("group_key"):
        persons = {}
        for pid, ptrips in grp.groupby("Person_index"):
            trips = []
            for _, row in ptrips.iterrows():
                try:
                    h = int(float(row["Departure time transfer"]))
                    m = int(float(row["Departure minute displacement"]))
                    time_str = f"{h:02d}:{m:02d}"
                except (ValueError, TypeError):
                    time_str = "00:00"
                trips.append({
                    "time":               time_str,
                    "departure_time_class": str(row.get("Departure time class", "")),
                    "mode":               str(row.get("Main mode of transport travel", "")),
                    "purpose":            str(row.get("Motive", "")),
                    "distance_class":     str(row.get("Travel distance class in the Netherlands", "")),
                    "travel":             True,
                    "plausible":          True,
                    "implausibility_reason": None,
                    "motivation":         None,
                    "destination":        None,
                })
            persons[pid] = trips
        group_persons[group_key] = list(persons.values())

    target_personas = [p for p in persona_objects if p.get("day_type") == day_type]
    records  = []
    counter  = 1

    for persona in target_personas:
        group_key  = persona["group_key"]
        pool       = group_persons.get(group_key, [])
        if not pool:
            print(f"  SKIP {group_key} — no real persons in training set")
            continue

        indices = rng.integers(0, len(pool), size=n_per_group)
        for idx in indices:
            records.append({
                "person_id":   f"resample_{day_type}_{counter:04d}",
                "group_key":   group_key,
                "day_type":    day_type,
                "is_atypical": False,
                "atypical_type": None,
                "trips":       pool[idx],
            })
            counter += 1

    avg_trips = sum(len(r["trips"]) for r in records) / len(records) if records else 0
    print(f"Resampling baseline: {len(records)} persons | avg {avg_trips:.2f} trips/person")
    return records


def _write_summary(results, output_file):
    all_records = [
        r for k, v in results.items()
        if isinstance(v, list)
        for r in v
    ]
    if not all_records:
        return
    n_total    = len(all_records)
    n_atypical = sum(1 for r in all_records if r.get("is_atypical"))
    n_zero     = sum(1 for r in all_records if r.get("atypical_type") == "zero_trip")
    avg_trips  = sum(len(r.get("trips", [])) for r in all_records) / n_total

    results["__summary__"] = {
        "total_persons":        n_total,
        "n_typical":            n_total - n_atypical,
        "n_atypical":           n_atypical,
        "n_zero_trip":          n_zero,
        "avg_trips_per_person": round(avg_trips, 2),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Summary] {n_total} persons | avg {avg_trips:.1f} trips | "
          f"{n_atypical} atypical ({n_zero} zero-trip)")
