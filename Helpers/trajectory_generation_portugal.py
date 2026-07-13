"""
Portugal (Oeiras) generation pipeline — self-contained, does not modify the
original Dutch framework.  trajectory_generation.py is untouched and can be
run independently to reproduce the NL results at any time.

Differences from the Dutch pipeline:
  1. MOTIVATIONAL_SUMMARY_PROMPT_PT replaces the NL motivational summary prompt
     so the reference is to "a traveller in the Área Metropolitana de Lisboa"
     instead of "an average Dutch traveller".
  2. DAILY_PLAN_PROMPT and RECURSIVE_REASONING_PROMPT have the Oeiras location
     context baked in as static text (no placeholder injection).
  3. "Take the bus to" is added to the movement verb list so bus plan steps are
     recognised as travel steps.
  4. available_modes always includes Train and Bus, regardless of the persona's
     Dutch ODiN distribution (Option A — documented PT override).
  5. run_generation_pipeline_portugal writes to a separate output file and
     person IDs are prefixed syn_pt_ so results are never mixed with NL output.

Everything else (LLM calls, retry logic, atypical sampling, distance sampling,
budget enforcement, plan parsing) is imported directly from the original module.
"""

import json, os, re, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_template.generation_prompt import (
    ATYPICAL_ZERO_TRIP_PLAN,
)
from Rural_test.portugal_context import (
    PORTUGAL_CONTEXT_FULL,
    PORTUGAL_CONTEXT_SHORT,
    build_portugal_context,
)
from Helpers.trajectory_generation import (
    _call_with_retry,
    _top_shares,
    _top_keys,
    _sample_distance,
    _sample_individual_budget,
    _append_distance_table,
    _parse_plan_steps,
    _is_return_home_step,
    _write_summary,
    format_persona_profile,
    DESTINATION_TYPES,
    _time_to_dep_class,
)
from Helpers.atypical_travelers import should_be_atypical

import numpy as np
import random


#  Portugal-specific movement verbs (adds "Take the bus to") 

MOVEMENT_VERBS_PT = [
    "Travel to",
    "Walk to",
    "Drive to",
    "Take the train to",
    "Take the bus to",
    "Return home by",
]

# Cycling is intentionally absent — no "Cycle to" verb.


#  Portugal-specific prompt templates (context baked in) 

DAILY_PLAN_PROMPT_PT = (
    """You are a daily planning expert, skilled at creating realistic daily plans for individuals based on their personal profile and travel behaviour.

<MOTIVATIONAL SUMMARY>
{motivational_summary}
</MOTIVATIONAL SUMMARY>

<INDIVIDUAL PROFILE>
{persona_profile}
</INDIVIDUAL PROFILE>

<MOBILITY PATTERNS>
{mobility_pattern}
</MOBILITY PATTERNS>
"""
    + "{portugal_context}"
    + """
Generate a plausible step-by-step daily plan for a typical {day_type}.

Trip count constraint: This person makes approximately {individual_trip_budget} trips today. Include no more than {individual_trip_budget} travel steps (outbound + return combined) in the plan. The motivational summary explains why and what kind of trips — use it to decide which trips to include, not to add more.

Constraints:
- Cover the full day from waking up to going to sleep; anchor the morning routine to the group's most common first departure time shown in the mobility patterns
- Each away-from-home activity must be followed by a return-home step
- Each return-home step must use the same mode as its corresponding outbound trip
- Every trip must use a mode drawn from the group's known modes in the mobility patterns
- The plan should not include any speculative or uncertain terms. If there are any unspecified contexts, you may make appropriate assumptions.
- Provide the plan directly, without explanations or summaries

Format each step strictly as:
[HH:MM] Activity

For any step involving physical movement, the activity MUST begin with one of these verbs:
Travel to / Walk to / Drive to / Take the train to / Take the bus to / Return home by
Examples: [08:00] Travel to work | [17:30] Return home by car | [10:00] Walk to the supermarket
"""
)

RECURSIVE_REASONING_PROMPT_PT = (
    """Current time: {current_time}
Plan step: {plan_step}

<MOTIVATIONAL SUMMARY>
{motivational_summary}
</MOTIVATIONAL SUMMARY>

Daily plan overview:
{daily_plan}

Earlier schedule today:
{earlier_schedule}

<INDIVIDUAL PROFILE>
{persona_profile}
</INDIVIDUAL PROFILE>

<MOBILITY PATTERNS>
{mobility_pattern}
</MOBILITY PATTERNS>
"""
    + "{portugal_context}"
    + """
The daily plan indicates this person is making a trip at {current_time}. Your task is to reason about the travel attributes for this specific trip using their profile, motivational summary, and group mobility patterns.

Plausibility check: If the earlier schedule shows this person has already returned home AND this plan step is a new outbound trip (not a return-home step), set PLAUSIBLE to NO and give a brief reason. In all other cases, set PLAUSIBLE to YES.

Available values — choose exactly one per field:
- Modes: {available_modes}
- Purposes: {available_purposes}
- Distances: {available_distances}
- Destinations: {available_destinations}

Reply in exactly this format — no other text:
PLAUSIBLE: YES or NO
IMPLAUSIBILITY_REASON: [brief reason if PLAUSIBLE is NO, otherwise NONE]
MOTIVATION: [2 sentences — (1) why this person is making this specific trip given their profile and motivational summary, (2) how the group's mobility patterns support this choice]
MODE: [exact value from Modes list]
PURPOSE: [exact value from Purposes list]
DISTANCE: [exact value from Distances list]
DESTINATION: [exact value from Destinations list]
"""
)


#  Portugal-specific motivational summary prompt 

MOTIVATIONAL_SUMMARY_PROMPT_PT = """You are a daily planning expert, skilled at creating clear and reasonable daily plans for individuals based on various factors (such as location, external conditions, personal attributes, and previous reflections).

<INDIVIDUAL PROFILE>
{persona_profile}
</INDIVIDUAL PROFILE>

<MOBILITY PATTERNS>
{mobility_pattern}
</MOBILITY PATTERNS>

Write a motivational summary (4–5 sentences) that explains this person's travel behaviour on a typical {day_type}. Address in order:
1. The primary driver(s) of their travel — what obligations or activities bring them out of the home on this type of day.
2. The constraints that shape when and how they travel — time pressure, mode availability, household role, income, or physical capacity.
3. The expected number of trips and dominant purposes for this person today.
4. Any behavioural tendency that distinguishes this group from a typical traveller in the Área Metropolitana de Lisboa.

Write in third person. Be specific to this profile. Interpret the patterns — do not restate the statistics.
"""


def _generate_motivational_summary_pt(persona, pattern, provider, use_persona=True):
    profile = format_persona_profile(persona, use_persona)
    prompt = MOTIVATIONAL_SUMMARY_PROMPT_PT.format(
        persona_profile=profile,
        mobility_pattern=pattern,
        day_type=persona.get("day_type", "weekday"),
    )
    return _call_with_retry(prompt, provider, max_tokens=350)


def _generate_zero_trip_trajectory_pt(persona, pattern, provider, use_persona=True):
    motivational_summary = _generate_motivational_summary_pt(persona, pattern, provider, use_persona=use_persona)
    profile = format_persona_profile(persona, use_persona)
    prompt = ATYPICAL_ZERO_TRIP_PLAN.format(
        motivational_summary=motivational_summary or "",
        persona_profile=profile,
        mobility_pattern=pattern,
        day_type=persona.get("day_type", "weekday"),
    )
    daily_plan = _call_with_retry(prompt, provider, max_tokens=600)
    if not daily_plan:
        return None
    return {
        "motivational_summary": motivational_summary,
        "daily_plan":           daily_plan,
        "reasoning_steps":      [],
        "trips":                [],
    }


#  PT-aware available modes 

def _available_modes_pt(persona):
    """Top-6 Dutch modes with Train and Bus appended if not already present.
    Documented intervention: personas' Dutch distributions exclude PT for most
    groups; forced inclusion reflects AML's 15.8% PT share (IMOB 2017)."""
    top6 = [k for k, _ in sorted(persona["mode_share"].items(), key=lambda x: -x[1])[:6]]
    for m in ("Train", "Bus"):
        if m not in top6:
            top6.append(m)
    return ", ".join(top6)


#  Step helpers 

def _is_travel_step_pt(activity: str) -> bool:
    low = activity.lower()
    return any(low.startswith(v.lower()) for v in MOVEMENT_VERBS_PT)


def _mode_from_return_step(plan_step: str):
    """Parse mode directly from 'Return home by X' plan text — no LLM needed."""
    s = plan_step.lower()
    if "by car" in s or "by driving" in s or "by passenger car" in s:
        return "Passenger car"
    if "by walking" in s or "by foot" in s or "on foot" in s or "walk" in s:
        return "On foot"
    if "taking the train" in s or "by train" in s:
        return "Train"
    if "public transport" in s:
        return "Train"   # Linha de Cascais is the primary PT mode in Oeiras
    if "by bus" in s:
        return "Bus"
    if "by non-electric bicycle" in s or "by bicycle" in s or "by bike" in s or "by cycling" in s:
        return "Non-electric bicycle"
    return None


def _parse_step_pt(raw, checkpoint, plan_step="", inherited_distance=None, inherited_purpose=None):
    is_return = _is_return_home_step(plan_step)
    result = {
        "time":                  checkpoint,
        "plan_step":             plan_step,
        "travel":                True,
        "plausible":             True,
        "implausibility_reason": None,
        "departure_time_class":  _time_to_dep_class(checkpoint),
    }
    if not raw:
        if not is_return:
            result["plausible"] = False
            result["implausibility_reason"] = "LLM call failed"
        return result

    def _extract(field):
        m = re.search(rf"^{field}:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
        val = m.group(1).strip() if m else None
        return None if val and val.upper() == "NONE" else val

    if not is_return:
        plausible_val = (_extract("PLAUSIBLE") or "YES").strip().upper()
        result["plausible"] = (plausible_val == "YES")
        result["implausibility_reason"] = (
            _extract("IMPLAUSIBILITY_REASON") if not result["plausible"] else None
        )

    result["motivation"]     = _extract("MOTIVATION") or None
    result["mode"]           = _mode_from_return_step(plan_step) if is_return else _extract("MODE")
    result["purpose"]        = inherited_purpose or _extract("PURPOSE")
    result["distance_class"] = inherited_distance or _extract("DISTANCE")
    result["destination"]    = "home" if is_return else _extract("DESTINATION")
    return result


#  LLM call wrappers 

def _generate_daily_plan_pt(
    persona, pattern, provider, motivational_summary="", individual_trip_budget=3, use_persona=True,
    portugal_context_full=PORTUGAL_CONTEXT_FULL,
):
    profile = format_persona_profile(persona, use_persona)
    prompt = DAILY_PLAN_PROMPT_PT.format(
        motivational_summary=motivational_summary or "",
        individual_trip_budget=individual_trip_budget,
        persona_profile=profile,
        mobility_pattern=pattern,
        day_type=persona.get("day_type", "weekday"),
        portugal_context=portugal_context_full,
    )
    return _call_with_retry(prompt, provider, max_tokens=700)


def _run_reasoning_step_pt(
    current_time, plan_step, daily_plan, earlier_steps,
    persona, pattern, provider,
    motivational_summary="", use_persona=True,
    portugal_context_short=PORTUGAL_CONTEXT_SHORT,
):
    earlier_text = "\n".join(earlier_steps) if earlier_steps else "None yet — day just started."
    profile = format_persona_profile(persona, use_persona)
    prompt = RECURSIVE_REASONING_PROMPT_PT.format(
        current_time=current_time,
        plan_step=plan_step,
        motivational_summary=motivational_summary or "",
        daily_plan=daily_plan,
        earlier_schedule=earlier_text,
        persona_profile=profile,
        mobility_pattern=pattern,
        available_modes=_available_modes_pt(persona),
        available_purposes=_top_keys(persona["purpose_share"], 6),
        available_distances=_top_shares(persona["distance_share"], 6),
        available_destinations=", ".join(DESTINATION_TYPES),
        portugal_context=portugal_context_short,
    )
    return _call_with_retry(prompt, provider, max_tokens=350)


#  Trajectory generation 

def _generate_full_trajectory_pt(
    persona, pattern, provider,
    individual_trip_budget=None,
    use_persona=True,
    portugal_context_full=PORTUGAL_CONTEXT_FULL,
    portugal_context_short=PORTUGAL_CONTEXT_SHORT,
):
    if individual_trip_budget is None:
        individual_trip_budget = _sample_individual_budget(persona)

    motivational_summary = _generate_motivational_summary_pt(persona, pattern, provider, use_persona=use_persona)
    if not motivational_summary:
        return None

    daily_plan = _generate_daily_plan_pt(
        persona, pattern, provider,
        motivational_summary=motivational_summary,
        individual_trip_budget=individual_trip_budget,
        use_persona=use_persona,
        portugal_context_full=portugal_context_full,
    )
    if not daily_plan:
        return None

    trips           = []
    reasoning_steps = []
    earlier_steps   = []

    for checkpoint, plan_activity in _parse_plan_steps(daily_plan):
        if _is_travel_step_pt(plan_activity):
            is_return = _is_return_home_step(plan_activity)

            if len(trips) >= individual_trip_budget and not is_return:
                earlier_steps.append(f"{checkpoint}: {plan_activity} [budget reached — skipped]")
                continue

            inherited_distance = None
            inherited_purpose  = None
            if is_return and trips:
                last_outbound = next(
                    (t for t in reversed(trips) if not _is_return_home_step(t.get("plan_step", ""))),
                    None,
                )
                if last_outbound:
                    inherited_distance = last_outbound.get("distance_class")
                    inherited_purpose  = last_outbound.get("purpose")

            raw  = _run_reasoning_step_pt(
                checkpoint, plan_activity, daily_plan, earlier_steps,
                persona, pattern, provider,
                motivational_summary=motivational_summary,
                use_persona=use_persona,
                portugal_context_short=portugal_context_short,
            )
            step = _parse_step_pt(raw, checkpoint, plan_activity,
                                   inherited_distance=inherited_distance,
                                   inherited_purpose=inherited_purpose)

            # if not is_return:
            #     step["distance_class"] = _sample_distance(persona, mode=step.get("mode"))

            reasoning_steps.append(step)

            if step["plausible"]:
                trips.append(step.copy())
                desc = f"{plan_activity} → [{step.get('mode')}] to {step.get('destination')}"
            else:
                desc = f"{plan_activity} [IMPLAUSIBLE: {step.get('implausibility_reason', '')}]"
        else:
            desc = plan_activity

        earlier_steps.append(f"{checkpoint}: {desc}")

    #  Guarantee day closure 
    # If the LLM plan omitted the final return-home step (or it was dropped as
    # implausible), inject one so that every synthetic PT day ends at home,
    # matching the NL pipeline's guarantee (Helpers/trajectory_generation.py).
    # Inherits mode/purpose/distance_class from the last outbound trip; flagged
    # with injected_return_home=True for transparency.
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
            "motivation":            None,
            "mode":                  last.get("mode"),
            "purpose":               last.get("purpose"),
            "distance_class":        last.get("distance_class"),
            "destination":           "home",
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


#  Public entry point 

def run_generation_pipeline_portugal(
    persona_objects,
    patterns,
    output_file,
    day_type       = "weekday",
    provider       = "groq_generation",
    n_per_group    = 3,
    atypical_rates = None,
    seed           = 42,
    use_persona    = True,
    use_pattern    = True,
    context_condition = "full",
):
    """
    Portugal-specific generation pipeline for the Oeiras transferability experiment.
    Signature mirrors run_generation_pipeline() for drop-in use in notebooks,
    but uses Portugal prompts and PT-aware mode lists throughout.
    use_daily_plan is always True — the ablation path is not relevant here.

    context_condition selects one of the four sensitivity-analysis conditions
    (see prompt_template/portugal_context.py): "full" (baseline), "no_commuting",
    "no_cycling", "minimal". Only the commuting-stats and cycling-note blocks
    differ between conditions — everything else in the prompt is identical.
    """
    random.seed(seed)
    np.random.seed(seed)

    portugal_context_full, portugal_context_short = build_portugal_context(context_condition)

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
            person_id = f"syn_pt_{day_type}_{syn_counter:04d}"
            syn_counter += 1

            atypical_type = should_be_atypical(persona, atypical_rates)
            is_atypical   = atypical_type is not None
            tag           = f"[{atypical_type.upper()}]" if is_atypical else ""
            print(f"  {person_id} {tag}")

            if atypical_type == "zero_trip":
                traj = _generate_zero_trip_trajectory_pt(persona, pattern, provider, use_persona=use_persona)
            else:
                individual_budget = _sample_individual_budget(persona)
                traj = _generate_full_trajectory_pt(
                    persona, pattern, provider,
                    individual_trip_budget=individual_budget,
                    use_persona=use_persona,
                    portugal_context_full=portugal_context_full,
                    portugal_context_short=portugal_context_short,
                )

            if traj is None:
                print(f"    FAILED — skipping {person_id}")
                continue

            group_results.append({
                "person_id":        person_id,
                "group_key":        group_key,
                "day_type":         day_type,
                "is_atypical":      is_atypical,
                "atypical_type":    atypical_type,
                "context_condition": context_condition,
                **traj,
            })

        results[group_key] = group_results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  -> saved ({len(group_results)} total for {group_key})")

    _write_summary(results, output_file)
    return results
