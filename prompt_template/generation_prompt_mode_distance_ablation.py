MOTIVATIONAL_SUMMARY_PROMPT = """You are a daily planning expert, skilled at creating clear and reasonable daily plans for individuals based on various factors (such as location, external conditions, personal attributes, and previous reflections).

<INDIVIDUAL PROFILE>
{persona_profile}
</INDIVIDUAL PROFILE>

<MOBILITY PATTERNS>
{mobility_pattern}
</MOBILITY PATTERNS>

Write a motivational summary (4–5 sentences) that explains this person's travel behaviour on a typical {day_type}. Address in order:
1. The primary driver(s) of their travel — what obligations or activities bring them out of the home on this type of day.
2. The constraints that shape when and how they travel — time pressure, mode availability, household role, income, or physical capacity.
3. The dominant trip purposes for this person today and whether their travel day is typically simple (one main activity) or complex (a chain of activities).
4. Any behavioural tendency that distinguishes this group from an average Dutch traveller.

Write in third person. Be specific to this profile. Interpret the patterns — do not restate the statistics.
"""


DAILY_PLAN_PROMPT = """You are a daily planning expert, skilled at creating realistic daily plans for individuals based on their personal profile and travel behaviour.

<MOTIVATIONAL SUMMARY>
{motivational_summary}
</MOTIVATIONAL SUMMARY>

<INDIVIDUAL PROFILE>
{persona_profile}
</INDIVIDUAL PROFILE>

<MOBILITY PATTERNS>
{mobility_pattern}
</MOBILITY PATTERNS>

Generate a plausible step-by-step daily plan for a typical {day_type}.

Trip count constraint: This person makes approximately {individual_trip_budget} trips today. Include no more than {individual_trip_budget} travel steps (outbound + return combined) in the plan. The motivational summary explains why and what kind of trips — use it to decide which trips to include, not to add more.

Constraints:
- Cover the full day from waking up to going to sleep; anchor the morning routine to the group's most common first departure time shown in the mobility patterns
- Trip chains are allowed (e.g. work → school → supermarket → home). Do not add a return-home step after every activity unless the person would genuinely go home in between.
- Every trip must use a mode drawn from the group's known modes in the mobility patterns; travel step descriptions must only reference those modes — do not invent modes not present in the mobility patterns
- The plan should not include any speculative or uncertain terms. If there are any unspecified contexts, you may make appropriate assumptions.
- Provide the plan directly, without explanations or summaries

Format each step strictly as:
[HH:MM] Activity

For any step involving physical movement, the activity MUST begin with one of these verbs:
Travel to / Walk to / Cycle to / Drive to / Take the train to / Return home by
Examples: [08:00] Travel to work | [17:30] Return home by car | [10:00] Walk to the supermarket
For all other steps (waking up, eating, working, resting), describe the activity directly.
"""


RECURSIVE_REASONING_PROMPT = """Current time: {current_time}
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

The daily plan indicates this person is making a trip at {current_time}. Your task is to reason about the travel attributes for this specific trip using their profile, motivational summary, and group mobility patterns.

Plausibility check: If the earlier schedule shows this person has already returned home AND this plan step is a new outbound trip (not a return-home step), set PLAUSIBLE to NO and give a brief reason. In all other cases, set PLAUSIBLE to YES.

Available values — choose exactly one per field:
- Mode (choose the most appropriate given this person's profile, trip purpose, and group mobility patterns): {available_modes}
- Distance class (choose the most appropriate given the trip purpose, destination, and this person's profile): {available_distances}
- Purposes: {available_purposes}
- Destinations: {available_destinations}

Reply in exactly this format — no other text:
PLAUSIBLE: YES or NO
IMPLAUSIBILITY_REASON: [brief reason if PLAUSIBLE is NO, otherwise NONE]
MOTIVATION: [2 sentences — (1) why this person is making this specific trip given their profile and motivational summary, (2) why the chosen mode and distance class are appropriate given the group's mobility patterns and trip context]
MODE: [exact value from Mode list]
DISTANCE: [exact value from Distance list]
PURPOSE: [exact value from Purposes list]
DESTINATION: [exact value from Destinations list]
"""


ATYPICAL_ZERO_TRIP_PLAN = """You are a daily planning expert.

<MOTIVATIONAL SUMMARY>
{motivational_summary}
</MOTIVATIONAL SUMMARY>

<INDIVIDUAL PROFILE>
{persona_profile}
</INDIVIDUAL PROFILE>

<MOBILITY PATTERNS>
{mobility_pattern}
</MOBILITY PATTERNS>

Today ({day_type}) this person makes no recorded trips outside their home address. In one sentence, explain why this specific person is not travelling today given their profile. Then generate a realistic stay-at-home plan with activities that do not require leaving the home address.

Format each step strictly as:
[HH:MM] Activity

Provide the opening explanation sentence, then the plan directly. No further explanations.
"""
