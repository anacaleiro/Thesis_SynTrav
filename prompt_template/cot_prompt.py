PATTERN_EXTRACTION_PROMPT = """                          
  You are analysing Dutch daily mobility data from the ODiN 2022 national travel survey.

  <GROUP PROFILE>
  Group: {group_name}
  N persons: {n_persons} | N trips: {n_trips} | Avg trips/person: {trips_per_person}

  Demographics:
  - Age: {age_dist}
  - Gender: {gender_dist}
  - Income: {income_dist}
  - Household composition: {household_dist}
  - Urbanisation: {urban_dist}

  Travel behaviour distributions:
  - Transport mode share: {mode_share}
  - Trip purpose share: {purpose_share}
  - Departure time share: {dep_time_share}
  - Distance class share: {distance_share}
  </GROUP PROFILE>

  Using step-by-step reasoning:
  1. For each demographic attribute (age, occupation, income, household), explain how it
     likely shapes this group's travel choices.
  2. Identify the dominant daily mobility rhythm: when they travel, how far, by what mode,
     and for what purpose.
  3. Note any notable secondary patterns (e.g. a minority that deviates from the dominant mode).
  4. Write a concise mobility pattern summary (3-5 sentences) describing what makes this group
     distinct from the broader Dutch travelling population.

  Think through each step internally, then output ONLY the final 3-5 sentence mobility pattern summary.
  """

COT_VALIDATE_REFINE_PROMPT = """                                                                                                                                            
  You are analysing Dutch daily mobility data from the ODiN 2022 national travel survey.                                                                                    
                                                                                                                                                                              
  <IDENTIFIED PATTERN>
  Group: {group_name}                                                                                                                                                         
  {pattern}       
  </IDENTIFIED PATTERN>

  <REAL TRAJECTORIES>
  Below are anonymised trip sequences from ODiN respondents in this group.
  Format: Departure time class | Transport mode | Trip purpose | Distance class

  {trajectories}
  </REAL TRAJECTORIES>

  <MASKED TRAJECTORIES>
  Below are trip sequences with one field per trip hidden as [MASKED].
  Format: Departure time class | Transport mode | Trip purpose | Distance class

  {masked_trajectories}
  </MASKED TRAJECTORIES>

  Please follow these steps:

   1. TRAJECTORY VALIDATION — For each real trajectory, one line per trajectory only: consistent or not,                                                                                     
       and why. If inconsistent, note whether it suggests a subgroup or gap in the pattern.                                                                                   
                                                                                                                                                                              
    2. MASKED COMPLETION — Fill each [MASKED] field. Format:
     Trajectory N: [field] → [value]
     No other text.

    3. PATTERN REFINEMENT — Write the final refined mobility pattern for "{group_name}".
       3-5 sentences. Incorporate any updates from steps 1 and 2.
       Output under the header: FINAL PATTERN:

"""