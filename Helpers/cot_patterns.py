import json
import os
import random
import sys
import time
import re

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import RateLimitError
from llm_config.llm_config import call_llm
from prompt_template.cot_prompt import *

MODE_COL   = 'Main mode of transport travel'
MOTIVE_COL = 'Motive'
DEP_COL    = 'Departure time class'
DIST_COL   = 'Travel distance class in the Netherlands'
PERSON_COL = 'Person_index'

DEMO_COLS = {
      'age':       'Age class OP',
      'gender':    'Gender OP',
      'income':    'income_class',
      'household': 'Household composition',
      'urban':     'Urbanization class of residential municipality',
  }


def _dist(series, top_n=6):
      counts = series.value_counts(dropna=False).head(top_n)
      pct = (counts / counts.sum() * 100).round(1)
      return {str(k): f"{v}%" for k, v in pct.items()}



def format_trajectory(person_trips, traj_id):
      lines = [f"Trajectory {traj_id}:"]
      for _, row in person_trips.iterrows():
          lines.append(
              f"  {row.get(DEP_COL, 'N/A')} | "
              f"{row.get(MODE_COL, 'N/A')} | "
              f"{row.get(MOTIVE_COL, 'N/A')} | "
              f"{row.get(DIST_COL, 'N/A')}"
          )
      return "\n".join(lines)


def mask_trajectory(person_trips, traj_id):
      maskable = [MOTIVE_COL, MODE_COL, DEP_COL]
      lines = [f"Trajectory {traj_id}:"]
      for _, row in person_trips.iterrows():
          dep   = str(row.get(DEP_COL,    'N/A'))
          mode  = str(row.get(MODE_COL,   'N/A'))
          motiv = str(row.get(MOTIVE_COL, 'N/A'))
          dist  = str(row.get(DIST_COL,   'N/A'))
          field = random.choice(maskable)
          if field == DEP_COL:    dep   = '[MASKED]'
          elif field == MODE_COL: mode  = '[MASKED]'
          else:                   motiv = '[MASKED]'
          lines.append(f"  {dep} | {mode} | {motiv} | {dist}")
      return "\n".join(lines)


def _sample_person_trips(group_df, n):
      person_ids = group_df['Person_index'].unique().tolist()
      sampled    = random.sample(person_ids, min(n, len(person_ids)))
      return [group_df[group_df['Person_index'] == pid] for pid in sampled]


def _call_with_retry(prompt, provider, max_tokens, retries=5, delay=2.1):
      for attempt in range(retries):
          try:
              result = call_llm(prompt, provider=provider, max_tokens=max_tokens)
              time.sleep(delay)
              return result
          except RateLimitError:
              wait = 15 * (attempt + 1)
              print(f"  Rate limit — waiting {wait}s...")
              time.sleep(wait)
      return None


def extract_initial_pattern(persona, provider='groq'):
      prompt = PATTERN_EXTRACTION_PROMPT.format(
          group_name       = persona['group_key'],
          n_persons        = persona['n_persons'],
          n_trips          = persona['n_trips'],
          trips_per_person = round(persona['n_trips'] / persona['n_persons'], 2),
          age_dist         = persona['age_class'],
          gender_dist      = persona['demographics'].get('gender', {}),
          income_dist      = persona.get('income_level', 'N/A'),
          household_dist   = persona['demographics'].get('household', {}),
          urban_dist       = persona['demographics'].get('urbanisation', {}),
          mode_share       = persona['mode_share'],
          purpose_share    = persona['purpose_share'],
          dep_time_share   = persona['dep_time_share'],
          distance_share   = persona['distance_share'],
      )
      return _call_with_retry(prompt, provider, max_tokens=600)



def run_cot_validate_refine(group_name, pattern, group_df, n_step1=8, n_step2=4, provider='groq'):
      real_samples   = _sample_person_trips(group_df, n_step1)
      masked_samples = _sample_person_trips(group_df, n_step2)
      prompt = COT_VALIDATE_REFINE_PROMPT.format(
          group_name          = group_name,
          pattern             = pattern,
          trajectories        = "\n\n".join(format_trajectory(t, i+1) for i, t in enumerate(real_samples)),
          masked_trajectories = "\n\n".join(mask_trajectory(t, i+1)   for i, t in enumerate(masked_samples)),
      )
      return _call_with_retry(prompt, provider, max_tokens=1400)



def run_cot_pipeline(persona_objects, subgroups, output_file,
                       provider='groq', n_step1=8, n_step2=4):
      sg_lookup = {sg['group_key']: sg['df'] for sg in subgroups}
      results   = json.load(open(output_file, encoding='utf-8')) if os.path.exists(output_file) else {}
      total     = len(persona_objects)

      for i, persona in enumerate(persona_objects):
          group_name = persona['group_key']
          if group_name in results:
              print(f"[{i+1}/{total}] SKIP {group_name}")
              continue

          print(f"[{i+1}/{total}] {group_name}")
          group_df        = sg_lookup[group_name]
          initial_pattern = extract_initial_pattern(persona, provider)
          combined_output  = run_cot_validate_refine(group_name, initial_pattern, group_df, n_step1, n_step2, provider)

          results[group_name] = {
              'initial_pattern': initial_pattern,
              'combined_output':  combined_output,
          }
          with open(output_file, 'w', encoding='utf-8') as f:
              json.dump(results, f, indent=2, ensure_ascii=False)
          print(f"  -> saved ({i+1}/{total})")

      return results


FINAL_PATTERN_RE = re.compile(r'#*\s*FINAL PATTERN\s*:?', re.IGNORECASE)

def extract_final_patterns(cot_results):
    """Parse COT JSON → {group_key: final_pattern_text}.

    Matches 'FINAL PATTERN' with optional leading markdown hashes and an
    optional trailing colon, and splits on the LAST occurrence so the
    prompt's own instruction echo isn't mistaken for the answer.
    """
    patterns = {}
    for group_key, val in cot_results.items():
        combined = val.get('combined_output', '') or ''
        matches = list(FINAL_PATTERN_RE.finditer(combined))
        if matches:
            patterns[group_key] = combined[matches[-1].end():].strip()
        else:
            patterns[group_key] = combined.strip()
    return patterns