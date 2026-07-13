import random

# Zero-trip rates by activity status, computed from ODiN 2022 weekday person-days
# (persons who stayed home / total persons per group, excluding public holidays)
_DEFAULT_RATES = {
    "employed":         {"zero_trip": 0.14},  # ODiN: 0.135
    "retired":          {"zero_trip": 0.27},  # ODiN: 0.266
    "homemaker":        {"zero_trip": 0.35},  # ODiN: 0.355 — was 0.22, largest fix
    "student":          {"zero_trip": 0.14},  # ODiN: 0.141
    "unemployed":       {"zero_trip": 0.35},  # ODiN: 0.347
    "incapacitated":    {"zero_trip": 0.29},  # ODiN: 0.290 — was 0.40, large fix
    "inactive":         {"zero_trip": 0.26},  # ODiN: 0.262
    "working_retired":  {"zero_trip": 0.11},  # ODiN: 0.108
}

_ATYPICAL_TYPES = ["zero_trip"]

ATYPICAL_DESCRIPTIONS = {
    "zero_trip": "Person makes no trips this day — stays at home.",
}


def should_be_atypical(persona, override_rates=None):
    """
    Returns 'zero_trip' or None.
    override_rates: {activity_status: {atypical_type: float}} to override defaults.
    """
    rates = {k: dict(v) for k, v in _DEFAULT_RATES.items()}
    if override_rates:
        for status, overrides in override_rates.items():
            rates.setdefault(status, {}).update(overrides)

    status  = persona.get("activity_status", "inactive")
    profile = rates.get(status, {"zero_trip": 0.20})

    for atype in _ATYPICAL_TYPES:
        if random.random() < profile.get(atype, 0.0):
            return atype
    return None
