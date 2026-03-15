from __future__ import annotations

from typing import Dict

SCOUTING_FIELDS = [
    "cycling",
    "scoring",
    "feeding",
    "defending",
    "immobile",
    "intake_type_ground",
    "intake_type_outpost_or_source",
    "traversal_trench",
    "traversal_bump",
    "auto_climb_status",
    "beached_on_fuel",
    "beached_on_bump",
    "scores_while_moving",
    "robot_broke",
    "estimated_points_scored",
]


def empty_scouting_payload() -> Dict[str, Dict[str, float | int | str]]:
    payload: Dict[str, Dict[str, float | int | str]] = {}
    for field in SCOUTING_FIELDS:
        default_value = 0
        if field == "auto_climb_status":
            default_value = "unknown"
        payload[field] = {"value": default_value, "confidence": 0.0}
    return payload
