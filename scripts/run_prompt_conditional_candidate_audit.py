from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RULES: dict[str, dict[str, Any]] = {
    "motorcycle": {
        "type": "object",
        "prompt_aliases": ["motorcycle", "motorbike", "scooter"],
        "candidate_aliases": ["motorcycle", "motorbike", "scooter"],
    },
    "bicycle": {
        "type": "object",
        "prompt_aliases": ["bicycle", "bike", "cyclist"],
        "candidate_aliases": ["bicycle", "bike", "cyclist"],
    },
    "pedestrian": {
        "type": "object",
        "prompt_aliases": ["pedestrian", "person", "people", "walker"],
        "candidate_aliases": ["pedestrian", "person", "people", "walker"],
    },
    "vehicle": {
        "type": "object",
        "prompt_aliases": ["vehicle", "car", "truck", "bus"],
        "candidate_aliases": ["vehicle", "car", "truck", "bus"],
    },
    "lane_change": {
        "type": "motion",
        "prompt_aliases": ["lane change", "lane-change", "changing lane", "changes lane", "change lanes"],
        "candidate_aliases": ["lane_change", "lane-related", "lane related", "cut_in", "cut-in"],
    },
    "cut_in": {
        "type": "motion",
        "prompt_aliases": ["cut in", "cut-in", "cuts in"],
        "candidate_aliases": ["cut_in", "cut-in", "cut in", "cuts in"],
    },
    "rainy": {
        "type": "environment",
        "prompt_aliases": ["rainy", "rain"],
        "candidate_aliases": ["rainy", "rain"],
    },
    "night": {
        "type": "environment",
        "prompt_aliases": ["night", "nighttime", "dark"],
        "candidate_aliases": ["night", "nighttime", "dark"],
    },
    "daytime": {
        "type": "environment",
        "prompt_aliases": ["daytime", "daylight", "clear day"],
        "candidate_aliases": ["daytime", "daylight", "clear"],
    },
    "urban": {
        "type": "scene",
        "prompt_aliases": ["urban", "city", "street"],
        "candidate_aliases": ["urban", "city", "street"],
    },
    "highway": {
        "type": "scene",
        "prompt_aliases": ["highway", "freeway"],
        "candidate_aliases": ["highway", "freeway"],
    },
}


PROMPT_CONDITIONAL_TYPES = {"object", "motion", "environment", "scene"}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def normalize_text(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ")


def contains_any(text: str, aliases: list[str]) -> bool:
    normalized = normalize_text(text)
    for alias in aliases:
        normalized_alias = normalize_text(alias).strip()
        if not normalized_alias:
            continue
        phrase_pattern = r"\s+".join(
            re.escape(part) for part in normalized_alias.split()
        )
        pattern = rf"(?<![a-z0-9]){phrase_pattern}(?![a-z0-9])"
        if re.search(pattern, normalized):
            return True
    return False


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for child in value.values():
            result.extend(flatten_strings(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(flatten_strings(child))
        return result
    return []


def requested_rules(prompt: str) -> list[str]:
    return [
        name
        for name, rule in RULES.items()
        if contains_any(prompt, list(rule["prompt_aliases"]))
    ]


def candidate_supports_rules(candidate: dict[str, Any]) -> list[str]:
    candidate_text = " ".join(flatten_strings(candidate))
    return [
        name
        for name, rule in RULES.items()
        if contains_any(candidate_text, list(rule["candidate_aliases"]))
    ]


def selection_reason_rules(candidate: dict[str, Any]) -> list[str]:
    reason_fields = [
        "selection_reason",
        "selection_reasons",
        "selection_reason_tags",
        "candidate_focus",
        "candidate_focus_tags",
        "retrieval_reason",
        "retrieval_tags",
    ]
    values: list[str] = []
    for field in reason_fields:
        if field in candidate:
            values.extend(flatten_strings(candidate[field]))
    reason_text = " ".join(values)
    return [
        name
        for name, rule in RULES.items()
        if reason_text and contains_any(reason_text, list(rule["candidate_aliases"]))
    ]


def audit_candidate(prompt: str, candidate: dict[str, Any]) -> dict[str, Any]:
    requested = requested_rules(prompt)
    supported = candidate_supports_rules(candidate)
    reason_rules = selection_reason_rules(candidate)

    missing_requested_support = [name for name in requested if name not in supported]
    unrequested_selection_bias = [
        name
        for name in reason_rules
        if name not in requested and RULES[name]["type"] in PROMPT_CONDITIONAL_TYPES
    ]

    allowed = not missing_requested_support and not unrequested_selection_bias

    if missing_requested_support:
        status_reason = "candidate does not support accepted prompt requirements"
    elif unrequested_selection_bias:
        status_reason = "candidate selection reason adds prompt-unsupported bias"
    else:
        status_reason = "candidate is prompt-conditional"

    return {
        "schema_version": "driveloop_prompt_conditional_candidate_audit.v0",
        "accepted_prompt": prompt,
        "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
        "allowed": allowed,
        "status": "allowed" if allowed else "blocked",
        "status_reason": status_reason,
        "requested_rules": requested,
        "candidate_supported_rules": supported,
        "selection_reason_rules": reason_rules,
        "missing_requested_support": missing_requested_support,
        "unrequested_selection_bias": unrequested_selection_bias,
        "claim_boundary": {
            "audit_selects_no_candidate_by_itself": True,
            "allowed_candidate_is_not_video_semantic_success": True,
            "source_candidate_support_is_not_generation_success": True,
            "accepted_prompt_must_drive_candidate_selection": True,
        },
        "next_required_steps": next_steps(allowed, missing_requested_support, unrequested_selection_bias),
    }


def next_steps(
    allowed: bool,
    missing_requested_support: list[str],
    unrequested_selection_bias: list[str],
) -> list[str]:
    if allowed:
        return [
            "use this only as candidate-support evidence",
            "preserve candidate metadata and audit output",
            "run DD2 runtime audit before any GPU claim",
            "review generated video before any semantic claim",
        ]

    steps = ["do not use this candidate for the accepted prompt"]
    if missing_requested_support:
        steps.append("search for a candidate that supports the missing prompt requirements")
    if unrequested_selection_bias:
        steps.append("remove prompt-unsupported selection bias or ask the user to confirm a changed prompt")
    steps.append("record a negative result if no suitable candidate exists")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether a dataset candidate is conditional on the accepted prompt.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = load_json(args.candidate)
    result = audit_candidate(prompt=args.prompt, candidate=candidate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
