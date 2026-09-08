from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_prompt_conditional_candidate_audit import RULES, contains_any


DEFAULT_BACKEND_SUMMARY = Path(
    "outputs/driveloop/motorcycle_manual_feedback_dd2_audit_only/"
    "motorcycle_manual_feedback_dd2_audit_only/backend_audit_only_summary.json"
)
DEFAULT_PAPER_ALIGNMENT_REPORT = Path(
    "outputs/driveloop/motorcycle_manual_feedback_dd2_audit_only/"
    "motorcycle_manual_feedback_dd2_audit_only/paper_alignment_report_00.json"
)


OBJECT_ALIASES = {
    "motorbike": "motorcycle",
    "scooter": "motorcycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "person": "pedestrian",
    "people": "pedestrian",
    "walker": "pedestrian",
    "vehicle": "car",
    "vehicles": "car",
    "truck": "car",
    "bus": "car",
}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def canonical_object(value: Any) -> str:
    text = str(value).lower().strip().replace("_", " ").replace("-", " ")
    return OBJECT_ALIASES.get(text, text)


def requested_prompt_objects(prompt: str) -> list[str]:
    objects: list[str] = []
    for name, rule in RULES.items():
        if rule.get("type") != "object":
            continue
        if contains_any(prompt, list(rule.get("prompt_aliases", []))):
            objects.append(canonical_object(name))
    return list(dict.fromkeys(objects))


def first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def extract_executable_condition(
    summary: dict[str, Any],
    paper_alignment_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    paper_alignment_report = paper_alignment_report or {}

    paper_stage_1 = first_dict(
        paper_alignment_report.get("stage_1_multimodal_prompt_grounding")
    )
    paper_stage_3 = first_dict(
        summary.get("paper_alignment_stage_3"),
        paper_alignment_report.get("stage_3_scene_consistent_generation"),
    )

    synthesized_from_paper_report: dict[str, Any] = {}
    if paper_stage_1.get("actor_controls") or paper_stage_3.get("structural_input_plan"):
        synthesized_from_paper_report = {
            "actor_controls": list(paper_stage_1.get("actor_controls", [])),
            "structural_input_plan": dict(paper_stage_3.get("structural_input_plan", {})),
        }

    return first_dict(
        summary.get("dd2_executable_condition"),
        metadata.get("dd2_executable_condition") if isinstance(metadata, dict) else None,
        summary.get("metadata", {}).get("dd2_condition", {}).get("executable_condition")
        if isinstance(summary.get("metadata", {}).get("dd2_condition"), dict)
        else None,
        synthesized_from_paper_report if synthesized_from_paper_report else None,
    )


def extract_runtime_input_audit(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    return first_dict(
        summary.get("runtime_input_audit"),
        summary.get("dd2_runtime_input_audit"),
        metadata.get("dd2_runtime_input_audit") if isinstance(metadata, dict) else None,
    )


def extract_override_audit(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    return first_dict(
        summary.get("override_audit"),
        summary.get("dd2_override_audit"),
        metadata.get("dd2_override_audit") if isinstance(metadata, dict) else None,
    )


def actor_categories(executable_condition: dict[str, Any]) -> list[str]:
    actors = executable_condition.get("actor_controls", [])
    if not isinstance(actors, list):
        return []
    return list(
        dict.fromkeys(
            canonical_object(actor.get("category"))
            for actor in actors
            if isinstance(actor, dict) and actor.get("category")
        )
    )


def structural_labels(executable_condition: dict[str, Any]) -> list[str]:
    plan = executable_condition.get("structural_input_plan", {})
    labels = plan.get("labels", {}) if isinstance(plan, dict) else {}
    values = labels.get("values", []) if isinstance(labels, dict) else []
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(canonical_object(value) for value in values))


def override_box_categories(override_audit: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    entries = override_audit.get("entries_preview", [])
    if not isinstance(entries, list):
        return []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for applied in entry.get("applied", []):
            if not isinstance(applied, dict):
                continue
            for accepted in applied.get("accepted_entries", []):
                if isinstance(accepted, dict) and accepted.get("category"):
                    categories.append(canonical_object(accepted.get("category")))

    return list(dict.fromkeys(categories))


def missing(requested: list[str], observed: list[str]) -> list[str]:
    observed_set = set(observed)
    return [item for item in requested if item not in observed_set]


def build_audit(
    prompt: str,
    backend_summary: dict[str, Any],
    paper_alignment_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested = requested_prompt_objects(prompt)
    executable_condition = extract_executable_condition(
        backend_summary,
        paper_alignment_report=paper_alignment_report,
    )
    runtime_audit = extract_runtime_input_audit(backend_summary)
    override_audit = extract_override_audit(backend_summary)

    observed_actor_categories = actor_categories(executable_condition)
    observed_structural_labels = structural_labels(executable_condition)
    observed_override_boxes = override_box_categories(override_audit)

    missing_actor_controls = missing(requested, observed_actor_categories)
    missing_structural_labels = missing(requested, observed_structural_labels)
    missing_override_boxes = missing(requested, observed_override_boxes)

    runtime_tensor_available = bool(
        runtime_audit.get("box_downsampler_input", {}).get("available")
        or runtime_audit.get("grounding_downsampler_input", {}).get("available")
    )
    runtime_class_label_observable = False

    blockers: list[str] = []
    if requested and missing_actor_controls:
        blockers.append("prompt_to_executable_actor_control_missing")
    if requested and missing_structural_labels:
        blockers.append("executable_condition_to_structural_label_missing")
    if requested and missing_override_boxes:
        blockers.append("structural_label_to_override_box_missing")
    if requested and runtime_tensor_available and not runtime_class_label_observable:
        blockers.append("runtime_tensor_class_label_not_directly_observable")

    if not requested:
        status = "not_applicable"
        status_reason = "accepted prompt does not request a known object class"
    elif blockers == ["runtime_tensor_class_label_not_directly_observable"]:
        status = "partially_verified"
        status_reason = "object transfer is visible through condition and override, but runtime tensor class labels are not directly observable"
    elif blockers:
        status = "blocked"
        status_reason = "object transfer is missing before or inside DD2 override"
    else:
        status = "verified_to_override"
        status_reason = "object transfer is visible through condition and override"

    return {
        "schema_version": "driveloop_prompt_object_transfer_audit.v0",
        "accepted_prompt": prompt,
        "requested_objects": requested,
        "status": status,
        "status_reason": status_reason,
        "checks": {
            "executable_actor_controls": {
                "observed_categories": observed_actor_categories,
                "missing_requested_objects": missing_actor_controls,
            },
            "structural_input_plan_labels": {
                "observed_labels": observed_structural_labels,
                "missing_requested_objects": missing_structural_labels,
            },
            "override_appended_boxes": {
                "observed_categories": observed_override_boxes,
                "missing_requested_objects": missing_override_boxes,
            },
            "runtime_tensor_class_labels": {
                "runtime_tensor_available": runtime_tensor_available,
                "class_label_observable": runtime_class_label_observable,
                "reason": "dd2_runtime_input_audit records tensor signatures, not decoded per-object class labels",
            },
        },
        "blockers": blockers,
        "claim_boundary": {
            "object_transfer_audit_is_not_video_semantic_success": True,
            "override_box_presence_is_not_object_visual_success": True,
            "runtime_tensor_signature_is_not_decoded_class_label": True,
            "semantic_success_requires_measured_passed_review": True,
        },
        "next_required_steps": next_steps(status, blockers),
    }


def next_steps(status: str, blockers: list[str]) -> list[str]:
    if status == "not_applicable":
        return ["use a prompt with explicit object requirements before object transfer audit"]
    if "prompt_to_executable_actor_control_missing" in blockers:
        return ["fix prompt grounding or condition adapter before DD2 audit"]
    if "executable_condition_to_structural_label_missing" in blockers:
        return ["fix structural_input_plan label transfer"]
    if "structural_label_to_override_box_missing" in blockers:
        return ["fix DD2 override box synthesis for requested object"]
    return [
        "preserve object transfer evidence",
        "do not claim visual object success from transfer evidence alone",
        "use explicit review/perception/VLM evidence to assess generated object identity",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generic prompt-requested object transfer into DD2 condition and override evidence.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--backend-summary", type=Path, default=DEFAULT_BACKEND_SUMMARY)
    parser.add_argument("--paper-alignment-report", type=Path, default=DEFAULT_PAPER_ALIGNMENT_REPORT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    audit = build_audit(
        prompt=args.prompt,
        backend_summary=load_json(args.backend_summary),
        paper_alignment_report=load_json(args.paper_alignment_report),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
