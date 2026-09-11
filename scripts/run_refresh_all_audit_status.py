from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_candidate70_gpu_readiness_gate import build_candidate70_readiness_gate, write_gate
from scripts.run_candidate70_prompt_bank import write_outputs as write_candidate70_prompt_bank_outputs
from scripts.run_candidate70_source_sample_binding_readiness import build_gate as build_candidate70_source_sample_binding_gate
from scripts.run_candidate_artifact_manifest import build_manifest
from scripts.run_candidate_bundle_validator import validate_bundle
from scripts.run_experiment_status_dashboard import build_dashboard
from scripts.run_gpu_smoke_readiness_gate import build_readiness_report
from scripts.run_gpu_smoke_runbook import render_runbook
from scripts.run_single_gpu_smoke_command_plan import (
    DEFAULT_ALIGNMENT_EVAL_DIR,
    DEFAULT_CONFIG_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POST_GATE_DIR,
    DEFAULT_PROMPT,
    DEFAULT_SCENARIO_ID,
    build_command_plan,
    expected_video_path,
)

DEFAULT_READINESS_OUTPUT = Path("outputs/driveloop/gpu_smoke_readiness/motorcycle_refined_candidate_gate.json")
DEFAULT_COMMAND_PLAN_OUTPUT = Path("outputs/driveloop/gpu_smoke_command_plan/motorcycle_refined_candidate_plan.json")
DEFAULT_RUNBOOK_OUTPUT = Path("outputs/driveloop/gpu_smoke_runbook/motorcycle_refined_candidate_runbook.md")
DEFAULT_MANIFEST_OUTPUT = Path("outputs/driveloop/candidate_artifact_manifest/motorcycle_refined_candidate_manifest.json")
DEFAULT_VALIDATION_OUTPUT = Path("outputs/driveloop/candidate_bundle_validation/motorcycle_refined_candidate_validation.json")
DEFAULT_DASHBOARD_OUTPUT = Path("outputs/driveloop/experiment_status_dashboard/motorcycle_refined_candidate_dashboard.json")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/driveloop/refresh_all_audit_status/motorcycle_refined_candidate_refresh.json")

DEFAULT_RUNTIME_COMPARE = Path("outputs/driveloop/dd2_runtime_hash_compare/motorcycle_earlier_vs_refined.json")
DEFAULT_MOTION_GAP = Path("outputs/driveloop/motion_control_gap_audit/motorcycle_manual_feedback_motion_gap.json")
DEFAULT_VELOCITY_SURFACE = Path("outputs/driveloop/dd2_velocity_surface_audit/mini_velocity_surface.json")
DEFAULT_TRAJECTORY_CONTRACT_DOC = Path("experiments/2026-06-28_trajectory_control_contract_v0.md")
DEFAULT_CONFIG_PATH = Path("dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py")
DEFAULT_LABELS_PATH = Path("/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2/labels/data.pkl")
DEFAULT_WEIGHTS_PATH = Path("/data/projects/DriveLoop/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin")
DEFAULT_EVIDENCE_INDEX = Path("experiments/2026-06-28_motorcycle_alignment_evidence_index.md")
DEFAULT_CLAIM_TABLE = Path("experiments/2026-06-28_paper_claim_table_v0.md")
DEFAULT_PROMPT_OBJECT_TRANSFER_AUDIT = Path("outputs/driveloop/prompt_object_transfer_audit/motorcycle_refined_object_transfer_audit.json")
DEFAULT_TRAJECTORY_RUNTIME_SURFACE_AUDIT = Path("outputs/driveloop/trajectory_runtime_surface_audit/motorcycle_refined_trajectory_runtime_surface_audit.json")
DEFAULT_RUNTIME_SURFACE_CODE_AUDIT = Path("outputs/driveloop/runtime_surface_code_audit/motorcycle_refined_runtime_surface_code_audit.json")
DEFAULT_MOTION_METADATA_RUNTIME_AUDIT = Path("outputs/driveloop/motorcycle_motion_metadata_audit_only/motorcycle_motion_metadata_audit_only/dd2_runtime_input_audit_00.json")
DEFAULT_ACTOR_IDENTITY_SURFACE_AUDIT = Path("outputs/driveloop/actor_identity_surface_audit/mini_actor_identity_surface_audit.json")
DEFAULT_CANDIDATE70_PROMPT_BANK_OUTPUT = Path("outputs/driveloop/prompt_bank/candidate70_prompt_bank_v0.json")
DEFAULT_CANDIDATE70_PROMPT_BANK_SUPPORT_AUDIT_OUTPUT = Path("outputs/driveloop/prompt_bank/candidate70_prompt_bank_support_audit_v0.json")
DEFAULT_CANDIDATE70_ACCEPTED_PROMPT_SELECTION = Path("outputs/driveloop/accepted_prompt/candidate70_accepted_prompt_v0.json")
DEFAULT_CANDIDATE70_GPU_READINESS_OUTPUT = Path("outputs/driveloop/gpu_smoke_readiness/candidate70_gpu_readiness_gate.json")
DEFAULT_CANDIDATE70_GPU_SMOKE_PLAN_DRAFT = Path("outputs/driveloop/gpu_smoke_command_plan/candidate70_night_cut_in_plan_draft.json")
DEFAULT_CANDIDATE70_SOURCE_SAMPLE_BINDING_READINESS = Path("outputs/driveloop/source_sample_binding_readiness/candidate70_source_sample_binding_readiness.json")
DEFAULT_CANDIDATE70_RUNTIME_SURFACE_AUDIT = Path("outputs/driveloop/runtime_surface_code_audit/candidate70_runtime_surface_code_audit.json")
DEFAULT_CANDIDATE70_TRAJECTORY_SURFACE_AUDIT = Path("outputs/driveloop/trajectory_runtime_surface_audit/candidate70_night_cut_in_trajectory_runtime_surface_audit.json")
DEFAULT_CANDIDATE70_DRY_RUN_REPLACEMENT_AUDIT = Path("outputs/driveloop/candidate70_hdmap_dry_run_replacement_surface_audit/candidate70_dry_run_raster_to_grounding_surface.json")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def artifact_entry(path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path), "exists_after_refresh": path.exists()}


def build_refresh_summary(
    *,
    prompt: str,
    scenario_id: str,
    readiness_output: Path,
    command_plan_output: Path,
    runbook_output: Path,
    manifest_output: Path,
    validation_output: Path,
    dashboard_output: Path,
    readiness: dict[str, Any],
    manifest: dict[str, Any],
    validation: dict[str, Any],
    dashboard: dict[str, Any],
    prompt_object_transfer_audit: Path,
    trajectory_runtime_surface_audit: Path,
    runtime_surface_code_audit: Path,
    motion_metadata_runtime_audit: Path,
    actor_identity_surface_audit: Path,
    candidate70_prompt_bank_output: Path,
    candidate70_prompt_bank_support_audit_output: Path,
    candidate70_gpu_readiness_output: Path,
    candidate70_gpu_smoke_plan_draft: Path,
    candidate70_source_sample_binding_readiness_output: Path,
    candidate70_gpu_smoke_plan: dict[str, Any],
    candidate70_gpu_readiness: dict[str, Any],
    candidate70_source_sample_binding_readiness: dict[str, Any],
) -> dict[str, Any]:
    dashboard_summary = dashboard.get("summary", {})
    if not isinstance(dashboard_summary, dict):
        dashboard_summary = {}

    legacy_scenario_id = scenario_id
    legacy_prompt = prompt
    candidate70_readiness_active = bool(candidate70_gpu_readiness) and (
        candidate70_gpu_readiness.get("schema_version") == "driveloop_candidate70_gpu_readiness_gate.v0"
        or "readiness_status" in candidate70_gpu_readiness
        or "gpu_smoke_allowed" in candidate70_gpu_readiness
    )
    candidate70_scenario_id = (
        candidate70_gpu_readiness.get("scenario_id")
        or candidate70_gpu_smoke_plan.get("scenario_id")
    )
    candidate70_prompt = (
        candidate70_gpu_smoke_plan.get("prompt")
        or candidate70_gpu_smoke_plan.get("accepted_prompt")
        or candidate70_gpu_smoke_plan.get("selected_prompt")
    )
    active_scenario_id = (
        (candidate70_scenario_id or legacy_scenario_id)
        if candidate70_readiness_active
        else legacy_scenario_id
    )
    active_prompt = (
        (candidate70_prompt or legacy_prompt)
        if candidate70_readiness_active
        else legacy_prompt
    )

    return {
        "schema_version": "driveloop_refresh_all_audit_status.v0",
        "scenario_id": active_scenario_id,
        "prompt": active_prompt,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "semantic_success_claim_allowed": False,
        "refreshed_artifacts": {
            "readiness_gate": artifact_entry(readiness_output, "gpu_smoke_readiness_gate"),
            "command_plan": artifact_entry(command_plan_output, "single_gpu_smoke_command_plan"),
            "runbook": artifact_entry(runbook_output, "gpu_smoke_runbook"),
            "candidate_manifest": artifact_entry(manifest_output, "candidate_artifact_manifest"),
            "bundle_validation": artifact_entry(validation_output, "candidate_bundle_validation"),
            "experiment_dashboard": artifact_entry(dashboard_output, "experiment_status_dashboard"),
            "prompt_object_transfer_audit": artifact_entry(prompt_object_transfer_audit, "prompt_object_transfer_audit"),
            "trajectory_runtime_surface_audit": artifact_entry(trajectory_runtime_surface_audit, "trajectory_runtime_surface_audit"),
            "runtime_surface_code_audit": artifact_entry(runtime_surface_code_audit, "runtime_surface_code_audit"),
            "motion_metadata_runtime_audit": artifact_entry(motion_metadata_runtime_audit, "motion_metadata_runtime_audit"),
            "actor_identity_surface_audit": artifact_entry(actor_identity_surface_audit, "actor_identity_surface_audit"),
            "candidate70_prompt_bank": artifact_entry(candidate70_prompt_bank_output, "candidate70_prompt_bank"),
            "candidate70_prompt_bank_support_audit": artifact_entry(candidate70_prompt_bank_support_audit_output, "candidate70_prompt_bank_support_audit"),
            "candidate70_gpu_readiness_gate": artifact_entry(candidate70_gpu_readiness_output, "candidate70_gpu_readiness_gate"),
            "candidate70_gpu_smoke_plan_draft": artifact_entry(candidate70_gpu_smoke_plan_draft, "candidate70_gpu_smoke_plan_draft"),
            "candidate70_source_sample_binding_readiness": artifact_entry(candidate70_source_sample_binding_readiness_output, "candidate70_source_sample_binding_readiness"),
        },
        "refresh_order": [
            "readiness_gate",
            "command_plan",
            "runbook",
            "candidate_manifest",
            "bundle_validation",
            "experiment_dashboard",
            "prompt_object_transfer_audit",
            "trajectory_runtime_surface_audit",
            "runtime_surface_code_audit",
            "motion_metadata_runtime_audit",
            "actor_identity_surface_audit",
            "candidate70_prompt_bank",
            "candidate70_prompt_bank_support_audit",
            "candidate70_gpu_readiness_gate",
            "candidate70_gpu_smoke_plan_draft",
            "candidate70_source_sample_binding_readiness",
        ],
        "status_summary": {
            "gpu_smoke_allowed": (
                dashboard.get("summary", {}).get("gpu_smoke_allowed")
                if isinstance(dashboard.get("summary"), dict)
                and "gpu_smoke_allowed" in dashboard.get("summary", {})
                else readiness.get("gpu_smoke_allowed")
            ),
            "legacy_gpu_smoke_allowed": readiness.get("gpu_smoke_allowed"),
            "active_gpu_smoke_readiness_source": (
                dashboard_summary.get("active_gpu_smoke_readiness_source")
                if isinstance(dashboard_summary, dict)
                else None
            ),
            "active_scenario_id": dashboard_summary.get("active_scenario_id", active_scenario_id),
            "active_prompt": dashboard_summary.get("active_prompt", active_prompt),
            "legacy_scenario_id": legacy_scenario_id,
            "legacy_prompt": legacy_prompt,
            "candidate70_scenario_id": candidate70_scenario_id,
            "candidate70_prompt": candidate70_prompt,
            "candidate70_prompt_id": candidate70_gpu_smoke_plan.get("selected_prompt_id"),
            "candidate_status": manifest.get("candidate_status"),
            "bundle_status": validation.get("bundle_status"),
            "dashboard_status": dashboard.get("dashboard_status"),
            "video_semantic_claim": dashboard.get("summary", {}).get(
                "video_semantic_claim", manifest.get("video_semantic_claim")
            ),
            "candidate_manifest_video_semantic_claim": manifest.get("video_semantic_claim"),
            "dashboard_semantic_success_claim_allowed": dashboard.get("summary", {}).get(
                "semantic_success_claim_allowed"
            ),
            "object_transfer_status": dashboard.get("summary", {}).get("object_transfer_status"),
            "trajectory_runtime_surface_status": dashboard.get("summary", {}).get(
                "trajectory_runtime_surface_status"
            ),
            "runtime_surface_code_audit_status": dashboard.get("summary", {}).get(
                "runtime_surface_code_audit_status"
            ),
            "motion_metadata_runtime_status": dashboard.get("summary", {}).get(
                "motion_metadata_runtime_status"
            ),
            "actor_identity_surface_status": dashboard.get("summary", {}).get(
                "actor_identity_surface_status"
            ),
            "candidate70_gpu_readiness_status": candidate70_gpu_readiness.get("readiness_status"),
            "candidate70_gpu_smoke_allowed": candidate70_gpu_readiness.get("gpu_smoke_allowed"),
            "candidate70_gpu_readiness_blockers": candidate70_gpu_readiness.get("blockers", []),
            "candidate70_gpu_smoke_plan_status": candidate70_gpu_smoke_plan.get("plan_status"),
            "candidate70_gpu_smoke_plan_selected_prompt_id": candidate70_gpu_smoke_plan.get("selected_prompt_id"),
            "candidate70_gpu_smoke_plan_allowed_at_plan_time": candidate70_gpu_smoke_plan.get("gpu_smoke_allowed_at_plan_time"),
            "candidate70_gpu_smoke_plan_blockers_at_plan_time": candidate70_gpu_smoke_plan.get("readiness_blockers_at_plan_time", []),
            "candidate70_source_sample_binding_readiness_status": candidate70_source_sample_binding_readiness.get("readiness_status"),
            "candidate70_source_sample_binding_gpu_smoke_allowed": candidate70_source_sample_binding_readiness.get("gpu_smoke_allowed"),
            "candidate70_source_sample_binding_blockers": candidate70_source_sample_binding_readiness.get("blockers", []),
        },
        "claim_boundary": {
            "refresh_all_is_audit_only": True,
            "video_generation_is_not_semantic_success": True,
            "runtime_tensor_audit_is_not_video_semantic_success": True,
            "semantic_success_requires_explicit_measured_passed_review": True,
            "candidate70_readiness_gate_is_not_gpu_approval": True,
            "candidate70_readiness_gate_is_not_video_semantic_success": True,
            "candidate70_source_sample_binding_gate_is_not_gpu_approval": True,
            "candidate70_converter_identity_subset_is_not_runtime_binding": True,
        },
    }


def refresh_all(
    *,
    prompt: str = DEFAULT_PROMPT,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    readiness_output: Path = DEFAULT_READINESS_OUTPUT,
    command_plan_output: Path = DEFAULT_COMMAND_PLAN_OUTPUT,
    runbook_output: Path = DEFAULT_RUNBOOK_OUTPUT,
    manifest_output: Path = DEFAULT_MANIFEST_OUTPUT,
    validation_output: Path = DEFAULT_VALIDATION_OUTPUT,
    dashboard_output: Path = DEFAULT_DASHBOARD_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    post_gate_dir: Path = DEFAULT_POST_GATE_DIR,
    alignment_eval_dir: Path = DEFAULT_ALIGNMENT_EVAL_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    runtime_compare: Path = DEFAULT_RUNTIME_COMPARE,
    motion_gap: Path = DEFAULT_MOTION_GAP,
    velocity_surface: Path = DEFAULT_VELOCITY_SURFACE,
    trajectory_contract_doc: Path = DEFAULT_TRAJECTORY_CONTRACT_DOC,
    config_path: Path = DEFAULT_CONFIG_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    evidence_index: Path = DEFAULT_EVIDENCE_INDEX,
    claim_table: Path = DEFAULT_CLAIM_TABLE,
    runtime_audit: Path | None = None,
    prompt_object_transfer_audit: Path = DEFAULT_PROMPT_OBJECT_TRANSFER_AUDIT,
    trajectory_runtime_surface_audit: Path = DEFAULT_TRAJECTORY_RUNTIME_SURFACE_AUDIT,
    runtime_surface_code_audit: Path = DEFAULT_RUNTIME_SURFACE_CODE_AUDIT,
    motion_metadata_runtime_audit: Path = DEFAULT_MOTION_METADATA_RUNTIME_AUDIT,
    actor_identity_surface_audit: Path = DEFAULT_ACTOR_IDENTITY_SURFACE_AUDIT,
    candidate70_prompt_bank_output: Path = DEFAULT_CANDIDATE70_PROMPT_BANK_OUTPUT,
    candidate70_prompt_bank_support_audit_output: Path = DEFAULT_CANDIDATE70_PROMPT_BANK_SUPPORT_AUDIT_OUTPUT,
    candidate70_accepted_prompt_selection: Path = DEFAULT_CANDIDATE70_ACCEPTED_PROMPT_SELECTION,
    candidate70_gpu_readiness_output: Path = DEFAULT_CANDIDATE70_GPU_READINESS_OUTPUT,
    candidate70_gpu_smoke_plan_draft: Path = DEFAULT_CANDIDATE70_GPU_SMOKE_PLAN_DRAFT,
    candidate70_source_sample_binding_readiness_output: Path = DEFAULT_CANDIDATE70_SOURCE_SAMPLE_BINDING_READINESS,
    candidate70_runtime_surface_audit: Path = DEFAULT_CANDIDATE70_RUNTIME_SURFACE_AUDIT,
    candidate70_trajectory_surface_audit: Path = DEFAULT_CANDIDATE70_TRAJECTORY_SURFACE_AUDIT,
    candidate70_dry_run_replacement_audit: Path = DEFAULT_CANDIDATE70_DRY_RUN_REPLACEMENT_AUDIT,
) -> dict[str, Any]:
    readiness = build_readiness_report(
        prompt=prompt,
        scenario_id=scenario_id,
        runtime_compare=runtime_compare,
        motion_gap=motion_gap,
        velocity_surface=velocity_surface,
        trajectory_contract_doc=trajectory_contract_doc,
        config_path=config_path,
        labels_path=labels_path,
        weights_path=weights_path,
    )
    write_json(readiness_output, readiness)

    command_plan = build_command_plan(
        prompt=prompt,
        scenario_id=scenario_id,
        output_dir=output_dir,
        readiness_output=readiness_output,
        post_gate_dir=post_gate_dir,
        alignment_eval_dir=alignment_eval_dir,
        config_name=config_name,
    )
    write_json(command_plan_output, command_plan)

    runbook = render_runbook(command_plan)
    write_text(runbook_output, runbook)

    video_path = expected_video_path(output_dir, scenario_id)
    runtime_audit_path = runtime_audit or (output_dir / "artifacts" / scenario_id / "dd2_runtime_input_audit_00.json")
    manifest = build_manifest(
        prompt=prompt,
        scenario_id=scenario_id,
        video_path=video_path,
        readiness_gate=readiness_output,
        command_plan=command_plan_output,
        runbook=runbook_output,
        post_gpu_gate=post_gate_dir / "post_gpu_review_gate.json",
        manual_report=post_gate_dir / "manual_review_pack" / "manual_alignment_report.json",
        alignment_eval=alignment_eval_dir / f"{scenario_id}_manual_review" / "prompt_video_alignment_evaluation.json",
        runtime_audit=runtime_audit_path,
    )
    write_json(manifest_output, manifest)

    validation = validate_bundle(manifest)
    write_json(validation_output, validation)

    write_candidate70_prompt_bank_outputs(
        prompt_bank_output=candidate70_prompt_bank_output,
        support_audit_output=candidate70_prompt_bank_support_audit_output,
    )
    candidate70_gpu_readiness = build_candidate70_readiness_gate(
        prompt_bank_audit_path=candidate70_prompt_bank_support_audit_output,
        accepted_prompt_selection_path=candidate70_accepted_prompt_selection,
        runtime_surface_audit_path=candidate70_runtime_surface_audit,
        trajectory_surface_audit_path=candidate70_trajectory_surface_audit,
        dry_run_replacement_audit_path=candidate70_dry_run_replacement_audit,
    )
    write_gate(candidate70_gpu_readiness_output, candidate70_gpu_readiness)
    candidate70_source_sample_binding_readiness = build_candidate70_source_sample_binding_gate()
    write_json(candidate70_source_sample_binding_readiness_output, candidate70_source_sample_binding_readiness)
    candidate70_gpu_smoke_plan = load_json(candidate70_gpu_smoke_plan_draft)

    dashboard = build_dashboard(
        readiness_path=readiness_output,
        manifest_path=manifest_output,
        bundle_validation_path=validation_output,
        runtime_compare_path=runtime_compare,
        motion_gap_path=motion_gap,
        velocity_audit_path=velocity_surface,
        evidence_index_path=evidence_index,
        claim_table_path=claim_table,
        prompt_object_transfer_audit_path=prompt_object_transfer_audit,
        trajectory_runtime_surface_audit_path=trajectory_runtime_surface_audit,
        runtime_surface_code_audit_path=runtime_surface_code_audit,
        motion_metadata_runtime_audit_path=motion_metadata_runtime_audit,
        actor_identity_surface_audit_path=actor_identity_surface_audit,
        candidate70_gpu_readiness_gate_path=candidate70_gpu_readiness_output,
        candidate70_gpu_smoke_plan_draft_path=candidate70_gpu_smoke_plan_draft,
        candidate70_source_sample_binding_readiness_path=candidate70_source_sample_binding_readiness_output,
    )
    write_json(dashboard_output, dashboard)

    summary = build_refresh_summary(
        prompt=prompt,
        scenario_id=scenario_id,
        readiness_output=readiness_output,
        command_plan_output=command_plan_output,
        runbook_output=runbook_output,
        manifest_output=manifest_output,
        validation_output=validation_output,
        dashboard_output=dashboard_output,
        readiness=readiness,
        manifest=manifest,
        validation=validation,
        dashboard=dashboard,
        prompt_object_transfer_audit=prompt_object_transfer_audit,
        trajectory_runtime_surface_audit=trajectory_runtime_surface_audit,
        runtime_surface_code_audit=runtime_surface_code_audit,
        motion_metadata_runtime_audit=motion_metadata_runtime_audit,
        actor_identity_surface_audit=actor_identity_surface_audit,
        candidate70_prompt_bank_output=candidate70_prompt_bank_output,
        candidate70_prompt_bank_support_audit_output=candidate70_prompt_bank_support_audit_output,
        candidate70_gpu_readiness_output=candidate70_gpu_readiness_output,
        candidate70_gpu_smoke_plan_draft=candidate70_gpu_smoke_plan_draft,
        candidate70_source_sample_binding_readiness_output=candidate70_source_sample_binding_readiness_output,
        candidate70_gpu_smoke_plan=candidate70_gpu_smoke_plan,
        candidate70_gpu_readiness=candidate70_gpu_readiness,
        candidate70_source_sample_binding_readiness=candidate70_source_sample_binding_readiness,
    )
    write_json(summary_output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh DriveLoop audit/status artifacts without running GPU generation."
    )
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()

    summary = refresh_all(summary_output=args.summary_output)
    print(args.summary_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
