from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from driveloop import DriveLoopRequest
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.condition_adapter import DriveDreamer2ConditionAdapter
from driveloop.grounding import RuleBasedGrounder
from driveloop.longtail import LongTailController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official DD2 backend in CPU/light audit-only mode.")
    parser.add_argument("--prompt", default="rainy night road with a traffic barrier blocking the lane")
    parser.add_argument("--scenario-id", default="paper_ch3_backend_audit_only")
    parser.add_argument("--output-dir", default="outputs/driveloop/dd2_backend_audit_only")
    parser.add_argument("--config-name", default="drivedreamer2_img_cond_mini_local")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = DriveLoopRequest(prompt=args.prompt, scenario_id=args.scenario_id)

    scene_spec = RuleBasedGrounder().ground(request)
    condition_plan = LongTailController().build(scene_spec)
    dd2_condition = DriveDreamer2ConditionAdapter().build(scene_spec, condition_plan)

    conditioned_request = DriveLoopRequest(
        prompt=args.prompt,
        scenario_id=args.scenario_id,
        condition={"dd2_condition": asdict(dd2_condition)},
    )

    backend = DriveDreamer2Backend(
        project_root=".",
        config_name=args.config_name,
        artifact_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        audit_only=True,
    )
    generation = backend.generate(conditioned_request, iteration=0)

    summary = {
        "scenario_id": args.scenario_id,
        "prompt": args.prompt,
        "artifacts": generation.artifacts,
        "dd2_audit_only": generation.metadata.get("dd2_audit_only"),
        "dd2_tensor_control_ready": generation.metadata.get("dd2_tensor_control_ready"),
        "runtime_input_audit": generation.metadata.get("dd2_runtime_input_audit"),
        "override_audit": generation.metadata.get("dd2_override_audit"),
        "paper_alignment_stage_3": generation.metadata.get("dd2_paper_alignment_report", {})
        .get("stage_3_scene_consistent_generation"),
    }

    output_path = Path(args.output_dir) / args.scenario_id / "backend_audit_only_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote backend audit-only summary: {output_path}")


if __name__ == "__main__":
    main()
