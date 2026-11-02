from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from driveloop import DriveLoopConfig, DriveLoopRequest, DriveLoopRunner
from driveloop.backends import DriveDreamer2Backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DriveLoop with DriveDreamer-2 backend.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--scenario-id", default="drivedreamer2_backend_demo")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--target-score", type=float, default=0.90)
    parser.add_argument("--output-dir", default="outputs/driveloop/drivedreamer2_backend_demo")
    parser.add_argument("--config-name", default="drivedreamer2_img_cond_mini_local")
    parser.add_argument("--dd2-batch-skip", type=int, default=0)
    parser.add_argument("--source-candidate-id", default=None)
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--scene-token", default=None)
    parser.add_argument("--instance-token", default=None)
    parser.add_argument("--source-identity-summary", default=None)
    parser.add_argument("--baseline-dataset-dir", default=None)
    parser.add_argument("--baseline-output-dir", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force-boxes3d-probe", action="store_true")
    parser.add_argument("--boxes3d-probe-category", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    request = DriveLoopRequest(
        prompt=args.prompt,
        scenario_id=args.scenario_id,
    )
    config = DriveLoopConfig(
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        output_dir=args.output_dir,
    )
    backend = DriveDreamer2Backend(
        project_root=".",
        config_name=args.config_name,
        artifact_dir=f"{args.output_dir}/artifacts",
        baseline_output_dir=args.baseline_output_dir or "/data/projects/DriveLoop/outputs/drivedreamer2_img_cond_mini",
        baseline_dataset_dir=args.baseline_dataset_dir or "/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2",
        audit_only=args.audit_only,
        batch_skip=args.dd2_batch_skip,
        source_candidate_id=args.source_candidate_id,
        sample_token=args.sample_token,
        scene_token=args.scene_token,
        instance_token=args.instance_token,
        source_identity_summary_path=args.source_identity_summary,
        force_boxes3d_probe=args.force_boxes3d_probe,
        boxes3d_probe_category=args.boxes3d_probe_category,
    )

    result = DriveLoopRunner(backend=backend, config=config).run(request)
    print(
        json.dumps(
            {
                "request": asdict(result.request),
                "best_generation": asdict(result.best_generation),
                "best_evaluation": asdict(result.best_evaluation),
                "returned_generation": asdict(result.returned_generation),
                "returned_evaluation": asdict(result.returned_evaluation),
                "return_status": result.return_status,
                "selection_metric": result.selection_metric,
                "acceptance_policy": result.acceptance_policy,
                "iterations": len(result.history),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
