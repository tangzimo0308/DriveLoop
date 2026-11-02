#!/usr/bin/env python
"""Render a case manifest on a source window, reading the window's binding
(dataset_dir + sample/scene/instance tokens + identity summary) byte-exact from
a no-injection baseline run directory, so tokens are never retyped.

The arm recipe -- released vs fine-tune checkpoint, and the real-track dims scale
-- is set by the caller through the environment, not here:

    DRIVELOOP_EGO_INJECTION=1                 required to inject the actor
    DRIVELOOP_DD2_SEED_BANK=0                  seed bank (bank0 = archived arms)
    DRIVELOOP_DD2_WEIGHT_PATH=<gligen.bin>     set for the fine-tune arm; unset = released
    DRIVELOOP_EGO_REAL_TRACK_DIMS_SCALE=1.5    set for the dims arm; unset = 1.0

This wrapper only assembles the window-binding arguments and invokes
run_driveloop_experiment. The dd2 batch skip is recomputed from the binding by
the backend, so it is not passed here. Use --print-only for a dry run that shows
the assembled command without touching the GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from scripts.run_driveloop_experiment import main as experiment_main
from driveloop.acceptance import POLICIES
from scripts.run_window_admission_probe import (
    resolve_baseline_video,
    source_config_from_baseline_dir,
)


def build_experiment_argv(config: Dict[str, Any], baseline_video: Any, args: argparse.Namespace) -> List[str]:
    argv = [
        "--cases", str(args.cases),
        "--output-dir", str(args.output_dir),
        "--backend", "drivedreamer2",
        "--config-name", args.config_name,
        "--source-candidate-id", str(config["source_candidate_id"]),
        "--baseline-dataset-dir", str(config["dataset_dir"]),
        "--max-iterations", str(args.max_iterations),
        "--target-score", str(args.target_score),
    ]
    for flag, key in (
        ("--scene-token", "scene_token"),
        ("--sample-token", "sample_token"),
        ("--instance-token", "instance_token"),
        ("--source-identity-summary", "identity_summary_path"),
    ):
        value = config.get(key)
        if value:
            argv += [flag, str(value)]
    if baseline_video:
        argv += ["--perception-baseline-video", str(baseline_video)]
    if args.perception_weights:
        argv += ["--perception-weights", args.perception_weights,
                 "--perception-confidence", str(args.perception_confidence)]
    if args.use_task_utility:
        argv += ["--use-task-utility"]
    argv += ["--acceptance-policy", getattr(args, "acceptance_policy", "score_and_diagnosis"),
             "--perception-acceptance-threshold", str(getattr(args, "perception_acceptance_threshold", 0.3))]
    if getattr(args, "fixed_budget", False):
        argv += ["--fixed-budget"]
    return argv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a case manifest on a window; binding read byte-exact from a "
                    "baseline run directory. Arm recipe (checkpoint / dims) is set via the environment."
    )
    parser.add_argument("--source-from-baseline-dir", required=True, type=Path,
                        help="no-injection baseline run dir to read the window binding from")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-video", type=Path, default=None,
                        help="override the resolved no-injection baseline video")
    parser.add_argument("--config-name", default="drivedreamer2_img_cond_mini_local")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--target-score", type=float, default=0.99)
    parser.add_argument("--acceptance-policy", choices=POLICIES, default="score_and_diagnosis")
    parser.add_argument("--perception-acceptance-threshold", type=float, default=0.3)
    parser.add_argument("--fixed-budget", action="store_true")
    parser.add_argument("--perception-weights", default=None)
    parser.add_argument("--perception-confidence", type=float, default=0.20)
    parser.add_argument("--use-task-utility", action="store_true")
    parser.add_argument("--print-only", action="store_true",
                        help="print the assembled experiment command and exit (no render)")
    args = parser.parse_args(argv)

    config = source_config_from_baseline_dir(args.source_from_baseline_dir)
    baseline_video = args.baseline_video or resolve_baseline_video(args.source_from_baseline_dir)
    experiment_argv = build_experiment_argv(config, baseline_video, args)
    print("window=%s  dataset_dir=%s  baseline_video=%s"
          % (config.get("source_candidate_id"), config.get("dataset_dir"), baseline_video))
    print("run_driveloop_experiment argv:\n  %s" % " ".join(experiment_argv))
    if args.print_only:
        return 0
    return experiment_main(experiment_argv)


if __name__ == "__main__":
    sys.exit(main())
