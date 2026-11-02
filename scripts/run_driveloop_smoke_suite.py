from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from driveloop import DriveLoopConfig, DriveLoopRequest, DriveLoopRunner
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.backends.mock import MockGenerationBackend
from driveloop.evaluators import RuleBasedEvaluator
from driveloop.intent.adapter import MultimodalInputBundle, RuleBasedIntentAdapter


SMOKE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_id": "smoke_rainy_cut_in",
        "prompt": "rainy night intersection, a pedestrian crosses in front while a car cuts in from the right",
        "metadata": {"modalities": ["text"]},
    },
    {
        "scenario_id": "smoke_foggy_cyclist",
        "prompt": "urban road with unusual hazard",
        "metadata": {
            "modalities": ["text", "image", "voice"],
            "image": {
                "filename": "foggy_night_pedestrian_crossing.png",
                "status": "placeholder",
            },
            "voice": {
                "transcript": "a cyclist cuts in from the left near an intersection",
                "status": "placeholder",
            },
        },
    },
    {
        "scenario_id": "smoke_stopped_vehicle",
        "prompt": "foggy road with a parked vehicle blocking the lane",
        "metadata": {"modalities": ["text"]},
    },
    {
        "scenario_id": "smoke_highway_lane_change",
        "prompt": "daytime highway scene where a vehicle changes lane near the ego car",
        "metadata": {"modalities": ["text"]},
    },
    {
        "scenario_id": "smoke_low_visibility_hazard",
        "prompt": "low visibility urban road with debris obstacle ahead",
        "metadata": {"modalities": ["text"]},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DriveLoop smoke scenarios.")
    parser.add_argument("--backend", choices=["mock", "drivedreamer2"], default="mock")
    parser.add_argument("--output-dir", default="outputs/driveloop/smoke_suite")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--target-score", type=float, default=0.9)
    parser.add_argument("--config-name", default="drivedreamer2_img_cond_mini_local")
    parser.add_argument("--scenario-id", default=None, help="Run only one named smoke scenario.")
    return parser.parse_args()


def build_backend(args: argparse.Namespace, scenario_output_dir: Path):
    if args.backend == "mock":
        return MockGenerationBackend()

    return DriveDreamer2Backend(
        project_root=".",
        config_name=args.config_name,
        artifact_dir=scenario_output_dir / "artifacts",
        timeout_seconds=1800,
    )


def run_scenario(args: argparse.Namespace, scenario: Dict[str, Any], root: Path) -> Dict[str, Any]:
    scenario_output_dir = root / scenario["scenario_id"]
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    structured_intent = RuleBasedIntentAdapter().parse_bundle(
        MultimodalInputBundle(
            text=scenario["prompt"],
            metadata=scenario["metadata"],
        )
    ).to_dict()

    request = DriveLoopRequest(
        prompt=scenario["prompt"],
        scenario_id=scenario["scenario_id"],
        metadata={
            **scenario["metadata"],
            "intent_backend": "rule_based",
            "structured_intent": structured_intent,
        },
    )

    runner = DriveLoopRunner(
        backend=build_backend(args, scenario_output_dir),
        evaluator=RuleBasedEvaluator(),
        config=DriveLoopConfig(
            max_iterations=args.max_iterations,
            target_score=args.target_score,
            output_dir=scenario_output_dir,
        ),
    )
    result = runner.run(request)

    best_generation = asdict(result.best_generation)
    best_evaluation = asdict(result.best_evaluation)
    generation_metadata = best_generation.get("metadata", {})
    dd2_condition = generation_metadata.get("dd2_condition", {})
    executable_condition = (
        dd2_condition.get("executable_condition", {})
        if isinstance(dd2_condition, dict)
        else {}
    )
    condition_trace = {
        "scene_specification": generation_metadata.get("scene_specification"),
        "long_tail_condition_plan": generation_metadata.get("long_tail_condition_plan"),
        "dd2_condition": dd2_condition,
    }

    summary = {
        "scenario_id": scenario["scenario_id"],
        "backend": args.backend,
        "prompt": scenario["prompt"],
        "metadata": scenario["metadata"],
        "structured_intent": structured_intent,
        "condition_trace": condition_trace,
        "executable_condition": executable_condition,
        "accepted": result.return_status == "accepted",
        "returned_generation": asdict(result.returned_generation),
        "returned_evaluation": asdict(result.returned_evaluation),
        "return_status": result.return_status,
        "best_score": best_evaluation["score"],
        "iterations": len(result.history),
        "best_generation": best_generation,
        "best_evaluation": best_evaluation,
    }

    (scenario_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    scenarios = SMOKE_SCENARIOS
    if args.scenario_id:
        scenarios = [scenario for scenario in SMOKE_SCENARIOS if scenario["scenario_id"] == args.scenario_id]
        if not scenarios:
            available = ", ".join(scenario["scenario_id"] for scenario in SMOKE_SCENARIOS)
            raise SystemExit(f"Unknown scenario_id: {args.scenario_id}. Available: {available}")

    summaries = [run_scenario(args, scenario, root) for scenario in scenarios]
    suite_summary = {
        "backend": args.backend,
        "num_scenarios": len(summaries),
        "accepted": sum(1 for item in summaries if item["accepted"]),
        "summaries": summaries,
    }
    (root / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(suite_summary, indent=2))


if __name__ == "__main__":
    main()
