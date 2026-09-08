from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
from typing import Any

from scripts.run_manual_alignment_review_pack import create_review_pack
from scripts.run_prompt_video_alignment_eval import build_generation, evaluate_generation


def build_not_measured_payload(
    prompt: str,
    scenario_id: str,
    video_path: Path,
    review_pack_manifest: dict[str, Any] | None = None,
    pass_threshold: float = 0.8,
) -> dict[str, Any]:
    generation = build_generation(
        Namespace(
            prompt=prompt,
            scenario_id=scenario_id,
            video_path=str(video_path),
            alignment_report=None,
        )
    )
    alignment_payload = evaluate_generation(generation, pass_threshold=pass_threshold)

    return {
        "schema_version": "driveloop_post_gpu_review_gate.v0",
        "scenario_id": scenario_id,
        "prompt": prompt,
        "video_path": str(video_path),
        "candidate_video_available": video_path.exists(),
        "alignment_evaluation": alignment_payload,
        "review_pack_manifest": review_pack_manifest,
        "review_status": "requires_manual_or_perception_review",
        "video_semantic_claim": "not_measured",
        "next_required_steps": [
            "inspect the generated video or contact sheet",
            "edit the manual alignment report with explicit pass/fail checks and evidence",
            "run scripts/run_prompt_video_alignment_eval.py with the completed report",
            "record measured_failed or measured_passed only from the explicit review report",
        ],
        "claim_boundary": (
            "A GPU-generated video is only a candidate artifact. This gate does not inspect pixels "
            "and does not permit prompt-video semantic success claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a GPU smoke video before semantic review.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--pass-threshold", type=float, default=0.8)
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(f"video artifact does not exist: {args.video_path}")

    review_manifest = None
    if not args.skip_review_pack:
        review_manifest = create_review_pack(
            video_path=args.video_path,
            prompt=args.prompt,
            output_dir=args.output_dir / "manual_review_pack",
        )

    payload = build_not_measured_payload(
        prompt=args.prompt,
        scenario_id=args.scenario_id,
        video_path=args.video_path,
        review_pack_manifest=review_manifest,
        pass_threshold=args.pass_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "post_gpu_review_gate.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
