from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manual prompt-video alignment review pack.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=180)
    return parser.parse_args()


def sample_frame_indices(frame_count: int, num_frames: int) -> List[int]:
    if frame_count <= 0:
        return []
    if num_frames <= 1:
        return [0]
    return sorted(set(int(i * max(frame_count - 1, 0) / (num_frames - 1)) for i in range(num_frames)))


def _is_candidate70_prompt(prompt: str) -> bool:
    normalized = prompt.lower().replace("_", " ").replace("-", " ")
    return (
        "candidate70" in normalized
        or (
            "night" in normalized
            and "motorcycle" in normalized
            and "cut in" in normalized
        )
    )


def _legacy_default_checks(prompt: str = "") -> List[Dict[str, Any]]:
    lighting_check = "lighting.night" if "night" in prompt.lower() else "lighting.daytime"
    return [
        {"name": "object_presence.motorcycle", "required": True, "passed": False, "score": 0.0, "evidence": "not_reviewed"},
        {"name": "spatial_relation.left_lane_change", "required": True, "passed": False, "score": 0.0, "evidence": "not_reviewed"},
        {"name": lighting_check, "required": True, "passed": False, "score": 0.0, "evidence": "not_reviewed"},
        {"name": "scene_type.urban_road", "required": True, "passed": False, "score": 0.0, "evidence": "not_reviewed"},
    ]


def default_checks(prompt: str = "") -> List[Dict[str, Any]]:
    if _is_candidate70_prompt(prompt):
        from scripts.run_candidate70_semantic_alignment_protocol import required_semantic_checks

        return required_semantic_checks()
    return _legacy_default_checks(prompt)


def build_report(video_path: Path, contact_sheet: Path, prompt: str) -> Dict[str, Any]:
    return {
        "status": "not_measured",
        "source": "manual_review_frame_pack_v0",
        "review_scope": {
            "video": str(video_path),
            "contact_sheet": str(contact_sheet),
            "prompt": prompt,
            "note": "Template defaults to not_measured until a human reviewer inspects the frame pack and updates evidence.",
        },
        "checks": default_checks(prompt),
        "semantic_success_claim_allowed": False,
        "claim_boundary": {
            "template_is_not_measured_review": True,
            "manual_report_requires_reviewer_updates": True,
            "semantic_success_claim_allowed": False,
        },
    }


def create_review_pack(
    video_path: Path,
    prompt: str,
    output_dir: Path,
    num_frames: int = 8,
    columns: int = 4,
    frame_width: int = 320,
    frame_height: int = 180,
) -> Dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(frame_count, num_frames)

    frames = []
    saved_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        frame_path = output_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved_frames.append(str(frame_path))

        small = cv2.resize(frame, (frame_width, frame_height))
        cv2.putText(small, f"frame {idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frames.append(small)

    cap.release()

    if not frames:
        raise RuntimeError(f"no frames extracted from video: {video_path}")

    rows = []
    for start in range(0, len(frames), columns):
        row = frames[start:start + columns]
        while len(row) < columns:
            row.append(row[-1])
        rows.append(cv2.hconcat(row))

    contact_sheet = output_dir / "contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet), cv2.vconcat(rows))

    report_path = output_dir / "manual_alignment_report_template.json"
    report = build_report(video_path, contact_sheet, prompt)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest = {
        "video": str(video_path),
        "prompt": prompt,
        "frame_count": frame_count,
        "requested_frame_count": num_frames,
        "extracted_frame_indices": indices,
        "saved_frames": saved_frames,
        "contact_sheet": str(contact_sheet),
        "report_template": str(report_path),
        "claim_boundary": "This pack supports human review only. It does not prove prompt-video semantic alignment by itself.",
    }
    manifest_path = output_dir / "review_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent / "alignment_review" / "frame_pack_v0"
    manifest = create_review_pack(
        video_path=video_path,
        prompt=args.prompt,
        output_dir=output_dir,
        num_frames=args.num_frames,
        columns=args.columns,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
