#!/usr/bin/env python
"""Pre-generation window-admission probe (C1, C2, C3 gates; C4 diagnostic).

Offline, read-only. It does NOT run the DD2 backend and does NOT render video.
For each (window, case) it answers questions decidable before spending a GPU
render:

  C1  binding readiness -- does the requested source window bind?
      build_source_sample_binding on the window's source config, the same call
      the DD2 backend makes before its GPU subprocess. Property of the window.

  C2  v10b measurability -- will v10b resolve a scorable view? A case is
      measurable only if grounding yields a lateral maneuver (cut_in /
      lane_change) whose actor_motion_surface_plan resolves target cams and a
      lateral side. The "approaching" primitive (m4 class) builds no surface
      plan and is unmeasurable. C2 reuses the real v10b _views_to_evaluate.

  C3  baseline sanity (detector-free) -- the no-injection baseline that the
      evaluator will subtract must exist, live under a no-injection baseline
      directory (not a per-run staging video, the block-220 trap), and belong to
      the intended window. The source-row (top band) fingerprint is reported as
      a diagnostic; it is weight-invariant and separates windows (block 204).

  C4  baseline super-class presence (diagnostic only, opt-in, needs YOLO) --
      super-class (motorcycle/bicycle/pedestrian) detections of the no-injection
      baseline in the case's restricted views. This is NOT a collision
      predictor. The evaluator's baseline subtraction is class-agnostic and
      IoU-based (_baseline_view_detections is unfiltered; _subtract_baseline
      matches boxes regardless of label), and archived subtracted counts are
      large (65-72 per case) and driven by scene objects such as cars, not by a
      super-class actor -- so a zero here does not mean the arm's actor survives.
      Predicting subtraction requires the injected actor's projected pixel box
      against the baseline detections at that box, which needs DD2's box3d->pixel
      projection and is left as separate future work. C4 is reported, never gated.

Verdict:

  REJECT           C1 fails            -- the window does not bind.
  BASELINE_SUSPECT C3 fails            -- baseline missing / staging / wrong window.
  WARN             C2 unmeasurable.
  ADMIT            C1, C2, C3 pass. C4 is diagnostic and does not change this.

Source config and the baseline video are read from a no-injection baseline run
directory (--source-from-baseline-dir), so tokens are taken byte-exact from the
archive rather than retyped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from driveloop.actor_motion import build_actor_motion_surface_plan
from driveloop.composite_perception import CompositeVideoLayout
from driveloop.condition_adapter import DriveDreamer2ConditionAdapter
from driveloop.grounding import RuleBasedGrounder
from driveloop.longtail import LongTailController
from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
from driveloop.perception_video import OpenCVFrameReader
from driveloop.schema import DriveLoopRequest, Generation
from driveloop.source_sample_binding import build_source_sample_binding

__all__ = [
    "scene_spec_for",
    "surface_plan_for",
    "c2_measurability",
    "c1_binding",
    "resolve_baseline_video",
    "source_row_fingerprint",
    "c3_baseline_check",
    "make_baseline_superclass_evaluator",
    "c4_baseline_superclass",
    "verdict",
    "source_config_from_baseline_dir",
    "load_cases",
    "build_report",
    "main",
]

_V10B_EVALUATOR = None


def _v10b_evaluator() -> ManeuverViewRestrictedSuperclassEvaluator:
    """A detector-free v10b evaluator, built lazily and reused. detector=None is
    safe: _views_to_evaluate never calls the detector."""
    global _V10B_EVALUATOR
    if _V10B_EVALUATOR is None:
        _V10B_EVALUATOR = ManeuverViewRestrictedSuperclassEvaluator(detector=None)
    return _V10B_EVALUATOR


# ---- C1 (binding) and C2 (measurability) ----

def scene_spec_for(prompt: str):
    """Ground a text prompt as the runner does. multimodal_preprocessor is None
    because these cases carry no auxiliary inputs; grounding of a text-only
    prompt is identical either way."""
    grounder = RuleBasedGrounder(multimodal_preprocessor=None)
    return grounder.ground(DriveLoopRequest(prompt=prompt, condition={}, metadata={}))


def surface_plan_for(prompt: str) -> Dict[str, Any]:
    """Run the same no-GPU chain the backend runs before generation:
    grounding -> long-tail -> condition adapter -> actor-motion surface plan."""
    spec = scene_spec_for(prompt)
    condition_plan = LongTailController().build(spec, requested_tags=[], history=[])
    dd2_condition = DriveDreamer2ConditionAdapter().build(spec, condition_plan)
    actor_motion_plan = dd2_condition.executable_condition.get("actor_motion_plan", {})
    return build_actor_motion_surface_plan(actor_motion_plan)


def c2_measurability(surface_plan: Dict[str, Any]) -> Dict[str, Any]:
    """C2: does v10b resolve a scorable view for this surface plan?"""
    metadata = {"dd2_override_candidate_plan": {"actor_motion_surface_plan": surface_plan}}
    allowed = list(_v10b_evaluator()._views_to_evaluate(metadata))
    available = surface_plan.get("available") is True
    return {
        "surface_plan_available": available,
        "allowed_view_count": len(allowed),
        "allowed_views": allowed,
        "measurable": bool(available and allowed),
        "target_cam_types": list(surface_plan.get("target_cam_types") or []),
        "lateral_side": surface_plan.get("lateral_side"),
    }


def c1_binding(config: Dict[str, Any]) -> Dict[str, Any]:
    """C1: does the requested source window bind? Same call as the backend."""
    binding = build_source_sample_binding(
        config["dataset_dir"],
        source_candidate_id=config.get("source_candidate_id"),
        sample_token=config.get("sample_token"),
        scene_token=config.get("scene_token"),
        instance_token=config.get("instance_token"),
        identity_summary_path=config.get("identity_summary_path"),
        frame_num=int(config.get("frame_num", 8)),
        hz_factor=int(config.get("hz_factor", 3)),
        video_split_rate=int(config.get("video_split_rate", 1)),
        multiview=bool(config.get("multiview", True)),
    )
    return {
        "ready": binding.get("ready") is True,
        "reason": binding.get("reason"),
        "matched_sample_tokens": list(binding.get("matched_sample_tokens", []) or []),
        "matched_scene_tokens": list(binding.get("matched_scene_tokens", []) or []),
        "front_record": binding.get("front_record", {}) or {},
        "dd2_batch_skip": binding.get("dd2_batch_skip"),
    }


# ---- C3 (baseline sanity) and C4 (baseline super-class presence, diagnostic) ----

def resolve_baseline_video(directory: Path) -> Optional[Path]:
    """Find the no-injection baseline video under a baseline run directory."""
    directory = Path(directory)
    videos = sorted(directory.rglob("*.mp4"))
    if not videos:
        return None
    preferred = [v for v in videos if "no_injection" in str(v).lower() and "iteration" in v.name.lower()]
    if preferred:
        return preferred[0]
    no_injection = [v for v in videos if "no_injection" in str(v).lower()]
    return (no_injection or videos)[0]


def source_row_fingerprint(baseline_video: Any, layout: CompositeVideoLayout = None,
                           reader: OpenCVFrameReader = None) -> Optional[List[float]]:
    """Per-view mean brightness of the mosaic's source row (top band). This band
    is invariant under injection and weights and separates windows (block 204);
    reported as a diagnostic, not a gate."""
    layout = layout or CompositeVideoLayout()
    reader = reader or OpenCVFrameReader()
    frames = reader.read(Path(baseline_video), max_frames=1)
    if not frames:
        return None
    frame = frames[0]
    height, width = frame.shape[:2]
    band = min(layout.generated_row_height, height)
    view_width = width // layout.num_views if layout.num_views else width
    if view_width <= 0:
        return None
    return [
        round(float(frame[0:band, i * view_width:(i + 1) * view_width].mean()), 3)
        for i in range(layout.num_views)
    ]


def c3_baseline_check(baseline_video: Any, window_label: str, source_config: Dict[str, Any],
                      layout: CompositeVideoLayout = None, reader: OpenCVFrameReader = None) -> Dict[str, Any]:
    """C3: the no-injection baseline exists, lives under a no-injection directory
    (not a per-run staging video), and belongs to the intended window."""
    flags: List[str] = []
    exists = baseline_video is not None and Path(baseline_video).exists()
    if not exists:
        flags.append("baseline_missing")
    path_str = str(baseline_video).lower() if baseline_video else ""
    is_no_injection = "no_injection" in path_str
    if baseline_video is not None and not is_no_injection:
        flags.append("baseline_not_no_injection")
    candidate = source_config.get("source_candidate_id")
    label = str(window_label or "")
    candidate_consistent = True
    if label.startswith("candidate") and candidate and candidate != label:
        candidate_consistent = False
        flags.append("window_candidate_mismatch")
    fingerprint = source_row_fingerprint(baseline_video, layout=layout, reader=reader) if exists else None
    return {
        "baseline_video": str(baseline_video) if baseline_video else None,
        "exists": exists,
        "is_no_injection": is_no_injection,
        "source_candidate_id": candidate,
        "candidate_consistent": candidate_consistent,
        "source_row_fingerprint": fingerprint,
        "pass": bool(exists and is_no_injection and candidate_consistent),
        "flags": flags,
    }


def make_baseline_superclass_evaluator(weights: str = "yolov8x.pt", confidence: float = 0.20):
    """Build a v10b evaluator with a YOLO detector (mirrors rescore's setup).
    baseline_video=None: the baseline is scored raw, with no subtraction."""
    from driveloop.perception_video import UltralyticsYOLODetector

    return ManeuverViewRestrictedSuperclassEvaluator(
        detector=UltralyticsYOLODetector(weights, confidence_threshold=confidence),
        confidence_threshold=confidence,
        baseline_video=None,
    )


def c4_baseline_superclass(evaluator: Any, baseline_video: Any, surface_plan: Dict[str, Any]) -> Dict[str, Any]:
    """C4 (diagnostic): super-class detections of the no-injection baseline in
    the case's restricted views. NOT a collision predictor -- baseline
    subtraction is class-agnostic and box-based, so a zero here does not imply
    the arm's actor will survive. Reported, never gated."""
    generation = Generation(
        iteration=0,
        prompt="",
        artifacts={"video": str(baseline_video)},
        metadata={"dd2_override_candidate_plan": {"actor_motion_surface_plan": surface_plan}},
    )
    evaluation = evaluator.evaluate(generation)
    count = float(evaluation.metrics.get("perception_superclass_detection_count", 0.0))
    return {
        "superclass_detection_count": count,
        "selected_view": int(evaluation.metrics.get("perception_selected_view", -1)),
        "allowed_view_count": int(evaluation.metrics.get("perception_allowed_view_count", 0)),
        "superclass_actor_present": count > 0.0,
    }


def verdict(c1: Dict[str, Any], c2: Dict[str, Any], c3: Dict[str, Any] = None) -> str:
    """C4 is diagnostic only and never affects the verdict."""
    if not c1.get("ready"):
        return "REJECT"
    if c3 is not None and not c3.get("pass"):
        return "BASELINE_SUSPECT"
    if not c2.get("measurable"):
        return "WARN"
    return "ADMIT"


# ---- source config extraction ----

def _iter_json_objects(value: Any) -> Iterable[dict]:
    """Yield every dict nested anywhere in a parsed JSON value."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_json_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_objects(item)


def _iter_json_documents(directory: Path):
    """Yield (document, path) for every *.json (whole file) and *.jsonl
    (one document per line) under the directory. Baseline runs archive the
    binding in result.json (v10w windows) or in history.jsonl / attempts.jsonl
    (older v9 runs), so both extensions must be read."""
    for path in sorted(directory.rglob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8")), path
        except (ValueError, OSError):
            continue
    for path in sorted(directory.rglob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), path
            except ValueError:
                continue


def _binding_to_config(binding: Any, path: Path):
    """Turn a dd2_source_sample_binding object into a source config, or None."""
    if not isinstance(binding, dict):
        return None
    selector = binding.get("selector")
    selector = selector if isinstance(selector, dict) else {}
    dataset_dir = binding.get("dataset_dir") or selector.get("dataset_dir")
    if not dataset_dir:
        return None
    return {
        "dataset_dir": dataset_dir,
        "source_candidate_id": selector.get("source_candidate_id"),
        "sample_token": selector.get("sample_token"),
        "scene_token": selector.get("scene_token"),
        "instance_token": selector.get("instance_token"),
        "identity_summary_path": selector.get("identity_summary_path"),
        "_source_metadata_path": str(path),
    }


def source_config_from_baseline_dir(directory: Path) -> Dict[str, Any]:
    """Extract the window source config (dataset_dir + selector tokens) from a
    baseline run directory's archived metadata, so tokens are read byte-exact
    from the archive rather than retyped. Reads both *.json and *.jsonl, finds a
    dd2_source_sample_binding, and prefers one whose selector carries at least
    one non-null token (falling back to the first with a dataset_dir)."""
    directory = Path(directory)
    token_keys = ("source_candidate_id", "sample_token", "scene_token", "instance_token")
    fallback = None
    for document, path in _iter_json_documents(directory):
        for obj in _iter_json_objects(document):
            config = _binding_to_config(obj.get("dd2_source_sample_binding"), path)
            if config is None:
                continue
            if any(config.get(key) for key in token_keys):
                return config
            fallback = fallback or config
    if fallback is not None:
        return fallback
    raise SystemExit("no dd2_source_sample_binding with a dataset_dir found under %s" % directory)


def load_cases(manifest_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = data["cases"] if isinstance(data, dict) and "cases" in data else data
    if not isinstance(rows, list):
        raise SystemExit("cases manifest must be a list or contain a 'cases' list")
    cases = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("name") or not row.get("prompt"):
            raise SystemExit("each case needs a name and a prompt")
        cases.append({"name": str(row["name"]), "prompt": str(row["prompt"])})
    return cases


def build_report(window_label: str, source_config: Dict[str, Any], cases: List[Dict[str, Any]],
                 baseline_video: Any = None, superclass_check: bool = False,
                 weights: str = "yolov8x.pt", confidence: float = 0.20,
                 superclass_evaluator: Any = None) -> Dict[str, Any]:
    c1 = c1_binding(source_config)  # window-level, case-independent
    c3 = c3_baseline_check(baseline_video, window_label, source_config) if baseline_video else None
    if superclass_check and baseline_video and superclass_evaluator is None:
        superclass_evaluator = make_baseline_superclass_evaluator(weights, confidence)
    rows = []
    for case in cases:
        surface_plan = surface_plan_for(case["prompt"])
        c2 = c2_measurability(surface_plan)
        c4 = None
        if superclass_check and baseline_video and c2["measurable"]:
            c4 = c4_baseline_superclass(superclass_evaluator, baseline_video, surface_plan)
        rows.append({
            "window": window_label,
            "case": case["name"],
            "verdict": verdict(c1, c2, c3),  # C4 is diagnostic, not a gate
            "c1_ready": c1["ready"],
            "c2_measurable": c2["measurable"],
            "c2_allowed_view_count": c2["allowed_view_count"],
            "c3_pass": (c3["pass"] if c3 else None),
            "c3_flags": (c3["flags"] if c3 else None),
            "c4_superclass_actor_present": (c4["superclass_actor_present"] if c4 else None),
            "c4_superclass_detection_count": (c4["superclass_detection_count"] if c4 else None),
        })
    return {"window": window_label, "c1": c1, "c3": c3, "source_config": source_config, "cases": rows}


def _print_report(report: Dict[str, Any]) -> None:
    c1 = report["c1"]
    print("window=%s  C1.ready=%s  reason=%s  batch_skip=%s"
          % (report["window"], c1["ready"], c1["reason"], c1["dd2_batch_skip"]))
    print("  matched_sample_tokens=%s  matched_scene_tokens=%s"
          % (c1["matched_sample_tokens"], c1["matched_scene_tokens"]))
    c3 = report.get("c3")
    if c3 is not None:
        print("  C3.baseline pass=%s  no_injection=%s  candidate=%s  flags=%s"
              % (c3["pass"], c3["is_no_injection"], c3["source_candidate_id"], c3["flags"]))
        print("       source_row_fingerprint=%s" % (c3["source_row_fingerprint"],))
    print("  %-30s %-16s %-13s %-18s %s"
          % ("case", "verdict", "c2_measurable", "c4_superclass_actor", "c4_superclass_count"))
    for r in report["cases"]:
        print("  %-30s %-16s %-13s %-18s %s"
              % (r["case"], r["verdict"], r["c2_measurable"],
                 r["c4_superclass_actor_present"], r["c4_superclass_detection_count"]))
    verdicts = [r["verdict"] for r in report["cases"]]
    print("  summary: ADMIT=%d WARN=%d REJECT=%d BASELINE_SUSPECT=%d"
          % (verdicts.count("ADMIT"), verdicts.count("WARN"),
             verdicts.count("REJECT"), verdicts.count("BASELINE_SUSPECT")))
    print("  NOTE: C4 is a diagnostic (baseline super-class presence in the maneuver views); "
          "it is not a collision predictor and does not affect the verdict.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-generation window-admission probe (C1 binding, C2 v10b measurability, "
                    "C3 baseline sanity; C4 baseline super-class presence is diagnostic). "
                    "Read-only; C4 needs YOLO."
    )
    parser.add_argument("--cases-manifest", required=True, type=Path,
                        help="JSON manifest with cases[].name and cases[].prompt")
    parser.add_argument("--source-from-baseline-dir", type=Path, default=None,
                        help="read the window source config and baseline video from a baseline run dir")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--source-candidate-id", default=None)
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--scene-token", default=None)
    parser.add_argument("--instance-token", default=None)
    parser.add_argument("--identity-summary-path", default=None)
    parser.add_argument("--baseline-video", type=Path, default=None,
                        help="override the resolved baseline video for C3/C4")
    parser.add_argument("--baseline-superclass-check", action="store_true",
                        help="run the C4 diagnostic (needs YOLO); off by default")
    parser.add_argument("--weights", default="yolov8x.pt")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--window-label", default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args(argv)

    baseline_video = args.baseline_video
    if args.source_from_baseline_dir is not None:
        source_config = source_config_from_baseline_dir(args.source_from_baseline_dir)
        if baseline_video is None:
            baseline_video = resolve_baseline_video(args.source_from_baseline_dir)
    elif args.dataset_dir:
        source_config = {
            "dataset_dir": args.dataset_dir,
            "source_candidate_id": args.source_candidate_id,
            "sample_token": args.sample_token,
            "scene_token": args.scene_token,
            "instance_token": args.instance_token,
            "identity_summary_path": args.identity_summary_path,
        }
    else:
        parser.error("provide --source-from-baseline-dir or --dataset-dir")

    window_label = args.window_label or source_config.get("source_candidate_id") or "window"
    cases = load_cases(args.cases_manifest)
    report = build_report(
        window_label, source_config, cases,
        baseline_video=baseline_video,
        superclass_check=args.baseline_superclass_check,
        weights=args.weights,
        confidence=args.confidence,
    )
    _print_report(report)

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in report["cases"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
