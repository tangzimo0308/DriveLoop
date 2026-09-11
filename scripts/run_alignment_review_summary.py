from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def unwrap_report(data: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "generation" in data and "evaluation" in data:
        generation = data.get("generation") if isinstance(data.get("generation"), dict) else {}
        metadata = generation.get("metadata") if isinstance(generation.get("metadata"), dict) else {}
        report = metadata.get("prompt_video_alignment")
        if not isinstance(report, dict):
            report = {}
        interpretation = data.get("interpretation") if isinstance(data.get("interpretation"), dict) else {}
        return "prompt_video_alignment_evaluation", report, generation, interpretation

    for key in ("prompt_video_alignment", "video_alignment_report", "perception_alignment"):
        value = data.get(key)
        if isinstance(value, dict):
            return key, value, {}, {}

    return "manual_alignment_report", data, {}, {}


def check_rows(report: dict[str, Any]) -> tuple[int, int, list[str], list[str], float]:
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return 0, 0, [], [], 0.0

    required = []
    failed = []
    not_reviewed = []
    scores = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("required") is not True:
            continue
        name = str(check.get("name", "unknown"))
        required.append(check)
        try:
            scores.append(float(check.get("score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
        if check.get("passed") is not True:
            failed.append(name)
        if check.get("evidence") == "not_reviewed":
            not_reviewed.append(name)

    passed_count = len(required) - len(failed)
    score = round(sum(scores) / len(required), 6) if required else 0.0
    return len(required), passed_count, failed, not_reviewed, score


def semantic_claim(report: dict[str, Any], interpretation: dict[str, Any]) -> str:
    existing = interpretation.get("video_semantic_claim")
    if isinstance(existing, str) and existing in {"measured_passed", "measured_failed", "not_measured"}:
        return existing

    if report.get("status") != "measured":
        return "not_measured"

    required_count, passed_count, _, _, score = check_rows(report)
    if required_count == 0:
        return "measured_failed"
    if passed_count == required_count and score >= 0.8:
        return "measured_passed"
    return "measured_failed"


def summarize_file(path: Path) -> dict[str, Any]:
    data = load_json(path)
    kind, report, generation, interpretation = unwrap_report(data)
    required_count, passed_count, failed, not_reviewed, score = check_rows(report)

    review_scope = report.get("review_scope") if isinstance(report.get("review_scope"), dict) else {}
    artifacts = generation.get("artifacts") if isinstance(generation.get("artifacts"), dict) else {}
    metadata = generation.get("metadata") if isinstance(generation.get("metadata"), dict) else {}

    return {
        "path": str(path),
        "kind": kind,
        "scenario_id": metadata.get("scenario_id"),
        "source": report.get("source"),
        "status": report.get("status"),
        "prompt": generation.get("prompt") or review_scope.get("prompt"),
        "video": artifacts.get("video") or review_scope.get("video"),
        "contact_sheet": review_scope.get("contact_sheet"),
        "video_semantic_claim": semantic_claim(report, interpretation),
        "required_check_count": required_count,
        "passed_required_check_count": passed_count,
        "required_score": score,
        "failed_required_checks": failed,
        "not_reviewed_required_checks": not_reviewed,
    }


def discover_json(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            found.append(path)
        elif path.is_dir():
            found.extend(path.rglob("*.json"))
    return sorted(dict.fromkeys(found))


def build_summary(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in discover_json(paths):
        try:
            data = load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        kind, report, _, _ = unwrap_report(data)
        is_evaluation = kind == "prompt_video_alignment_evaluation"
        is_review_report = isinstance(report.get("checks"), list) and "status" in report
        if not is_evaluation and not is_review_report:
            continue
        row = summarize_file(path)
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        claim = row["video_semantic_claim"]
        counts[claim] = counts.get(claim, 0) + 1

    return {
        "schema_version": "driveloop_alignment_review_summary.v0",
        "row_count": len(rows),
        "claim_counts": counts,
        "rows": rows,
        "claim_boundary": (
            "This summary aggregates existing manual/perception/evaluator reports; "
            "it does not inspect video pixels or prove new video semantics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DriveLoop alignment review reports.")
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    summary = build_summary(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
