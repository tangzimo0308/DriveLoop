from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two DD2 backend audit-only runs.")
    parser.add_argument("--prompt-a", default="rainy night road with a traffic barrier blocking the lane")
    parser.add_argument("--prompt-b", default="foggy night road with a bicycle cutting in from the left")
    parser.add_argument("--scenario-a", default="paper_ch3_compare_barrier")
    parser.add_argument("--scenario-b", default="paper_ch3_compare_bicycle")
    parser.add_argument("--output-dir", default="outputs/driveloop/dd2_backend_audit_compare")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def run_audit(prompt: str, scenario_id: str, output_dir: str, timeout_seconds: int) -> dict:
    subprocess.run(
        [
            "python",
            "scripts/run_dd2_backend_audit_only.py",
            "--prompt",
            prompt,
            "--scenario-id",
            scenario_id,
            "--output-dir",
            output_dir,
            "--timeout-seconds",
            str(timeout_seconds),
        ],
        check=True,
    )
    summary_path = Path(output_dir) / scenario_id / "backend_audit_only_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def tensor_hash(summary: dict, key: str) -> str | None:
    value = summary["runtime_input_audit"].get(key, {})
    return value.get("sha256")


def main() -> None:
    args = parse_args()
    a = run_audit(args.prompt_a, args.scenario_a, args.output_dir, args.timeout_seconds)
    b = run_audit(args.prompt_b, args.scenario_b, args.output_dir, args.timeout_seconds)

    comparison = {
        "schema_version": "driveloop_dd2_backend_audit_compare.v0",
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "scenario_a": args.scenario_a,
        "scenario_b": args.scenario_b,
        "audit_only": {
            "a": a["dd2_audit_only"],
            "b": b["dd2_audit_only"],
        },
        "video_generated": {
            "a": "video" in a["artifacts"],
            "b": "video" in b["artifacts"],
        },
        "runtime_tensor_hash_changed": {
            "prompt_embed": tensor_hash(a, "prompt_embed") != tensor_hash(b, "prompt_embed"),
            "box_downsampler_input": tensor_hash(a, "box_downsampler_input") != tensor_hash(b, "box_downsampler_input"),
            "grounding_downsampler_input": tensor_hash(a, "grounding_downsampler_input") != tensor_hash(b, "grounding_downsampler_input"),
            "img_cond": tensor_hash(a, "img_cond") != tensor_hash(b, "img_cond"),
        },
        "override_changed_counts": {
            "a": a["override_audit"].get("changed_counts"),
            "b": b["override_audit"].get("changed_counts"),
        },
        "runtime_tensor_signatures": {
            "a": {
                "prompt_embed": a["runtime_input_audit"].get("prompt_embed"),
                "box_downsampler_input": a["runtime_input_audit"].get("box_downsampler_input"),
                "grounding_downsampler_input": a["runtime_input_audit"].get("grounding_downsampler_input"),
                "img_cond": a["runtime_input_audit"].get("img_cond"),
            },
            "b": {
                "prompt_embed": b["runtime_input_audit"].get("prompt_embed"),
                "box_downsampler_input": b["runtime_input_audit"].get("box_downsampler_input"),
                "grounding_downsampler_input": b["runtime_input_audit"].get("grounding_downsampler_input"),
                "img_cond": b["runtime_input_audit"].get("img_cond"),
            },
        },
        "paper_interpretation": (
            "Different prompts changed DD2 text embedding and box structural conditioning while fixed mini baseline image and HDMap inputs stayed unchanged."
        ),
    }

    output_path = Path(args.output_dir) / "backend_audit_compare_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(json.dumps(comparison, indent=2))
    print(f"\nWrote comparison: {output_path}")

    expected = comparison["runtime_tensor_hash_changed"]
    if not expected["prompt_embed"]:
        raise SystemExit("Expected prompt_embed hash to change.")
    if not expected["box_downsampler_input"]:
        raise SystemExit("Expected box_downsampler_input hash to change.")
    if expected["grounding_downsampler_input"]:
        raise SystemExit("Expected grounding_downsampler_input hash to remain fixed.")
    if expected["img_cond"]:
        raise SystemExit("Expected img_cond hash to remain fixed.")


if __name__ == "__main__":
    main()
