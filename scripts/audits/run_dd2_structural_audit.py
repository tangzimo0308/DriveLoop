from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dreamer_datasets import load_dataset

from driveloop import DriveLoopRequest
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.condition_adapter import DriveDreamer2ConditionAdapter
from driveloop.grounding import RuleBasedGrounder
from driveloop.longtail import LongTailController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only audit from DriveLoop executable_condition to DD2 structural transform tensors."
    )
    parser.add_argument(
        "--prompt",
        default="rainy night road with a traffic barrier blocking the lane",
    )
    parser.add_argument("--scenario-id", default="dd2_structural_audit")
    parser.add_argument(
        "--config-path",
        default="dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/driveloop/dd2_structural_audit/latest.json",
    )
    parser.add_argument("--require-change", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("dd2_mini_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.config


def tensor_sig(value: Any) -> dict:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().float().numpy()
    else:
        arr = np.asarray(value)
    arr = np.ascontiguousarray(arr)
    arr64 = arr.astype(np.float64) if arr.size else arr
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sum": float(arr64.sum()) if arr.size else 0.0,
        "mean": float(arr64.mean()) if arr.size else 0.0,
        "std": float(arr64.std()) if arr.size else 0.0,
        "nonzero": int(np.count_nonzero(arr)) if arr.size else 0,
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def make_transform(transform_cfg: dict):
    from projects.DriveDreamer2.drivedreamer2 import drivedreamer2_transforms

    cfg = copy.deepcopy(transform_cfg)
    transform_type = cfg.pop("type")
    return getattr(drivedreamer2_transforms, transform_type)(**cfg)


def transform_first_sample(data_cfg: dict, override_json: dict | None, audit_path: Path | None):
    if override_json:
        os.environ["DRIVELOOP_DD2_OVERRIDE_JSON"] = json.dumps(override_json, sort_keys=True)
        os.environ["DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH"] = str(audit_path)
        if audit_path and audit_path.exists():
            audit_path.unlink()
    else:
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_JSON", None)
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH", None)

    dataset = load_dataset(data_cfg["data_or_config"])
    dataset.set_transform(make_transform(data_cfg["transform"]))
    return dataset[0]


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config_path))
    data_cfg = config["dataloaders"]["test"]

    request = DriveLoopRequest(prompt=args.prompt, scenario_id=args.scenario_id)
    scene_spec = RuleBasedGrounder().ground(request)
    condition_plan = LongTailController().build(scene_spec)
    dd2_condition = DriveDreamer2ConditionAdapter().build(scene_spec, condition_plan)

    backend = DriveDreamer2Backend(project_root=".")
    baseline_snapshot = backend._build_baseline_structural_snapshot()
    executable_condition = dd2_condition.executable_condition
    structural_plan = executable_condition["structural_input_plan"]
    trace_metadata = executable_condition["trace_metadata"]

    structural_diff = backend._build_structural_request_diff(
        structural_input_plan=structural_plan,
        baseline_structural_snapshot=baseline_snapshot,
        trace_metadata=trace_metadata,
    )
    override_candidate_plan = backend._build_override_candidate_plan(
        structural_input_plan=structural_plan,
        structural_request_diff=structural_diff,
        baseline_structural_snapshot=baseline_snapshot,
    )
    override_json = backend._build_override_json(
        dd2_prompt=dd2_condition.text_prompt,
        structural_input_plan=structural_plan,
        override_candidate_plan=override_candidate_plan,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    override_audit_path = output_path.with_suffix(".override_audit.jsonl")

    baseline = transform_first_sample(data_cfg, override_json=None, audit_path=None)
    override = transform_first_sample(data_cfg, override_json=override_json, audit_path=override_audit_path)

    baseline_sigs = {
        "box_downsampler_input": tensor_sig(baseline["box_downsampler_input"]),
        "grounding_downsampler_input": tensor_sig(baseline["grounding_downsampler_input"]),
        "input_image": tensor_sig(baseline["input_image"]),
    }
    override_sigs = {
        "box_downsampler_input": tensor_sig(override["box_downsampler_input"]),
        "grounding_downsampler_input": tensor_sig(override["grounding_downsampler_input"]),
        "input_image": tensor_sig(override["input_image"]),
    }
    changed = {
        key: baseline_sigs[key]["sha256"] != override_sigs[key]["sha256"]
        for key in baseline_sigs
    }

    audit_preview = []
    if override_audit_path.exists():
        audit_preview = [
            json.loads(line)
            for line in override_audit_path.read_text(encoding="utf-8").splitlines()[:3]
            if line.strip()
        ]

    report = {
        "schema_version": "driveloop_dd2_structural_audit.v0",
        "scenario_id": args.scenario_id,
        "paper_alignment": {
            "chapter_3_prompt_grounding": {
                "prompt": args.prompt,
                "scene_specification": asdict(scene_spec),
                "long_tail_condition_plan": asdict(condition_plan),
            },
            "chapter_3_executable_condition": {
                "dd2_text_prompt": dd2_condition.text_prompt,
                "executable_condition": executable_condition,
                "override_json": override_json,
                "override_candidate_plan": override_candidate_plan,
                "structural_request_diff": structural_diff,
            },
            "chapter_3_dd2_structural_mapping": {
                "box_downsampler_input_changed": changed["box_downsampler_input"],
                "grounding_downsampler_input_changed": changed["grounding_downsampler_input"],
                "input_image_changed": changed["input_image"],
                "interpretation": (
                    "DriveLoop executable_condition changed DD2 box structural conditioning while keeping baseline image and HDMap fixed."
                    if changed["box_downsampler_input"]
                    else "No DD2 box structural tensor change was observed for this prompt; inspect requested labels versus baseline labels."
                ),
            },
        },
        "baseline_signatures": baseline_sigs,
        "override_signatures": override_sigs,
        "transform_override_audit_path": str(override_audit_path),
        "transform_override_audit_preview": audit_preview,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["paper_alignment"], indent=2))
    print(f"\nWrote audit report: {output_path}")

    if args.require_change and not changed["box_downsampler_input"]:
        raise SystemExit("Expected box_downsampler_input to change, but it did not.")


if __name__ == "__main__":
    main()
