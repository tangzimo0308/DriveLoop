from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from driveloop.backends.base import GenerationBackend
from driveloop.actor_motion import build_actor_motion_surface_plan
from driveloop.dd2_override import read_override_audit
from driveloop.ego_injection import apply_trajectory_tangent_heading, cam_box9_to_ego_entry
from driveloop.schema import DriveLoopRequest, Generation
from driveloop.source_sample_binding import build_source_sample_binding
from driveloop.real_track_guard import check_real_track


def real_track_dims_scale() -> float:
    """Env-gated reinforcement magnitude for real-track ego entries.

    DRIVELOOP_EGO_REAL_TRACK_DIMS_SCALE scales the reinforced actor
    dims (w/h/d) in the conditioning only; position and heading stay
    untouched. Default 1.0 keeps the default path byte-identical.
    Invalid or non-positive values fall back to 1.0.
    """
    raw = os.environ.get("DRIVELOOP_EGO_REAL_TRACK_DIMS_SCALE", "1.0")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if value <= 0.0:
        return 1.0
    return value


class DriveDreamer2Backend(GenerationBackend):
    """Command-line wrapper for the DriveDreamer-2 mini baseline."""

    def __init__(
        self,
        project_root: str | Path = ".",
        config_name: str = "drivedreamer2_img_cond_mini_local",
        baseline_output_dir: str | Path = "/data/projects/DriveLoop/outputs/drivedreamer2_img_cond_mini",
        baseline_dataset_dir: str | Path = "/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2",
        artifact_dir: str | Path = "outputs/driveloop/drivedreamer2_backend/artifacts",
        python_executable: str = "python",
        timeout_seconds: Optional[int] = None,
        audit_only: bool = False,
        batch_skip: int = 0,
        source_candidate_id: Optional[str] = None,
        sample_token: Optional[str] = None,
        scene_token: Optional[str] = None,
        instance_token: Optional[str] = None,
        source_identity_summary_path: str | Path | None = None,
        source_selector_frame_num: int = 8,
        source_selector_hz_factor: int = 3,
        source_selector_video_split_rate: int = 1,
        source_selector_multiview: bool = True,
        force_boxes3d_probe: bool = False,
        boxes3d_probe_category: Optional[str] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_name = config_name
        self.baseline_output_dir = Path(baseline_output_dir)
        self.baseline_dataset_dir = Path(baseline_dataset_dir)
        self.artifact_dir = Path(artifact_dir)
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.audit_only = audit_only
        self.batch_skip = batch_skip
        self.source_candidate_id = source_candidate_id
        self.sample_token = sample_token
        self.scene_token = scene_token
        self.instance_token = instance_token
        self.source_identity_summary_path = Path(source_identity_summary_path) if source_identity_summary_path else None
        self.source_selector_frame_num = source_selector_frame_num
        self.source_selector_hz_factor = source_selector_hz_factor
        self.source_selector_video_split_rate = source_selector_video_split_rate
        self.source_selector_multiview = source_selector_multiview
        self.force_boxes3d_probe = force_boxes3d_probe
        self.boxes3d_probe_category = boxes3d_probe_category

    def generate(self, request: DriveLoopRequest, iteration: int) -> Generation:
        run_artifact_dir = self.artifact_dir
        if request.scenario_id:
            run_artifact_dir = run_artifact_dir / request.scenario_id
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        # DD2 tester staging output for this run: overwritten on every run and
        # copied into artifacts below. This is the arm's own render, NOT the
        # no-injection support baseline of --perception-baseline-video.
        baseline_video = self.baseline_output_dir / "000000.mp4"

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        generation_parameter_env = self._build_generation_parameter_env(request.condition, iteration)
        env.update(generation_parameter_env)

        dd2_condition = request.condition.get("dd2_condition", {})
        dd2_prompt = dd2_condition.get("text_prompt") if isinstance(dd2_condition, dict) else None
        executable_condition = (
            dd2_condition.get("executable_condition", {})
            if isinstance(dd2_condition, dict)
            else {}
        )
        trace_metadata = (
            executable_condition.get("trace_metadata", {})
            if isinstance(executable_condition, dict)
            else {}
        )
        structural_input_plan = (
            executable_condition.get("structural_input_plan", {})
            if isinstance(executable_condition, dict)
            else {}
        )
        actor_motion_plan = (
            executable_condition.get("actor_motion_plan", {})
            if isinstance(executable_condition, dict)
            else {}
        )

        self._synthetic_trajectory_escalation = bool(
            isinstance(request.condition, dict)
            and request.condition.get("synthetic_trajectory_escalation")
        )
        runtime_sample_selector = self._build_runtime_sample_selector(request)
        source_sample_binding = self._build_source_sample_binding(runtime_sample_selector)
        effective_batch_skip = (
            source_sample_binding.get("dd2_batch_skip")
            if source_sample_binding.get("ready") is True
            else self.batch_skip
        )
        baseline_structural_snapshot = self._build_baseline_structural_snapshot(
            selected_label_index=source_sample_binding.get("front_record_index")
            if source_sample_binding.get("ready") is True
            else None
        )
        structural_request_diff = self._build_structural_request_diff(
            structural_input_plan=structural_input_plan,
            baseline_structural_snapshot=baseline_structural_snapshot,
            trace_metadata=trace_metadata,
        )
        override_candidate_plan = self._build_override_candidate_plan(
            structural_input_plan=structural_input_plan,
            structural_request_diff=structural_request_diff,
            baseline_structural_snapshot=baseline_structural_snapshot,
            actor_motion_plan=actor_motion_plan,
        )
        override_json = self._build_override_json(
            dd2_prompt=dd2_prompt,
            structural_input_plan=structural_input_plan,
            override_candidate_plan=override_candidate_plan,
            source_sample_binding=source_sample_binding,
        )
        audit_path = run_artifact_dir / f"dd2_runtime_input_audit_{iteration:02d}.json"
        override_audit_path = run_artifact_dir / f"dd2_override_audit_{iteration:02d}.jsonl"
        for stale_audit_path in (audit_path, override_audit_path):
            if stale_audit_path.exists():
                stale_audit_path.unlink()
        env["DRIVELOOP_DD2_AUDIT_PATH"] = str(audit_path)
        env["DRIVELOOP_DD2_DATA_OR_CONFIG"] = str(self.baseline_dataset_dir)
        effective_audit_only = self.audit_only or env.get("DRIVELOOP_DD2_AUDIT_ONLY") == "1"
        if (
            not effective_audit_only
            and source_sample_binding.get("requested") is True
            and source_sample_binding.get("ready") is not True
        ):
            raise RuntimeError(
                "source sample selector requested but binding is not ready"
                " (reason: %s); refusing to start generation on an unbound"
                " window" % source_sample_binding.get("reason")
            )
        if baseline_video.exists() and not effective_audit_only:
            baseline_video.unlink()
        if dd2_prompt:
            env["DRIVELOOP_DD2_PROMPT"] = str(dd2_prompt)
        if effective_audit_only:
            env["DRIVELOOP_DD2_AUDIT_ONLY"] = "1"
        if effective_batch_skip is not None:
            env["DRIVELOOP_DD2_BATCH_SKIP"] = str(effective_batch_skip)
        if source_sample_binding.get("ready") is True:
            env["DRIVELOOP_DD2_SOURCE_BOUND"] = "1"
        if override_json.get("available"):
            env["DRIVELOOP_DD2_OVERRIDE_JSON"] = json.dumps(override_json, sort_keys=True)
            env["DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH"] = str(override_audit_path)

        cmd = [
            self.python_executable,
            "./dreamer-train/projects/launch.py",
            "--project_name",
            "DriveDreamer2",
            "--config_name",
            self.config_name,
            "--runners",
            "drivedreamer2.DriveDreamer2_Tester",
        ]

        completed = subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            check=True,
            text=True,
            timeout=self.timeout_seconds,
        )

        artifact_video = None
        if not effective_audit_only:
            if not baseline_video.exists():
                raise FileNotFoundError(f"DriveDreamer-2 did not create {baseline_video}")

            artifact_video = run_artifact_dir / f"iteration_{iteration:02d}.mp4"
            shutil.copy2(baseline_video, artifact_video)

        override_audit = (
            read_override_audit(override_audit_path)
            if override_json.get("available")
            else {"available": False, "reason": "override_json_not_available"}
        )
        dd2_runtime_input_audit = {}
        if audit_path.exists():
            dd2_runtime_input_audit = json.loads(audit_path.read_text(encoding="utf-8"))

        paper_alignment_report = self._build_paper_alignment_report(
            dd2_prompt=dd2_prompt,
            executable_condition=executable_condition,
            trace_metadata=trace_metadata,
            structural_input_plan=structural_input_plan,
            structural_request_diff=structural_request_diff,
            override_candidate_plan=override_candidate_plan,
        )
        report_path = run_artifact_dir / f"paper_alignment_report_{iteration:02d}.json"
        report_path.write_text(json.dumps(paper_alignment_report, indent=2), encoding="utf-8")

        return Generation(
            iteration=iteration,
            prompt=request.prompt,
            artifacts={
                **({"video": str(artifact_video)} if artifact_video else {}),
                "paper_alignment_report": str(report_path),
                "dd2_runtime_input_audit": str(audit_path),
            },
            metadata={
                "backend": "drivedreamer2",
                "config_name": self.config_name,
                "dd2_raw_output_video": str(baseline_video),
                "returncode": completed.returncode,
                "dd2_audit_only": effective_audit_only,
                "dd2_batch_skip": effective_batch_skip,
                "dd2_runtime_sample_selector": runtime_sample_selector,
                "dd2_source_sample_binding": source_sample_binding,
                "dd2_prompt": str(dd2_prompt) if dd2_prompt else None,
                "dd2_executable_condition": executable_condition,
                "dd2_actor_motion_plan": actor_motion_plan,
                "dd2_condition_schema_version": executable_condition.get("schema_version")
                if isinstance(executable_condition, dict)
                else None,
                "dd2_tensor_control_ready": trace_metadata.get("tensor_control_ready")
                if isinstance(trace_metadata, dict)
                else None,
                "dd2_structural_input_plan": structural_input_plan,
                "dd2_structural_control_level": structural_input_plan.get("control_level")
                if isinstance(structural_input_plan, dict)
                else None,
                "dd2_baseline_structural_snapshot": baseline_structural_snapshot,
                "dd2_structural_request_diff": structural_request_diff,
                "dd2_override_candidate_plan": override_candidate_plan,
                "dd2_override_json": override_json,
                "dd2_generation_parameter_env": generation_parameter_env,
                "dd2_override_audit_path": str(override_audit_path)
                if override_json.get("available")
                else None,
                "dd2_override_audit": override_audit,
                "dd2_paper_alignment_report": paper_alignment_report,
                "dd2_runtime_input_audit": dd2_runtime_input_audit,
            },
        )

    def _build_runtime_sample_selector(self, request: DriveLoopRequest) -> dict:
        metadata = request.metadata if isinstance(request.metadata, dict) else {}
        condition = request.condition if isinstance(request.condition, dict) else {}

        return {
            "source_candidate_id": metadata.get("source_candidate_id")
            or condition.get("source_candidate_id")
            or self.source_candidate_id,
            "sample_token": metadata.get("sample_token")
            or condition.get("sample_token")
            or self.sample_token,
            "scene_token": metadata.get("scene_token")
            or condition.get("scene_token")
            or self.scene_token,
            "instance_token": metadata.get("instance_token")
            or condition.get("instance_token")
            or self.instance_token,
            "candidate_offset": (
                (condition.get("source_rebinding") or {}).get("candidate_offset", 0)
                if isinstance(condition.get("source_rebinding"), dict)
                else 0
            ),
            "identity_summary_path": str(
                metadata.get("source_identity_summary_path")
                or condition.get("source_identity_summary_path")
                or self.source_identity_summary_path
                or ""
            )
            or None,
        }

    def _build_source_sample_binding(self, selector: dict) -> dict:
        return build_source_sample_binding(
            self.baseline_dataset_dir,
            source_candidate_id=selector.get("source_candidate_id"),
            sample_token=selector.get("sample_token"),
            scene_token=selector.get("scene_token"),
            instance_token=selector.get("instance_token"),
            identity_summary_path=selector.get("identity_summary_path"),
            candidate_offset=int(selector.get("candidate_offset") or 0),
            frame_num=self.source_selector_frame_num,
            hz_factor=self.source_selector_hz_factor,
            video_split_rate=self.source_selector_video_split_rate,
            multiview=self.source_selector_multiview,
        )

    def _build_source_bound_sample_identities(
        self,
        source_sample_binding: dict | None,
    ) -> list[dict]:
        if not isinstance(source_sample_binding, dict) or source_sample_binding.get("ready") is not True:
            return []

        labels_path_value = source_sample_binding.get("labels_path")
        dd2_batch_skip = source_sample_binding.get("dd2_batch_skip")
        if labels_path_value is None or dd2_batch_skip is None:
            return []

        try:
            from scripts.run_dd2_batch_sampler_audit import (
                candidate_camera_starts,
                load_records,
                selected_frame_indices,
            )

            labels_path = Path(labels_path_value)
            records = load_records(labels_path)
            starts = candidate_camera_starts(
                records,
                frame_num=self.source_selector_frame_num,
                hz_factor=self.source_selector_hz_factor,
                video_split_rate=self.source_selector_video_split_rate,
                multiview=self.source_selector_multiview,
            )
            selected_indices = selected_frame_indices(
                starts[int(dd2_batch_skip)],
                frame_num=self.source_selector_frame_num,
                hz_factor=self.source_selector_hz_factor,
            )
        except Exception as exc:
            return [
                {
                    "available": False,
                    "reason": "source_bound_frame_mapping_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]

        identities = []
        for position, record_index in enumerate(selected_indices):
            if record_index < 0 or record_index >= len(records):
                continue
            record = records[record_index]
            identity = {
                "relative_step": int(position % self.source_selector_frame_num),
                "record_index": int(record_index),
                "frame_idx": record.get("frame_idx"),
                "cam_type": record.get("cam_type"),
                "sample_token": record.get("sample_token"),
                "scene_token": record.get("scene_token"),
            }
            calib = record.get("calib")
            if isinstance(calib, dict):
                extrinsics = {}
                for key in ("cam2ego", "ego2global"):
                    value = calib.get(key)
                    if value is not None:
                        extrinsics[key] = value.tolist() if hasattr(value, "tolist") else value
                if extrinsics:
                    identity["calib"] = extrinsics
            identities.append(identity)
        return identities

    def _build_source_bound_front_records(
        self,
        source_sample_binding: dict | None,
    ) -> list[dict]:
        """Selected-window cam_front records (with boxes/labels/calib),
        one per relative step, for real-track ego injection."""
        if not isinstance(source_sample_binding, dict) or source_sample_binding.get("ready") is not True:
            return []
        labels_path_value = source_sample_binding.get("labels_path")
        dd2_batch_skip = source_sample_binding.get("dd2_batch_skip")
        if labels_path_value is None or dd2_batch_skip is None:
            return []
        try:
            from scripts.run_dd2_batch_sampler_audit import (
                candidate_camera_starts,
                load_records,
                selected_frame_indices,
            )

            records = load_records(Path(labels_path_value))
            starts = candidate_camera_starts(
                records,
                frame_num=self.source_selector_frame_num,
                hz_factor=self.source_selector_hz_factor,
                video_split_rate=self.source_selector_video_split_rate,
                multiview=self.source_selector_multiview,
            )
            selected_indices = selected_frame_indices(
                starts[int(dd2_batch_skip)],
                frame_num=self.source_selector_frame_num,
                hz_factor=self.source_selector_hz_factor,
            )
        except Exception:
            return []
        front_records = []
        for position, record_index in enumerate(selected_indices):
            if record_index < 0 or record_index >= len(records):
                continue
            record = records[record_index]
            if str(record.get("cam_type") or "").lower() != "cam_front":
                continue
            front_records.append(
                {
                    "relative_step": int(position % self.source_selector_frame_num),
                    "record_index": int(record_index),
                    "record": record,
                }
            )
        return front_records

    def _map_real_track_to_ego_entries(
        self,
        per_frame_boxes: list[dict],
        source_sample_binding: dict | None,
    ) -> tuple[list[dict], dict]:
        """Real-track ego injection: when the source-bound window already
        contains a real actor of the requested category, lift ITS per-frame
        cam_front box into the ego frame and emit it on the C4 surface
        (true per-view projections), suppressing the synthetic stand-in.
        A synthetic duplicate next to a real target produces overlapping
        conditioning (measured 2026-07-09: min gap 3.7 m, crossing tracks).
        Selection requires the binding instance_token when supplied; otherwise
        it uses category and track continuity.
        Disable with DRIVELOOP_EGO_REAL_TRACK=0 to force the synthetic path."""
        if os.environ.get("DRIVELOOP_EGO_REAL_TRACK", "1") == "0":
            return [], {"available": False, "reason": "real_track_mode_disabled"}
        category = next(
            (str(entry.get("category")) for entry in per_frame_boxes if isinstance(entry, dict) and entry.get("category")),
            None,
        )
        if not category:
            return [], {"available": False, "reason": "no_requested_category"}
        front_records = self._build_source_bound_front_records(source_sample_binding)
        if not front_records:
            return [], {"available": False, "reason": "no_source_bound_front_records"}
        identities = self._build_source_bound_sample_identities(source_sample_binding)
        identities_by_step: dict[int, list[dict]] = {}
        for identity in identities:
            if identity.get("available") is False:
                continue
            identities_by_step.setdefault(int(identity["relative_step"]), []).append(identity)

        selector = source_sample_binding.get("selector", {}) if isinstance(source_sample_binding, dict) else {}
        instance_token = str(
            selector.get("instance_token")
            or source_sample_binding.get("instance_token")
            or ""
        )

        dims_scale = real_track_dims_scale()
        entries = []
        track_points = []
        missing_steps = []
        selection_basis = "category_nearest_to_previous"
        previous_box = None
        for item in sorted(front_records, key=lambda x: x["relative_step"]):
            record = item["record"]
            step = item["relative_step"]
            boxes = record.get("boxes3d")
            labels = list(record.get("ori_labels3d", []))
            if boxes is None or len(labels) == 0:
                missing_steps.append(step)
                continue
            candidates = [
                (index, [float(v) for v in boxes[index]])
                for index in range(min(len(labels), len(boxes)))
                if category.lower() in str(labels[index]).lower()
            ]
            if not candidates:
                missing_steps.append(step)
                continue
            record_instance_tokens = record.get("instance_tokens")
            if instance_token:
                if record_instance_tokens is None or len(record_instance_tokens) != len(labels):
                    missing_steps.append(step)
                    continue
                candidates = [(i, b) for i, b in candidates if str(record_instance_tokens[i]) == instance_token]
                if not candidates:
                    missing_steps.append(step)
                    continue
                selection_basis = "binding_instance_token"
            if previous_box is not None and len(candidates) > 1:
                candidates.sort(key=lambda ib: (ib[1][0] - previous_box[0]) ** 2 + (ib[1][2] - previous_box[2]) ** 2)
            else:
                candidates.sort(key=lambda ib: ib[1][0] ** 2 + ib[1][2] ** 2)
            box_index, box9 = candidates[0]
            previous_box = box9
            calib = record.get("calib", {})
            cam2ego = calib.get("cam2ego")
            ego2global = calib.get("ego2global")
            if cam2ego is None or ego2global is None:
                missing_steps.append(step)
                continue
            cam2ego = cam2ego.tolist() if hasattr(cam2ego, "tolist") else cam2ego
            ego2global = ego2global.tolist() if hasattr(ego2global, "tolist") else ego2global
            ego_payload = cam_box9_to_ego_entry(box9, cam2ego)
            if dims_scale != 1.0:
                ego_payload["dims"] = [float(v) * dims_scale for v in ego_payload["dims"]]
            matches = identities_by_step.get(step, [])
            entries.append(
                {
                    "relative_frame_idx": step,
                    "frame_idx": record.get("frame_idx"),
                    "actor_id": instance_token or f"real_track_{category}",
                    "synthetic_track_id": None,
                    "category": category,
                    "ego": ego_payload,
                    "ref_ego2global": ego2global,
                    "sample_identities": [
                        {
                            "cam_type": identity.get("cam_type"),
                            "frame_idx": identity.get("frame_idx"),
                            "sample_token": identity.get("sample_token"),
                            "scene_token": identity.get("scene_token"),
                        }
                        for identity in matches
                    ],
                    "source_record_indices": {
                        str(identity.get("cam_type")): identity.get("record_index")
                        for identity in matches
                    },
                    "frame_mapping": {
                        "mode": "real_track_relative_step_to_all_cam_sample_identities",
                        "relative_step": step,
                    },
                    "source": "source_bound_real_track",
                    "provenance": "driveloop_real_track_ego_injection",
                    "motion_surface": "boxes3d.per_frame_append_ego",
                    "heading": {"mode": "real_track_annotation"},
                    "source_box_index": box_index,
                }
            )
            track_points.append(
                {
                    "relative_step": step,
                    "frame_idx": record.get("frame_idx"),
                    "cam_x": round(box9[0], 3),
                    "cam_z": round(box9[2], 3),
                }
            )

        return (
            entries,
            {
                "available": bool(entries),
                "mode": "source_bound_real_track_ego",
                "reason": None if entries else "no_real_track_boxes_for_category",
                "category": category,
                "selection_basis": selection_basis if entries else None,
                "mapped_entry_count": len(entries),
                "missing_relative_steps": sorted(set(missing_steps)),
                "track_points": track_points,
                "synthetic_suppressed_count": len(per_frame_boxes) if entries else 0,
                "dims_scale": dims_scale,
                "heading_mode": "real_track_annotation" if entries else None,
                "claim_boundary": "Real-track ego mapping reinforces existing scene actors through per-view conditioning; it is not video semantic proof.",
            },
        )

    def _map_per_frame_actor_boxes_to_source_bound_samples(
        self,
        per_frame_boxes: list[dict],
        source_sample_binding: dict | None,
        target_cam_types: list[str] | None = None,
    ) -> tuple[list[dict], dict]:
        identities = self._build_source_bound_sample_identities(source_sample_binding)
        mapping_errors = [item for item in identities if item.get("available") is False]
        valid_identities = [item for item in identities if item.get("available") is not False]

        if not per_frame_boxes:
            return [], {"available": False, "reason": "no_per_frame_actor_boxes3d"}
        if not valid_identities:
            return (
                [dict(entry) for entry in per_frame_boxes],
                {
                    "available": False,
                    "reason": "no_source_bound_sample_identity_mapping",
                    "errors": mapping_errors,
                },
            )

        identities_by_step: dict[int, list[dict]] = {}
        for identity in valid_identities:
            identities_by_step.setdefault(int(identity["relative_step"]), []).append(identity)

        allowed_cam_types = {str(item).lower() for item in (target_cam_types or [])}
        view_filtered_count = 0
        mapped_entries = []
        unmapped_relative_steps = []
        for entry in per_frame_boxes:
            relative_step = int(entry.get("frame_idx"))
            matches = identities_by_step.get(relative_step, [])
            if not matches:
                unmapped_relative_steps.append(relative_step)
                continue

            for identity in matches:
                identity_cam = str(identity.get("cam_type") or "").lower()
                if allowed_cam_types and identity_cam not in allowed_cam_types:
                    view_filtered_count += 1
                    continue
                mapped = dict(entry)
                mapped["relative_frame_idx"] = relative_step
                mapped["frame_idx"] = identity.get("frame_idx")
                mapped["sample_identity"] = {
                    "cam_type": identity.get("cam_type"),
                    "frame_idx": identity.get("frame_idx"),
                    "sample_token": identity.get("sample_token"),
                    "scene_token": identity.get("scene_token"),
                }
                mapped["source_record_index"] = identity.get("record_index")
                mapped["frame_mapping"] = {
                    "mode": "source_bound_relative_step_to_sample_identity",
                    "relative_step": relative_step,
                }
                mapped_entries.append(mapped)

        return (
            mapped_entries,
            {
                "available": True,
                "mode": "source_bound_relative_step_to_sample_identity",
                "source_identity_count": len(valid_identities),
                "input_per_frame_count": len(per_frame_boxes),
                "mapped_entry_count": len(mapped_entries),
                "view_filter": {
                    "target_cam_types": sorted(allowed_cam_types),
                    "filtered_out_count": view_filtered_count,
                    "all_entries_filtered": bool(
                        per_frame_boxes and not mapped_entries and view_filtered_count
                    ),
                },
                "unmapped_relative_frame_idx": sorted(set(unmapped_relative_steps)),
                "claim_boundary": "Frame mapping connects structural actor boxes to source-bound DD2 samples; it is not video semantic proof.",
            },
        )

    def _map_per_frame_actor_boxes_to_ego_entries(
        self,
        per_frame_boxes: list[dict],
        source_sample_binding: dict | None,
    ) -> tuple[list[dict], dict]:
        """Ego-frame injection surface (C4): ONE ego-frame entry per video
        frame. The plan's cam_front-frame box9 is lifted into the ego frame
        using the per-frame cam_front record's cam2ego; that record's
        ego2global is embedded as the reference so every camera record can
        convert the entry into its own frame at consumption (true per-view
        projections, no clones). No view filter: behind-camera boxes are
        culled by the existing z>0 crop at canvas time."""
        identities = self._build_source_bound_sample_identities(source_sample_binding)
        mapping_errors = [item for item in identities if item.get("available") is False]
        valid_identities = [item for item in identities if item.get("available") is not False]

        if not per_frame_boxes:
            return [], {"available": False, "reason": "no_per_frame_actor_boxes3d"}
        if not valid_identities:
            return (
                [],
                {
                    "available": False,
                    "mode": "ego_frame_one_entry_per_video_frame",
                    "reason": "no_source_bound_sample_identity_mapping",
                    "errors": mapping_errors,
                },
            )

        identities_by_step: dict[int, list[dict]] = {}
        for identity in valid_identities:
            identities_by_step.setdefault(int(identity["relative_step"]), []).append(identity)

        mapped_entries = []
        unmapped_relative_steps = []
        missing_front_calib_steps = []
        for entry in per_frame_boxes:
            relative_step = int(entry.get("frame_idx"))
            matches = identities_by_step.get(relative_step, [])
            if not matches:
                unmapped_relative_steps.append(relative_step)
                continue

            front_identity = next(
                (
                    identity
                    for identity in matches
                    if str(identity.get("cam_type") or "").lower() == "cam_front"
                ),
                None,
            )
            front_calib = front_identity.get("calib") if isinstance(front_identity, dict) else None
            if (
                not isinstance(front_calib, dict)
                or front_calib.get("cam2ego") is None
                or front_calib.get("ego2global") is None
            ):
                missing_front_calib_steps.append(relative_step)
                continue

            ego_payload = cam_box9_to_ego_entry(entry.get("box3d"), front_calib["cam2ego"])
            mapped_entries.append(
                {
                    "relative_frame_idx": relative_step,
                    "frame_idx": front_identity.get("frame_idx"),
                    "actor_id": entry.get("actor_id"),
                    "synthetic_track_id": entry.get("synthetic_track_id"),
                    "category": entry.get("category"),
                    "ego": ego_payload,
                    "ref_ego2global": front_calib["ego2global"],
                    "sample_identities": [
                        {
                            "cam_type": identity.get("cam_type"),
                            "frame_idx": identity.get("frame_idx"),
                            "sample_token": identity.get("sample_token"),
                            "scene_token": identity.get("scene_token"),
                        }
                        for identity in matches
                    ],
                    "source_record_indices": {
                        str(identity.get("cam_type")): identity.get("record_index")
                        for identity in matches
                    },
                    "frame_mapping": {
                        "mode": "ego_frame_relative_step_to_all_cam_sample_identities",
                        "relative_step": relative_step,
                    },
                    "source": entry.get("source", "actor_motion_plan.per_frame_actor_boxes3d"),
                    "provenance": entry.get("provenance", "driveloop_ego_injection_surface"),
                    "motion_surface": "boxes3d.per_frame_append_ego",
                    "maneuver": entry.get("maneuver"),
                }
            )

        heading_mode = "plan_yaw_tangent_disabled"
        if os.environ.get("DRIVELOOP_EGO_TANGENT_HEADING", "1") != "0":
            heading_mode = apply_trajectory_tangent_heading(mapped_entries)

        return (
            mapped_entries,
            {
                "available": bool(mapped_entries),
                "mode": "ego_frame_one_entry_per_video_frame",
                "heading_mode": heading_mode,
                "source_identity_count": len(valid_identities),
                "input_per_frame_count": len(per_frame_boxes),
                "mapped_entry_count": len(mapped_entries),
                "cam_coverage": sorted(
                    {
                        str(identity.get("cam_type"))
                        for identity in valid_identities
                        if identity.get("cam_type")
                    }
                ),
                "view_filter": {
                    "target_cam_types": [],
                    "filtered_out_count": 0,
                    "policy": "no_view_filter_geometry_culls_behind_camera_boxes",
                },
                "unmapped_relative_frame_idx": sorted(set(unmapped_relative_steps)),
                "missing_front_calib_relative_frame_idx": sorted(set(missing_front_calib_steps)),
                "claim_boundary": "Ego-frame mapping connects structural actor boxes to source-bound DD2 samples; it is not video semantic proof.",
            },
        )

    def _build_generation_parameter_env(
        self,
        condition: dict | None,
        iteration: int,
    ) -> dict:
        """Map the refiner's generation_escalation (and the per-attempt
        seed offset) onto the DD2 tester env overrides. This is the
        closed-loop lever that reaches the generation itself: v9 showed
        that with real-track injection and canned-prompt collapse, the
        prior levers (synthetic geometry escalation, prompt additions)
        never change the conditioning, making all attempts and arms
        bit-identical under a frozen seed."""
        # Run-level seed bank for repeat experiments: attempts within a
        # run vary by iteration; whole runs vary by DRIVELOOP_DD2_SEED_BANK
        # (bank 0 reproduces all pre-bank runs byte-identically).
        try:
            seed_bank = int(os.environ.get("DRIVELOOP_DD2_SEED_BANK", "0"))
        except ValueError:
            seed_bank = 0
        parameter_env = {"DRIVELOOP_DD2_SEED_OFFSET": str(seed_bank * 100 + int(iteration))}
        generation_escalation = (
            condition.get("generation_escalation") if isinstance(condition, dict) else None
        )
        if isinstance(generation_escalation, dict):
            mapping = {
                "num_inf_steps": "DRIVELOOP_DD2_NUM_INF_STEPS",
                "min_guidance_scale": "DRIVELOOP_DD2_MIN_GUIDANCE",
                "max_guidance_scale": "DRIVELOOP_DD2_MAX_GUIDANCE",
            }
            for key, env_key in mapping.items():
                value = generation_escalation.get(key)
                if value is not None:
                    parameter_env[env_key] = str(value)
        return parameter_env

    def _build_override_json(
        self,
        dd2_prompt: str | None,
        structural_input_plan: dict,
        override_candidate_plan: dict,
        source_sample_binding: dict | None = None,
    ) -> dict:
        if not isinstance(structural_input_plan, dict) or not structural_input_plan:
            return {
                "available": False,
                "reason": "missing_structural_input_plan",
            }

        append_boxes = []
        box_synthesis_draft = (
            override_candidate_plan.get("box_synthesis_plan", {})
            .get("box_synthesis_draft", {})
            if isinstance(override_candidate_plan, dict)
            else {}
        )
        for entry in box_synthesis_draft.get("draft_boxes3d", []):
            append_boxes.append(
                {
                    "category": entry.get("category"),
                    "box3d": list(entry.get("box3d", [])),
                    "source": entry.get("source", "unknown"),
                    "provenance": "driveloop_executable_condition",
                    "placement_policy": entry.get("placement_policy"),
                    "requires_projection": entry.get("requires_projection", True),
                }
            )

        per_frame_append_boxes = []
        actor_motion_surface_plan = (
            override_candidate_plan.get("actor_motion_surface_plan", {})
            if isinstance(override_candidate_plan, dict)
            else {}
        )
        for entry in actor_motion_surface_plan.get("per_frame_boxes3d", []):
            per_frame_append_boxes.append(
                {
                    "frame_idx": entry.get("frame_idx"),
                    "actor_id": entry.get("actor_id"),
                    "synthetic_track_id": entry.get("synthetic_track_id"),
                    "category": entry.get("category"),
                    "box3d": list(entry.get("box3d", [])),
                    "source": entry.get("source", "actor_motion_plan.per_frame_actor_boxes3d"),
                    "provenance": entry.get("provenance", "driveloop_actor_motion_surface"),
                    "motion_surface": entry.get("motion_surface"),
                    "maneuver": entry.get("maneuver"),
                }
            )

        actor_motion_frame_mapping = {
            "available": False,
            "reason": "no_actor_motion_per_frame_entries",
        }
        ego_injection_enabled = os.environ.get("DRIVELOOP_EGO_INJECTION") == "1"
        per_frame_append_ego_boxes: list[dict] = []
        if per_frame_append_boxes and ego_injection_enabled:
            if getattr(self, "_synthetic_trajectory_escalation", False):
                real_track_entries, real_track_mapping = [], {
                    "available": False,
                    "reason": "synthetic_trajectory_escalation_requested",
                }
            else:
                (
                    real_track_entries,
                    real_track_mapping,
                ) = self._map_real_track_to_ego_entries(
                    per_frame_append_boxes,
                    source_sample_binding,
                )
            if real_track_entries:
                planned_entries, _ = self._map_per_frame_actor_boxes_to_ego_entries(
                    per_frame_append_boxes, source_sample_binding,
                )
                maneuver = actor_motion_surface_plan.get("maneuver") or per_frame_append_boxes[0].get("maneuver")
                compatibility = check_real_track(real_track_entries, planned_entries, maneuver)
                if maneuver in {"cut_in", "lane_change"} and (
                    len(planned_entries) != len(per_frame_append_boxes)
                    or real_track_mapping.get("selection_basis") != "binding_instance_token"
                ):
                    compatibility.update(compatible=False, reason="unverified_identity_or_incomplete_plan")
                real_track_mapping["trajectory_compatibility"] = compatibility
                if not compatibility["compatible"]:
                    real_track_entries = []
                    real_track_mapping.update(available=False, reason=compatibility["reason"],
                                              synthetic_suppressed_count=0)
            if real_track_entries:
                per_frame_append_ego_boxes = real_track_entries
                actor_motion_frame_mapping = real_track_mapping
            else:
                if os.environ.get("DRIVELOOP_EGO_REQUIRE_REAL_TRACK") == "1":
                    raise RuntimeError(
                        "real-track mapping is empty (reason: %s) and"
                        " DRIVELOOP_EGO_REQUIRE_REAL_TRACK=1; refusing the"
                        " silent synthetic fallback"
                        % real_track_mapping.get("reason")
                    )
                (
                    per_frame_append_ego_boxes,
                    actor_motion_frame_mapping,
                ) = self._map_per_frame_actor_boxes_to_ego_entries(
                    per_frame_append_boxes,
                    source_sample_binding,
                )
                actor_motion_frame_mapping["real_track_fallback_reason"] = real_track_mapping.get("reason")
                actor_motion_frame_mapping["rejected_real_track_audit"] = real_track_mapping
            # The legacy per-cam clone surface is suppressed when the
            # ego-frame surface is active to avoid double injection.
            per_frame_append_boxes = []
        elif per_frame_append_boxes:
            (
                per_frame_append_boxes,
                actor_motion_frame_mapping,
            ) = self._map_per_frame_actor_boxes_to_source_bound_samples(
                per_frame_append_boxes,
                source_sample_binding,
                actor_motion_surface_plan.get("target_cam_types"),
            )

        scene_description = structural_input_plan.get("scene_description", {})
        scene_value = scene_description.get("value") if isinstance(scene_description, dict) else dd2_prompt

        ground_mode = os.environ.get("DRIVELOOP_EGO_GROUND_ALIGNMENT", "off")
        if ground_mode not in {"off", "local_vehicle_bottom_v1"}:
            raise ValueError("unknown DRIVELOOP_EGO_GROUND_ALIGNMENT mode: " + ground_mode)
        if ground_mode != "off" and per_frame_append_ego_boxes:
            from driveloop.ground_alignment import align_to_local_vehicle_bottom
            per_frame_append_ego_boxes, height_audit = align_to_local_vehicle_bottom(
                per_frame_append_ego_boxes, self._build_source_bound_front_records(source_sample_binding),
            )
            actor_motion_frame_mapping["ground_alignment"] = height_audit

        return {
            "available": True,
            "schema_version": "driveloop_dd2_override.v0",
            "source": "DriveLoop.executable_condition",
            "scene_description": {
                "mode": "replace",
                "value": scene_value or dd2_prompt,
                "source": scene_description.get("source", "text_control.prompt")
                if isinstance(scene_description, dict)
                else "text_control.prompt",
            },
            "boxes3d": {
                "mode": (
                    "append_and_per_frame_append_ego"
                    if per_frame_append_ego_boxes
                    else "append_and_per_frame_append"
                    if per_frame_append_boxes
                    else "append"
                ),
                "append": append_boxes,
                "per_frame_append": per_frame_append_boxes,
                "per_frame_append_ego": per_frame_append_ego_boxes,
                "ego_injection": {
                    "enabled": ego_injection_enabled,
                    "env_flag": "DRIVELOOP_EGO_INJECTION",
                    "surface": "boxes3d.per_frame_append_ego",
                },
                "source": structural_input_plan.get("boxes3d", {}).get(
                    "source",
                    "executable_condition_tensor_override",
                ),
            },
            "image_box": {
                "mode": "derive_from_boxes3d_after_override",
                "source": structural_input_plan.get("image_box", {}).get(
                    "source",
                    "derived_from_boxes3d_override",
                ),
            },
            "image_hdmap": {
                "mode": "keep_baseline",
                "source": structural_input_plan.get("image_hdmap", {}).get(
                    "source",
                    "runtime_dataset_baseline",
                ),
                "reason": structural_input_plan.get("image_hdmap", {}).get(
                    "reason",
                    "no_verified_hdmap_override_source",
                ),
            },
            "audit": {
                "control_level": (
                    "tensor_override_runtime"
                    if append_boxes or per_frame_append_boxes or per_frame_append_ego_boxes or structural_input_plan.get("image_hdmap", {}).get("source") != "runtime_dataset_baseline"
                    else "runtime_surface_observation"
                ),
                "actor_motion_frame_mapping": actor_motion_frame_mapping,
                "limitations": [
                    "boxes3d_override_not_applied" if not append_boxes and not per_frame_append_boxes and not per_frame_append_ego_boxes else "box_positions_are_draft_until_projection_and_scene_geometry_are_verified",
                    "per_frame_actor_boxes3d_runtime_surface_connected_ego_frame"
                    if per_frame_append_ego_boxes
                    else "per_frame_actor_boxes3d_runtime_surface_connected"
                    if per_frame_append_boxes
                    else "actor_motion_surface_not_applied",
                    "hdmap_kept_baseline_without_explicit_verified_override",
                ],
            },
        }

    def _build_paper_alignment_report(
        self,
        dd2_prompt: str | None,
        executable_condition: dict,
        trace_metadata: dict,
        structural_input_plan: dict,
        structural_request_diff: dict,
        override_candidate_plan: dict,
    ) -> dict:
        tensor_ready = trace_metadata.get("tensor_control_ready") is True
        structural_level = structural_input_plan.get("control_level")
        actor_controls = executable_condition.get("actor_controls", [])
        environment_controls = executable_condition.get("environment_controls", {})
        risk_controls = executable_condition.get("risk_controls", {})

        stage_3_status = "tensor_control_ready" if tensor_ready else "text_and_plan_only"
        blockers = []
        if not tensor_ready:
            blockers.extend(trace_metadata.get("limitations", []))
        if structural_level in (None, "plan_only", "schema_only"):
            blockers.append("dd2_structural_inputs_not_overridden")

        return {
            "schema_version": "driveloop_paper_alignment_report.v0",
            "paper_reference": "DriveLoop methodology Section 3",
            "stage_1_multimodal_prompt_grounding": {
                "status": "available",
                "dd2_text_prompt_available": bool(dd2_prompt),
                "actor_controls": actor_controls,
                "environment_controls": environment_controls,
            },
            "stage_2_long_tail_conditioning": {
                "status": "available" if risk_controls.get("long_tail_tags") else "no_long_tail_tags",
                "risk_controls": risk_controls,
            },
            "stage_3_scene_consistent_generation": {
                "status": stage_3_status,
                "tensor_control_ready": tensor_ready,
                "structural_control_level": structural_level,
                "structural_input_plan": structural_input_plan,
                "structural_request_diff": structural_request_diff,
                "override_candidate_plan": override_candidate_plan,
                "blockers": list(dict.fromkeys(blockers)),
            },
            "stage_4_perception_evaluation_and_refinement": {
                "status": "requires_perception_evaluator_for_visual_alignment",
                "current_guardrail": "dd2_outputs_are_not_accepted_when_tensor_control_ready_is_false",
            },
            "experiment_readiness": {
                "main_experiment_ready": tensor_ready,
                "allowed_use": "prototype_trace_and_ablation_only" if not tensor_ready else "controlled_generation_candidate",
            },
        }

    def _build_override_candidate_plan(
        self,
        structural_input_plan: dict,
        structural_request_diff: dict,
        baseline_structural_snapshot: dict | None = None,
        actor_motion_plan: dict | None = None,
    ) -> dict:
        if not structural_request_diff.get("available", False):
            return {
                "available": False,
                "reason": structural_request_diff.get("reason", "missing_structural_request_diff"),
            }

        actor_label_actions = [
            {"type": "add_actor_label", "label": label}
            for label in structural_request_diff.get("missing_requested_labels", [])
        ]
        actor_label_actions.extend(
            {
                "type": "mark_extra_baseline_label",
                "label": label,
            }
            for label in structural_request_diff.get("extra_baseline_labels", [])
        )
        scene_description_action = {
            "type": "keep_baseline_text",
            "target_value": structural_request_diff.get("baseline_scene_description"),
        }
        if structural_request_diff.get("scene_description_changed"):
            scene_description_action = {
                "type": "replace_text_prompt",
                "target_value": structural_request_diff.get("requested_scene_description"),
            }

        boxes3d_plan = structural_input_plan.get("boxes3d", {})
        force_boxes3d_probe = (
            boxes3d_plan.get("force_probe") is True
            if isinstance(boxes3d_plan, dict)
            else False
        ) or self.force_boxes3d_probe
        boxes3d_probe_category = (
            boxes3d_plan.get("probe_category")
            if isinstance(boxes3d_plan, dict)
            else None
        ) or self.boxes3d_probe_category or "motorcycle"
        probe_actor_labels = [str(boxes3d_probe_category)] if force_boxes3d_probe else []
        actor_label_actions.extend(
            {"type": "probe_target_box", "label": label}
            for label in probe_actor_labels
        )

        requires_box_synthesis = bool(
            structural_request_diff.get("missing_requested_labels")
        ) or force_boxes3d_probe
        actor_motion_surface_plan = build_actor_motion_surface_plan(actor_motion_plan)
        actor_motion_surface_available = actor_motion_surface_plan.get("available") is True

        control_level = (
            "tensor_override_runtime"
            if requires_box_synthesis
            or actor_motion_surface_available
            or structural_input_plan.get("image_hdmap", {}).get("source") != "runtime_dataset_baseline"
            else "runtime_surface_observation"
        )

        return {
            "available": True,
            "control_level": control_level,
            "scene_description_action": scene_description_action,
            "actor_label_actions": actor_label_actions,
            "requires_box_synthesis": requires_box_synthesis,
            "force_boxes3d_probe": force_boxes3d_probe,
            "boxes3d_probe_category": boxes3d_probe_category if force_boxes3d_probe else None,
            "actor_motion_surface_plan": actor_motion_surface_plan,
            "box_synthesis_plan": self._build_box_synthesis_plan(
                structural_request_diff=structural_request_diff,
                requires_box_synthesis=requires_box_synthesis,
                baseline_structural_snapshot=baseline_structural_snapshot,
                probe_actor_labels=probe_actor_labels,
            ),
            "requires_hdmap_override": structural_input_plan.get("image_hdmap", {}).get("source")
            != "runtime_dataset_baseline",
            "baseline_sources": {
                "image_hdmap": structural_input_plan.get("image_hdmap", {}).get("source"),
                "image_box": structural_input_plan.get("image_box", {}).get("source"),
                "boxes3d": structural_input_plan.get("boxes3d", {}).get("source"),
            },
            "limitations": [
                "box_positions_are_draft_until_projection_and_scene_geometry_are_verified",
                "hdmap_override_requires_explicit_verified_source",
            ],
        }

    def _build_box_synthesis_plan(
        self,
        structural_request_diff: dict,
        requires_box_synthesis: bool,
        baseline_structural_snapshot: dict | None = None,
        probe_actor_labels: list[str] | None = None,
    ) -> dict:
        if not requires_box_synthesis:
            return {
                "available": False,
                "reason": "box_synthesis_not_required",
            }

        actors_to_synthesize = []
        seen_actor_labels = set()
        for label in structural_request_diff.get("missing_requested_labels", []):
            if label in seen_actor_labels:
                continue
            seen_actor_labels.add(label)
            actors_to_synthesize.append(
                {
                    "category": label,
                    "source_action": "add_actor_label",
                    "confidence": "low",
                    "reason": "missing_requested_label",
                }
            )
        for label in probe_actor_labels or []:
            if label in seen_actor_labels:
                continue
            seen_actor_labels.add(label)
            actors_to_synthesize.append(
                {
                    "category": label,
                    "source_action": "probe_target_box",
                    "confidence": "audit_only",
                    "reason": "explicit_boxes3d_probe",
                }
            )
        placement_policy = "front_adjacent_lane_cut_in"
        box_template_source = "class_default_dimensions"

        return {
            "available": True,
            "control_level": "tensor_override_runtime",
            "target_tensor": "boxes3d",
            "derived_tensor": "image_box",
            "placement_policy": placement_policy,
            "box_template_source": box_template_source,
            "requires_manual_review": True,
            "actors_to_synthesize": actors_to_synthesize,
            "box_synthesis_draft": self._build_box_synthesis_draft(
                actors_to_synthesize=actors_to_synthesize,
                placement_policy=placement_policy,
                box_template_source=box_template_source,
                baseline_structural_snapshot=baseline_structural_snapshot,
            ),
            "limitations": [
                "3d_position_uses_audited_draft_policy",
                "camera_projection_validated_when_baseline_intrinsic_is_available",
            ],
        }

    def _build_box_synthesis_draft(
        self,
        actors_to_synthesize: list,
        placement_policy: str,
        box_template_source: str,
        baseline_structural_snapshot: dict | None = None,
    ) -> dict:
        default_dimensions = {
            "bicycle": {"width": 0.6, "height": 1.6, "depth": 1.8},
            "motorcycle": {
                "width": 0.8,
                "height": 1.5,
                "depth": 2.2,
            },
            "pedestrian": {"width": 0.6, "height": 1.7, "depth": 0.6},
            "car": {"width": 1.8, "height": 1.6, "depth": 4.5},
            "truck": {"width": 2.5, "height": 3.0, "depth": 7.0},
            "bus": {"width": 2.6, "height": 3.2, "depth": 10.0},
            "barrier": {"width": 1.2, "height": 0.9, "depth": 0.45},
        }

        draft_boxes3d = []
        for actor in actors_to_synthesize:
            category = actor["category"]
            dims = default_dimensions.get(category)
            if not dims:
                continue
            draft_boxes3d.append(
                {
                    "category": category,
                    "box3d": [
                        8.0,
                        1.8,
                        18.0,
                        dims["width"],
                        dims["height"],
                        dims["depth"],
                        0.0,
                        0.0,
                        -0.25,
                    ],
                    "placement_policy": placement_policy,
                    "source": box_template_source,
                    "requires_projection": True,
                }
            )

        sample = baseline_structural_snapshot.get("sample", {}) if isinstance(baseline_structural_snapshot, dict) else {}
        cam_intrinsic = sample.get("cam_intrinsic") if isinstance(sample, dict) else None
        validation = self._validate_box_synthesis_draft(
            draft_boxes3d,
            cam_intrinsic=cam_intrinsic,
        )

        return {
            "available": bool(draft_boxes3d),
            "control_level": "draft_only",
            "coordinate_frame": "dd2_dataset_frame_unverified",
            "coordinate_frame_verified": False,
            "units": "meters",
            "boxes3d_format": "x_y_z_width_height_depth_rotX_rotY_rotZ",
            "default_dimensions": default_dimensions,
            "draft_boxes3d": draft_boxes3d,
            "validation": validation,
            "limitations": [
                "written_to_runtime_sample_only",
                "3d_position_uses_audited_draft_policy",
                "camera_projection_validated_when_baseline_intrinsic_is_available",
            ],
        }

    def _validate_box_synthesis_draft(
        self,
        draft_boxes3d: list,
        cam_intrinsic: list | None = None,
    ) -> dict:
        entries = []
        projection_available = self._projection_matrix_available(cam_intrinsic)

        for entry in draft_boxes3d:
            raw_box = entry.get("box3d", []) if isinstance(entry, dict) else []
            numeric_box = []
            numeric = True
            for value in raw_box:
                try:
                    numeric_box.append(float(value))
                except (TypeError, ValueError):
                    numeric = False

            shape_ok = len(raw_box) == 9
            dims_positive = (
                len(numeric_box) == 9
                and numeric_box[3] > 0
                and numeric_box[4] > 0
                and numeric_box[5] > 0
            )
            z_positive = len(numeric_box) == 9 and numeric_box[2] > 0
            projection = None
            if shape_ok and numeric and dims_positive and z_positive and projection_available:
                projection = self._project_box3d_axis_aligned(numeric_box, cam_intrinsic)

            category = entry.get("category") if isinstance(entry, dict) else None
            entries.append(
                {
                    "category": category,
                    "shape_ok": shape_ok,
                    "float32_convertible": numeric and shape_ok,
                    "dimensions_positive": dims_positive,
                    "mean_z_positive": z_positive,
                    "projection_finite": projection.get("finite") if projection else None,
                    "projected_2d_range": projection.get("range") if projection else None,
                    "requires_projection_validation": not projection_available,
                    "image_box_canvas_dry_run": self._build_image_box_canvas_dry_run(
                        category=category,
                        projection=projection,
                    ),
                }
            )

        limitations = [
            "image_box_canvas_not_rendered",
            "dataset_not_written",
        ]
        if projection_available:
            limitations.append("projection_validator_uses_axis_aligned_corners")
        else:
            limitations.append("projection_not_run")

        return {
            "available": bool(entries),
            "control_level": "validator_only",
            "projection_control_level": "validator_only" if projection_available else "not_run",
            "all_entries_valid": bool(entries)
            and all(
                item["shape_ok"]
                and item["float32_convertible"]
                and item["dimensions_positive"]
                and item["mean_z_positive"]
                and (item["projection_finite"] is not False)
                for item in entries
            ),
            "entries": entries,
            "limitations": limitations,
        }

    def _projection_matrix_available(self, cam_intrinsic: list | None) -> bool:
        return (
            isinstance(cam_intrinsic, list)
            and len(cam_intrinsic) >= 3
            and all(isinstance(row, list) and len(row) >= 3 for row in cam_intrinsic[:3])
        )

    def _build_image_box_canvas_dry_run(
        self,
        category: str | None,
        projection: dict | None,
    ) -> dict:
        class_channels = {
            "animal": 0,
            "pedestrian": 1,
            "bicycle": 5,
            "motorcycle": 6,
            "car": 8,
            "truck": 9,
            "bus": 11,
            "barrier": 17,
            "traffic_cone": 18,
        }
        class_channel = class_channels.get(category)
        projected_range = projection.get("range") if projection else None
        projected_box_drawable = (
            bool(projection.get("finite"))
            and projected_range is not None
            and class_channel is not None
        ) if projection else None

        return {
            "control_level": "validator_only",
            "target_shape": [19, 900, 1600],
            "class_channel": class_channel,
            "projected_box_drawable": projected_box_drawable,
            "projected_2d_range": projected_range,
            "canvas_rendered": False,
            "dataset_written": False,
        }

    def _project_box3d_axis_aligned(self, box3d: list, cam_intrinsic: list) -> dict:
        x, y, z, width, height, depth = box3d[:6]
        half_w, half_h, half_d = width / 2.0, height / 2.0, depth / 2.0
        corners = [
            (x + sx * half_w, y + sy * half_h, z + sz * half_d)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]

        projected = []
        finite = True
        for cx, cy, cz in corners:
            px = cam_intrinsic[0][0] * cx + cam_intrinsic[0][1] * cy + cam_intrinsic[0][2] * cz
            py = cam_intrinsic[1][0] * cx + cam_intrinsic[1][1] * cy + cam_intrinsic[1][2] * cz
            pz = cam_intrinsic[2][0] * cx + cam_intrinsic[2][1] * cy + cam_intrinsic[2][2] * cz
            if pz == 0:
                finite = False
                continue
            u = px / pz
            v = py / pz
            finite = finite and math.isfinite(u) and math.isfinite(v)
            projected.append((u, v))

        if not projected:
            return {"finite": False, "range": None}

        xs = [point[0] for point in projected]
        ys = [point[1] for point in projected]
        return {
            "finite": finite,
            "range": {
                "min": [round(min(xs), 2), round(min(ys), 2)],
                "max": [round(max(xs), 2), round(max(ys), 2)],
            },
        }


    def _build_structural_request_diff(
        self,
        structural_input_plan: dict,
        baseline_structural_snapshot: dict,
        trace_metadata: dict,
    ) -> dict:
        if not isinstance(structural_input_plan, dict) or not structural_input_plan:
            return {
                "available": False,
                "reason": "missing_structural_input_plan",
            }

        sample = baseline_structural_snapshot.get("sample", {})
        if not sample.get("available", False):
            return {
                "available": False,
                "reason": "missing_baseline_sample",
            }

        requested_labels = list(
            structural_input_plan.get("labels", {}).get("values", [])
        )
        baseline_labels = self._canonicalize_baseline_labels(
            sample.get("labels3d_preview", [])
        )

        requested_scene_description = structural_input_plan.get(
            "scene_description", {}
        ).get("value")
        baseline_scene_description = sample.get("scene_description")

        requested_set = set(requested_labels)
        baseline_set = set(baseline_labels)

        return {
            "available": True,
            "requested_labels": requested_labels,
            "baseline_labels": baseline_labels,
            "missing_requested_labels": sorted(requested_set - baseline_set),
            "extra_baseline_labels": sorted(baseline_set - requested_set),
            "requested_scene_description": requested_scene_description,
            "baseline_scene_description": baseline_scene_description,
            "scene_description_changed": requested_scene_description != baseline_scene_description,
            "tensor_override_ready": trace_metadata.get("tensor_control_ready")
            if isinstance(trace_metadata, dict)
            else None,
        }

    def _canonicalize_baseline_labels(self, labels3d_preview: list) -> list:
        labels = []
        aliases = {
            "vehicle": None,
            "human": None,
            "adult": "pedestrian",
            "child": "pedestrian",
            "bicycle": "bicycle",
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "pedestrian": "pedestrian",
        }

        for label in labels3d_preview:
            if isinstance(label, str):
                raw_parts = label.split(".")
            elif isinstance(label, (list, tuple)):
                raw_parts = [str(part) for part in label]
            else:
                continue

            for part in raw_parts:
                canonical = aliases.get(part)
                if canonical:
                    labels.append(canonical)
                    break

        return list(dict.fromkeys(labels))

    def _build_baseline_structural_snapshot(self, selected_label_index: int | None = None) -> dict:
        dataset_dir = self.baseline_dataset_dir
        snapshot = {
            "dataset_dir": str(dataset_dir),
            "available": dataset_dir.exists(),
        }
        if not dataset_dir.exists():
            return snapshot

        dataset_config = self._read_json(dataset_dir / "config.json")
        labels_config = self._read_json(dataset_dir / "labels" / "config.json")
        images_config = self._read_json(dataset_dir / "images" / "config.json")
        hdmaps_config = self._read_json(dataset_dir / "hdmaps" / "config.json")

        snapshot.update(
            {
                "dataset_config": self._summarize_config(dataset_config),
                "labels_config": self._summarize_config(labels_config),
                "images_config": self._summarize_config(images_config),
                "hdmaps_config": self._summarize_config(hdmaps_config),
                "sample": self._summarize_first_label_sample(
                    dataset_dir / "labels" / "data.pkl",
                    selected_index=selected_label_index,
                ),
            }
        )
        return snapshot

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {"available": False, "path": str(path)}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True
        data["path"] = str(path)
        return data

    def _summarize_config(self, config: dict) -> dict:
        return {
            "available": config.get("available", False),
            "path": config.get("path"),
            "class_name": config.get("_class_name"),
            "key_names": list(config.get("_key_names", [])),
            "data_size": config.get("data_size"),
            "data_type": config.get("data_type"),
            "data_name": config.get("data_name"),
            "config_paths": list(config.get("config_paths", [])),
        }

    def _summarize_first_label_sample(self, path: Path, selected_index: int | None = None) -> dict:
        if not path.exists():
            return {"available": False, "path": str(path)}

        with path.open("rb") as f:
            data = pickle.load(f)

        if not data:
            return {"available": False, "path": str(path), "reason": "empty_data"}

        sample_index = int(selected_index or 0)
        if sample_index < 0 or sample_index >= len(data):
            return {
                "available": False,
                "path": str(path),
                "selected_label_index": sample_index,
                "reason": "selected_index_out_of_range",
            }

        sample = data[sample_index]
        boxes3d = sample.get("boxes3d")
        ori_labels3d = list(sample.get("ori_labels3d", []))
        labels3d = list(sample.get("labels3d", []))
        calib = sample.get("calib", {})
        cam_intrinsic = (
            calib.get("cam_intrinsic")
            if isinstance(calib, dict)
            else None
        )

        return {
            "available": True,
            "path": str(path),
            "selected_label_index": sample_index,
            "sample_token": sample.get("sample_token"),
            "scene_token": sample.get("scene_token"),
            "cam_type": sample.get("cam_type"),
            "frame_idx": sample.get("frame_idx"),
            "scene_description": sample.get("scene_description"),
            "boxes3d_shape": list(boxes3d.shape) if hasattr(boxes3d, "shape") else None,
            "boxes3d_dtype": str(boxes3d.dtype) if hasattr(boxes3d, "dtype") else None,
            "ori_labels3d_count": len(ori_labels3d),
            "ori_labels3d_preview": ori_labels3d[:8],
            "labels3d_count": len(labels3d),
            "labels3d_preview": labels3d[:8],
            "cam_intrinsic_shape": list(cam_intrinsic.shape)
            if hasattr(cam_intrinsic, "shape")
            else None,
            "cam_intrinsic": cam_intrinsic.tolist()
            if hasattr(cam_intrinsic, "tolist")
            else None,
        }
