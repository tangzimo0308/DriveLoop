from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from dd_scripts.converters.nuscenes_converter import (
    NuScenesConverter,
    get_map_geom,
    line_geoms_to_vectors,
    poly_geoms_to_vectors,
    preprocess_map,
    quaternion_yaw,
    view_points_depth,
)


DEFAULT_PROBE = Path("outputs/driveloop/candidate70_hdmap_raster_probe/candidate70_hdmap_raster_probe_summary.json")
DEFAULT_OUTPUT = Path("outputs/driveloop/candidate70_hdmap_geometry_introspection/candidate70_hdmap_geometry_introspection_summary.json")
DEFAULT_IMAGES_DIR = Path("outputs/driveloop/candidate70_hdmap_geometry_introspection/images")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def array_signature(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sum": float(array.astype(np.float64).sum()) if array.size else 0.0,
        "nonzero": int(np.count_nonzero(array)) if array.size else 0,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def vector_stats(vectors: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for vector in vectors:
        type_name = str(vector.get("type_name", "unknown"))
        pts = np.asarray(vector.get("pts", []), dtype=np.float64)
        pts_num = int(vector.get("pts_num", 0) or 0)
        entry = by_type.setdefault(
            type_name,
            {
                "count": 0,
                "visible_count": 0,
                "total_pts_num": 0,
                "total_visible_pts_num": 0,
                "min_xy": None,
                "max_xy": None,
            },
        )
        entry["count"] += 1
        entry["total_pts_num"] += pts_num
        if pts_num >= 2:
            entry["visible_count"] += 1
            entry["total_visible_pts_num"] += pts_num
        if pts.size:
            xy = pts[:, :2]
            min_xy = xy.min(axis=0).tolist()
            max_xy = xy.max(axis=0).tolist()
            if entry["min_xy"] is None:
                entry["min_xy"] = min_xy
                entry["max_xy"] = max_xy
            else:
                entry["min_xy"] = [min(entry["min_xy"][i], min_xy[i]) for i in range(2)]
                entry["max_xy"] = [max(entry["max_xy"][i], max_xy[i]) for i in range(2)]
    return by_type


def geometry_counts(line_vector_dict: dict[str, Any], ped_vector_list: list[Any], poly_bound_list: list[Any]) -> dict[str, Any]:
    return {
        "road_divider": len(line_vector_dict.get("road_divider", [])),
        "lane_divider": len(line_vector_dict.get("lane_divider", [])),
        "ped_crossing": len(ped_vector_list),
        "contours": len(poly_bound_list),
    }


def build_filtered_vectors(converter: NuScenesConverter, cam_token: str, scene_token: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cam_record = converter.nusc.get("sample_data", cam_token)
    pose_record = converter.nusc.get("ego_pose", cam_record["ego_pose_token"])
    cs_record = converter.nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])
    cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    ego2global_translation = np.array(pose_record["translation"])
    rotation = Quaternion(pose_record["rotation"])
    map_pose = ego2global_translation[:2]
    patch_box = (map_pose[0], map_pose[1], 102.4, 102.4)
    patch_angle = quaternion_yaw(rotation) / np.pi * 180
    location = converter.nusc.get("log", converter.nusc.get("scene", scene_token)["log_token"])["location"]
    nusc_map = converter.nusc_maps[location]
    map_explorer = converter.map_explorer[location]

    line_geom = get_map_geom(patch_box, patch_angle, ["road_divider", "lane_divider"], nusc_map, map_explorer)
    line_vector_dict = line_geoms_to_vectors(line_geom)
    ped_geom = get_map_geom(patch_box, patch_angle, ["ped_crossing"], nusc_map, map_explorer)
    ped_vector_list = line_geoms_to_vectors(ped_geom)["ped_crossing"]
    polygon_geom = get_map_geom(patch_box, patch_angle, ["road_segment", "lane"], nusc_map, map_explorer)
    poly_bound_list = poly_geoms_to_vectors(polygon_geom)

    vectors = []
    for line_type, vects in line_vector_dict.items():
        for line, length in vects:
            label = converter.class2label.get(line_type, -1)
            if label != -1:
                vectors.append({
                    "pts": line.astype(float),
                    "pts_num": int(length),
                    "type": int(label),
                    "type_name": line_type,
                })
    for ped_line, length in ped_vector_list:
        label = converter.class2label.get("ped_crossing", -1)
        if label != -1:
            vectors.append({
                "pts": ped_line.astype(float),
                "pts_num": int(length),
                "type": int(label),
                "type_name": "ped_crossing",
            })
    for contour, length in poly_bound_list:
        label = converter.class2label.get("contours", -1)
        if label != -1:
            vectors.append({
                "pts": contour.astype(float),
                "pts_num": int(length),
                "type": int(label),
                "type_name": "contours",
            })

    for vector in vectors:
        pts = np.asarray(vector["pts"], dtype=np.float64)
        vector["pts"] = np.concatenate((pts, np.zeros((pts.shape[0], 1))), axis=1)

    for vector in vectors:
        pts3d = np.asarray(vector["pts"], dtype=np.float64).T
        pts3d = pts3d - np.array(cs_record["translation"]).reshape((-1, 1))
        pts3d = np.dot(Quaternion(cs_record["rotation"]).rotation_matrix.T, pts3d)
        projected, depth = view_points_depth(pts3d, cam_intrinsic, normalize=True)
        visible = depth > 1e-3
        vector["source_pts_num"] = int(vector["pts_num"])
        vector["pts_num"] = int(vector["pts_num"] - (depth <= 1e-3).sum())
        vector["visible_depth_positive"] = int(visible.sum())
        vector["pts"] = projected[:2, visible].T

    metadata = {
        "location": location,
        "patch_box": list(patch_box),
        "patch_angle": float(patch_angle),
        "imsize": [cam_record["width"], cam_record["height"]],
        "raw_geometry_counts": geometry_counts(line_vector_dict, ped_vector_list, poly_bound_list),
    }
    return vectors, metadata


def rasterize_vectors(vectors: list[dict[str, Any]], imsize: list[int]) -> Image.Image:
    map_canvas_size = [imsize[1], imsize[0]]
    semantic_masks = preprocess_map(vectors, map_canvas_size, max_channel=3, thickness=10)
    color_base_map = 255 * np.ones((imsize[1], imsize[0], 3), dtype=np.uint8)
    color_base_map[..., 0] *= ~semantic_masks[0]
    color_base_map[..., 1] *= ~semantic_masks[1]
    color_base_map[..., 2] *= ~semantic_masks[2]
    color_base_map = 255 - color_base_map
    color_base_map = color_base_map[:, :, ::-1]
    return Image.fromarray(color_base_map)


def build_frame_record(converter: NuScenesConverter, probe_record: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    cam_token = str(probe_record["cam_token"])
    scene_token = str(probe_record["scene_token"])
    vectors, metadata = build_filtered_vectors(converter, cam_token, scene_token)
    raster = rasterize_vectors(vectors, metadata["imsize"])

    frame_index = int(probe_record.get("i", 0))
    data_index = probe_record.get("data_index")
    output_path = images_dir / f"candidate70_frame_{frame_index:02d}_idx_{data_index}_geometry_rebuilt_hdmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raster.save(output_path)

    raster_signature = array_signature(raster)
    converter_signature = probe_record.get("converter_signature", {})
    matches_converter = raster_signature.get("sha256") == converter_signature.get("sha256")

    return {
        "i": frame_index,
        "data_index": data_index,
        "frame_idx": probe_record.get("frame_idx"),
        "cam_token": cam_token,
        "scene_token": scene_token,
        "geometry_metadata": metadata,
        "vector_stats": vector_stats(vectors),
        "total_vector_count": len(vectors),
        "visible_vector_count": sum(1 for vector in vectors if int(vector.get("pts_num", 0) or 0) >= 2),
        "rebuilt_raster_path": str(output_path),
        "rebuilt_raster_signature": raster_signature,
        "converter_hdmap_path": probe_record.get("converter_hdmap_path"),
        "converter_signature": converter_signature,
        "rebuilt_matches_converter_signature": matches_converter,
        "claim_boundary": {
            "geometry_introspection_is_not_lane_geometry_override": True,
            "rebuilt_raster_match_is_not_semantic_success": True,
            "does_not_run_gpu": True,
        },
    }


def build_summary(records: list[dict[str, Any]], probe_path: Path, raw_root: str) -> dict[str, Any]:
    match_true = sum(1 for record in records if record.get("rebuilt_matches_converter_signature") is True)
    match_false = sum(1 for record in records if record.get("rebuilt_matches_converter_signature") is False)
    layer_visible_counts: dict[str, int] = {}
    for record in records:
        stats = record.get("vector_stats", {})
        if not isinstance(stats, dict):
            continue
        for layer, layer_stats in stats.items():
            if not isinstance(layer_stats, dict):
                continue
            layer_visible_counts[layer] = layer_visible_counts.get(layer, 0) + int(layer_stats.get("visible_count", 0) or 0)

    return {
        "schema_version": "candidate70_hdmap_geometry_introspection_audit.v0",
        "audit_only": True,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "does_not_modify_model_inputs": True,
        "probe_path": str(probe_path),
        "raw_root": raw_root,
        "frame_count": len(records),
        "rebuilt_match_true": match_true,
        "rebuilt_match_false": match_false,
        "all_rebuilt_match_converter": bool(records) and match_false == 0,
        "layer_visible_counts": layer_visible_counts,
        "records": records,
        "claim": {
            "candidate70_hdmap_geometry_introspected": bool(records),
            "candidate70_geometry_rebuild_matches_converter": bool(records) and match_false == 0,
            "candidate70_true_lane_geometry_replacement_available": False,
            "hdmap_lane_geometry_override_verified": False,
            "lane_change_control_verified": False,
            "runtime_motion_control_connected": False,
            "semantic_success_claim_allowed": False,
        },
        "claim_boundary": {
            "geometry_introspection_is_not_lane_geometry_override": True,
            "geometry_rebuild_match_is_not_replacement": True,
            "hdmap_raster_hash_change_is_not_lane_change_control": True,
            "runtime_or_raster_audit_is_not_video_semantic_success": True,
        },
        "next_required_steps": [
            "Use these geometry/vector stats to decide whether a true lane-geometry replacement raster can be constructed.",
            "Do not claim lane geometry override until a replacement raster is explicitly constructed and audited.",
            "Do not run GPU from this audit alone.",
        ],
    }


def run_audit(probe_path: Path, output_path: Path, images_dir: Path, max_frames: int | None) -> dict[str, Any]:
    probe = load_json(probe_path)
    raw_root = str(probe.get("raw_root", "/data/projects/DriveLoop/data/raw/nuscenes"))
    records = probe.get("records", [])
    if not isinstance(records, list) or not records:
        raise SystemExit(f"No records found in probe: {probe_path}")

    version = "v1.0-mini"
    dataroot = Path(raw_root)
    if dataroot.name == version:
        dataroot = dataroot.parent
    converter = NuScenesConverter(
        data_dir=str(dataroot),
        version=version,
        save_path="/tmp/driveloop_candidate70_hdmap_geometry_introspection_unused",
        save_version="v0.0.1",
    )

    selected_records = records[:max_frames] if max_frames is not None else records
    frame_records = [
        build_frame_record(converter, record, images_dir)
        for record in selected_records
        if isinstance(record, dict)
    ]
    summary = build_summary(frame_records, probe_path=probe_path, raw_root=raw_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate70 HDMap geometry/vector surfaces without GPU or model inference.")
    parser.add_argument("--probe-path", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    summary = run_audit(
        probe_path=args.probe_path,
        output_path=args.output,
        images_dir=args.images_dir,
        max_frames=args.max_frames,
    )
    print(args.output)
    print(json.dumps({
        "schema_version": summary["schema_version"],
        "frame_count": summary["frame_count"],
        "rebuilt_match_true": summary["rebuilt_match_true"],
        "rebuilt_match_false": summary["rebuilt_match_false"],
        "all_rebuilt_match_converter": summary["all_rebuilt_match_converter"],
        "layer_visible_counts": summary["layer_visible_counts"],
        "claim": summary["claim"],
        "claim_boundary": summary["claim_boundary"],
    }, indent=2))


if __name__ == "__main__":
    main()
