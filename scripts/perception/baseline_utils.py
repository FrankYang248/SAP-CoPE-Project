#!/usr/bin/env python3
"""Shared helpers for learning-based baseline adapters."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .geometry import (
    build_projection_matrix,
    invert_se3,
    load_camera_calibration,
    load_lidar_calibration,
    project_world_to_image,
    transform_points,
    world_from_pixel_and_height,
)
from .image_ops import rotate_angle_for_side, rotate_image_for_inference, rotate_keypoints_back


H36M_17_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16],
    ],
    dtype=np.int32,
)

COCO_EDGES = np.array(
    [
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [11, 13],
        [13, 15],
        [12, 14],
        [14, 16],
        [5, 6],
        [5, 11],
        [6, 12],
        [11, 12],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
    ],
    dtype=np.int32,
)

SMPL_EDGES = np.array(
    [
        [0, 1],
        [1, 4],
        [4, 7],
        [7, 10],
        [0, 2],
        [2, 5],
        [5, 8],
        [8, 11],
        [0, 3],
        [3, 6],
        [6, 9],
        [9, 12],
        [12, 15],
        [9, 13],
        [13, 16],
        [16, 18],
        [18, 20],
        [9, 14],
        [14, 17],
        [17, 19],
        [19, 21],
    ],
    dtype=np.int32,
)

LICAMPOSE_EDGES = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [3, 4],
        [4, 5],
        [0, 9],
        [9, 10],
        [10, 11],
        [2, 6],
        [6, 7],
        [7, 8],
        [2, 12],
        [12, 13],
        [13, 14],
    ],
    dtype=np.int32,
)


class ResultWriter:
    """Write baseline rows in the schema consumed by the evaluator."""

    def __init__(self, out_dir: Path, fields: Iterable[str] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []
        self.csv_path = self.out_dir / "results.csv"
        self.jsonl_path = self.out_dir / "results.jsonl"
        self.debug_dir = self.out_dir / "debug_vis"
        self.npz_dir = self.out_dir / "per_image_npz"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.npz_dir.mkdir(parents=True, exist_ok=True)
        self.fields = list(
            fields
            or [
                "scenario",
                "slice",
                "group",
                "node_id",
                "side",
                "image_path",
                "image_stem",
                "person_idx",
                "method",
                "det_score",
                "center_x",
                "center_y",
                "center_z",
                "yaw_rad",
                "yaw_deg",
                "yaw_valid",
            ]
        )
        for path in (self.csv_path, self.jsonl_path):
            if path.exists():
                path.unlink()

    def add_row(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)

    def write_jsonl(self, obj: Dict[str, Any]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(obj) + "\n")

    def finalize(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({field: row.get(field, "") for field in self.fields})
        print(f"[INFO] Wrote CSV : {self.csv_path}")
        print(f"[INFO] Wrote JSONL: {self.jsonl_path}")


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_node_side_pairs(nodes: str, sides: str) -> List[Tuple[int, str]]:
    node_ids = parse_int_list(nodes)
    side_names = [item.lower() for item in parse_str_list(sides)]
    if len(side_names) == 1 and len(node_ids) > 1:
        side_names = side_names * len(node_ids)
    if len(node_ids) != len(side_names):
        raise ValueError(f"node IDs and sides must have matching lengths: {node_ids} vs {side_names}")
    return list(zip(node_ids, side_names))


def resolve_groups(slice_dir: Path, groups: str) -> List[str]:
    pcimage_root = slice_dir / "PcImage"
    if not pcimage_root.exists():
        raise FileNotFoundError(f"PcImage directory not found: {pcimage_root}")
    if groups.strip().upper() == "ALL":
        resolved = sorted(path.name for path in pcimage_root.iterdir() if path.is_dir())
    else:
        resolved = parse_str_list(groups)
    if not resolved:
        raise RuntimeError(f"No groups resolved under {pcimage_root}")
    return resolved


def iter_images(
    data_root: Path,
    scenario: str,
    slice_name: str,
    groups: Iterable[str],
    node_side_pairs: Iterable[Tuple[int, str]],
):
    slice_dir = Path(data_root) / scenario / slice_name
    for group in groups:
        camera_dir = slice_dir / "PcImage" / group / "camera"
        if not camera_dir.exists():
            print(f"[WARN] Missing camera directory, skipping: {camera_dir}")
            continue
        for node_id, side in node_side_pairs:
            for image_path in sorted(camera_dir.glob(f"{node_id}_{side}_*.jpg")):
                yield group, int(node_id), side, image_path


def build_camera_world_transforms(camera_yaml: Path, lidar_yaml: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return K, world-to-camera transform, and camera-to-world transform."""
    intrinsic, camera_from_lidar = load_camera_calibration(camera_yaml)
    lidar_from_ground, world_from_ground, _ = load_lidar_calibration(lidar_yaml)
    ground_from_world = invert_se3(world_from_ground)
    camera_from_world = camera_from_lidar @ lidar_from_ground @ ground_from_world
    world_from_camera = invert_se3(camera_from_world)
    return intrinsic, camera_from_world, world_from_camera


def embed_rotation(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    return transform


def rotated_camera_intrinsics_and_frame(
    intrinsic: np.ndarray,
    width: int,
    height: int,
    angle: int,
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """
    Build positive-axis intrinsics for an OpenCV-rotated image and the
    cam-rot-to-physical-camera rotation.
    """
    intrinsic = np.asarray(intrinsic, dtype=float)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])

    if angle == 0:
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float), width, height, np.eye(3)
    if angle == 90:
        k_rot = np.array([[fy, 0, (height - 1) - cy], [0, fx, cx], [0, 0, 1]], dtype=float)
        r_rot_to_phys = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        return k_rot, height, width, r_rot_to_phys
    if angle == 270:
        k_rot = np.array([[fy, 0, cy], [0, fx, (width - 1) - cx], [0, 0, 1]], dtype=float)
        r_rot_to_phys = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        return k_rot, height, width, r_rot_to_phys
    if angle == 180:
        k_rot = np.array([[fx, 0, (width - 1) - cx], [0, fy, (height - 1) - cy], [0, 0, 1]], dtype=float)
        r_rot_to_phys = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        return k_rot, width, height, r_rot_to_phys
    raise ValueError("Rotation angle must be one of 0, 90, 180, 270")


def canonicalize_intrinsics(intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use positive focal lengths and return p_original = R @ p_model."""
    k_model = np.asarray(intrinsic, dtype=float).copy()
    sx = -1.0 if k_model[0, 0] < 0 else 1.0
    sy = -1.0 if k_model[1, 1] < 0 else 1.0
    k_model[0, 0] = abs(k_model[0, 0])
    k_model[1, 1] = abs(k_model[1, 1])
    return k_model, np.diag([sx, sy, 1.0]).astype(float)


def project_points(projection_or_k: np.ndarray, transform_or_points: np.ndarray, points: np.ndarray | None = None) -> np.ndarray:
    """Project either with P and points, or with K, T_c_w, and points."""
    if points is None:
        projection = np.asarray(projection_or_k, dtype=float)
        pts = np.asarray(transform_or_points, dtype=float)
        homo = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=float)])
        uvw = (projection @ homo.T).T
    else:
        intrinsic = np.asarray(projection_or_k, dtype=float)
        camera_from_world = np.asarray(transform_or_points, dtype=float)
        pts = np.asarray(points, dtype=float)
        homo = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=float)])
        camera_points = (camera_from_world @ homo.T).T[:, :3]
        uvw = (intrinsic @ camera_points.T).T
    z = np.where(np.abs(uvw[:, 2:3]) < 1e-6, 1e-6, uvw[:, 2:3])
    return uvw[:, :2] / z


def draw_skeleton(image_bgr: np.ndarray, keypoints_xy: np.ndarray, edges: np.ndarray, color_bgr, radius: int = 3, thickness: int = 2) -> None:
    import cv2

    keypoints = np.asarray(keypoints_xy, dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return
    for point in keypoints[:, :2]:
        if np.isfinite(point).all():
            cv2.circle(image_bgr, tuple(np.round(point).astype(int)), radius, color_bgr, -1)
    for a, b in edges:
        if a < keypoints.shape[0] and b < keypoints.shape[0]:
            p0, p1 = keypoints[int(a), :2], keypoints[int(b), :2]
            if np.isfinite([p0[0], p0[1], p1[0], p1[1]]).all():
                cv2.line(image_bgr, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color_bgr, thickness)


def center_yaw_from_pairs(
    joints_world: np.ndarray,
    *,
    center_indices: Iterable[int],
    yaw_pairs: Iterable[Tuple[int, int]],
) -> Tuple[np.ndarray, float, bool]:
    joints = np.asarray(joints_world, dtype=float)
    valid = np.isfinite(joints).all(axis=1)
    center_candidates = [joints[idx] for idx in center_indices if idx < joints.shape[0] and bool(valid[idx])]
    if center_candidates:
        center = np.mean(np.stack(center_candidates, axis=0), axis=0)
    else:
        center = np.mean(joints[valid] if np.any(valid) else joints, axis=0)

    for left_idx, right_idx in yaw_pairs:
        if left_idx >= joints.shape[0] or right_idx >= joints.shape[0]:
            continue
        if not (bool(valid[left_idx]) and bool(valid[right_idx])):
            continue
        direction = joints[right_idx, :2] - joints[left_idx, :2]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            direction = direction / norm
            return center, float(math.atan2(direction[1], direction[0]) + math.pi / 2.0), True
    return center, 0.0, False


def center_yaw_h36m17(joints_world: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    return center_yaw_from_pairs(joints_world, center_indices=[0, 1, 4], yaw_pairs=[(1, 4), (11, 14)])


def center_yaw_coco17(joints_world: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    return center_yaw_from_pairs(joints_world, center_indices=[11, 12, 5, 6], yaw_pairs=[(11, 12), (5, 6)])


def center_yaw_smpl(joints_world: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    return center_yaw_from_pairs(joints_world, center_indices=[0, 1, 2], yaw_pairs=[(1, 2), (16, 17)])


def center_yaw_licampose15(joints_world: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    return center_yaw_from_pairs(joints_world, center_indices=[2], yaw_pairs=[(6, 12), (3, 9)])


def bbox_world_center_from_image(
    bbox_xyxy: np.ndarray,
    projection: np.ndarray,
    *,
    z_world: float = 0.8,
) -> Optional[np.ndarray]:
    bbox = np.asarray(bbox_xyxy, dtype=float).reshape(4)
    u = 0.5 * (bbox[0] + bbox[2])
    v = 0.5 * (bbox[1] + bbox[3])
    try:
        point, _ = world_from_pixel_and_height(float(u), float(v), float(z_world), projection)
        return point
    except ValueError:
        return None


def bbox_from_keypoints(keypoints_xy: np.ndarray, scores: np.ndarray, threshold: float = 0.2) -> Optional[np.ndarray]:
    valid = np.asarray(scores, dtype=float) > float(threshold)
    if not np.any(valid):
        return None
    points = np.asarray(keypoints_xy, dtype=float)[valid]
    return np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])], dtype=float)


def load_group_lidar_world(slice_dir: Path, group: str, lidar_nodes: Iterable[int]) -> np.ndarray:
    all_points = []
    for node_id in lidar_nodes:
        lidar_dir = slice_dir / "PcImage" / group / "lidar"
        files = sorted(lidar_dir.glob(f"{int(node_id)}_*.npy"))
        if not files:
            continue
        _, _, world_from_lidar = load_lidar_calibration(slice_dir / "Calibration" / "lidar" / f"{int(node_id)}.yaml")
        raw = np.load(files[0])
        all_points.append(transform_points(np.asarray(raw, dtype=float)[:, :3], world_from_lidar))
    if not all_points:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(all_points)


def make_result_row(
    *,
    scenario: str,
    slice_name: str,
    group: str,
    node_id: int | str = "",
    side: str = "",
    image_path: Path | str = "",
    person_idx: int,
    method: str,
    det_score: float,
    center: np.ndarray,
    yaw: float,
    yaw_valid: bool,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row = {
        "scenario": scenario,
        "slice": slice_name,
        "group": group,
        "node_id": node_id,
        "side": side,
        "image_path": str(image_path),
        "image_stem": Path(image_path).stem if image_path else "",
        "person_idx": int(person_idx),
        "method": method,
        "det_score": float(det_score),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "yaw_rad": float(yaw),
        "yaw_deg": float(math.degrees(yaw)),
        "yaw_valid": int(bool(yaw_valid)),
    }
    if extra:
        row.update(extra)
    return row


__all__ = [
    "COCO_EDGES",
    "H36M_17_EDGES",
    "LICAMPOSE_EDGES",
    "SMPL_EDGES",
    "ResultWriter",
    "bbox_from_keypoints",
    "bbox_world_center_from_image",
    "build_camera_world_transforms",
    "build_projection_matrix",
    "canonicalize_intrinsics",
    "center_yaw_coco17",
    "center_yaw_h36m17",
    "center_yaw_licampose15",
    "center_yaw_smpl",
    "draw_skeleton",
    "embed_rotation",
    "iter_images",
    "load_group_lidar_world",
    "make_result_row",
    "parse_int_list",
    "parse_str_list",
    "project_points",
    "project_world_to_image",
    "resolve_groups",
    "rotate_angle_for_side",
    "rotate_image_for_inference",
    "rotate_keypoints_back",
    "rotated_camera_intrinsics_and_frame",
]

