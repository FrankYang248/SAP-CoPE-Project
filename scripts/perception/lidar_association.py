#!/usr/bin/env python3
"""Sparse LiDAR-to-joint association utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

from .geometry import project_world_to_image, transform_points
from .pose_refinement import LidarJointMeasurements, NUM_COCO_KEYPOINTS


def load_bev_roi(slice_dir: Path, node_id: int):
    """Load optional BEV transform metadata and node ROI mask."""
    import cv2

    hdmap_dir = slice_dir / "HDmap"
    transform_yaml = hdmap_dir / "transformation.yaml"
    bev_cfg = None
    if transform_yaml.exists():
        with open(transform_yaml, "r", encoding="utf-8") as file:
            info = yaml.safe_load(file) or {}
        bev_cfg = {
            "img_size": int(info["img_size"]),
            "scale": float(info["scale"]),
            "min_x": float(info["min_x"]),
            "max_y": float(info["max_y"]),
        }

    roi_path = hdmap_dir / "ROI" / f"{node_id}_roi.png"
    roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path.exists() else None
    return bev_cfg, roi_mask


def filter_points_by_roi(points_world: np.ndarray, bev_cfg, roi_mask: np.ndarray | None) -> np.ndarray:
    """Filter world-frame points by a BEV ROI mask when one is available."""
    if bev_cfg is None or roi_mask is None or points_world.size == 0:
        return points_world
    x = points_world[:, 0]
    y = points_world[:, 1]
    img_x = ((x - bev_cfg["min_x"]) * bev_cfg["scale"]).astype(np.int32)
    img_y = ((bev_cfg["max_y"] - y) * bev_cfg["scale"]).astype(np.int32)
    inside = (
        (img_x >= 0)
        & (img_x < bev_cfg["img_size"])
        & (img_y >= 0)
        & (img_y < bev_cfg["img_size"])
    )
    if not np.any(inside):
        return np.zeros((0, 3), dtype=float)
    filtered = points_world[inside]
    img_x = img_x[inside]
    img_y = img_y[inside]
    return filtered[roi_mask[img_y, img_x] > 0]


def project_lidar_points(projection: np.ndarray, points_world: np.ndarray, image_shape: tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Project world-frame LiDAR points into an image and keep in-frame points."""
    height, width = image_shape[:2]
    uv_rows = []
    xyz_rows = []
    for point in points_world:
        uv = project_world_to_image(projection, point)
        if uv is None:
            continue
        u, v = uv
        if 0.0 <= u < width and 0.0 <= v < height:
            uv_rows.append([u, v])
            xyz_rows.append(point)
    if not uv_rows:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)
    return np.asarray(uv_rows, dtype=float), np.asarray(xyz_rows, dtype=float)


def associate_lidar_to_keypoints(
    *,
    keypoints_xy: np.ndarray,
    keypoint_valid: np.ndarray,
    lidar_uv: np.ndarray,
    lidar_xyz_world: np.ndarray,
    initial_joints_world: np.ndarray,
    pixel_threshold: float = 10.0,
    world_threshold: float = 0.7,
    lidar_sigma_m: float = 0.03,
) -> LidarJointMeasurements:
    """Associate projected LiDAR points to valid 2D joints by nearest image distance."""
    mean = np.zeros((NUM_COCO_KEYPOINTS, 3), dtype=float)
    covariance = np.zeros((NUM_COCO_KEYPOINTS, 3, 3), dtype=float)
    valid = np.zeros(NUM_COCO_KEYPOINTS, dtype=bool)
    if lidar_uv.size == 0:
        return LidarJointMeasurements(mean=mean, covariance=covariance, valid=valid)

    for joint_idx in range(NUM_COCO_KEYPOINTS):
        if not bool(keypoint_valid[joint_idx]):
            continue
        delta = lidar_uv - keypoints_xy[joint_idx]
        distances = np.linalg.norm(delta, axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) > float(pixel_threshold):
            continue
        lidar_point = lidar_xyz_world[nearest]
        if float(np.linalg.norm(lidar_point - initial_joints_world[joint_idx])) > float(world_threshold):
            continue
        mean[joint_idx] = lidar_point
        covariance[joint_idx] = np.eye(3, dtype=float) * float(lidar_sigma_m) ** 2
        valid[joint_idx] = True

    return LidarJointMeasurements(mean=mean, covariance=covariance, valid=valid)


def load_world_lidar_points(
    lidar_npy: Path,
    world_from_lidar: np.ndarray,
    *,
    slice_dir: Path | None = None,
    node_id: int | None = None,
) -> np.ndarray:
    """Load a LiDAR npy file, transform points to world, and apply optional ROI filtering."""
    raw = np.load(str(lidar_npy))
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"LiDAR npy must be Nx3 or wider, got {raw.shape}")
    points_world = transform_points(raw[:, :3], world_from_lidar)
    if slice_dir is not None and node_id is not None:
        bev_cfg, roi_mask = load_bev_roi(slice_dir, int(node_id))
        points_world = filter_points_by_roi(points_world, bev_cfg, roi_mask)
    return points_world

