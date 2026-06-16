#!/usr/bin/env python3
"""Calibration, projection, and uncertainty propagation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import yaml


def invert_se3(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def as_homogeneous_transform(value: Any, *, name: str) -> np.ndarray:
    """Convert a YAML 3x4 or 4x4 transform field to a 4x4 matrix."""
    transform = np.asarray(value, dtype=float)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, [0.0, 0.0, 0.0, 1.0]])
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 3x4 or 4x4 transform, got {transform.shape}")
    return transform


def load_camera_calibration(camera_yaml: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics and the LiDAR-to-camera transform.

    Supported calibration keys:
      - intrinsic_matrix: 3x3 camera matrix
      - camera_to_lidar: camera-to-LiDAR transform, inverted internally
      - T_camera_lidar: LiDAR-to-camera transform
    """
    with open(camera_yaml, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if "intrinsic_matrix" not in data:
        raise ValueError(f"Missing 'intrinsic_matrix' in {camera_yaml}")
    intrinsic = np.asarray(data["intrinsic_matrix"], dtype=float).reshape(3, 3)

    if "camera_to_lidar" in data:
        lidar_from_camera = as_homogeneous_transform(
            data["camera_to_lidar"], name="camera_to_lidar"
        )
        camera_from_lidar = invert_se3(lidar_from_camera)
    elif "T_camera_lidar" in data:
        camera_from_lidar = as_homogeneous_transform(
            data["T_camera_lidar"], name="T_camera_lidar"
        )
    else:
        raise ValueError(f"Missing camera/LiDAR extrinsics in {camera_yaml}")

    return intrinsic, camera_from_lidar


def load_lidar_calibration(lidar_yaml: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load LiDAR, ground, and world transforms.

    Returns:
      - lidar_from_ground
      - world_from_ground
      - world_from_lidar
    """
    with open(lidar_yaml, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if "lidar_to_ground" not in data:
        raise ValueError(f"Missing 'lidar_to_ground' in {lidar_yaml}")

    ground_from_lidar = as_homogeneous_transform(
        data["lidar_to_ground"], name="lidar_to_ground"
    )
    lidar_from_ground = invert_se3(ground_from_lidar)

    if "ground_to_global" in data:
        world_from_ground = as_homogeneous_transform(
            data["ground_to_global"], name="ground_to_global"
        )
    else:
        world_from_ground = np.eye(4, dtype=float)

    world_from_lidar = world_from_ground @ ground_from_lidar
    return lidar_from_ground, world_from_ground, world_from_lidar


def build_projection_matrix(camera_yaml: Path, lidar_yaml: Path) -> np.ndarray:
    """Build P such that image homogeneous pixels satisfy u ~ P [x_w,y_w,z_w,1]."""
    intrinsic, camera_from_lidar = load_camera_calibration(camera_yaml)
    lidar_from_ground, world_from_ground, _ = load_lidar_calibration(lidar_yaml)
    ground_from_world = invert_se3(world_from_ground)
    camera_from_world = camera_from_lidar @ lidar_from_ground @ ground_from_world
    return intrinsic @ camera_from_world[:3, :]


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an Nx3 point array."""
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=float)
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError(f"Expected an Nx3 point array, got {points_xyz.shape}")
    homogeneous = np.hstack([points_xyz[:, :3], np.ones((points_xyz.shape[0], 1))])
    return (np.asarray(transform, dtype=float) @ homogeneous.T).T[:, :3]


def project_world_to_image(projection: np.ndarray, point_xyz: np.ndarray) -> Tuple[float, float] | None:
    """Project one world-frame point to image pixels."""
    point = np.asarray(point_xyz, dtype=float).reshape(3)
    uvw = np.asarray(projection, dtype=float) @ np.array(
        [point[0], point[1], point[2], 1.0], dtype=float
    )
    if abs(float(uvw[2])) < 1e-9:
        return None
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def world_from_pixel_and_height(
    xp: float,
    yp: float,
    zw: float,
    projection: np.ndarray,
    *,
    min_denominator: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover world x,y for a pixel and known world z, plus the analytic Jacobian.

    The returned Jacobian maps perturbations in (xp, yp, zw) to
    perturbations in (xw, yw, zw), matching the uncertainty propagation in
    the SAP-CoPE paper.
    """
    projection = np.asarray(projection, dtype=float)
    if projection.shape != (3, 4):
        raise ValueError(f"Projection matrix must be 3x4, got {projection.shape}")

    h11, h12, h13, h14 = projection[0]
    h21, h22, h23, h24 = projection[1]
    h31, h32, h33, h34 = projection[2]

    a11 = h31 * xp - h11
    a12 = h32 * xp - h12
    a21 = h31 * yp - h21
    a22 = h32 * yp - h22
    b1 = h13 * zw - h33 * xp * zw - h34 * xp + h14
    b2 = h23 * zw - h33 * yp * zw - h34 * yp + h24

    numerator_x = b1 * a22 - b2 * a12
    numerator_y = b2 * a11 - b1 * a21
    denominator = a11 * a22 - a12 * a21
    if abs(float(denominator)) < min_denominator:
        raise ValueError("Degenerate projection geometry for the requested pixel/height")

    xw = numerator_x / denominator
    yw = numerator_y / denominator

    denominator_sq = denominator * denominator
    dxw_dxp = (
        -denominator * (a22 * (h33 * zw + h34) + b2 * h32)
        + numerator_x * (a21 * h32 - a22 * h31)
    ) / denominator_sq
    dxw_dyp = (
        denominator * (a12 * (h33 * zw + h34) + b1 * h32)
        - numerator_x * (a11 * h32 - a12 * h31)
    ) / denominator_sq
    dxw_dzw = (-a12 * (h23 - h33 * yp) + a22 * (h13 - h33 * xp)) / denominator

    dyw_dxp = (
        denominator * (a21 * (h33 * zw + h34) + b2 * h31)
        + numerator_y * (a21 * h32 - a22 * h31)
    ) / denominator_sq
    dyw_dyp = (
        -denominator * (a11 * (h33 * zw + h34) + b1 * h31)
        - numerator_y * (a11 * h32 - a12 * h31)
    ) / denominator_sq
    dyw_dzw = (a11 * (h23 - h33 * yp) - a21 * (h13 - h33 * xp)) / denominator

    jacobian = np.array(
        [
            [dxw_dxp, dxw_dyp, dxw_dzw],
            [dyw_dxp, dyw_dyp, dyw_dzw],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return np.array([xw, yw, zw], dtype=float), jacobian


def propagate_pixel_height_covariance(
    jacobian: np.ndarray,
    pixel_sigma: float | Tuple[float, float] = 10.0,
    height_sigma: float = 0.2,
) -> np.ndarray:
    """Propagate diagonal pixel/height uncertainty to 3D world covariance."""
    if isinstance(pixel_sigma, tuple):
        sigma_x, sigma_y = float(pixel_sigma[0]), float(pixel_sigma[1])
    else:
        sigma_x = sigma_y = float(pixel_sigma)
    input_covariance = np.diag([sigma_x * sigma_x, sigma_y * sigma_y, height_sigma * height_sigma])
    covariance = np.asarray(jacobian, dtype=float) @ input_covariance @ np.asarray(jacobian, dtype=float).T
    return 0.5 * (covariance + covariance.T)

