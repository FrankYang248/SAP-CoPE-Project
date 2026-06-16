#!/usr/bin/env python3
"""Uncertainty-aware 3D human pose estimation with optional LiDAR constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .geometry import propagate_pixel_height_covariance, world_from_pixel_and_height

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - handled at runtime.
    least_squares = None


NUM_COCO_KEYPOINTS = 17

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_SKELETON_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

SYMMETRIC_ORIENTATION_PAIRS = (
    (1, 2, 0.5),
    (3, 4, 0.5),
    (5, 6, 3.0),
    (11, 12, 2.0),
    (13, 14, 0.3),
    (15, 16, 0.2),
)

DEFAULT_KEYPOINT_HEIGHT_MEAN_M = np.array(
    [1.63, 1.67, 1.67, 1.64, 1.64, 1.42, 1.42, 1.20, 1.20, 0.90, 0.90, 0.95, 0.95, 0.50, 0.50, 0.10, 0.10],
    dtype=float,
)

DEFAULT_KEYPOINT_HEIGHT_SIGMA_M = np.array(
    [0.20, 0.20, 0.20, 0.20, 0.20, 0.15, 0.15, 0.35, 0.35, 0.50, 0.50, 0.10, 0.10, 0.05, 0.05, 0.02, 0.02],
    dtype=float,
)


@dataclass
class HumanPoseEstimate:
    """A single person pose in world coordinates."""

    bbox_xyxy: np.ndarray
    score: float
    valid: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    refined: np.ndarray
    center: np.ndarray
    center_covariance: np.ndarray
    yaw_rad: float = 0.0
    yaw_valid: bool = False


@dataclass
class LidarJointMeasurements:
    """Optional LiDAR-derived Gaussian joint constraints."""

    mean: np.ndarray
    covariance: np.ndarray
    valid: np.ndarray


def _regularized_inverse(matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.inv(matrix + regularization * np.eye(matrix.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix + regularization * np.eye(matrix.shape[0]))


def _sqrt_information(covariance: np.ndarray) -> np.ndarray:
    information = _regularized_inverse(covariance)
    try:
        return np.linalg.cholesky(information).T
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(information)
        values = np.maximum(values, 1e-9)
        return np.diag(np.sqrt(values)) @ vectors.T


def default_bone_priors(
    edges: Iterable[Tuple[int, int]] = COCO_SKELETON_EDGES,
    height_means: np.ndarray = DEFAULT_KEYPOINT_HEIGHT_MEAN_M,
    *,
    fallback_length_m: float = 0.35,
    sigma_fraction: float = 0.25,
    min_sigma_m: float = 0.05,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Build conservative bone-length priors from the height template.

    User-supplied anthropometric priors can be passed to `refine_pose`; this
    fallback keeps the release code runnable without private training metadata.
    """
    priors: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for a, b in edges:
        length = abs(float(height_means[a] - height_means[b]))
        if length < min_sigma_m:
            length = fallback_length_m
        priors[(a, b)] = (length, max(min_sigma_m, sigma_fraction * length))
    return priors


def estimate_orientation_from_coco_joints(joints_xyz: np.ndarray, valid: np.ndarray) -> Tuple[float, bool]:
    """Estimate planar yaw from left/right COCO joint pairs."""
    vector = np.zeros(2, dtype=float)
    total_weight = 0.0
    for left, right, weight in SYMMETRIC_ORIENTATION_PAIRS:
        if bool(valid[left]) and bool(valid[right]):
            direction = np.asarray(joints_xyz[right, :2] - joints_xyz[left, :2], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                vector += float(weight) * direction / norm
                total_weight += float(weight)
    if total_weight <= 4.0:
        return 0.0, False
    vector /= total_weight
    return math.atan2(float(vector[1]), float(vector[0])) + math.pi / 2.0, True


def build_pose_from_coco_detection(
    bbox_xyxy: np.ndarray,
    score: float,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    projection: np.ndarray,
    *,
    keypoint_conf_threshold: float = 0.5,
    pixel_sigma: float = 10.0,
    center_height_m: float = 0.85,
    center_height_sigma_m: float = 0.20,
    keypoint_height_mean_m: np.ndarray = DEFAULT_KEYPOINT_HEIGHT_MEAN_M,
    keypoint_height_sigma_m: np.ndarray = DEFAULT_KEYPOINT_HEIGHT_SIGMA_M,
) -> HumanPoseEstimate:
    """Lift one COCO 2D pose detection to uncertain 3D world-frame joints."""
    bbox_xyxy = np.asarray(bbox_xyxy, dtype=float).reshape(4)
    keypoints_xy = np.asarray(keypoints_xy, dtype=float)
    keypoints_conf = np.asarray(keypoints_conf, dtype=float).reshape(-1)
    if keypoints_xy.shape != (NUM_COCO_KEYPOINTS, 2):
        raise ValueError(f"Expected keypoints_xy shape (17,2), got {keypoints_xy.shape}")
    if keypoints_conf.shape[0] != NUM_COCO_KEYPOINTS:
        raise ValueError(f"Expected 17 keypoint confidences, got {keypoints_conf.shape[0]}")

    center_pixel = np.array(
        [0.5 * (bbox_xyxy[0] + bbox_xyxy[2]), 0.5 * (bbox_xyxy[1] + bbox_xyxy[3])],
        dtype=float,
    )
    center, center_jacobian = world_from_pixel_and_height(
        float(center_pixel[0]), float(center_pixel[1]), center_height_m, projection
    )
    center_covariance = propagate_pixel_height_covariance(
        center_jacobian, pixel_sigma=pixel_sigma, height_sigma=center_height_sigma_m
    )

    valid = keypoints_conf >= float(keypoint_conf_threshold)
    means = np.zeros((NUM_COCO_KEYPOINTS, 3), dtype=float)
    covariances = np.zeros((NUM_COCO_KEYPOINTS, 3, 3), dtype=float)
    for joint_idx in range(NUM_COCO_KEYPOINTS):
        if not bool(valid[joint_idx]):
            continue
        point, jacobian = world_from_pixel_and_height(
            float(keypoints_xy[joint_idx, 0]),
            float(keypoints_xy[joint_idx, 1]),
            float(keypoint_height_mean_m[joint_idx]),
            projection,
        )
        means[joint_idx] = point
        covariances[joint_idx] = propagate_pixel_height_covariance(
            jacobian,
            pixel_sigma=pixel_sigma,
            height_sigma=float(keypoint_height_sigma_m[joint_idx]),
        )

    pose = HumanPoseEstimate(
        bbox_xyxy=bbox_xyxy,
        score=float(score),
        valid=valid,
        mean=means,
        covariance=covariances,
        refined=means.copy(),
        center=center,
        center_covariance=center_covariance,
    )
    pose.yaw_rad, pose.yaw_valid = estimate_orientation_from_coco_joints(pose.refined, pose.valid)
    return pose


def refine_pose(
    pose: HumanPoseEstimate,
    *,
    lidar: LidarJointMeasurements | None = None,
    bone_priors: Mapping[Tuple[int, int], Tuple[float, float]] | None = None,
    lambda_camera: float = 1.0,
    lambda_bone: float = 1.0,
    lambda_lidar: float = 1.0,
    max_nfev: int = 200,
) -> HumanPoseEstimate:
    """
    Refine a pose using the paper's normalized camera, bone, and LiDAR terms.

    The optimization variables are valid COCO joints only. Residuals are
    covariance-whitened before being passed to SciPy, so the least-squares
    objective equals the intended negative log-likelihood up to constants.
    """
    if least_squares is None:
        raise RuntimeError("scipy is required for pose refinement. Install scipy or skip refinement.")

    valid_indices = np.flatnonzero(pose.valid)
    if valid_indices.size == 0:
        return pose

    local_index = {int(joint_idx): local for local, joint_idx in enumerate(valid_indices.tolist())}
    observations = pose.mean[valid_indices]
    x0 = observations.reshape(-1)

    camera_sqrt_info = [_sqrt_information(pose.covariance[joint_idx]) for joint_idx in valid_indices]
    camera_scale = math.sqrt(max(float(lambda_camera), 0.0) / max(len(valid_indices), 1))

    priors = dict(bone_priors or default_bone_priors())
    active_bones = [
        (a, b, float(length), float(sigma))
        for (a, b), (length, sigma) in priors.items()
        if a in local_index and b in local_index and sigma > 0.0
    ]
    bone_scale = math.sqrt(max(float(lambda_bone), 0.0) / max(len(active_bones), 1))

    active_lidar: List[Tuple[int, np.ndarray, np.ndarray]] = []
    if lidar is not None:
        for joint_idx in valid_indices.tolist():
            if bool(lidar.valid[joint_idx]):
                active_lidar.append(
                    (
                        int(joint_idx),
                        np.asarray(lidar.mean[joint_idx], dtype=float),
                        _sqrt_information(lidar.covariance[joint_idx]),
                    )
                )
    lidar_scale = math.sqrt(max(float(lambda_lidar), 0.0) / max(len(active_lidar), 1))

    def residuals(flat_joints: np.ndarray) -> np.ndarray:
        joints = flat_joints.reshape((-1, 3))
        residual_parts: List[np.ndarray] = []

        if camera_scale > 0.0:
            for local, observed in enumerate(observations):
                residual_parts.append(camera_scale * (camera_sqrt_info[local] @ (joints[local] - observed)))

        if bone_scale > 0.0:
            for a, b, length, sigma in active_bones:
                delta = joints[local_index[a]] - joints[local_index[b]]
                residual_parts.append(np.array([bone_scale * ((np.linalg.norm(delta) - length) / sigma)]))

        if lidar_scale > 0.0:
            for joint_idx, lidar_mean, lidar_sqrt_info in active_lidar:
                residual_parts.append(
                    lidar_scale * (lidar_sqrt_info @ (joints[local_index[joint_idx]] - lidar_mean))
                )

        if not residual_parts:
            return np.zeros(0, dtype=float)
        return np.concatenate(residual_parts).astype(float)

    result = least_squares(residuals, x0, method="trf", max_nfev=int(max_nfev))
    refined_valid = result.x.reshape((-1, 3))

    pose.refined = pose.mean.copy()
    for local, joint_idx in enumerate(valid_indices):
        pose.refined[int(joint_idx)] = refined_valid[local]

    pose.center, pose.center_covariance = estimate_body_center(pose)
    pose.yaw_rad, pose.yaw_valid = estimate_orientation_from_coco_joints(pose.refined, pose.valid)
    return pose


def estimate_body_center(pose: HumanPoseEstimate) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a stable body center from refined hips/shoulders, with bbox fallback."""
    torso_indices = [idx for idx in (5, 6, 11, 12) if bool(pose.valid[idx])]
    if torso_indices:
        center = np.mean(pose.refined[torso_indices], axis=0)
        center[2] = 0.10
        covariance = np.mean(pose.covariance[torso_indices], axis=0)
        return center, covariance
    return pose.center, pose.center_covariance


def poses_to_csv_rows(
    poses: Sequence[HumanPoseEstimate],
    *,
    method: str,
    group: str,
    node_id: int,
    side: str,
) -> List[Dict[str, object]]:
    """Create evaluator-compatible rows for pose detections."""
    rows: List[Dict[str, object]] = []
    for det_idx, pose in enumerate(poses):
        rows.append(
            {
                "method": method,
                "group": group,
                "node_id": int(node_id),
                "side": side,
                "det_idx": det_idx,
                "det_score": float(pose.score),
                "center_x": float(pose.center[0]),
                "center_y": float(pose.center[1]),
                "center_z": float(pose.center[2]),
                "yaw_rad": float(pose.yaw_rad),
                "yaw_valid": int(bool(pose.yaw_valid)),
                "valid_keypoints": int(np.count_nonzero(pose.valid)),
            }
        )
    return rows

