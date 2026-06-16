#!/usr/bin/env python3
"""Run SAP-CoPE camera-only or camera+LiDAR pose estimation on processed data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from .dataset import iter_camera_frames, list_groups, parse_csv_list, parse_node_side_pairs
from .geometry import build_projection_matrix, load_lidar_calibration, project_world_to_image
from .image_ops import (
    rotate_angle_for_side,
    rotate_boxes_back_xyxy,
    rotate_image_for_inference,
    rotate_keypoints_back,
)
from .lidar_association import (
    associate_lidar_to_keypoints,
    load_world_lidar_points,
    project_lidar_points,
)
from .pose_refinement import (
    HumanPoseEstimate,
    build_pose_from_coco_detection,
    poses_to_csv_rows,
    refine_pose,
)


METHOD_OUTPUT_NAMES = {
    "camera": "camera_only_refine",
    "camera_lidar": "camera_lidar_refine",
}


def _load_cv2_and_yolo():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for image loading and debug visualization") from exc
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required to run YOLO pose inference") from exc
    return cv2, YOLO


def draw_debug_overlay(image_bgr: np.ndarray, projection: np.ndarray, poses: List[HumanPoseEstimate]) -> np.ndarray:
    """Draw detections, centers, and yaw arrows on the original image."""
    cv2, _ = _load_cv2_and_yolo()
    output = image_bgr.copy()
    for pose in poses:
        x1, y1, x2, y2 = pose.bbox_xyxy.astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 180, 0), 2)
        for joint_idx in np.flatnonzero(pose.valid):
            uv = project_world_to_image(projection, pose.refined[int(joint_idx)])
            if uv is not None:
                cv2.circle(output, (int(round(uv[0])), int(round(uv[1]))), 3, (0, 0, 255), -1)
        center_uv = project_world_to_image(projection, pose.center)
        if center_uv is None:
            continue
        center_px = (int(round(center_uv[0])), int(round(center_uv[1])))
        cv2.circle(output, center_px, 5, (0, 255, 255), -1)
        if pose.yaw_valid:
            tip_world = pose.center + np.array(
                [0.8 * math.cos(pose.yaw_rad), 0.8 * math.sin(pose.yaw_rad), 0.0],
                dtype=float,
            )
            tip_uv = project_world_to_image(projection, tip_world)
            if tip_uv is not None:
                cv2.arrowedLine(
                    output,
                    center_px,
                    (int(round(tip_uv[0])), int(round(tip_uv[1]))),
                    (255, 0, 0),
                    2,
                    tipLength=0.25,
                )
    return output


def run_yolo_on_image(model, image_bgr: np.ndarray, side: str, conf: float):
    """Run YOLO on the rotated image and return detections in the original frame."""
    cv2, _ = _load_cv2_and_yolo()
    angle = rotate_angle_for_side(side)
    image_yolo = rotate_image_for_inference(image_bgr, angle)
    image_rgb = cv2.cvtColor(image_yolo, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, verbose=False, conf=float(conf))
    if not results:
        return [], [], [], []
    result = results[0]
    if result.boxes is None or result.keypoints is None or result.keypoints.xy is None:
        return [], [], [], []

    boxes_rot = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    keypoints_rot = result.keypoints.xy.cpu().numpy()
    keypoint_conf = result.keypoints.conf.cpu().numpy()
    boxes = rotate_boxes_back_xyxy(boxes_rot, image_yolo.shape, angle)
    keypoints = rotate_keypoints_back(keypoints_rot, image_yolo.shape, angle)
    return boxes, classes, scores, keypoints, keypoint_conf


def process_frame(
    *,
    image_path: Path,
    camera_yaml: Path,
    lidar_yaml: Path,
    lidar_npy: Path | None,
    node_id: int,
    side: str,
    method: str,
    model,
    yolo_conf: float,
    keypoint_conf: float,
    lidar_pixel_threshold: float,
    lidar_world_threshold: float,
    lidar_sigma: float,
    lambda_camera: float,
    lambda_bone: float,
    lambda_lidar: float,
    debug_dir: Path | None,
) -> List[HumanPoseEstimate]:
    cv2, _ = _load_cv2_and_yolo()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    projection = build_projection_matrix(camera_yaml, lidar_yaml)
    boxes, classes, scores, keypoints, keypoint_scores = run_yolo_on_image(
        model, image_bgr, side, yolo_conf
    )

    lidar_uv = np.zeros((0, 2), dtype=float)
    lidar_world = np.zeros((0, 3), dtype=float)
    if method == "camera_lidar" and lidar_npy is not None:
        _, _, world_from_lidar = load_lidar_calibration(lidar_yaml)
        points_world = load_world_lidar_points(
            lidar_npy,
            world_from_lidar,
            slice_dir=lidar_yaml.parents[2],
            node_id=node_id,
        )
        lidar_uv, lidar_world = project_lidar_points(projection, points_world, image_bgr.shape)

    poses: List[HumanPoseEstimate] = []
    for det_idx, cls_id in enumerate(classes):
        if int(cls_id) != 0:
            continue
        pose = build_pose_from_coco_detection(
            boxes[det_idx],
            scores[det_idx],
            keypoints[det_idx],
            keypoint_scores[det_idx],
            projection,
            keypoint_conf_threshold=keypoint_conf,
        )

        lidar = None
        if method == "camera_lidar" and lidar_uv.size > 0:
            lidar = associate_lidar_to_keypoints(
                keypoints_xy=keypoints[det_idx],
                keypoint_valid=pose.valid,
                lidar_uv=lidar_uv,
                lidar_xyz_world=lidar_world,
                initial_joints_world=pose.mean,
                pixel_threshold=lidar_pixel_threshold,
                world_threshold=lidar_world_threshold,
                lidar_sigma_m=lidar_sigma,
            )

        pose = refine_pose(
            pose,
            lidar=lidar,
            lambda_camera=lambda_camera,
            lambda_bone=lambda_bone,
            lambda_lidar=lambda_lidar,
        )
        poses.append(pose)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_image = draw_debug_overlay(image_bgr, projection, poses)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_{method}_debug.jpg"), debug_image)

    return poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Processed dataset root.")
    parser.add_argument("--scenario", required=True, help="Scenario folder, e.g. WithBed.")
    parser.add_argument("--slice", required=True, dest="slice_name", help="Slice folder, e.g. slice0.")
    parser.add_argument("--groups", default=None, help="Comma-separated PcImage groups. Default: all groups.")
    parser.add_argument("--nodes", default="1,3", help="Comma-separated node IDs.")
    parser.add_argument("--sides", default="left,right", help="Comma-separated sides matching --nodes.")
    parser.add_argument("--method", choices=["camera", "camera_lidar"], default="camera_lidar")
    parser.add_argument("--pose-model", default="yolo11s-pose.pt", help="Ultralytics pose model path.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--yolo-conf", type=float, default=0.30)
    parser.add_argument("--keypoint-conf", type=float, default=0.50)
    parser.add_argument("--lidar-pixel-threshold", type=float, default=10.0)
    parser.add_argument("--lidar-world-threshold", type=float, default=0.7)
    parser.add_argument("--lidar-sigma", type=float, default=0.03)
    parser.add_argument("--lambda-camera", type=float, default=1.0)
    parser.add_argument("--lambda-bone", type=float, default=1.0)
    parser.add_argument("--lambda-lidar", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true", help="Write debug overlay images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2, YOLO = _load_cv2_and_yolo()
    del cv2

    slice_dir = args.data_root / args.scenario / args.slice_name
    pcimage_root = slice_dir / "PcImage"
    groups = list_groups(pcimage_root, parse_csv_list(args.groups))
    node_side_pairs = parse_node_side_pairs(args.nodes, args.sides)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.out_dir / "debug_vis" if args.debug else None
    csv_path = args.out_dir / "results.csv"
    jsonl_path = args.out_dir / "results.jsonl"

    model = YOLO(args.pose_model)
    rows: List[Dict[str, object]] = []

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for group, node_id, side, image_path, camera_yaml, lidar_yaml, lidar_npy in iter_camera_frames(
            args.data_root,
            args.scenario,
            args.slice_name,
            groups,
            node_side_pairs,
        ):
            if args.method == "camera_lidar" and lidar_npy is None:
                print(f"[WARN] Running camera-only for missing LiDAR: group={group}, node={node_id}")
            try:
                poses = process_frame(
                    image_path=image_path,
                    camera_yaml=camera_yaml,
                    lidar_yaml=lidar_yaml,
                    lidar_npy=lidar_npy,
                    node_id=node_id,
                    side=side,
                    method=args.method,
                    model=model,
                    yolo_conf=args.yolo_conf,
                    keypoint_conf=args.keypoint_conf,
                    lidar_pixel_threshold=args.lidar_pixel_threshold,
                    lidar_world_threshold=args.lidar_world_threshold,
                    lidar_sigma=args.lidar_sigma,
                    lambda_camera=args.lambda_camera,
                    lambda_bone=args.lambda_bone,
                    lambda_lidar=args.lambda_lidar,
                    debug_dir=debug_dir,
                )
            except Exception as exc:
                print(f"[WARN] Failed group={group}, node={node_id}, side={side}: {exc}")
                continue

            frame_rows = poses_to_csv_rows(
                poses,
                method=METHOD_OUTPUT_NAMES[args.method],
                group=group,
                node_id=node_id,
                side=side,
            )
            rows.extend(frame_rows)
            jsonl_file.write(
                json.dumps(
                    {
                        "group": group,
                        "node_id": node_id,
                        "side": side,
                        "image": str(image_path),
                        "method": METHOD_OUTPUT_NAMES[args.method],
                        "detections": frame_rows,
                    }
                )
                + "\n"
            )

    fieldnames = [
        "method",
        "group",
        "node_id",
        "side",
        "det_idx",
        "det_score",
        "center_x",
        "center_y",
        "center_z",
        "yaw_rad",
        "yaw_valid",
        "valid_keypoints",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} detections to {csv_path}")
    print(f"Wrote per-frame JSONL to {jsonl_path}")


if __name__ == "__main__":
    main()
