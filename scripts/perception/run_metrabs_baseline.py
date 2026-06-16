#!/usr/bin/env python3
"""Run the MeTRAbs learning-based baseline on processed SAP-CoPE data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .baseline_utils import (
    COCO_EDGES,
    H36M_17_EDGES,
    ResultWriter,
    build_camera_world_transforms,
    center_yaw_coco17,
    center_yaw_h36m17,
    draw_skeleton,
    embed_rotation,
    iter_images,
    make_result_row,
    parse_node_side_pairs,
    project_points,
    resolve_groups,
    rotate_angle_for_side,
    rotate_image_for_inference,
    rotated_camera_intrinsics_and_frame,
)
from .geometry import invert_se3


def decode_metrabs_box(box: np.ndarray):
    left, top, width, height, confidence = [float(value) for value in box.tolist()]
    return left, top, left + width, top + height, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--slice", required=True, dest="slice_name")
    parser.add_argument("--groups", default="ALL")
    parser.add_argument("--nodes", default="1,1,3,3")
    parser.add_argument("--sides", default="left,right,left,right")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tfhub-url", default="https://bit.ly/metrabs_l")
    parser.add_argument("--skeleton", choices=["h36m_17", "coco_17"], default="h36m_17")
    parser.add_argument("--method-name", default="metrabs")
    parser.add_argument("--det-thr", type=float, default=0.3)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--num-aug", type=int, default=1)
    parser.add_argument("--no-avg-aug", action="store_true")
    parser.add_argument("--extrinsic-mode", choices=["cam2world", "none"], default="cam2world")
    parser.add_argument("--save-debug-vis", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import tensorflow as tf
        import tensorflow_hub as tfhub
    except Exception as exc:
        raise RuntimeError(
            "MeTRAbs baseline requires opencv-python, tensorflow, and tensorflow_hub."
        ) from exc

    slice_dir = args.data_root / args.scenario / args.slice_name
    groups = resolve_groups(slice_dir, args.groups)
    node_side_pairs = parse_node_side_pairs(args.nodes, args.sides)
    edges = H36M_17_EDGES if args.skeleton == "h36m_17" else COCO_EDGES
    yaw_fn = center_yaw_h36m17 if args.skeleton == "h36m_17" else center_yaw_coco17

    print(f"[INFO] Loading MeTRAbs model: {args.tfhub_url}")
    model = tfhub.load(args.tfhub_url)
    writer = ResultWriter(args.out_dir)
    calib_cache: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}

    processed = 0
    for group, node_id, side, image_path in iter_images(
        args.data_root, args.scenario, args.slice_name, groups, node_side_pairs
    ):
        if args.max_images > 0 and processed >= args.max_images:
            break
        processed += 1
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        height0, width0 = image_bgr.shape[:2]
        angle = rotate_angle_for_side(side)
        image_rot = rotate_image_for_inference(image_bgr, angle)
        cache_key = (node_id, side)
        if cache_key not in calib_cache:
            camera_yaml = slice_dir / "Calibration" / "camera" / f"{node_id}_{side}.yaml"
            lidar_yaml = slice_dir / "Calibration" / "lidar" / f"{node_id}.yaml"
            intrinsic, camera_from_world, world_from_camera = build_camera_world_transforms(camera_yaml, lidar_yaml)
            calib_cache[cache_key] = {
                "intrinsic": intrinsic,
                "camera_from_world": camera_from_world,
                "world_from_camera": world_from_camera,
            }

        intrinsic = calib_cache[cache_key]["intrinsic"]
        world_from_camera = calib_cache[cache_key]["world_from_camera"]
        k_rot, _, _, r_rot_to_phys = rotated_camera_intrinsics_and_frame(intrinsic, width0, height0, angle)
        world_from_camera_rot = world_from_camera @ embed_rotation(r_rot_to_phys)
        camera_rot_from_world = invert_se3(world_from_camera_rot)

        extrinsic = None
        if args.extrinsic_mode == "cam2world":
            extrinsic = world_from_camera_rot.copy()
            extrinsic[:3, 3] *= 1000.0

        image_rgb = cv2.cvtColor(image_rot, cv2.COLOR_BGR2RGB)
        kwargs = {
            "image": tf.convert_to_tensor(image_rgb, dtype=tf.uint8),
            "intrinsic_matrix": tf.constant(k_rot.astype(np.float32)),
            "distortion_coeffs": tf.constant(np.zeros(5, dtype=np.float32)),
            "world_up_vector": tf.constant(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            "skeleton": args.skeleton,
            "detector_threshold": float(args.det_thr),
            "num_aug": int(args.num_aug),
            "average_aug": not bool(args.no_avg_aug),
            "suppress_implausible_poses": True,
        }
        if extrinsic is not None:
            kwargs["extrinsic_matrix"] = tf.constant(extrinsic.astype(np.float32))

        try:
            pred = model.detect_poses(**kwargs)
        except Exception as exc:
            print(f"[WARN] MeTRAbs inference failed for {image_path}: {exc}")
            continue

        boxes = pred["boxes"].numpy()
        poses2d = pred["poses2d"].numpy()
        poses3d = pred["poses3d"].numpy()
        record: Dict[str, Any] = {
            "scenario": args.scenario,
            "slice": args.slice_name,
            "group": group,
            "node_id": node_id,
            "side": side,
            "image_path": str(image_path),
            "persons": [],
        }
        debug = image_rot.copy() if args.save_debug_vis else None

        for person_idx in range(int(boxes.shape[0])):
            x1, y1, x2, y2, confidence = decode_metrabs_box(boxes[person_idx])
            if confidence < args.min_conf:
                continue
            joints3d = np.asarray(poses3d[person_idx], dtype=float)
            if args.extrinsic_mode == "none":
                joints_cam_m = joints3d / 1000.0
                homo = np.hstack([joints_cam_m, np.ones((joints_cam_m.shape[0], 1), dtype=float)])
                joints_world = (world_from_camera_rot @ homo.T).T[:, :3]
            else:
                joints_world = joints3d / 1000.0
            center, yaw, yaw_valid = yaw_fn(joints_world)

            row = make_result_row(
                scenario=args.scenario,
                slice_name=args.slice_name,
                group=group,
                node_id=node_id,
                side=side,
                image_path=image_path,
                person_idx=person_idx,
                method=args.method_name,
                det_score=confidence,
                center=center,
                yaw=yaw,
                yaw_valid=yaw_valid,
            )
            writer.add_row(row)
            record["persons"].append(row)

            if debug is not None:
                cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                draw_skeleton(debug, poses2d[person_idx], edges, (0, 255, 0))
                draw_skeleton(debug, project_points(k_rot, camera_rot_from_world, joints_world), edges, (0, 0, 255))

        writer.write_jsonl(record)
        if debug is not None:
            out_path = writer.debug_dir / f"{image_path.stem}_group{group}_node{node_id}_{side}_metrabs.jpg"
            cv2.imwrite(str(out_path), debug)

    writer.finalize()


if __name__ == "__main__":
    main()
