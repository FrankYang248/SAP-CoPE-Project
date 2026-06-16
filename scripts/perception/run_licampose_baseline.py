#!/usr/bin/env python3
"""Run the LiCamPose learning-based baseline adapter on processed SAP-CoPE data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .baseline_utils import (
    LICAMPOSE_EDGES,
    bbox_from_keypoints,
    bbox_world_center_from_image,
    build_camera_world_transforms,
    center_yaw_licampose15,
    draw_skeleton,
    load_group_lidar_world,
    make_result_row,
    parse_int_list,
    parse_str_list,
    project_points,
    resolve_groups,
    rotate_angle_for_side,
    rotate_image_for_inference,
    rotate_keypoints_back,
)
from .geometry import build_projection_matrix


def namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def make_licampose_cfg(
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    grid_size: Tuple[float, float, float],
    cube_size: Tuple[int, int, int],
    beta: float,
    sigma: float,
    mid_fdim: int,
):
    return namespace(
        NUM_JOINTS=15,
        THU=True,
        NETWORK=namespace(
            IMAGE_SIZE=np.array(image_size, dtype=np.int32),
            HEATMAP_SIZE=np.array(heatmap_size, dtype=np.int32),
            SIGMA=float(sigma),
            MID_FDIM=int(mid_fdim),
        ),
        PICT_STRUCT=namespace(
            GRID_SIZE=np.array(grid_size, dtype=np.float32),
            CUBE_SIZE=np.array(cube_size, dtype=np.int32),
        ),
        TRAIN=namespace(BETA=float(beta)),
    )


def parse_mmpose_2d_instances(result2d: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions = result2d.get("predictions", [])
    if not predictions:
        return []
    per_image = predictions[0]
    if isinstance(per_image, dict):
        per_image = per_image.get("instances", []) or per_image.get("pred_instances", []) or []

    output = []
    for instance in per_image:
        keypoints = instance.get("keypoints")
        if keypoints is None:
            continue
        keypoints = np.asarray(keypoints, dtype=float)
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if keypoints.shape != (17, 2):
            continue
        scores = instance.get("keypoint_scores", instance.get("keypoints_score"))
        scores = np.ones(17, dtype=float) if scores is None else np.asarray(scores, dtype=float).reshape(-1)
        if scores.shape[0] != 17:
            scores = np.ones(17, dtype=float)
        bbox = instance.get("bbox")
        bbox_xyxy = None
        if bbox is not None:
            bbox_array = np.asarray(bbox, dtype=float).reshape(-1)
            if bbox_array.shape[0] >= 4:
                bbox_xyxy = bbox_array[:4]
        score = float(instance.get("bbox_score", instance.get("score", np.nanmean(scores))))
        output.append(
            {
                "keypoints": keypoints,
                "keypoint_scores": scores,
                "bbox_xyxy": bbox_xyxy,
                "score": score,
            }
        )
    return output


def coco17_to_licampose15(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    output = np.zeros((15, 3), dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)

    def put(dst: int, src: int) -> None:
        output[dst, :2] = keypoints[src]
        output[dst, 2] = scores[src]

    output[0, :2] = 0.5 * (keypoints[5] + keypoints[6])
    output[0, 2] = min(scores[5], scores[6])
    put(1, 0)
    output[2, :2] = 0.5 * (keypoints[11] + keypoints[12])
    output[2, 2] = min(scores[11], scores[12])
    put(3, 5)
    put(4, 7)
    put(5, 9)
    put(6, 11)
    put(7, 13)
    put(8, 15)
    put(9, 6)
    put(10, 8)
    put(11, 10)
    put(12, 12)
    put(13, 14)
    put(14, 16)
    return output


def compute_human_scale(pose_heatmap: np.ndarray, visibility: np.ndarray) -> float:
    valid = visibility[:, 0] >= 0.2
    if not np.any(valid):
        return 0.0
    min_x, max_x = np.min(pose_heatmap[valid, 0]), np.max(pose_heatmap[valid, 0])
    min_y, max_y = np.min(pose_heatmap[valid, 1]), np.max(pose_heatmap[valid, 1])
    return float(np.clip(max(max_y - min_y, max_x - min_x) ** 2, 0.25 * 96**2, 4 * 96**2))


def generate_input_heatmaps(poses15: List[np.ndarray], image_size: np.ndarray, heatmap_size: np.ndarray, sigma: float):
    import torch

    feat_stride = image_size / heatmap_size
    heatmaps = []
    for pose in poses15:
        target = np.zeros((15, int(heatmap_size[1]), int(heatmap_size[0])), dtype=np.float32)
        visibility = pose[:, 2:3]
        human_scale = 2.0 * compute_human_scale(pose[:, :2] / feat_stride, visibility)
        if human_scale <= 0:
            heatmaps.append(torch.from_numpy(target))
            continue
        current_sigma = float(sigma) * math.sqrt(human_scale / (96.0 * 96.0))
        tmp_size = int(current_sigma * 3.0)
        for joint_idx in range(15):
            mu_x = int(pose[joint_idx, 0] / feat_stride[0])
            mu_y = int(pose[joint_idx, 1] / feat_stride[1])
            ul = [mu_x - tmp_size, mu_y - tmp_size]
            br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
            if pose[joint_idx, 2] <= 0.2 or ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
                continue
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            center = size // 2
            gaussian = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * current_sigma**2)) * float(pose[joint_idx, 2])
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
            target[joint_idx][img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
                target[joint_idx][img_y[0] : img_y[1], img_x[0] : img_x[1]],
                gaussian[g_y[0] : g_y[1], g_x[0] : g_x[1]],
            )
        heatmaps.append(torch.from_numpy(np.clip(target, 0, 1)))
    return heatmaps


def generate_3d_input(points_world: np.ndarray, space_center: np.ndarray, grid_size: np.ndarray, cube_size: np.ndarray) -> np.ndarray:
    if points_world.shape[0] == 0:
        return np.zeros(tuple(cube_size.tolist()), dtype=np.float32)
    grids = [
        np.linspace(-grid_size[i] / 2.0, grid_size[i] / 2.0, int(cube_size[i])) + float(space_center[i])
        for i in range(3)
    ]
    ix = np.clip(np.searchsorted(grids[0], points_world[:, 0]), 0, int(cube_size[0]) - 1)
    iy = np.clip(np.searchsorted(grids[1], points_world[:, 1]), 0, int(cube_size[1]) - 1)
    iz = np.clip(np.searchsorted(grids[2], points_world[:, 2]), 0, int(cube_size[2]) - 1)
    unique_indices = np.unique(np.stack([ix, iy, iz], axis=1), axis=0)
    voxels = np.zeros(tuple(cube_size.tolist()), dtype=np.float32)
    voxels[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = 1.0
    return voxels


def build_person_clusters(
    detections_by_view: List[List[Dict[str, Any]]],
    projections_by_view: List[np.ndarray],
    min_det_score: float,
    cluster_dist_m: float,
) -> List[Dict[str, Any]]:
    candidates = []
    for view_idx, detections in enumerate(detections_by_view):
        for det_idx, detection in enumerate(detections):
            score = float(detection.get("score", 0.0))
            if score < min_det_score:
                continue
            bbox = detection.get("bbox_xyxy")
            if bbox is None:
                bbox = bbox_from_keypoints(detection["keypoints"], detection["keypoint_scores"])
            if bbox is None:
                continue
            center = bbox_world_center_from_image(bbox, projections_by_view[view_idx])
            if center is None or not np.isfinite(center).all():
                continue
            candidates.append({"view_idx": view_idx, "det_idx": det_idx, "det": detection, "center": center, "score": score})

    clusters: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        best_cluster = None
        best_distance = float("inf")
        for cluster in clusters:
            distance = float(np.linalg.norm(candidate["center"][:2] - cluster["center"][:2]))
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        if best_cluster is not None and best_distance <= cluster_dist_m:
            best_cluster["members"].append(candidate)
            best_cluster["weight_sum"] += candidate["score"]
            best_cluster["center_sum"] += candidate["score"] * candidate["center"]
            best_cluster["center"] = best_cluster["center_sum"] / max(best_cluster["weight_sum"], 1e-6)
            best_cluster["score"] = max(float(best_cluster["score"]), candidate["score"])
        else:
            clusters.append(
                {
                    "center": candidate["center"].copy(),
                    "center_sum": candidate["score"] * candidate["center"].copy(),
                    "weight_sum": max(candidate["score"], 1e-6),
                    "score": candidate["score"],
                    "members": [candidate],
                }
            )
    clusters.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1])))
    for idx, cluster in enumerate(clusters):
        cluster["person_idx"] = idx
    return clusters


def choose_cluster_detections(
    cluster_center: np.ndarray,
    detections_by_view: List[List[Dict[str, Any]]],
    projections_by_view: List[np.ndarray],
    min_det_score: float,
    match_dist_m: float,
):
    chosen, matched, distances = [], [], []
    for view_idx, detections in enumerate(detections_by_view):
        best = None
        best_distance = float("inf")
        for detection in detections:
            if float(detection.get("score", 0.0)) < min_det_score:
                continue
            bbox = detection.get("bbox_xyxy")
            if bbox is None:
                bbox = bbox_from_keypoints(detection["keypoints"], detection["keypoint_scores"])
            if bbox is None:
                continue
            center = bbox_world_center_from_image(bbox, projections_by_view[view_idx])
            if center is None:
                continue
            distance = float(np.linalg.norm(center[:2] - cluster_center[:2]))
            if distance < best_distance:
                best = detection
                best_distance = distance
        if best is not None and best_distance <= match_dist_m:
            chosen.append(best)
            matched.append(True)
            distances.append(best_distance)
        else:
            chosen.append(None)
            matched.append(False)
            distances.append(float("inf"))
    return chosen, matched, distances


def make_camera_dict(intrinsic: np.ndarray, camera_from_world: np.ndarray, batch: int, device):
    import torch

    def batched(value):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        return tensor.unsqueeze(0).repeat(batch, *([1] * tensor.ndim))

    return {
        "R": batched(camera_from_world[:3, :3]),
        "T": batched(camera_from_world[:3, 3].reshape(3, 1)),
        "fx": torch.full((batch,), float(intrinsic[0, 0]), dtype=torch.float32, device=device),
        "fy": torch.full((batch,), float(intrinsic[1, 1]), dtype=torch.float32, device=device),
        "cx": torch.full((batch,), float(intrinsic[0, 2]), dtype=torch.float32, device=device),
        "cy": torch.full((batch,), float(intrinsic[1, 2]), dtype=torch.float32, device=device),
        "k": torch.zeros((batch, 3, 1), dtype=torch.float32, device=device),
        "p": torch.zeros((batch, 2, 1), dtype=torch.float32, device=device),
    }


def load_licampose_model(repo_dir: Path, cfg, checkpoint: Path, device):
    model_file = repo_dir / "model" / "voxel_fusion_net.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Cannot find LiCamPose model code at {model_file}")
    sys.path.insert(0, str(repo_dir))
    from model.voxel_fusion_net import VoxelFusionNet

    import torch

    model = VoxelFusionNet(cfg).to(device)
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
    state = checkpoint_data.get("state_dict", checkpoint_data)
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)}")
    model.eval()
    return model


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--slice", required=True, dest="slice_name")
    parser.add_argument("--groups", default="ALL")
    parser.add_argument("--views", default="1_left,1_right,3_left,3_right")
    parser.add_argument("--primary-view", default="3_right")
    parser.add_argument("--lidar-nodes", default="1,3")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--licampose-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", default="", type=Path, help="Empty string writes adapter tensors without inference.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-groups", type=int, default=-1)
    parser.add_argument("--min-det-score", type=float, default=0.25)
    parser.add_argument("--person-cluster-dist-m", type=float, default=0.5)
    parser.add_argument("--view-match-dist-m", type=float, default=0.7)
    parser.add_argument("--crop-radius-m", type=float, default=0.8)
    parser.add_argument("--z-min", type=float, default=-0.2)
    parser.add_argument("--z-max", type=float, default=2.5)
    parser.add_argument("--grid-size", default="2.0,2.0,2.0")
    parser.add_argument("--cube-size", default="64,64,64")
    parser.add_argument("--heatmap-size", default="512,384")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--mid-fdim", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import torch
        from mmpose.apis import MMPoseInferencer
    except Exception as exc:
        raise RuntimeError("LiCamPose baseline requires opencv-python, torch, and mmpose.") from exc

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    slice_dir = args.data_root / args.scenario / args.slice_name
    groups = resolve_groups(slice_dir, args.groups)
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    views = parse_str_list(args.views)
    view_pairs = [(int(view.split("_")[0]), view.split("_", 1)[1]) for view in views]
    if args.primary_view not in views:
        raise ValueError(f"--primary-view must be one of {views}")
    primary_idx = views.index(args.primary_view)
    lidar_nodes = parse_int_list(args.lidar_nodes)

    sample_image = next((slice_dir / "PcImage" / groups[0] / "camera").glob(f"{args.primary_view}_*.jpg"))
    sample = cv2.imread(str(sample_image))
    if sample is None:
        raise RuntimeError(f"Failed to read sample image: {sample_image}")
    image_size = (int(sample.shape[1]), int(sample.shape[0]))
    cfg = make_licampose_cfg(
        image_size=image_size,
        heatmap_size=tuple(int(item) for item in args.heatmap_size.split(",")),
        grid_size=tuple(float(item) for item in args.grid_size.split(",")),
        cube_size=tuple(int(item) for item in args.cube_size.split(",")),
        beta=args.beta,
        sigma=args.sigma,
        mid_fdim=args.mid_fdim,
    )

    model = None
    if str(args.checkpoint):
        model = load_licampose_model(args.licampose_repo, cfg, args.checkpoint, device)
    else:
        print("[WARN] No --checkpoint supplied. Writing adapter tensors and debug views only.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.out_dir / "debug_vis"
    tensor_dir = args.out_dir / "per_person_npz"
    debug_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "results.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    infer2d = MMPoseInferencer(pose2d="human", device=str(device))
    rows: List[Dict[str, Any]] = []

    for group_idx, group in enumerate(groups, start=1):
        print(f"[INFO] group {group_idx}/{len(groups)}: {group}")
        camera_infos = []
        detections_by_view: List[List[Dict[str, Any]]] = []
        projections_by_view = []

        for node_id, side in view_pairs:
            image_paths = sorted((slice_dir / "PcImage" / group / "camera").glob(f"{node_id}_{side}_*.jpg"))
            if not image_paths:
                raise FileNotFoundError(f"No image for view {node_id}_{side} in group {group}")
            image_path = image_paths[0]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read {image_path}")
            angle = rotate_angle_for_side(side)
            image_rot = rotate_image_for_inference(image, angle)
            result_gen = infer2d(image_rot, show=False, out_dir=None, return_vis=False)
            try:
                result2d = next(result_gen)
            except StopIteration:
                result2d = {"predictions": []}
            detections_rot = parse_mmpose_2d_instances(result2d)
            detections = []
            for det in detections_rot:
                keypoints = rotate_keypoints_back(det["keypoints"], image_rot.shape, angle)
                bbox = bbox_from_keypoints(keypoints, det["keypoint_scores"])
                detections.append(
                    {
                        "keypoints": keypoints,
                        "keypoint_scores": det["keypoint_scores"],
                        "bbox_xyxy": bbox,
                        "score": det["score"],
                    }
                )
            detections_by_view.append(detections)

            camera_yaml = slice_dir / "Calibration" / "camera" / f"{node_id}_{side}.yaml"
            lidar_yaml = slice_dir / "Calibration" / "lidar" / f"{node_id}.yaml"
            intrinsic, camera_from_world, _ = build_camera_world_transforms(camera_yaml, lidar_yaml)
            projection = build_projection_matrix(camera_yaml, lidar_yaml)
            camera_infos.append(
                {
                    "node_id": node_id,
                    "side": side,
                    "image_path": image_path,
                    "image": image,
                    "intrinsic": intrinsic,
                    "camera_from_world": camera_from_world,
                }
            )
            projections_by_view.append(projection)

        clusters = build_person_clusters(
            detections_by_view,
            projections_by_view,
            min_det_score=args.min_det_score,
            cluster_dist_m=args.person_cluster_dist_m,
        )
        all_lidar_world = load_group_lidar_world(slice_dir, group, lidar_nodes)
        camera_dicts = [
            make_camera_dict(info["intrinsic"], info["camera_from_world"], batch=1, device=device)
            for info in camera_infos
        ]
        record = {"scenario": args.scenario, "slice": args.slice_name, "group": group, "method": "licampose", "persons": []}

        for cluster in clusters:
            person_idx = int(cluster["person_idx"])
            cluster_center = np.asarray(cluster["center"], dtype=float)
            chosen_by_view, matched, match_distances = choose_cluster_detections(
                cluster_center,
                detections_by_view,
                projections_by_view,
                min_det_score=args.min_det_score,
                match_dist_m=args.view_match_dist_m,
            )
            if not any(matched):
                continue

            poses15 = [
                np.zeros((15, 3), dtype=np.float32)
                if detection is None
                else coco17_to_licampose15(detection["keypoints"], detection["keypoint_scores"])
                for detection in chosen_by_view
            ]

            if all_lidar_world.shape[0]:
                dxy = all_lidar_world[:, :2] - cluster_center[:2].reshape(1, 2)
                crop_mask = np.linalg.norm(dxy, axis=1) <= args.crop_radius_m
                crop_mask &= (all_lidar_world[:, 2] >= args.z_min) & (all_lidar_world[:, 2] <= args.z_max)
                points_crop = all_lidar_world[crop_mask]
            else:
                points_crop = np.zeros((0, 3), dtype=float)

            if points_crop.shape[0] > 0:
                grid_center = 0.5 * (np.max(points_crop, axis=0) + np.min(points_crop, axis=0))
            else:
                grid_center = np.array([cluster_center[0], cluster_center[1], 0.8], dtype=float)

            voxels = generate_3d_input(points_crop, grid_center, cfg.PICT_STRUCT.GRID_SIZE, cfg.PICT_STRUCT.CUBE_SIZE)
            heatmaps = generate_input_heatmaps(poses15, cfg.NETWORK.IMAGE_SIZE, cfg.NETWORK.HEATMAP_SIZE, cfg.NETWORK.SIGMA)
            input3d = torch.from_numpy(voxels).unsqueeze(0).unsqueeze(0).to(device)
            heatmaps_b = [heatmap.unsqueeze(0).to(device) for heatmap in heatmaps]
            grid_centers = torch.as_tensor(grid_center.reshape(1, 3), dtype=torch.float32, device=device)

            pred_joints = None
            entropy_mean = None
            if model is not None:
                with torch.no_grad():
                    pred, _ = model(input3d.float(), heatmaps_b, camera_dicts, grid_centers)
                entropy_mean = float(torch.mean(pred[..., 3]).detach().cpu().item())
                pred_joints = pred[0, :, :3].detach().cpu().numpy()
                center, yaw, yaw_valid = center_yaw_licampose15(pred_joints)
                det_score = max(float(det["score"]) for det in chosen_by_view if det is not None)
                row = make_result_row(
                    scenario=args.scenario,
                    slice_name=args.slice_name,
                    group=group,
                    person_idx=person_idx,
                    method="licampose",
                    det_score=det_score,
                    center=center,
                    yaw=yaw,
                    yaw_valid=yaw_valid,
                    extra={
                        "entropy_mean": entropy_mean,
                        "num_lidar_points": int(points_crop.shape[0]),
                        "matched_views": int(sum(bool(value) for value in matched)),
                    },
                )
                rows.append(row)
                record["persons"].append(row)

            np.savez(
                tensor_dir / f"{group}_person{person_idx:03d}.npz",
                input3d=voxels,
                grid_center=grid_center,
                points_crop=points_crop,
                pred_joints=np.asarray([]) if pred_joints is None else pred_joints,
                cluster_center=cluster_center,
                matched=np.asarray(matched, dtype=np.bool_),
                match_dists=np.asarray(match_distances, dtype=np.float32),
                **{f"pose15_view{view_idx}": poses15[view_idx] for view_idx in range(len(poses15))},
            )

            debug_idx = primary_idx if matched[primary_idx] else next(idx for idx, value in enumerate(matched) if value)
            debug_info = camera_infos[debug_idx]
            debug = debug_info["image"].copy()
            draw_skeleton(debug, poses15[debug_idx], LICAMPOSE_EDGES, (0, 255, 0))
            if points_crop.shape[0] > 0:
                uv_lidar = project_points(debug_info["intrinsic"], debug_info["camera_from_world"], points_crop)
                height, width = debug.shape[:2]
                valid_uv = (uv_lidar[:, 0] >= 0) & (uv_lidar[:, 0] < width) & (uv_lidar[:, 1] >= 0) & (uv_lidar[:, 1] < height)
                stride = max(1, int(np.sum(valid_uv) / 400))
                for x, y in uv_lidar[valid_uv][::stride]:
                    cv2.circle(debug, (int(round(x)), int(round(y))), 1, (255, 255, 0), -1)
            if pred_joints is not None:
                uv_pose = project_points(debug_info["intrinsic"], debug_info["camera_from_world"], pred_joints)
                draw_skeleton(debug, uv_pose, LICAMPOSE_EDGES, (0, 0, 255))
            cv2.imwrite(str(debug_dir / f"{group}_person{person_idx:03d}_{views[debug_idx]}.jpg"), debug)

        with open(jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    fields = [
        "scenario",
        "slice",
        "group",
        "person_idx",
        "method",
        "det_score",
        "center_x",
        "center_y",
        "center_z",
        "yaw_rad",
        "yaw_deg",
        "yaw_valid",
        "entropy_mean",
        "num_lidar_points",
        "matched_views",
    ]
    write_csv(args.out_dir / "results.csv", rows, fields)
    print(f"[INFO] Wrote CSV : {args.out_dir / 'results.csv'}")
    print(f"[INFO] Wrote JSONL: {jsonl_path}")
    print(f"[INFO] Wrote tensors: {tensor_dir}")
    print(f"[INFO] Wrote debug visuals: {debug_dir}")


if __name__ == "__main__":
    main()
