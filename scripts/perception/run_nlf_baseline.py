#!/usr/bin/env python3
"""Run the NLF learning-based baseline on processed SAP-CoPE data."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .baseline_utils import (
    ResultWriter,
    SMPL_EDGES,
    build_camera_world_transforms,
    canonicalize_intrinsics,
    center_yaw_smpl,
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


def ensure_model(model_path: Path, model_url: str) -> None:
    if model_path.exists():
        return
    if not model_url:
        raise FileNotFoundError(f"NLF model not found: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading NLF model from {model_url} to {model_path}")
    urllib.request.urlretrieve(model_url, model_path)


def ensure_torchvision_ops_ready(device: str) -> None:
    import torch

    try:
        import torchvision
        from torchvision.ops import nms
    except Exception as exc:
        raise RuntimeError(
            "NLF requires torchvision because its TorchScript detector calls torchvision::nms."
        ) from exc

    try:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)
        scores = torch.tensor([1.0], device=device)
        _ = nms(boxes, scores, 0.5)
    except Exception as exc:
        raise RuntimeError(
            "torchvision.ops.nms is not usable. Install matching torch and torchvision builds."
        ) from exc
    print(f"[INFO] TorchVision ops ready: torchvision={torchvision.__version__}, torch={torch.__version__}")


def first_tensor(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            output = first_tensor(item)
            if output is not None:
                return output
        return None
    if hasattr(value, "detach") or hasattr(value, "cpu"):
        return value
    return None


def tensor_to_numpy(value):
    tensor = first_tensor(value)
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    array = tensor.numpy()
    if array.ndim == 4:
        array = array[0]
    return np.asarray(array)


def choose_prediction_array(pred: Dict[str, Any], keys: List[str]):
    for key in keys:
        if key in pred:
            array = tensor_to_numpy(pred[key])
            if array is not None and array.size > 0:
                return key, array
    return "", None


def run_model(model, image_chw, device: str, model_name: str, intrinsic: np.ndarray, distortion: np.ndarray):
    import torch

    frame_batch = image_chw.unsqueeze(0).to(device)
    kwargs = {
        "intrinsic_matrix": torch.tensor(intrinsic, dtype=torch.float32, device=device).unsqueeze(0),
    }
    if distortion.size > 0:
        kwargs["distortion_coeffs"] = torch.tensor(distortion, dtype=torch.float32, device=device).unsqueeze(0)
    if model_name:
        kwargs["model_name"] = model_name
    with torch.inference_mode():
        return model.detect_smpl_batched(frame_batch, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--slice", required=True, dest="slice_name")
    parser.add_argument("--groups", default="ALL")
    parser.add_argument("--nodes", default="1,1,3,3")
    parser.add_argument("--sides", default="left,right,left,right")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/nlf_l_multi.torchscript"))
    parser.add_argument("--model-url", default="https://bit.ly/nlf_l_pt")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model-name", default="", help="Optional NLF body model name, e.g. smplx.")
    parser.add_argument("--method-name", default="nlf")
    parser.add_argument("--unit-scale", type=float, default=0.001)
    parser.add_argument("--dist", default="0,0,0,0,0")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--save-debug-vis", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import torch
    except Exception as exc:
        raise RuntimeError("NLF baseline requires opencv-python and torch.") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"

    slice_dir = args.data_root / args.scenario / args.slice_name
    groups = resolve_groups(slice_dir, args.groups)
    node_side_pairs = parse_node_side_pairs(args.nodes, args.sides)
    distortion = np.array([float(item) for item in args.dist.split(",") if item.strip()], dtype=np.float32)

    ensure_model(args.model_path, args.model_url)
    ensure_torchvision_ops_ready(args.device)
    print(f"[INFO] Loading NLF TorchScript: {args.model_path}")
    model = torch.jit.load(str(args.model_path), map_location=args.device).eval()

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
            intrinsic, _, world_from_camera = build_camera_world_transforms(camera_yaml, lidar_yaml)
            calib_cache[cache_key] = {"intrinsic": intrinsic, "world_from_camera": world_from_camera}

        intrinsic = calib_cache[cache_key]["intrinsic"]
        world_from_camera = calib_cache[cache_key]["world_from_camera"]
        k_rot, _, _, r_rot_to_phys = rotated_camera_intrinsics_and_frame(intrinsic, width0, height0, angle)
        k_model, r_model_to_rot = canonicalize_intrinsics(k_rot)
        world_from_camera_rot = world_from_camera @ embed_rotation(r_rot_to_phys)
        camera_rot_from_world = invert_se3(world_from_camera_rot)

        image_rgb = cv2.cvtColor(image_rot, cv2.COLOR_BGR2RGB)
        image_chw = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).to(torch.uint8)
        try:
            pred = run_model(model, image_chw, args.device, args.model_name, k_model, distortion)
        except Exception as exc:
            print(f"[WARN] NLF inference failed for {image_path}: {exc}")
            continue

        k3_key, joints3d = choose_prediction_array(pred, ["joints3d", "joints3d_nonparam"])
        _, joints2d = choose_prediction_array(pred, ["joints2d", "joints2d_nonparam"])
        _, boxes = choose_prediction_array(pred, ["boxes"])
        if joints3d is None or joints3d.ndim != 3 or joints3d.shape[-1] != 3:
            print(f"[WARN] No valid 3D joints found for {image_path}")
            continue

        record: Dict[str, Any] = {
            "scenario": args.scenario,
            "slice": args.slice_name,
            "group": group,
            "node_id": node_id,
            "side": side,
            "image_path": str(image_path),
            "nlf_3d_key": k3_key,
            "persons": [],
        }
        debug = image_rot.copy() if args.save_debug_vis else None

        for person_idx in range(int(joints3d.shape[0])):
            confidence = 1.0
            if boxes is not None and boxes.ndim >= 2 and boxes.shape[0] > person_idx:
                confidence = float(boxes[person_idx, -1])
            if confidence < args.min_conf:
                continue

            joints_model_m = np.asarray(joints3d[person_idx], dtype=float) * float(args.unit_scale)
            joints_rot_m = (r_model_to_rot @ joints_model_m.T).T
            homo = np.hstack([joints_rot_m, np.ones((joints_rot_m.shape[0], 1), dtype=float)])
            joints_world = (world_from_camera_rot @ homo.T).T[:, :3]
            center, yaw, yaw_valid = center_yaw_smpl(joints_world)
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
                if joints2d is not None and joints2d.ndim == 3 and joints2d.shape[0] > person_idx:
                    draw_skeleton(debug, joints2d[person_idx], SMPL_EDGES, (0, 255, 0))
                draw_skeleton(debug, project_points(k_rot, camera_rot_from_world, joints_world), SMPL_EDGES, (0, 0, 255))

        writer.write_jsonl(record)
        if debug is not None:
            out_path = writer.debug_dir / f"{image_path.stem}_group{group}_node{node_id}_{side}_nlf.jpg"
            cv2.imwrite(str(out_path), debug)

    writer.finalize()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
