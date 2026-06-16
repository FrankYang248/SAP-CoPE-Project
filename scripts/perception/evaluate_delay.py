#!/usr/bin/env python3
"""Evaluate delay-aware cooperative perception from pose result CSV files."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback is used.
    linear_sum_assignment = None


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_rad(a: float, b: float) -> float:
    return wrap_angle_rad(a - b)


def interp_angle_rad(a0: float, a1: float, alpha: float) -> float:
    return wrap_angle_rad(a0 + alpha * angle_diff_rad(a1, a0))


@dataclass
class Detection:
    t: float
    x: float
    y: float
    z: float
    yaw: float
    yaw_valid: bool
    score: float


@dataclass
class FusedDetection:
    t: float
    x: float
    y: float
    z: float
    yaw: float
    yaw_valid: bool
    score: float
    members: int


@dataclass
class GTPose:
    x: float
    y: float
    yaw: float


def cvat_px_to_world_xy(cx: float, cy: float, scale: float, min_x: float, max_y: float) -> Tuple[float, float]:
    return cx / scale + min_x, max_y - cy / scale


def cvat_rotation_to_yaw(rot_deg: float) -> float:
    yaw_deg = (90.0 - rot_deg) % 360.0
    yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
    return math.radians(yaw_deg)


def parse_cvat_tracks(
    xml_path: Path,
    *,
    scale: float,
    min_x: float,
    max_y: float,
    person_label: str = "Person",
) -> Dict[int, Dict[int, GTPose]]:
    tree = ET.parse(str(xml_path))
    tracks: Dict[int, Dict[int, GTPose]] = {}
    for track in tree.getroot().findall("track"):
        if track.attrib.get("label") != person_label:
            continue
        track_id = int(track.attrib.get("id", "0"))
        frame_map: Dict[int, GTPose] = {}
        for ellipse in track.findall("ellipse"):
            if int(ellipse.attrib.get("outside", "0")) == 1:
                continue
            frame = int(ellipse.attrib["frame"])
            x, y = cvat_px_to_world_xy(
                float(ellipse.attrib["cx"]),
                float(ellipse.attrib["cy"]),
                scale,
                min_x,
                max_y,
            )
            yaw = cvat_rotation_to_yaw(float(ellipse.attrib.get("rotation", "0.0")))
            frame_map[frame] = GTPose(x=x, y=y, yaw=yaw)
        if frame_map:
            tracks[track_id] = frame_map
    return tracks


class GTAccessor:
    def __init__(self, tracks: Dict[int, Dict[int, GTPose]]):
        self.tracks = tracks
        self.frames = {track_id: sorted(frame_map.keys()) for track_id, frame_map in tracks.items()}

    def interpolated(self, frame_f: float) -> List[GTPose]:
        poses: List[GTPose] = []
        for track_id, frame_map in self.tracks.items():
            frames = self.frames[track_id]
            idx = bisect.bisect_left(frames, frame_f)
            if idx == 0:
                if frame_f == frames[0]:
                    poses.append(frame_map[frames[0]])
                continue
            if idx >= len(frames):
                if frame_f == frames[-1]:
                    poses.append(frame_map[frames[-1]])
                continue
            f0, f1 = frames[idx - 1], frames[idx]
            p0, p1 = frame_map[f0], frame_map[f1]
            alpha = (frame_f - f0) / max(float(f1 - f0), 1e-9)
            poses.append(
                GTPose(
                    x=p0.x + alpha * (p1.x - p0.x),
                    y=p0.y + alpha * (p1.y - p0.y),
                    yaw=interp_angle_rad(p0.yaw, p1.yaw, alpha),
                )
            )
        return poses

    def nearest_labeled(self, frame_f: float) -> List[GTPose]:
        frame_i = int(round(frame_f))
        return [frame_map[frame_i] for frame_map in self.tracks.values() if frame_i in frame_map]


def to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_detections_by_method_time(
    csv_path: Path,
    methods: List[str],
    *,
    min_score: float,
) -> Dict[str, Dict[float, List[Detection]]]:
    output: Dict[str, Dict[float, List[Detection]]] = {method: {} for method in methods}
    with open(csv_path, "r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            method = (row.get("method") or "").strip().lower()
            if method not in output:
                continue
            score = to_float(row.get("det_score", "1.0"), 1.0)
            if score < min_score:
                continue
            t = to_float(row.get("group", ""))
            x = to_float(row.get("center_x", ""))
            y = to_float(row.get("center_y", ""))
            z = to_float(row.get("center_z", "0.0"), 0.0)
            yaw = to_float(row.get("yaw_rad", "0.0"), 0.0)
            if not all(math.isfinite(v) for v in (t, x, y, z, yaw)):
                continue
            yaw_valid = bool(int(float(row.get("yaw_valid", "1"))))
            output[method].setdefault(t, []).append(
                Detection(t=t, x=x, y=y, z=z, yaw=yaw, yaw_valid=yaw_valid, score=score)
            )
    return output


def fuse_detections(detections: List[Detection], fuse_dist_m: float) -> List[FusedDetection]:
    """Cluster same-timestamp multi-node detections by spatial proximity."""
    unused = list(detections)
    fused: List[FusedDetection] = []
    while unused:
        seed = unused.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            remaining = []
            for det in unused:
                if any(math.hypot(det.x - member.x, det.y - member.y) <= fuse_dist_m for member in cluster):
                    cluster.append(det)
                    changed = True
                else:
                    remaining.append(det)
            unused = remaining

        weights = np.array([max(member.score, 1e-6) for member in cluster], dtype=float)
        xyz = np.array([[member.x, member.y, member.z] for member in cluster], dtype=float)
        mean_xyz = np.average(xyz, axis=0, weights=weights)
        yaw_members = [member for member in cluster if member.yaw_valid]
        if yaw_members:
            yaw_weights = np.array([max(member.score, 1e-6) for member in yaw_members], dtype=float)
            yaw = math.atan2(
                float(np.sum(np.sin([member.yaw for member in yaw_members]) * yaw_weights)),
                float(np.sum(np.cos([member.yaw for member in yaw_members]) * yaw_weights)),
            )
            yaw_valid = True
        else:
            yaw = seed.yaw
            yaw_valid = False
        fused.append(
            FusedDetection(
                t=seed.t,
                x=float(mean_xyz[0]),
                y=float(mean_xyz[1]),
                z=float(mean_xyz[2]),
                yaw=yaw,
                yaw_valid=yaw_valid,
                score=float(np.mean([member.score for member in cluster])),
                members=len(cluster),
            )
        )
    return fused


class PersonEKF:
    """Small EKF used for delay compensation in the paper evaluation."""

    def __init__(self, measurement: FusedDetection, track_id: int):
        self.id = track_id
        self.t = measurement.t
        self.x = np.array([measurement.x, measurement.y, measurement.yaw, 0.0, 0.0, 0.0, measurement.z], dtype=float)
        self.P = np.diag([0.4, 0.4, math.radians(45), 2.0, 2.0, math.radians(90), 0.5]) ** 2
        self.hits = 1
        self.time_since_update = 0

    @staticmethod
    def clip_dt(dt: float, max_dt: float = 0.3) -> float:
        return max(-max_dt, min(max_dt, float(dt)))

    def _F(self, dt: float) -> np.ndarray:
        dt = self.clip_dt(dt)
        yaw, vx, vy = float(self.x[2]), float(self.x[3]), float(self.x[4])
        return np.array(
            [
                [1, 0, -dt * (vx * math.sin(yaw) + vy * math.cos(yaw)), dt * math.cos(yaw), -dt * math.sin(yaw), 0, 0],
                [0, 1, dt * (vx * math.cos(yaw) - vy * math.sin(yaw)), dt * math.sin(yaw), dt * math.cos(yaw), 0, 0],
                [0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )

    def _Q(self, dt: float) -> np.ndarray:
        dt = abs(self.clip_dt(dt))
        base = np.diag([0.02, 0.02, math.radians(3), 0.10, 0.10, math.radians(8), 0.02]) ** 2
        time_scaled = (np.diag([0.10, 0.10, math.radians(6), 0.60, 0.60, math.radians(35), 0.10]) * dt) ** 2
        return base + time_scaled

    def predict(self, t_new: float) -> None:
        predicted = self.predict_at(t_new)
        dt = self.clip_dt(t_new - self.t)
        self.x[:3] = predicted[:3]
        self.P = self._F(dt) @ self.P @ self._F(dt).T + self._Q(dt)
        self.t = float(t_new)
        self.time_since_update += 1

    def predict_at(self, t_query: float) -> np.ndarray:
        dt = self.clip_dt(t_query - self.t)
        yaw, vx, vy, yaw_rate = float(self.x[2]), float(self.x[3]), float(self.x[4]), float(self.x[5])
        output = self.x.copy()
        output[0] += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        output[1] += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        output[2] = wrap_angle_rad(yaw + yaw_rate * dt)
        return output

    def update(self, measurement: FusedDetection) -> None:
        H = np.zeros((4, 7), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 6] = 1.0
        z = np.array([measurement.x, measurement.y, measurement.yaw, measurement.z], dtype=float)
        innovation = z - H @ self.x
        innovation[2] = angle_diff_rad(z[2], (H @ self.x)[2])
        R = np.diag([0.10, 0.10, math.radians(30), 0.10]) ** 2
        if not measurement.yaw_valid:
            R[2, 2] = math.radians(180) ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap_angle_rad(float(self.x[2]))
        self.P = (np.eye(7) - K @ H) @ self.P
        self.hits += 1
        self.time_since_update = 0


class MultiObjectTracker:
    def __init__(self, *, max_age: int, min_hits: int, assoc_dist_m: float):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.assoc_dist_m = float(assoc_dist_m)
        self.tracks: List[PersonEKF] = []
        self.next_id = 0

    def step(self, t: float, measurements: List[FusedDetection]) -> None:
        for track in self.tracks:
            track.predict(t)

        matches, unmatched_measurements = associate_tracks(self.tracks, measurements, self.assoc_dist_m)
        for track_idx, measurement_idx in matches:
            self.tracks[track_idx].update(measurements[measurement_idx])
        for measurement_idx in unmatched_measurements:
            self.tracks.append(PersonEKF(measurements[measurement_idx], self.next_id))
            self.next_id += 1
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

    def confirmed(self) -> List[PersonEKF]:
        return [track for track in self.tracks if track.hits >= self.min_hits]


def hungarian_pairs(cost: np.ndarray):
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    flat = [(float(cost[i, j]), i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    flat.sort(key=lambda item: item[0])
    used_i, used_j, pairs = set(), set(), []
    for _, i, j in flat:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


def associate_tracks(
    tracks: List[PersonEKF],
    measurements: List[FusedDetection],
    assoc_dist_m: float,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    if not tracks:
        return [], list(range(len(measurements)))
    if not measurements:
        return [], []
    cost = np.zeros((len(tracks), len(measurements)), dtype=float)
    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            cost[i, j] = math.hypot(float(track.x[0]) - measurement.x, float(track.x[1]) - measurement.y)
    matches = []
    unmatched_measurements = set(range(len(measurements)))
    for i, j in hungarian_pairs(cost):
        if float(cost[i, j]) <= assoc_dist_m:
            matches.append((i, j))
            unmatched_measurements.discard(j)
    return matches, sorted(unmatched_measurements)


def match_predictions_to_gt(
    predictions: List[Tuple[int, float, float, float]],
    gt_poses: List[GTPose],
    max_dist_m: float,
) -> Tuple[List[Tuple[int, int, float, float]], int, int]:
    if not predictions or not gt_poses:
        return [], len(predictions), len(gt_poses)
    cost = np.zeros((len(predictions), len(gt_poses)), dtype=float)
    for i, (_, px, py, _) in enumerate(predictions):
        for j, gt in enumerate(gt_poses):
            cost[i, j] = math.hypot(px - gt.x, py - gt.y)
    matches = []
    for i, j in hungarian_pairs(cost):
        distance = float(cost[i, j])
        if distance > max_dist_m:
            continue
        yaw_error = abs(math.degrees(angle_diff_rad(predictions[i][3], gt_poses[j].yaw)))
        matches.append((i, j, distance, yaw_error))
    return matches, len(predictions), len(gt_poses)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def rmse(values: List[float]) -> float:
    return float(math.sqrt(sum(value * value for value in values) / len(values))) if values else float("nan")


def summarize_accumulator(acc: Dict[str, object]) -> Dict[str, float]:
    gt_total = int(acc["gt_total"])
    pred_total = int(acc["pred_total"])
    match_total = int(acc["match_total"])
    precision = match_total / pred_total if pred_total else 0.0
    recall = match_total / gt_total if gt_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 1e-12 else 0.0
    pos_err = acc["pos_err"]
    yaw_err = acc["yaw_err"]
    return {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "match_total": match_total,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pos_mae_m": mean(pos_err),
        "pos_rmse_m": rmse(pos_err),
        "yaw_mae_deg": mean(yaw_err),
        "yaw_rmse_deg": rmse(yaw_err),
        "eval_pairs": int(len(pos_err)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--gt-video-xml", required=True, type=Path)
    parser.add_argument("--tf-yaml", required=True, type=Path, help="HD map transformation YAML with scale/min_x/max_y.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--methods", default="camera_only_refine,camera_lidar_refine")
    parser.add_argument("--delays-ms", default="0,50,100,200")
    parser.add_argument("--frame-dt", type=float, default=0.1)
    parser.add_argument("--t0", type=float, default=None)
    parser.add_argument("--min-det-score", type=float, default=0.5)
    parser.add_argument("--fuse-dist-m", type=float, default=0.6)
    parser.add_argument("--assoc-dist-m", type=float, default=1.0)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--max-match-dist-m", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    delays_ms = [int(float(item.strip())) for item in args.delays_ms.split(",") if item.strip()]
    delays_s = [delay / 1000.0 for delay in delays_ms]

    with open(args.tf_yaml, "r", encoding="utf-8") as file:
        transform = yaml.safe_load(file) or {}
    gt = GTAccessor(
        parse_cvat_tracks(
            args.gt_video_xml,
            scale=float(transform["scale"]),
            min_x=float(transform["min_x"]),
            max_y=float(transform["max_y"]),
        )
    )

    detections_by_method = load_detections_by_method_time(
        args.results_csv,
        methods,
        min_score=args.min_det_score,
    )
    timestamps = [t for method in methods for t in detections_by_method[method].keys()]
    if not timestamps:
        raise RuntimeError("No valid detections found. Check methods, score threshold, and CSV schema.")
    t0 = float(args.t0) if args.t0 is not None else min(timestamps)

    def frame_at_time(t: float) -> float:
        return (t - t0) / float(args.frame_dt)

    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    per_match_rows: List[Dict[str, object]] = []

    for method in methods:
        fused_by_time = {
            timestamp: fuse_detections(dets, args.fuse_dist_m)
            for timestamp, dets in detections_by_method[method].items()
        }
        ts_sorted = sorted(fused_by_time.keys())
        summary[method] = {}

        mode = "delay_aware"
        acc = {delay: {"gt_total": 0, "pred_total": 0, "match_total": 0, "pos_err": [], "yaw_err": []} for delay in delays_ms}
        tracker = MultiObjectTracker(max_age=args.max_age, min_hits=args.min_hits, assoc_dist_m=args.assoc_dist_m)
        for t in ts_sorted:
            tracker.step(t, fused_by_time[t])
            for delay_ms, delay_s in zip(delays_ms, delays_s):
                t_eval = t + delay_s
                gt_poses = gt.interpolated(frame_at_time(t_eval))
                predictions = [
                    (track.id, float(pred[0]), float(pred[1]), float(pred[2]))
                    for track in tracker.confirmed()
                    for pred in [track.predict_at(t_eval)]
                ]
                matches, pred_total, gt_total = match_predictions_to_gt(predictions, gt_poses, args.max_match_dist_m)
                acc[delay_ms]["gt_total"] += gt_total
                acc[delay_ms]["pred_total"] += pred_total
                acc[delay_ms]["match_total"] += len(matches)
                for pred_idx, gt_idx, pos_error, yaw_error in matches:
                    acc[delay_ms]["pos_err"].append(pos_error)
                    acc[delay_ms]["yaw_err"].append(yaw_error)
                    pred_id, px, py, pyaw = predictions[pred_idx]
                    gt_pose = gt_poses[gt_idx]
                    per_match_rows.append(
                        {
                            "method": method,
                            "mode": mode,
                            "delay_ms": delay_ms,
                            "t_meas": t,
                            "t_eval": t_eval,
                            "pred_id": pred_id,
                            "pred_x": px,
                            "pred_y": py,
                            "pred_yaw_rad": pyaw,
                            "gt_x": gt_pose.x,
                            "gt_y": gt_pose.y,
                            "gt_yaw_rad": gt_pose.yaw,
                            "pos_err_m": pos_error,
                            "yaw_err_deg": yaw_error,
                            "gt_count": gt_total,
                            "pred_count": pred_total,
                        }
                    )
        summary[method][mode] = {str(delay): summarize_accumulator(acc[delay]) for delay in delays_ms}

        mode = "baseline_no_delay_comp"
        acc = {delay: {"gt_total": 0, "pred_total": 0, "match_total": 0, "pos_err": [], "yaw_err": []} for delay in delays_ms}
        for t in ts_sorted:
            predictions = [(idx, det.x, det.y, det.yaw) for idx, det in enumerate(fused_by_time[t])]
            for delay_ms, delay_s in zip(delays_ms, delays_s):
                t_eval = t + delay_s
                gt_poses = gt.nearest_labeled(frame_at_time(t_eval))
                matches, pred_total, gt_total = match_predictions_to_gt(predictions, gt_poses, args.max_match_dist_m)
                acc[delay_ms]["gt_total"] += gt_total
                acc[delay_ms]["pred_total"] += pred_total
                acc[delay_ms]["match_total"] += len(matches)
                for pred_idx, gt_idx, pos_error, yaw_error in matches:
                    acc[delay_ms]["pos_err"].append(pos_error)
                    acc[delay_ms]["yaw_err"].append(yaw_error)
                    pred_id, px, py, pyaw = predictions[pred_idx]
                    gt_pose = gt_poses[gt_idx]
                    per_match_rows.append(
                        {
                            "method": method,
                            "mode": mode,
                            "delay_ms": delay_ms,
                            "t_meas": t,
                            "t_eval": t_eval,
                            "pred_id": pred_id,
                            "pred_x": px,
                            "pred_y": py,
                            "pred_yaw_rad": pyaw,
                            "gt_x": gt_pose.x,
                            "gt_y": gt_pose.y,
                            "gt_yaw_rad": gt_pose.yaw,
                            "pos_err_m": pos_error,
                            "yaw_err_deg": yaw_error,
                            "gt_count": gt_total,
                            "pred_count": pred_total,
                        }
                    )
        summary[method][mode] = {str(delay): summarize_accumulator(acc[delay]) for delay in delays_ms}

    per_match_path = args.out_dir / "per_match_delay.csv"
    if per_match_rows:
        with open(per_match_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_match_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_match_rows)

    summary_path = args.out_dir / "summary_delay.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "methods": methods,
                "modes": ["delay_aware", "baseline_no_delay_comp"],
                "delays_ms": delays_ms,
                "frame_dt": args.frame_dt,
                "t0": t0,
                "summary": summary,
            },
            file,
            indent=2,
        )

    print(f"Wrote summary to {summary_path}")
    if per_match_rows:
        print(f"Wrote per-match rows to {per_match_path}")
    for method in methods:
        for mode in ("delay_aware", "baseline_no_delay_comp"):
            for delay_ms in delays_ms:
                stats = summary[method][mode][str(delay_ms)]
                print(
                    f"[{method}][{mode}][{delay_ms}ms] "
                    f"F1={stats['f1']:.4f} R={stats['recall']:.4f} "
                    f"DE={stats['pos_mae_m']:.4f} AE={stats['yaw_mae_deg']:.4f}"
                )


if __name__ == "__main__":
    main()
