#!/usr/bin/env python3
"""Dataset layout helpers for processed CoInfra/SAP-CoPE perception data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple


def parse_csv_list(value: str | None) -> List[str] | None:
    """Parse a comma-separated CLI value; empty/None means all."""
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_node_side_pairs(nodes: str, sides: str) -> List[Tuple[int, str]]:
    """Parse matching comma-separated node IDs and camera sides."""
    node_ids = [int(item.strip()) for item in nodes.split(",") if item.strip()]
    side_names = [item.strip().lower() for item in sides.split(",") if item.strip()]
    if len(side_names) == 1 and len(node_ids) > 1:
        side_names = side_names * len(node_ids)
    if len(node_ids) != len(side_names):
        raise ValueError(f"nodes and sides must have matching lengths: {node_ids} vs {side_names}")
    for side in side_names:
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported camera side '{side}'. Expected left or right.")
    return list(zip(node_ids, side_names))


def list_groups(pcimage_root: Path, requested_groups: Sequence[str] | None) -> List[str]:
    """Resolve requested group names or list all PcImage timestamp groups."""
    if not pcimage_root.exists():
        raise FileNotFoundError(f"PcImage directory not found: {pcimage_root}")
    if requested_groups is not None:
        return list(requested_groups)
    groups = sorted(path.name for path in pcimage_root.iterdir() if path.is_dir())
    if not groups:
        raise RuntimeError(f"No PcImage groups found in {pcimage_root}")
    return groups


def iter_camera_frames(
    data_root: Path,
    scenario: str,
    slice_name: str,
    groups: Iterable[str],
    node_side_pairs: Sequence[Tuple[int, str]],
) -> Iterator[Tuple[str, int, str, Path, Path, Path, Path | None]]:
    """
    Yield image/calibration/LiDAR paths for each requested view.

    The LiDAR npy path is optional because camera-only runs should still work.
    """
    slice_dir = data_root / scenario / slice_name
    for group in groups:
        camera_dir = slice_dir / "PcImage" / group / "camera"
        lidar_dir = slice_dir / "PcImage" / group / "lidar"
        if not camera_dir.exists():
            print(f"[WARN] Skipping group without camera directory: {camera_dir}")
            continue
        for node_id, side in node_side_pairs:
            camera_yaml = slice_dir / "Calibration" / "camera" / f"{node_id}_{side}.yaml"
            lidar_yaml = slice_dir / "Calibration" / "lidar" / f"{node_id}.yaml"
            for required in (camera_yaml, lidar_yaml):
                if not required.exists():
                    raise FileNotFoundError(f"Required calibration file not found: {required}")
            image_paths = sorted(camera_dir.glob(f"{node_id}_{side}_*.jpg"))
            if not image_paths:
                print(f"[WARN] No image found for group={group}, node={node_id}, side={side}")
                continue
            lidar_paths = sorted(lidar_dir.glob(f"{node_id}_*.npy")) if lidar_dir.exists() else []
            yield group, node_id, side, image_paths[0], camera_yaml, lidar_yaml, (lidar_paths[0] if lidar_paths else None)

