#!/usr/bin/env python3
"""Image rotation helpers matching the infrastructure camera convention."""

from __future__ import annotations

import numpy as np


def rotate_angle_for_side(side: str) -> int:
    """The left camera images are rotated 90 deg clockwise before YOLO inference."""
    return 90 if side.lower() == "left" else 0


def rotate_image_for_inference(image_bgr: np.ndarray, angle: int) -> np.ndarray:
    import cv2

    if angle == 0:
        return image_bgr
    if angle == 90:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Rotation angle must be one of 0, 90, 180, 270")


def rotate_boxes_back_xyxy(boxes_xyxy: np.ndarray, rotated_shape: tuple[int, ...], angle: int) -> np.ndarray:
    """Rotate xyxy boxes from the inference image back to the calibration image."""
    h_rot, w_rot = rotated_shape[:2]
    boxes = np.asarray(boxes_xyxy, dtype=float)
    out = np.zeros_like(boxes, dtype=float)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        width = x2 - x1
        height = y2 - y1
        if angle == 90:
            cx_back, cy_back = cy, w_rot - cx
            width_back, height_back = height, width
        elif angle == 180:
            cx_back, cy_back = w_rot - cx, h_rot - cy
            width_back, height_back = width, height
        elif angle == 270:
            cx_back, cy_back = h_rot - cy, cx
            width_back, height_back = height, width
        else:
            cx_back, cy_back = cx, cy
            width_back, height_back = width, height
        out[idx] = [
            cx_back - 0.5 * width_back,
            cy_back - 0.5 * height_back,
            cx_back + 0.5 * width_back,
            cy_back + 0.5 * height_back,
        ]
    return out


def rotate_keypoints_back(keypoints_xy: np.ndarray, rotated_shape: tuple[int, ...], angle: int) -> np.ndarray:
    """Rotate keypoints from the inference image back to the calibration image."""
    h_rot, w_rot = rotated_shape[:2]
    points = np.asarray(keypoints_xy, dtype=float).copy()
    if angle == 0:
        return points
    x = points[..., 0].copy()
    y = points[..., 1].copy()
    if angle == 90:
        points[..., 0] = y
        points[..., 1] = w_rot - x
    elif angle == 180:
        points[..., 0] = w_rot - x
        points[..., 1] = h_rot - y
    elif angle == 270:
        points[..., 0] = h_rot - y
        points[..., 1] = x
    else:
        raise ValueError("Rotation angle must be one of 0, 90, 180, 270")
    return points

