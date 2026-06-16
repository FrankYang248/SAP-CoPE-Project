# SAP-CoPE Perception Release Code

This directory contains the release-oriented perception code for the SAP-CoPE paper draft:

- uncertainty propagation from 2D COCO joints to absolute 3D world-frame joint Gaussians;
- joint-coherence refinement with optional sparse LiDAR joint measurements;
- camera-only and camera+LiDAR batch inference over processed CoInfra/ICID-style folders;
- delay-aware cooperative perception evaluation with multi-node fusion, EKF tracking, and timestamp compensation.

## Dependencies

Core evaluation:

```bash
pip install numpy scipy pyyaml
```

Pose inference:

```bash
pip install numpy scipy pyyaml opencv-python ultralytics
```

The YOLO pose model can be an Ultralytics model name such as `yolo11s-pose.pt` or a local fine-tuned checkpoint.

Learning-based baselines have separate upstream dependencies:

- MeTRAbs: `tensorflow`, `tensorflow_hub`, `opencv-python`.
- NLF: `torch`, `torchvision`, `opencv-python`; the default model URL is `https://bit.ly/nlf_l_pt`.
- LiCamPose: `torch`, `mmpose`, and a local clone of the official LiCamPose repository plus a trained checkpoint.

## Expected Data Layout

The runner expects the processed dataset layout used in the experiments:

```text
DATA_ROOT/
  SCENARIO/
    slice0/
      Calibration/
        camera/
          1_left.yaml
          3_right.yaml
        lidar/
          1.yaml
          3.yaml
      HDmap/
        transformation.yaml
        ROI/
          1_roi.png
          3_roi.png
      PcImage/
        1765487998.3/
          camera/
            1_left_*.jpg
            3_right_*.jpg
          lidar/
            1_*.npy
            3_*.npy
```

`HDmap/ROI` is optional. If ROI masks are present, LiDAR points are filtered before image-plane association.

## Run Pose Estimation

Camera-only:

```bash
python -m scripts.perception.run_pose_estimation \
  --data-root /path/to/DATA_ROOT \
  --scenario WithBed \
  --slice slice0 \
  --nodes 1,3 \
  --sides left,right \
  --method camera \
  --pose-model /path/to/yolo11s-pose.pt \
  --out-dir eval_out/camera
```

Camera+LiDAR:

```bash
python -m scripts.perception.run_pose_estimation \
  --data-root /path/to/DATA_ROOT \
  --scenario WithBed \
  --slice slice0 \
  --nodes 1,3 \
  --sides left,right \
  --method camera_lidar \
  --pose-model /path/to/yolo11s-pose.pt \
  --out-dir eval_out/camera_lidar \
  --debug
```

Outputs:

- `results.csv`: evaluator-compatible detections with `method, group, center_x, center_y, center_z, yaw_rad, yaw_valid, det_score`;
- `results.jsonl`: per-frame metadata and detections;
- `debug_vis/*.jpg`: optional visual overlays.

For a small run, pass `--groups timestamp0,timestamp1`.

## Learning-Based Baselines

The paper table also reports learning-based baselines. Their runners write the same evaluator-compatible `results.csv` schema as the proposed method.

MeTRAbs:

```bash
python -m scripts.perception.run_metrabs_baseline \
  --data-root /path/to/DATA_ROOT \
  --scenario WithBed \
  --slice slice0 \
  --groups ALL \
  --nodes 1,1,3,3 \
  --sides left,right,left,right \
  --out-dir eval_out/metrabs \
  --skeleton h36m_17 \
  --method-name metrabs
```

NLF:

```bash
python -m scripts.perception.run_nlf_baseline \
  --data-root /path/to/DATA_ROOT \
  --scenario WithBed \
  --slice slice0 \
  --groups ALL \
  --nodes 1,1,3,3 \
  --sides left,right,left,right \
  --model-path models/nlf_l_multi.torchscript \
  --out-dir eval_out/nlf \
  --method-name nlf
```

LiCamPose:

```bash
python -m scripts.perception.run_licampose_baseline \
  --data-root /path/to/DATA_ROOT \
  --scenario WithBed \
  --slice slice0 \
  --groups ALL \
  --views 1_left,1_right,3_left,3_right \
  --primary-view 3_right \
  --lidar-nodes 1,3 \
  --licampose-repo /path/to/LiCamPose \
  --checkpoint /path/to/licampose_checkpoint.pt \
  --out-dir eval_out/licampose
```

LiCamPose can also be run with an empty `--checkpoint ""` to write adapter tensors and debug visualizations without model inference. This is useful for checking that the multi-view 2D poses, LiDAR crop, voxel grid, and camera dictionaries are being constructed correctly before loading a checkpoint.

## Delay-Robustness Evaluation

Use a result CSV containing one or more methods and a CVAT video XML annotation file:

```bash
python -m scripts.perception.evaluate_delay \
  --results-csv eval_out/results.csv \
  --gt-video-xml /path/to/gt_annotations_video.xml \
  --tf-yaml /path/to/HDmap/transformation.yaml \
  --methods camera_only_refine,camera_lidar_refine,metrabs,nlf,licampose \
  --delays-ms 0,50,100,200 \
  --out-dir eval_out/delay
```

Outputs:

- `summary_delay.json`: precision, recall, F1, position MAE/RMSE, and yaw MAE/RMSE for each delay;
- `per_match_delay.csv`: matched prediction/ground-truth pairs for further plotting.

The evaluator reports two modes:

- `delay_aware`: same-timestamp multi-node fusion, EKF tracking, and prediction to `t + delay`;
- `baseline_no_delay_comp`: same-timestamp fusion without tracking or timestamp compensation.

## Implementation Notes

The optimizer in `pose_refinement.py` implements the paper objective as a proper least-squares residual problem:

- camera residuals are whitened by each propagated 3D joint covariance;
- bone residuals use adjacent-joint length priors and uncertainty;
- LiDAR residuals are enabled only for accepted LiDAR-to-joint associations;
- residual groups are normalized by the number of active terms, matching the paper's negative-log-likelihood form.

Default bone priors are conservative and derived from the released height template so the code can run without private anthropometric metadata. If calibrated priors are available, pass them into `refine_pose(...)` from downstream code.
