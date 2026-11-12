#!/usr/bin/env python3
# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
# SPDX-License-Identifier: GPL-3.0-only
"""Visualize calibration results in Rerun (post-hoc).

Loads a saved calibration result + detections + thumbnails and replays them in the Rerun
viewer. No re-calibration needed. Like `mvis.analysis.regenerate_html`, the run's
``detections.mvisdb`` comes first (its latest solved run and its frames); the YAML (read with
the safe numpy-aware loader) and a legacy ``detections.json`` are the fallbacks.

Usage:
    python3 scripts/visualize_result.py results/<run>/
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_result(run_dir: Path):
    """CalibrationResult of a run directory: the mvisdb's latest solved run when it has board
    poses, else the YAML. Also returns the YAML's board points and the parsed YAML."""
    from mvis.core.types import (
        CalibrationResult, CameraParams, ExtrinsicTransform,
    )
    from mvis.io.safe_yaml import load_yaml_file_safe

    # Find the full YAML (has board_poses)
    full_yamls = sorted(run_dir.glob("*_full.yaml"))
    if not full_yamls:
        # not the cam-IMU board-shape sidecar: it sorts before cam_imu_calib.yaml
        full_yamls = sorted(y for y in run_dir.glob("*.yaml") if not y.stem.endswith("_board_shape"))
    if not full_yamls:
        raise FileNotFoundError(f"No YAML found in {run_dir}")

    yaml_path = full_yamls[0]
    logger.info(f"Loading: {yaml_path}")
    raw = load_yaml_file_safe(yaml_path) or {}

    # Cameras
    cameras = []
    for c in raw.get("cameras", []):
        intr = c["intrinsics"]
        dist = c["distortion"]
        cameras.append(CameraParams(
            camera_id=c["id"],
            intrinsics=np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]]),
            distortion=np.array(dist["coeffs"]),
            distortion_model=dist["model"],
            image_size=tuple(c.get("image_size", [0, 0])),
            rolling_shutter_readout=c.get("rolling_shutter_readout", 0.0),
        ))

    # Extrinsics
    extrinsics = []
    ext_raw = raw.get("extrinsics", {})
    transforms = ext_raw.get("transforms", []) if isinstance(ext_raw, dict) else ext_raw
    for e in transforms:
        T = np.array(e["T_to_from"])
        so3 = np.array(e.get("so3", [0, 0, 0]))
        trans = np.array(e.get("translation", T[:3, 3] if T.ndim == 2 else [0, 0, 0]))
        extrinsics.append(ExtrinsicTransform(
            from_camera=e["from"], to_camera=e["to"],
            T_to_from=T, so3=so3, translation=trans,
        ))

    # Board poses — handle both dict format and raw array format
    board_poses = []
    board_pose_times = []
    for bp in raw.get("board_poses", []):
        if bp is None:
            board_poses.append(None)
            board_pose_times.append(0.0)
        elif isinstance(bp, dict):
            # Format: {frame_index, timestamp, T_board_cam}
            T = bp.get("T_board_cam") or bp.get("T") or bp.get("pose")
            if T is not None:
                board_poses.append(np.array(T))
            else:
                board_poses.append(None)
            board_pose_times.append(float(bp.get("timestamp", 0.0)))
        elif isinstance(bp, (list, np.ndarray)):
            arr = np.array(bp)
            if arr.shape == (4, 4):
                board_poses.append(arr)
            else:
                board_poses.append(None)
            board_pose_times.append(0.0)
        else:
            board_poses.append(None)
            board_pose_times.append(0.0)

    # Fallback: use raw board_pose_times if present
    raw_times = raw.get("board_pose_times", [])
    if raw_times:
        board_pose_times = [float(t) for t in raw_times]

    ref_cam = raw.get("reference_camera",
                       ext_raw.get("reference_camera", 0) if isinstance(ext_raw, dict) else 0)

    result = CalibrationResult(
        cameras=cameras,
        extrinsics=extrinsics,
        reference_camera=ref_cam,
        board_poses=board_poses,
        board_pose_times=board_pose_times,
    )

    mvisdb_path = run_dir / "detections.mvisdb"
    if mvisdb_path.exists():
        from mvis.analysis.regenerate_html import _result_from_mvisdb
        try:
            db_result = _result_from_mvisdb(mvisdb_path)
        except Exception as e:
            logger.warning(f"mvisdb result read failed: {e}")
            db_result = None
        if db_result is not None and db_result.board_poses:
            logger.info(f"Result from {mvisdb_path.name} ({len(db_result.board_poses)} poses)")
            result = db_result

    # Board 3D points from YAML
    board_pts = None
    board_raw = raw.get("board", {})
    if board_raw and "points_3d" in board_raw:
        board_pts = np.array(board_raw["points_3d"])

    return result, board_pts, raw


def load_frames_and_images(run_dir: Path):
    """Detections (``detections.mvisdb`` first, then a legacy ``detections.json``), the image
    size per camera from its first frame, and the thumbnails."""
    from mvis.analysis.regenerate_html import _frames_from_mvisdb, _load_thumbnails
    from mvis.io.detection_cache import legacy_json_frames

    all_frames = {}
    mvisdb_path = run_dir / "detections.mvisdb"
    if mvisdb_path.exists():
        try:
            all_frames = _frames_from_mvisdb(mvisdb_path)
        except Exception as e:
            logger.warning(f"mvisdb frame reload failed: {e}")
    if not all_frames:
        all_frames = legacy_json_frames(run_dir / "detections.json")
    if all_frames:
        logger.info(f"Loaded detections: {', '.join(f'cam{k}={len(v)} frames' for k, v in all_frames.items())}")
    image_sizes = {cid: frames[0].image_size for cid, frames in all_frames.items() if frames}

    detection_images = _load_thumbnails(run_dir / "thumbnails")
    if detection_images:
        logger.info(f"Loaded thumbnails: {', '.join(f'cam{k}={len(v)}' for k, v in detection_images.items())}")

    return all_frames, image_sizes, detection_images


def main():
    parser = argparse.ArgumentParser(
        description="Visualize calibration results in Rerun")
    parser.add_argument("run_dir", type=Path,
                        help="Path to a calibration run directory")
    parser.add_argument("--no-spawn", action="store_true",
                        help="Don't spawn Rerun viewer (connect to existing)")
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: {args.run_dir} not found")
        return 1

    # Load data
    result, board_pts, raw = load_result(args.run_dir)
    all_frames, image_sizes, detection_images = load_frames_and_images(args.run_dir)

    # If no board_pose_times in YAML, reconstruct from detection timestamps
    if not result.board_pose_times and all_frames:
        ref_frames = all_frames.get(result.reference_camera, [])
        if ref_frames:
            # Map poses to frame timestamps (valid poses only)
            times = []
            for i, bp in enumerate(result.board_poses):
                if bp is not None and i < len(ref_frames):
                    times.append(ref_frames[i].timestamp)
                else:
                    times.append(0.0)
            result.board_pose_times = times
            logger.info(f"Reconstructed {sum(1 for t in times if t > 0)} pose timestamps from detections")

    # Launch Rerun
    from mvis.analysis.rerun_viz import RerunVisualizer
    viz = RerunVisualizer("mvis_replay", spawn=not args.no_spawn)
    if not viz.enabled:
        print("Error: Rerun not available. Install with: pip install rerun-sdk")
        return 1

    # Create a minimal board object for log_board_points
    class SimpleBoard:
        def __init__(self, pts):
            self._pts = pts
        def all_points_3d(self):
            return self._pts
        def num_points(self):
            return len(self._pts)

    board = SimpleBoard(board_pts) if board_pts is not None else None

    logger.info("Logging to Rerun...")
    viz.log_calibration_result(
        result, all_frames, board, image_sizes, detection_images)
    viz.log_text(f"Replay: {args.run_dir.name}")

    n_poses = sum(1 for bp in result.board_poses if bp is not None and not np.allclose(np.asarray(bp), np.eye(4)))
    n_cams = len(result.cameras)
    n_imgs = sum(len(v) for v in detection_images.values())
    logger.info(f"Logged: {n_poses} poses, {n_cams} cameras, {n_imgs} images")
    logger.info("Use the Rerun viewer to explore. Press Ctrl+C to exit.")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
