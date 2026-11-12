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
"""Synthetic camera-only chessboard observations."""
import numpy as np
import cv2

from mvis.core.types import FrameObservations, Observation
from mvis.detection.chessboard import Chessboard

_K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
_DIST = np.zeros(5)
_IMG = (640, 480)


def _gen_frames(board, n=16, seed=0):
    """Synthetic single-camera chessboard detections via cv2.projectPoints."""
    rng = np.random.RandomState(seed)
    pts = board.all_points_3d().astype(np.float64)
    frames, t = [], 0.0
    attempts = 0
    while len(frames) < n and attempts < n * 20:
        attempts += 1
        rvec = rng.uniform(-0.3, 0.3, 3)
        tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1),
                         rng.uniform(0.6, 0.9)])
        uv, _ = cv2.projectPoints(pts, rvec, tvec, _K, _DIST)
        uv = uv.reshape(-1, 2)
        inb = ((uv[:, 0] >= 0) & (uv[:, 0] < _IMG[0])
               & (uv[:, 1] >= 0) & (uv[:, 1] < _IMG[1]))
        if inb.sum() < 0.85 * len(pts):
            continue
        ids = np.where(inb)[0]
        obs = [Observation(point_id=int(i),
                           pixel=uv[i] + rng.standard_normal(2) * 0.2,
                           timestamp=t)
               for i in ids]
        frames.append(FrameObservations(camera_id=0, timestamp=t,
                                        observations=obs, image_size=_IMG))
        t += 0.1
    return frames


def _generate_synthetic_observations(
    board: Chessboard,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    num_frames: int = 15,
    image_size: tuple[int, int] = (640, 480),
    noise_sigma: float = 0.5,
    seed: int = 42,
) -> tuple[list[FrameObservations], list[np.ndarray]]:
    """Generate synthetic board observations with known ground truth.

    Returns:
        (frames, poses): list of FrameObservations and ground truth 4x4 poses.
    """
    rng = np.random.RandomState(seed)
    board_pts = board.all_points_3d().astype(np.float32)
    frames = []
    gt_poses = []

    for i in range(num_frames):
        # Random board pose: looking at the board from ~0.5-1.0m away
        rvec = rng.uniform(-0.5, 0.5, 3).astype(np.float64)
        tvec = np.array([
            rng.uniform(-0.15, 0.15),
            rng.uniform(-0.15, 0.15),
            rng.uniform(0.4, 1.0),
        ], dtype=np.float64)

        # Project points
        img_pts, _ = cv2.projectPoints(
            board_pts, rvec, tvec, K, dist_coeffs
        )
        img_pts = img_pts.reshape(-1, 2)

        # Check all points are within image
        w, h = image_size
        valid = (
            (img_pts[:, 0] > 10)
            & (img_pts[:, 0] < w - 10)
            & (img_pts[:, 1] > 10)
            & (img_pts[:, 1] < h - 10)
        )

        if not np.all(valid):
            continue

        # Add noise
        img_pts += rng.randn(*img_pts.shape) * noise_sigma

        observations = []
        for idx in range(len(img_pts)):
            observations.append(
                Observation(
                    point_id=idx,
                    pixel=img_pts[idx],
                    row=int(img_pts[idx, 1]),
                )
            )

        frames.append(
            FrameObservations(
                camera_id=0,
                timestamp=float(i),
                observations=observations,
                image_size=image_size,
            )
        )

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        gt_poses.append(T)

    return frames, gt_poses
