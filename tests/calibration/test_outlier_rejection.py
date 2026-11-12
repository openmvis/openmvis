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
"""Tests for the outlier_rejection module.

filter_outliers has two stages:
    1. Frame-level: drop frames whose median residual is far above the rest.
    2. Observation-level: drop individual obs whose residual is far above the rest.
"""

import numpy as np
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.outlier_rejection import filter_outliers
from mvis.core.types import FrameObservations, Observation


def _frame(obs_count: int, cam_id: int = 0, ts: float = 0.0):
    """Build a frame with `obs_count` observations on a 7x5 chessboard."""
    pixels = [(float(i * 10 + 50), float(50 + i * 10)) for i in range(obs_count)]
    obs = [Observation(point_id=i, pixel=np.array(pixels[i]), row=int(pixels[i][1]))
           for i in range(obs_count)]
    return FrameObservations(
        camera_id=cam_id, timestamp=ts, observations=obs, image_size=(640, 480))


def test_filter_drops_large_obs_residuals():
    """Catastrophic-outlier observations (<1% of data) should be removed."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    # Build 200 observations total: 198 good, 2 huge outliers (1%)
    # Split across 6 frames of 33 obs (P99 catches the 2 worst)
    residuals_flat = [0.3] * 198 + [50.0] * 2
    # Put everything in one big synthetic frame with 35 obs,
    # but with 33 frames of 6 = 198 obs to make P99 meaningful
    n_frames = 6
    per_frame = 33
    # Populate in order: last 2 obs of the last frame are the outliers
    frames_list = [_frame(per_frame, ts=float(fi)) for fi in range(n_frames)]
    # Residuals: 33 obs per frame, all 0.3 except last-frame positions -2 and -1
    frame_res = [[0.3] * per_frame for _ in range(n_frames)]
    frame_res[-1][-1] = 50.0
    frame_res[-1][-2] = 50.0
    frames = {0: frames_list}
    residuals = {0: frame_res}

    filtered, stats = filter_outliers(
        frames, residuals, board,
        absolute_threshold_px=2.0, percentile=99.0,
        frame_median_threshold_px=2.0, frame_median_percentile=99.0)

    s = stats[0]
    # 2 outlier observations should be dropped
    assert s.n_observations_before == n_frames * per_frame
    # Frame-level: the outlier frame has 2 bad out of 33 → median is still 0.3 → frame kept
    assert s.n_frames_after == n_frames
    # Obs threshold should pick up the 2 huge residuals
    assert s.n_observations_after == s.n_observations_before - 2


def test_filter_drops_whole_frame_with_high_median():
    """A frame whose median residual is huge should be dropped entirely."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    # Three frames: two good, one terrible
    frames = {0: [_frame(35), _frame(35), _frame(35)]}
    residuals = {0: [
        [0.3] * 35,           # good
        [0.4] * 35,           # good
        [50.0] * 35,          # all-bad frame → median = 50
    ]}
    filtered, stats = filter_outliers(
        frames, residuals, board,
        absolute_threshold_px=2.0, percentile=99.0,
        frame_median_threshold_px=2.0)

    assert stats[0].n_frames_before == 3
    assert stats[0].n_frames_after == 2
    # Good frames untouched
    assert len(filtered[0]) == 2
    assert len(filtered[0][0].observations) == 35


def test_filter_drops_underdetected_frame():
    """If None-residuals leave <min_points_for_pose, drop the whole frame."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)  # min_points=4
    # Frame with 4 observations, 3 of which had projection failures (None)
    frame = _frame(4)
    frames = {0: [frame]}
    residuals = {0: [[0.3, None, None, None]]}
    filtered, stats = filter_outliers(
        frames, residuals, board,
        absolute_threshold_px=2.0, percentile=95.0)
    # None residuals are always dropped; only 1 obs remains → below min_points → frame dropped
    assert stats[0].n_frames_after == 0
    assert len(filtered[0]) == 0


def test_filter_none_residuals_dropped():
    """Observations with projection failure (None residual) should be dropped."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    frames = {0: [_frame(35)]}
    residuals = {0: [[None] * 5 + [0.3] * 30]}
    filtered, stats = filter_outliers(
        frames, residuals, board,
        absolute_threshold_px=2.0, percentile=99.0)
    assert stats[0].n_observations_after == 30


def test_filter_preserves_good_data():
    """Clean residuals (<1 px) should not be pruned."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    frames = {0: [_frame(35), _frame(35)]}
    residuals = {0: [[0.5] * 35, [0.6] * 35]}
    filtered, stats = filter_outliers(
        frames, residuals, board,
        absolute_threshold_px=2.0, percentile=99.0)
    assert stats[0].n_frames_after == 2
    assert stats[0].n_observations_after == 70


def test_filter_threshold_floor():
    """Even if P99 is tiny, we never prune below absolute_threshold_px."""
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    # All residuals 0.1 px — P99 = 0.1, but floor is 2.0
    frames = {0: [_frame(35)]}
    residuals = {0: [[0.1] * 35]}
    filtered, stats = filter_outliers(
        frames, residuals, board,
        absolute_threshold_px=2.0, percentile=99.0)
    # threshold = max(2.0, 0.1) = 2.0 → nothing dropped
    assert stats[0].n_observations_after == 35
    assert stats[0].threshold_px == pytest.approx(2.0, abs=1e-9)


def test_board_pose_timeline_skips_missing_and_identity_poses():
    from types import SimpleNamespace

    import gtsam

    from mvis.calibration.outlier_rejection import board_pose_timeline

    good = gtsam.Pose3(gtsam.Rot3.Rz(0.1), gtsam.Point3(0, 0, 1)).matrix()
    res = SimpleNamespace(board_pose_times=[0.0, 0.1, 0.2, 0.3], board_poses=[good, None, np.eye(4), 2 * good - good])
    idx, t, poses = board_pose_timeline(res)
    assert idx == [0, 3] and t.tolist() == [0.0, 0.3] and set(poses) == {0, 3}
    assert isinstance(poses[0], gtsam.Pose3) and np.allclose(poses[3].matrix(), good)
    idx, t, poses = board_pose_timeline(SimpleNamespace(board_pose_times=[0.0], board_poses=[None]))
    assert idx == [] and t.shape == (0,) and poses == {}
