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
"""Tests for vanishing point intrinsics initialization and temporal sync.

Tests:
- Circle fitting (_fit_circle)
- Circle intersection (_intersect_circles)
- Vanishing point focal length estimation (init_intrinsics_opencv)
- Temporal sync: angular velocity cross-correlation
"""

import numpy as np
import cv2
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.initializer import Initializer, _fit_circle, _intersect_circles
from mvis.calibration.pose_alignment import (
    compute_angular_velocities,
    estimate_time_offset_rotation,
)
from mvis.core.types import Observation, FrameObservations


# ======================================================================
# Helper: generate synthetic observations with known intrinsics
# ======================================================================

def _make_frames(board, K, dist, num_frames=20, image_size=(640, 480),
                 noise_sigma=0.3, seed=42):
    rng = np.random.RandomState(seed)
    board_pts = board.all_points_3d().astype(np.float32)
    frames = []
    attempts = 0
    while len(frames) < num_frames and attempts < num_frames * 5:
        attempts += 1
        rvec = rng.uniform(-0.4, 0.4, 3).astype(np.float64)
        tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1),
                         rng.uniform(0.5, 0.9)], dtype=np.float64)
        pts, _ = cv2.projectPoints(board_pts, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 2)
        w, h = image_size
        if not np.all((pts > 20) & (pts < np.array([w - 20, h - 20]))):
            continue
        pts += rng.randn(*pts.shape) * noise_sigma
        obs = [Observation(point_id=i, pixel=pts[i], row=int(pts[i, 1]))
               for i in range(len(pts))]
        frames.append(FrameObservations(
            camera_id=0, timestamp=float(len(frames)) * 0.1,
            observations=obs, image_size=image_size))
    return frames


# ======================================================================
# Test: Circle fitting
# ======================================================================

class TestCircleFitting:
    def test_perfect_circle(self):
        """Points on a perfect circle should recover center and radius."""
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        cx, cy, r = 100.0, 200.0, 50.0
        pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
        result = _fit_circle(pts)
        assert result is not None
        fc, fy, fr = result
        assert abs(fc - cx) < 0.1
        assert abs(fy - cy) < 0.1
        assert abs(fr - r) < 0.1

    def test_noisy_circle(self):
        """Circle fitting should handle moderate noise."""
        rng = np.random.RandomState(42)
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        cx, cy, r = 300.0, 250.0, 100.0
        pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
        pts += rng.randn(*pts.shape) * 2.0  # 2px noise
        result = _fit_circle(pts)
        assert result is not None
        fc, fy, fr = result
        assert abs(fc - cx) < 10
        assert abs(fy - cy) < 10
        assert abs(fr - r) < 10

    def test_too_few_points(self):
        """Less than 3 points should return None."""
        pts = np.array([[0, 0], [1, 1]])
        assert _fit_circle(pts) is None

    def test_collinear_points(self):
        """Collinear points should return None or very large radius."""
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float64)
        result = _fit_circle(pts)
        # Collinear points produce degenerate circle (None or huge radius)
        if result is not None:
            assert result[2] > 1000  # radius should be enormous


# ======================================================================
# Test: Circle intersection
# ======================================================================

class TestCircleIntersection:
    def test_two_intersections(self):
        """Two overlapping circles should give 2 intersection points."""
        # Circles at (0,0) r=5 and (6,0) r=5 overlap
        pts = _intersect_circles(0, 0, 5, 6, 0, 5)
        assert len(pts) == 2
        # Intersection points should be symmetric about x-axis
        assert abs(pts[0][0] - pts[1][0]) < 0.01  # same x
        assert abs(pts[0][1] + pts[1][1]) < 0.01  # opposite y

    def test_no_intersection(self):
        """Separate circles should return empty list."""
        pts = _intersect_circles(0, 0, 1, 10, 0, 1)
        assert len(pts) == 0

    def test_contained_circles(self):
        """One circle inside another should return empty list."""
        pts = _intersect_circles(0, 0, 10, 1, 0, 2)
        assert len(pts) == 0

    def test_identical_circles(self):
        """Identical circles should return empty."""
        pts = _intersect_circles(0, 0, 5, 0, 0, 5)
        assert len(pts) == 0

    def test_known_intersection(self):
        """Test with known analytical solution."""
        # Two unit circles: (0,0) r=1 and (1,0) r=1
        # Intersect at (0.5, ±√3/2)
        pts = _intersect_circles(0, 0, 1, 1, 0, 1)
        assert len(pts) == 2
        for pt in pts:
            assert abs(pt[0] - 0.5) < 1e-10
            assert abs(abs(pt[1]) - np.sqrt(3) / 2) < 1e-10


# ======================================================================
# Test: Vanishing point intrinsics initialization
# ======================================================================

class TestVanishingPointInit:
    @pytest.fixture
    def setup(self):
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K_gt = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist_gt = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        return board, K_gt, dist_gt

    def test_radtan_uses_opencv(self, setup):
        """Radtan model should use OpenCV calibrateCamera, not VP."""
        board, K_gt, dist_gt = setup
        frames = _make_frames(board, K_gt, dist_gt, num_frames=20)

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "radtan")

        # OpenCV should give fx close to GT (500)
        assert abs(intrinsics[0] - 500) < 50
        assert abs(intrinsics[1] - 500) < 50

    def test_principal_point_reasonable(self, setup):
        board, K_gt, dist_gt = setup
        frames = _make_frames(board, K_gt, dist_gt, num_frames=10)

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "radtan")

        # cx/cy should be near image center (VP init) or near GT (OpenCV fallback)
        assert abs(intrinsics[2] - 320) < 30  # cx
        assert abs(intrinsics[3] - 240) < 30  # cy

    def test_distortion_finite(self, setup):
        board, K_gt, dist_gt = setup
        frames = _make_frames(board, K_gt, dist_gt, num_frames=10)

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "radtan")
        assert np.all(np.isfinite(dist))
        assert len(dist) == 4

    def test_equidistant_model(self, setup):
        """Vanishing point init should work for equidistant model too."""
        board, K_gt, dist_gt = setup
        frames = _make_frames(board, K_gt, dist_gt, num_frames=20)

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "equidistant")

        assert intrinsics[0] > 50
        np.testing.assert_array_equal(dist, np.zeros(4))

    def test_equidistant_single_frame(self, setup):
        """VP init should work with just one frame for equidistant."""
        board, K_gt, dist_gt = setup
        frames = _make_frames(board, K_gt, dist_gt, num_frames=5)
        frames = frames[:1]

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "equidistant")
        assert intrinsics[0] > 0  # at least positive (may use heuristic fallback)

    def test_empty_frames_raises(self, setup):
        board, _, _ = setup
        initializer = Initializer(board)
        with pytest.raises(ValueError, match="Need at least 1 frame"):
            initializer.init_intrinsics_opencv([], (640, 480), "equidistant")


# ======================================================================
# Test: Angular velocity computation
# ======================================================================

class TestAngularVelocity:
    def test_stationary_gives_zero(self):
        """Stationary camera should give zero angular velocity."""
        pose = gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 1]))
        poses = [pose] * 10
        timestamps = list(range(10))
        times, omega = compute_angular_velocities(poses, timestamps)
        assert len(omega) == 9
        np.testing.assert_allclose(omega, 0.0, atol=1e-10)

    def test_rotating_gives_nonzero(self):
        """Rotating camera should give nonzero angular velocity."""
        poses = []
        for i in range(20):
            r = gtsam.Rot3.Rz(i * 0.1)  # 0.1 rad/step rotation around z
            poses.append(gtsam.Pose3(r, np.array([0, 0, 1])))
        timestamps = [i * 0.1 for i in range(20)]

        times, omega = compute_angular_velocities(poses, timestamps)
        # omega should be approximately 1 rad/s (0.1 rad / 0.1 s)
        assert len(omega) == 19
        np.testing.assert_allclose(omega, 1.0, atol=0.1)


# ======================================================================
# Test: Temporal sync — cross-correlation
# ======================================================================

class TestTemporalSync:
    def _make_trajectory(self, n_frames=100, dt=0.1, time_offset=0.0):
        """Create a synthetic rotating trajectory with optional time offset.

        Uses a non-periodic chirp-like signal so cross-correlation has a
        unique peak at the correct offset.
        """
        rng = np.random.RandomState(123)
        poses = []
        timestamps = []
        # Pre-generate a smooth random angle trajectory
        # (same for all calls — deterministic from the time values)
        for i in range(n_frames):
            t = i * dt + time_offset
            # Non-periodic: chirp (increasing frequency) + exponential decay
            angle_z = 0.5 * np.sin(2 * np.pi * (0.2 * t + 0.05 * t * t)) * np.exp(-0.05 * t)
            angle_x = 0.3 * np.cos(2 * np.pi * (0.15 * t + 0.03 * t * t)) * np.exp(-0.03 * t)
            r = gtsam.Rot3.Rz(angle_z).compose(gtsam.Rot3.Rx(angle_x))
            poses.append(gtsam.Pose3(r, np.array([0, 0, 1])))
            timestamps.append(t)
        return poses, timestamps

    def test_zero_offset_synced(self):
        """Synced cameras should give dt ≈ 0."""
        poses_ref, ts_ref = self._make_trajectory(200, dt=0.05, time_offset=0.0)
        poses_cam, ts_cam = self._make_trajectory(200, dt=0.05, time_offset=0.0)

        dt = estimate_time_offset_rotation(poses_ref, ts_ref, poses_cam, ts_cam)
        assert abs(dt) < 0.15, f"Expected dt≈0, got {dt}"

    def test_correlation_peak_quality(self):
        """Cross-correlation peak should be strong (>0.3) for matching signals."""
        from mvis.calibration.pose_alignment import _median_filter

        # Two identical bursty signals — should give high peak correlation
        dt_frame = 0.05
        t = np.arange(0, 15, dt_frame)
        omega = np.zeros_like(t)
        omega[(t > 2) & (t < 4)] = 2.0
        omega[(t > 8) & (t < 10)] = 3.0
        omega += np.random.RandomState(42).randn(len(t)) * 0.1
        omega = np.maximum(omega, 0)

        s = _median_filter(omega, 3)
        s -= s.mean()
        norm = np.linalg.norm(s)
        if norm > 0:
            s /= norm
        corr = np.correlate(s, s, "full")
        # Autocorrelation peak should be 1.0
        assert corr.max() > 0.99

    def test_synced_cameras_give_near_zero(self):
        """Same-rate synced cameras should give dt≈0 (not a false positive)."""
        poses, ts = self._make_trajectory(200, dt=0.05, time_offset=0.0)
        dt = estimate_time_offset_rotation(poses, ts, poses, ts)
        assert abs(dt) < 0.1

    def test_weak_signal_returns_zero(self):
        """Very slow motion (near-zero angular velocity) should return 0."""
        poses = []
        for i in range(100):
            r = gtsam.Rot3.Rz(i * 0.001)  # very slow rotation
            poses.append(gtsam.Pose3(r, np.array([0, 0, 1])))
        ts = [i * 0.1 for i in range(100)]
        dt = estimate_time_offset_rotation(poses, ts, poses, ts)
        # Should either return 0 or a near-zero value
        assert abs(dt) < 0.5

    def test_no_motion_returns_zero(self):
        """Static cameras should return 0 (no signal to correlate)."""
        pose = gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 1]))
        poses = [pose] * 50
        ts = [i * 0.1 for i in range(50)]

        dt = estimate_time_offset_rotation(poses, ts, poses, ts)
        assert dt == 0.0

    def test_too_few_poses(self):
        """Too few poses should return 0."""
        pose = gtsam.Pose3()
        dt = estimate_time_offset_rotation([pose, pose], [0, 1], [pose, pose], [0, 1])
        assert dt == 0.0

    def test_max_expected_dt_clamp(self):
        """Should not return offsets beyond max_expected_dt."""
        poses_ref, ts_ref = self._make_trajectory(400, dt=0.05, time_offset=0.0)
        poses_cam, ts_cam = self._make_trajectory(400, dt=0.05, time_offset=2.0)

        # max_expected_dt=0.5, but true offset is 2.0 — should be clamped
        dt = estimate_time_offset_rotation(poses_ref, ts_ref, poses_cam, ts_cam,
                                            max_expected_dt=0.5)
        assert abs(dt) <= 0.5


# ======================================================================
# Test: VP init on a chessboard viewed by a fisheye (equidistant) lens
# ======================================================================

def _make_fisheye_frames(board, fx, k, num_frames=30, image_size=(640, 480), seed=0):
    """Equidistant (KB4) chessboard views; frames keep >= 80% of corners."""
    rng = np.random.RandomState(seed)
    pts3 = board.all_points_3d().astype(np.float64).reshape(-1, 1, 3)
    K = np.array([[fx, 0, 322.0], [0, fx + 1, 238.0], [0, 0, 1]], dtype=np.float64)
    w, h = image_size
    frames = []
    while len(frames) < num_frames:
        rvec = rng.uniform(-0.6, 0.6, 3).reshape(3, 1)
        tvec = np.array([[rng.uniform(-0.15, 0.15)], [rng.uniform(-0.1, 0.1)],
                         [rng.uniform(0.25, 0.45)]])
        uv, _ = cv2.fisheye.projectPoints(pts3, rvec, tvec, K, np.asarray(k, float))
        uv = uv.reshape(-1, 2)
        ok = (uv[:, 0] > 5) & (uv[:, 0] < w - 5) & (uv[:, 1] > 5) & (uv[:, 1] < h - 5)
        if ok.sum() < 0.8 * len(uv):
            continue
        uv = uv + rng.randn(*uv.shape) * 0.1
        obs = [Observation(point_id=i, pixel=uv[i]) for i in np.flatnonzero(ok)]
        frames.append(FrameObservations(camera_id=0, timestamp=0.1 * len(frames),
                                        observations=obs, image_size=image_size))
    return frames


@pytest.mark.parametrize("model", ["equidistant", "fov", "double_sphere", "eucm", "ucm", "kb4"])
def test_vp_init_chessboard_focal(model, caplog):
    """Chessboard aliases _tag_size (detection/chessboard.py), which made
    _is_tag_grid() True for it: the VP init grouped chessboard corners as tag
    rows (point_id // 4 ...), fit circles to non-collinear points and logged
    f ~ 14 px, so every non-radtan chessboard init fell back to 0.35 * w."""
    import logging
    board = Chessboard(rows=7, cols=9, square_size=0.04)
    frames = _make_fisheye_frames(board, 300.0, [0.02, -0.01, 0.003, -0.0005])
    with caplog.at_level(logging.INFO, logger="mvis.calibration.initializer"):
        intr, _ = Initializer(board).init_intrinsics_opencv(frames, (640, 480), model)
    assert "VP init failed" not in caplog.text
    assert abs(intr[0] - 300.0) < 0.2 * 300.0, intr
