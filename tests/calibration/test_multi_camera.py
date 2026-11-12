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
"""Tests for multi-camera calibration with extrinsics."""

import numpy as np
import cv2
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.core.types import Observation, FrameObservations


def _generate_stereo_data(
    board: Chessboard,
    K0: np.ndarray,
    dist0: np.ndarray,
    K1: np.ndarray,
    dist1: np.ndarray,
    T_cam1_cam0: np.ndarray,
    num_frames: int = 20,
    image_size: tuple[int, int] = (640, 480),
    noise_sigma: float = 0.5,
    seed: int = 42,
):
    """Generate synthetic stereo observations.

    Camera 0 is the reference. T_cam1_cam0 transforms points from cam0 to cam1.

    Returns:
        (frames_cam0, frames_cam1, gt_rvecs, gt_tvecs)
    """
    rng = np.random.RandomState(seed)
    board_pts = board.all_points_3d().astype(np.float32)
    R_10 = T_cam1_cam0[:3, :3]
    t_10 = T_cam1_cam0[:3, 3]
    rvec_10, _ = cv2.Rodrigues(R_10)

    frames0 = []
    frames1 = []
    gt_rvecs = []
    gt_tvecs = []

    attempts = 0
    while len(frames0) < num_frames and attempts < num_frames * 10:
        attempts += 1
        rvec = rng.uniform(-0.3, 0.3, 3).astype(np.float64)
        tvec = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(0.5, 0.8),
        ], dtype=np.float64)

        # Project to cam0
        pts0, _ = cv2.projectPoints(board_pts, rvec, tvec, K0, dist0)
        pts0 = pts0.reshape(-1, 2)

        # Transform to cam1 frame: rvec1, tvec1 from the composed transform
        R0, _ = cv2.Rodrigues(rvec)
        T0 = np.eye(4)
        T0[:3, :3] = R0
        T0[:3, 3] = tvec
        T1 = T_cam1_cam0 @ T0  # T_cam1_board = T_cam1_cam0 * T_cam0_board
        rvec1, _ = cv2.Rodrigues(T1[:3, :3])
        tvec1 = T1[:3, 3]

        pts1, _ = cv2.projectPoints(board_pts, rvec1, tvec1, K1, dist1)
        pts1 = pts1.reshape(-1, 2)

        # Check all points visible in both cameras
        w, h = image_size
        valid0 = np.all((pts0 > 20) & (pts0 < np.array([w - 20, h - 20])), axis=1)
        valid1 = np.all((pts1 > 20) & (pts1 < np.array([w - 20, h - 20])), axis=1)

        if not (np.all(valid0) and np.all(valid1)):
            continue

        pts0 += rng.randn(*pts0.shape) * noise_sigma
        pts1 += rng.randn(*pts1.shape) * noise_sigma

        ts = float(len(frames0))

        obs0 = [Observation(point_id=j, pixel=pts0[j], row=int(pts0[j, 1]))
                 for j in range(len(pts0))]
        obs1 = [Observation(point_id=j, pixel=pts1[j], row=int(pts1[j, 1]))
                 for j in range(len(pts1))]

        frames0.append(FrameObservations(
            camera_id=0, timestamp=ts, observations=obs0, image_size=image_size))
        frames1.append(FrameObservations(
            camera_id=1, timestamp=ts, observations=obs1, image_size=image_size))
        gt_rvecs.append(rvec)
        gt_tvecs.append(tvec)

    return frames0, frames1, gt_rvecs, gt_tvecs


class TestMultiCameraOptimizer:
    @pytest.fixture
    def stereo_setup(self):
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K0 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        K1 = np.array([[480, 0, 310], [0, 480, 235], [0, 0, 1]], dtype=np.float64)
        dist0 = np.array([-0.05, 0.005, 0.001, -0.001, 0.0])
        dist1 = np.array([-0.08, 0.01, 0.0, 0.0, 0.0])

        # Ground truth extrinsic: cam1 is 10cm to the right of cam0
        T_cam1_cam0 = np.eye(4)
        T_cam1_cam0[0, 3] = 0.1  # 10cm baseline

        return board, K0, dist0, K1, dist1, T_cam1_cam0

    def test_stereo_calibration(self, stereo_setup):
        """Test stereo calibration recovers extrinsics."""
        board, K0, dist0, K1, dist1, T_gt = stereo_setup

        frames0, frames1, _, _ = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_gt,
            num_frames=20, noise_sigma=0.3,
        )
        assert len(frames0) >= 10, "Need at least 10 valid stereo frames"

        initializer = Initializer(board)

        # Init cam0
        intr0, dc0 = initializer.init_intrinsics_opencv(frames0, (640, 480), "radtan")
        poses0 = initializer.init_board_poses(frames0, intr0, dc0, "radtan")

        # Init cam1
        intr1, dc1 = initializer.init_intrinsics_opencv(frames1, (640, 480), "radtan")
        poses1 = initializer.init_board_poses(frames1, intr1, dc1, "radtan")

        # Init extrinsics
        all_frames = {0: frames0, 1: frames1}
        all_poses = {0: poses0, 1: poses1}
        extrinsics = initializer.init_extrinsics(all_frames, all_poses, reference_camera=0)

        assert 1 in extrinsics, "Should have extrinsic for camera 1"

        # Optimize
        opts = OptimizerOptions(max_iterations=100, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)
        result = optimizer.solve_multi_camera(
            all_frames=all_frames,
            all_intrinsics={0: intr0, 1: intr1},
            all_dist_coeffs={0: dc0, 1: dc1},
            all_board_poses=all_poses,
            extrinsics=extrinsics,
            distortion_models={0: "radtan", 1: "radtan"},
            reference_camera=0,
            image_sizes={0: (640, 480), 1: (640, 480)},
        )

        # Check results
        assert len(result.cameras) == 2
        assert len(result.extrinsics) == 1

        # Check extrinsic is close to ground truth
        ext = result.extrinsics[0]
        assert ext.from_camera == 1
        assert ext.to_camera == 0

        # Translation should be approximately [0.1, 0, 0]
        # Tolerance is generous because the unified solver uses numerical Jacobians
        t_err = np.linalg.norm(ext.translation - np.array([0.1, 0, 0]))
        assert t_err < 0.25, f"Translation error {t_err:.4f} too large (GT=[0.1, 0, 0])"

        # Reprojection error should be reasonable
        assert result.overall_rms < 3.0

    def test_extrinsic_init(self, stereo_setup):
        """Test extrinsic initialization from co-visible frames."""
        board, K0, dist0, K1, dist1, T_gt = stereo_setup

        frames0, frames1, _, _ = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_gt, num_frames=15, noise_sigma=0.3,
        )

        initializer = Initializer(board)
        intr0, dc0 = initializer.init_intrinsics_opencv(frames0, (640, 480), "radtan")
        intr1, dc1 = initializer.init_intrinsics_opencv(frames1, (640, 480), "radtan")
        poses0 = initializer.init_board_poses(frames0, intr0, dc0, "radtan")
        poses1 = initializer.init_board_poses(frames1, intr1, dc1, "radtan")

        extrinsics = initializer.init_extrinsics(
            {0: frames0, 1: frames1},
            {0: poses0, 1: poses1},
            reference_camera=0,
        )

        assert 1 in extrinsics
        t_init = extrinsics[1].translation()
        t_init_np = np.array([t_init[0], t_init[1], t_init[2]])
        # Should be roughly [0.1, 0, 0] but initialization may be noisy
        assert np.linalg.norm(t_init_np - np.array([0.1, 0, 0])) < 0.15

    @staticmethod
    def _unsynced_init(stereo_setup, shift_frac, keep_exact=(), ref_keep=None):
        """init_extrinsics with cam1 stamps shifted by shift_frac periods.

        keep_exact: cam1 frame indices left at their exact (synced) stamp.
        ref_keep: if given, reference poses outside this index set are None
        (PnP failures), while the reference frames themselves stay.
        Returns (extrinsic of cam1, synced-stamp extrinsic of cam1).
        """
        import dataclasses
        board, K0, dist0, K1, dist1, T_gt = stereo_setup
        frames0, frames1, _, _ = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_gt, num_frames=15, noise_sigma=0.3,
        )
        period = float(np.median(np.diff([f.timestamp for f in frames0])))
        shifted = [f if i in keep_exact else
                   dataclasses.replace(f, timestamp=f.timestamp + shift_frac * period)
                   for i, f in enumerate(frames1)]

        initializer = Initializer(board)
        intr0, dc0 = initializer.init_intrinsics_opencv(frames0, (640, 480), "radtan")
        intr1, dc1 = initializer.init_intrinsics_opencv(frames1, (640, 480), "radtan")
        poses0 = initializer.init_board_poses(frames0, intr0, dc0, "radtan")
        poses1 = initializer.init_board_poses(frames1, intr1, dc1, "radtan")
        if ref_keep is not None:
            poses0 = [p if i in ref_keep else None for i, p in enumerate(poses0)]
        synced = initializer.init_extrinsics(
            {0: frames0, 1: frames1}, {0: poses0, 1: poses1}, reference_camera=0)
        unsynced = initializer.init_extrinsics(
            {0: frames0, 1: shifted}, {0: poses0, 1: poses1}, reference_camera=0)
        return unsynced[1], synced[1]

    @pytest.mark.parametrize("shift_frac,paired", [
        (0.2, True), (-0.2, True), (0.45, True), (100.0, False), (-100.0, False)])
    def test_extrinsic_init_unsynced_stamps(self, stereo_setup, shift_frac, paired):
        """Separate devices never share an exact stamp: pair by nearest time
        within half a reference period; with nothing in tolerance, identity.

        Correct nearest-neighbour pairing reproduces the synced init exactly;
        a wrong neighbour (off by one frame) gives ~1.5 cm / 0.03 rad error,
        which a loose "err < 3 cm" bound would not catch.
        """
        ext, synced = self._unsynced_init(stereo_setup, shift_frac)
        if paired:
            assert np.allclose(ext.matrix(), synced.matrix(), atol=1e-9)
            t = np.array(ext.translation())
            assert np.linalg.norm(t - np.array([0.1, 0, 0])) < 0.012
            assert np.linalg.norm(gtsam.Rot3.Logmap(ext.rotation())) < 0.01
        else:
            assert np.allclose(ext.matrix(), np.eye(4))  # identity fallback

    def test_extrinsic_init_partial_sync_pairs_all_frames(self, stereo_setup, caplog):
        """One accidental exact-stamp collision must not leave a one-sample
        init: below MIN_EXACT_EXTRINSIC_PAIRS exact pairs, nearest-time
        pairing (which includes the exact pair) is used for every frame."""
        import logging
        with caplog.at_level(logging.INFO, logger="mvis.calibration.initializer"):
            ext, synced = self._unsynced_init(stereo_setup, 0.2, keep_exact={0})
        assert "extrinsic init from 15 nearest-time pairs" in caplog.text
        assert "1 exact co-visible stamps" in caplog.text
        assert np.allclose(ext.matrix(), synced.matrix(), atol=1e-9)

    def test_extrinsic_init_exact_pairs_above_threshold_skip_nearest(
            self, stereo_setup, caplog):
        """With >= MIN_EXACT_EXTRINSIC_PAIRS exact matches the camera counts
        as synced: only exact pairs are used, no nearest-time pairing."""
        import logging
        n = Initializer.MIN_EXACT_EXTRINSIC_PAIRS
        with caplog.at_level(logging.INFO, logger="mvis.calibration.initializer"):
            self._unsynced_init(stereo_setup, 0.2, keep_exact=set(range(n)))
        assert f"extrinsic init from {n} co-visible frames" in caplog.text
        assert "by nearest time" not in caplog.text

    def test_extrinsic_init_tolerance_ignores_reference_pnp_failures(
            self, stereo_setup, caplog):
        """Reference poses only on every 3rd frame (PnP failures): the
        tolerance stays half the reference frame period, taken from all
        reference stamps. With a +0.2 period shift only the 5 cam1 frames next
        to a posed reference frame pair; a tolerance from the posed stamps
        alone (1.5 periods) would pair all 15 with boards up to 1.2 periods
        apart."""
        import logging
        with caplog.at_level(logging.INFO, logger="mvis.calibration.initializer"):
            ext, synced = self._unsynced_init(
                stereo_setup, 0.2, ref_keep=set(range(0, 15, 3)))
        assert "extrinsic init from 5 nearest-time pairs" in caplog.text
        assert np.allclose(ext.matrix(), synced.matrix(), atol=1e-9)

    def test_extrinsic_init_large_offset_pairs_wrong_neighbour(self, stereo_setup):
        """Pinned limitation: only the offset modulo one period is bounded.
        A 3.3-period clock offset pairs each frame with the reference frame
        three periods away (non-identity, wrong init), not identity."""
        ext, synced = self._unsynced_init(stereo_setup, 3.3)
        assert not np.allclose(ext.matrix(), np.eye(4))
        assert not np.allclose(ext.matrix(), synced.matrix(), atol=1e-6)


def _poses(n):
    return [gtsam.Pose3(gtsam.Rot3.Rz(0.1 * k), gtsam.Point3(k, 2.0 * k, 0.5))
            for k in range(n)]


def _cam_frames(stamps):
    return [FrameObservations(camera_id=1, timestamp=t, observations=[],
                              image_size=(640, 480)) for t in stamps]


class TestNearestTimeEstimates:
    """Direct tests of Initializer._nearest_time_estimates (cam pose = identity,
    so each estimate equals the paired reference pose)."""

    @staticmethod
    def _run(ref_by_time, stamps, **kw):
        frames = _cam_frames(stamps)
        return Initializer._nearest_time_estimates(
            frames, [gtsam.Pose3()] * len(frames), ref_by_time, 1, 0, **kw)

    def test_picks_nearest_neighbour(self):
        P = _poses(3)
        ref = {0.0: P[0], 1.0: P[1], 2.0: P[2]}
        got = self._run(ref, [0.3, 0.8, 1.6, 0.5, -0.2, 2.2])
        want = [P[0], P[1], P[2], P[0], P[0], P[2]]  # 0.5 tie -> earlier frame
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g.equals(w, 1e-12)

    def test_tolerance_is_half_period(self):
        ref = dict(zip([0.0, 1.0, 2.0], _poses(3)))
        assert len(self._run(ref, [2.45])) == 1
        assert len(self._run(ref, [2.55])) == 0
        assert len(self._run(ref, [-0.55])) == 0

    def test_explicit_ref_period_overrides_posed_interval(self):
        # Posed reference stamps 0, 2, 4 (dropouts): the posed interval is 2,
        # but the true frame period is 1, so 4.9 must be rejected.
        ref = dict(zip([0.0, 2.0, 4.0], _poses(3)))
        assert len(self._run(ref, [4.9])) == 1          # tol from posed: 1.0
        assert len(self._run(ref, [4.9], ref_period=1.0)) == 0
        assert len(self._run(ref, [4.45], ref_period=1.0)) == 1

    def test_too_few_reference_stamps(self):
        P = _poses(1)
        assert self._run({0.0: P[0]}, [0.1]) == []
        assert self._run({}, [0.1]) == []
        assert self._run({0.0: P[0]}, [0.1], ref_period=0.0) == []

    def test_nanosecond_stamps(self):
        """The tolerance comes from the stamps, so any unit pairs the same."""
        P = _poses(3)
        ns = 33_333_333.0
        ref = {0.0: P[0], ns: P[1], 2 * ns: P[2]}
        got = self._run(ref, [0.2 * ns, 1.3 * ns, 2.6 * ns])
        assert len(got) == 2
        assert got[0].equals(P[0], 1e-12) and got[1].equals(P[1], 1e-12)

    def test_skips_frames_without_pose(self):
        P = _poses(2)
        frames = _cam_frames([0.1, 0.9])
        got = Initializer._nearest_time_estimates(
            frames, [None, gtsam.Pose3()], {0.0: P[0], 1.0: P[1]}, 1, 0)
        assert len(got) == 1 and got[0].equals(P[1], 1e-12)


def test_median_interval_ignores_duplicates():
    assert Initializer._median_interval([0.0, 1.0, 1.0, 2.0]) == 1.0
    assert Initializer._median_interval([3.0, 3.0]) is None
    assert Initializer._median_interval([]) is None
