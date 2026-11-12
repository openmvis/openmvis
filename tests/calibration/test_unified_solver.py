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
"""Tests for the unified GTSAM optimizer (solve()) and unified projection factor.

Tests:
- Unified solve() for single-camera (equivalent to old solve_single_camera)
- Unified solve() for multi-camera stereo (equivalent to old solve_multi_camera)
- Intrinsics actually get refined during optimization
- Unified factor evaluates correctly
- Pipeline end-to-end with synthetic data
"""

import numpy as np
import cv2
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions, _make_unified_factor
from mvis.calibration.factors.pose_interpolation import se3_interpolate as _se3_interpolate
from mvis.calibration.factors.camera_intrinsics import _cal_to_vector, _cal_from_vector
from mvis.core.types import Observation, FrameObservations


# ======================================================================
# Helpers
# ======================================================================

def _make_board():
    return Chessboard(rows=7, cols=5, square_size=0.03)


def _make_K(fx=500, fy=500, cx=320, cy=240):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _generate_single_cam_data(
    board, K, dist, num_frames=20, image_size=(640, 480),
    noise_sigma=0.5, seed=42,
):
    """Generate synthetic single-camera observations."""
    rng = np.random.RandomState(seed)
    board_pts = board.all_points_3d().astype(np.float32)
    frames = []

    attempts = 0
    while len(frames) < num_frames and attempts < num_frames * 5:
        attempts += 1
        rvec = rng.uniform(-0.4, 0.4, 3).astype(np.float64)
        tvec = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(0.5, 0.9),
        ], dtype=np.float64)

        pts, _ = cv2.projectPoints(board_pts, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 2)

        w, h = image_size
        if not np.all((pts > 20) & (pts < np.array([w - 20, h - 20]))):
            continue

        pts += rng.randn(*pts.shape) * noise_sigma

        observations = [
            Observation(point_id=i, pixel=pts[i], row=int(pts[i, 1]))
            for i in range(len(pts))
        ]
        frames.append(FrameObservations(
            camera_id=0, timestamp=float(len(frames)) * 0.1,
            observations=observations, image_size=image_size,
        ))

    return frames


def _generate_stereo_data(
    board, K0, dist0, K1, dist1, T_cam1_cam0,
    num_frames=20, image_size=(640, 480), noise_sigma=0.5, seed=42,
):
    """Generate synthetic stereo data with synchronized timestamps."""
    rng = np.random.RandomState(seed)
    board_pts = board.all_points_3d().astype(np.float32)
    frames0, frames1 = [], []

    attempts = 0
    while len(frames0) < num_frames and attempts < num_frames * 10:
        attempts += 1
        rvec = rng.uniform(-0.3, 0.3, 3).astype(np.float64)
        tvec = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(0.5, 0.8),
        ], dtype=np.float64)

        pts0, _ = cv2.projectPoints(board_pts, rvec, tvec, K0, dist0)
        pts0 = pts0.reshape(-1, 2)

        R0, _ = cv2.Rodrigues(rvec)
        T0 = np.eye(4); T0[:3, :3] = R0; T0[:3, 3] = tvec
        T1 = T_cam1_cam0 @ T0
        rvec1, _ = cv2.Rodrigues(T1[:3, :3])
        tvec1 = T1[:3, 3]

        pts1, _ = cv2.projectPoints(board_pts, rvec1, tvec1, K1, dist1)
        pts1 = pts1.reshape(-1, 2)

        w, h = image_size
        if not (np.all((pts0 > 20) & (pts0 < np.array([w - 20, h - 20])))
                and np.all((pts1 > 20) & (pts1 < np.array([w - 20, h - 20])))):
            continue

        pts0 += rng.randn(*pts0.shape) * noise_sigma
        pts1 += rng.randn(*pts1.shape) * noise_sigma

        ts = float(len(frames0)) * 0.1

        obs0 = [Observation(point_id=j, pixel=pts0[j], row=int(pts0[j, 1]))
                for j in range(len(pts0))]
        obs1 = [Observation(point_id=j, pixel=pts1[j], row=int(pts1[j, 1]))
                for j in range(len(pts1))]

        frames0.append(FrameObservations(
            camera_id=0, timestamp=ts, observations=obs0, image_size=image_size))
        frames1.append(FrameObservations(
            camera_id=1, timestamp=ts, observations=obs1, image_size=image_size))

    return frames0, frames1


# ======================================================================
# Test: SE3 interpolation helpers
# ======================================================================

class TestSE3Helpers:
    def test_cal_vector_roundtrip_radtan(self):
        cal = gtsam.Cal3DS2(500, 500, 0, 320, 240, -0.1, 0.01, 0.001, -0.001)
        vec = _cal_to_vector(cal)
        assert vec.shape == (8,)
        assert vec[0] == pytest.approx(500.0)
        assert vec[4] == pytest.approx(-0.1)
        cal2 = _cal_from_vector(vec, is_fisheye=False)
        assert cal2.fx() == pytest.approx(500.0)
        assert cal2.k1() == pytest.approx(-0.1)

    def test_cal_vector_roundtrip_fisheye(self):
        cal = gtsam.Cal3Fisheye(400, 400, 0, 300, 200, 0.01, -0.001, 0.0001, -0.00001)
        vec = _cal_to_vector(cal)
        cal2 = _cal_from_vector(vec, is_fisheye=True)
        assert cal2.fx() == pytest.approx(400.0)
        assert cal2.k1() == pytest.approx(0.01)

    def test_se3_interpolate_endpoints(self):
        p0 = gtsam.Pose3(gtsam.Rot3.Rx(0.1), np.array([1, 0, 0]))
        p1 = gtsam.Pose3(gtsam.Rot3.Rx(0.3), np.array([2, 0, 0]))
        # At lambda=0, should be p0
        r0 = _se3_interpolate(p0, p1, 0.0)
        np.testing.assert_allclose(r0.translation(), p0.translation(), atol=1e-10)
        # At lambda=1, should be p1
        r1 = _se3_interpolate(p0, p1, 1.0)
        np.testing.assert_allclose(r1.translation(), p1.translation(), atol=1e-10)

    def test_se3_interpolate_midpoint(self):
        p0 = gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 0]))
        p1 = gtsam.Pose3(gtsam.Rot3(), np.array([2, 0, 0]))
        mid = _se3_interpolate(p0, p1, 0.5)
        np.testing.assert_allclose(mid.translation(), [1, 0, 0], atol=1e-10)


# ======================================================================
# Test: Unified factor
# ======================================================================

class TestUnifiedFactor:
    def test_factor_creates_and_evaluates(self):
        """Test that unified factor can be created and evaluated without crash."""
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        bp0_key = gtsam.symbol("B", 0)
        bp1_key = gtsam.symbol("B", 1)
        lm_key = gtsam.symbol("L", 0)
        cal_key = gtsam.symbol("K", 0)

        board = _make_board()
        pt = board.point_3d(0)

        factor, _ = _make_unified_factor(
            measured=np.array([320.0, 240.0]),
            board_point=pt,
            pixel_row=240.0,
            image_height=480.0,
            time_meas=0.0,
            time_0=0.0,
            time_1=0.1,
            noise_model=noise,
            bp0_key=bp0_key, bp1_key=bp1_key,
            lm_key=lm_key, cal_key=cal_key,
        )

        # Build values
        values = gtsam.Values()
        pose = gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 0.5]))
        values.insert(bp0_key, pose)
        values.insert(bp1_key, pose)
        values.insert(lm_key, gtsam.Point3(pt[0], pt[1], pt[2]))
        values.insert(cal_key, np.array([500, 500, 320, 240, 0, 0, 0, 0], dtype=np.float64))

        # Evaluate — should not crash
        err = factor.error(values)
        assert np.isfinite(err)

    def test_factor_with_extrinsic(self):
        """Test that unified factor works with an extrinsic key."""
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        bp0_key = gtsam.symbol("B", 0)
        bp1_key = gtsam.symbol("B", 1)
        lm_key = gtsam.symbol("L", 0)
        cal_key = gtsam.symbol("K", 0)
        ext_key = gtsam.symbol("E", 1)

        board = _make_board()
        pt = board.point_3d(0)

        factor, _ = _make_unified_factor(
            measured=np.array([320.0, 240.0]),
            board_point=pt,
            pixel_row=240.0, image_height=480.0,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            noise_model=noise,
            bp0_key=bp0_key, bp1_key=bp1_key,
            lm_key=lm_key, cal_key=cal_key,
            ext_key=ext_key,
        )

        values = gtsam.Values()
        pose = gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 0.5]))
        values.insert(bp0_key, pose)
        values.insert(bp1_key, pose)
        values.insert(lm_key, gtsam.Point3(pt[0], pt[1], pt[2]))
        values.insert(cal_key, np.array([500, 500, 320, 240, 0, 0, 0, 0], dtype=np.float64))
        values.insert(ext_key, gtsam.Pose3(gtsam.Rot3(), np.array([0.1, 0, 0])))

        err = factor.error(values)
        assert np.isfinite(err)

    def test_factor_with_all_keys(self):
        """Test unified factor with extrinsic + td + tr keys."""
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        bp0_key = gtsam.symbol("B", 0)
        bp1_key = gtsam.symbol("B", 1)
        lm_key = gtsam.symbol("L", 0)
        cal_key = gtsam.symbol("K", 0)
        ext_key = gtsam.symbol("E", 1)
        td_key = gtsam.symbol("D", 1)
        tr_key = gtsam.symbol("R", 0)

        board = _make_board()
        pt = board.point_3d(0)

        factor, _ = _make_unified_factor(
            measured=np.array([320.0, 240.0]),
            board_point=pt,
            pixel_row=240.0, image_height=480.0,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            noise_model=noise,
            bp0_key=bp0_key, bp1_key=bp1_key,
            lm_key=lm_key, cal_key=cal_key,
            ext_key=ext_key, td_key=td_key, tr_key=tr_key,
        )

        values = gtsam.Values()
        pose = gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 0.5]))
        values.insert(bp0_key, pose)
        values.insert(bp1_key, pose)
        values.insert(lm_key, gtsam.Point3(pt[0], pt[1], pt[2]))
        values.insert(cal_key, np.array([500, 500, 320, 240, 0, 0, 0, 0], dtype=np.float64))
        values.insert(ext_key, gtsam.Pose3(gtsam.Rot3(), np.array([0.1, 0, 0])))
        values.insert(td_key, np.array([0.0]))
        values.insert(tr_key, np.array([0.01]))

        err = factor.error(values)
        assert np.isfinite(err)


# ======================================================================
# Test: Unified solve() — single camera
# ======================================================================

class TestUnifiedSolveSingleCamera:
    @pytest.fixture
    def setup(self):
        board = _make_board()
        K = _make_K()
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        frames = _generate_single_cam_data(board, K, dist, num_frames=20, noise_sigma=0.5)
        return board, K, dist, frames

    def test_solve_single_camera(self, setup):
        """Test unified solve() for single camera reduces error."""
        board, K_gt, dist_gt, frames = setup
        assert len(frames) >= 10

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "radtan")
        board_poses = initializer.init_board_poses(frames, intrinsics, dist, "radtan")

        opts = OptimizerOptions(max_iterations=50, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)

        result = optimizer.solve(
            all_frames={0: frames},
            all_intrinsics={0: intrinsics},
            all_dist_coeffs={0: dist},
            all_board_poses={0: board_poses},
            distortion_models={0: "radtan"},
            reference_camera=0,
            image_sizes={0: (640, 480)},
        )

        assert len(result.cameras) == 1
        assert result.overall_rms < 2.0
        assert len(result.board_poses) > 0

    def test_intrinsics_get_optimized(self, setup):
        """Test that intrinsics converge toward ground truth during optimization."""
        board, K_gt, dist_gt, frames = setup
        gt_fx = 500.0

        # Use vanishing point init (gives rough estimate)
        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "radtan")

        # Use GT intrinsics for PnP so board poses are good
        gt_intr = np.array([500.0, 500.0, 320.0, 240.0])
        gt_dist = np.array([-0.1, 0.01, 0.001, -0.001])
        board_poses = initializer.init_board_poses(frames, gt_intr, gt_dist, "radtan")

        opts = OptimizerOptions(max_iterations=15, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)

        result = optimizer.solve(
            all_frames={0: frames},
            all_intrinsics={0: intrinsics},  # start from vanishing point init
            all_dist_coeffs={0: dist},
            all_board_poses={0: board_poses},
            distortion_models={0: "radtan"},
            reference_camera=0,
            image_sizes={0: (640, 480)},
        )

        opt_fx = result.cameras[0].intrinsics[0]
        init_fx = intrinsics[0]
        # Optimized fx should be closer to GT than the init value
        assert abs(opt_fx - gt_fx) < abs(init_fx - gt_fx), \
            f"opt_fx={opt_fx:.1f} not closer to GT={gt_fx} than init={init_fx:.1f}"

    def test_backward_compat_solve_single_camera(self, setup):
        """Test that solve_single_camera wrapper still works."""
        board, K_gt, dist_gt, frames = setup

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(frames, (640, 480), "radtan")
        board_poses = initializer.init_board_poses(frames, intrinsics, dist, "radtan")

        opts = OptimizerOptions(max_iterations=50, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)

        result = optimizer.solve_single_camera(
            frames, intrinsics, dist, board_poses, "radtan", (640, 480))

        assert len(result.cameras) == 1
        assert result.overall_rms < 2.0


# ======================================================================
# Test: Unified solve() — multi camera (stereo)
# ======================================================================

class TestUnifiedSolveMultiCamera:
    @pytest.fixture
    def stereo_setup(self):
        board = _make_board()
        K0 = _make_K(500, 500, 320, 240)
        K1 = _make_K(480, 480, 310, 235)
        dist0 = np.array([-0.05, 0.005, 0.001, -0.001, 0.0])
        dist1 = np.array([-0.08, 0.01, 0.0, 0.0, 0.0])
        T_cam1_cam0 = np.eye(4)
        T_cam1_cam0[0, 3] = 0.1  # 10cm baseline
        return board, K0, dist0, K1, dist1, T_cam1_cam0

    def test_stereo_solve(self, stereo_setup):
        """Test unified solve() for stereo pair."""
        board, K0, dist0, K1, dist1, T_gt = stereo_setup
        frames0, frames1 = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_gt, num_frames=20, noise_sigma=0.3)
        assert len(frames0) >= 10

        initializer = Initializer(board)
        intr0, dc0 = initializer.init_intrinsics_opencv(frames0, (640, 480), "radtan")
        intr1, dc1 = initializer.init_intrinsics_opencv(frames1, (640, 480), "radtan")
        poses0 = initializer.init_board_poses(frames0, intr0, dc0, "radtan")
        poses1 = initializer.init_board_poses(frames1, intr1, dc1, "radtan")
        extrinsics = initializer.init_extrinsics(
            {0: frames0, 1: frames1}, {0: poses0, 1: poses1}, 0)
        assert 1 in extrinsics

        opts = OptimizerOptions(max_iterations=100, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)

        result = optimizer.solve(
            all_frames={0: frames0, 1: frames1},
            all_intrinsics={0: intr0, 1: intr1},
            all_dist_coeffs={0: dc0, 1: dc1},
            all_board_poses={0: poses0, 1: poses1},
            distortion_models={0: "radtan", 1: "radtan"},
            reference_camera=0,
            image_sizes={0: (640, 480), 1: (640, 480)},
            extrinsics=extrinsics,
        )

        assert len(result.cameras) == 2
        assert len(result.extrinsics) == 1
        assert result.overall_rms < 3.0

        # Extrinsic translation should be near [0.1, 0, 0]
        # Tolerance is generous because numerical Jacobians converge differently
        ext = result.extrinsics[0]
        t_err = np.linalg.norm(ext.translation - np.array([0.1, 0, 0]))
        assert t_err < 0.25, f"Translation error {t_err:.4f}"

    def test_stereo_jittered_stamps_use_synced_ref_pose(self, stereo_setup):
        """Synced stereo whose cam1 stamps trail cam0 by 3 ms, frames 250 ms apart
        (TUM-VI camera-only on record time, #32): with snap_sync_tolerance_s the
        cam1 frames use their synced ref pose and the fit sits at the noise floor;
        interpolating at the cam1 stamp mixes in the next pose and inflates it."""
        import dataclasses
        board, K0, dist0, K1, dist1, T_gt = stereo_setup
        frames0, frames1 = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_gt, num_frames=20, noise_sigma=0.3)
        frames0 = [dataclasses.replace(f, timestamp=0.25 * k) for k, f in enumerate(frames0)]
        frames1 = [dataclasses.replace(f, timestamp=0.25 * k) for k, f in enumerate(frames1)]
        init = Initializer(board)
        intr0, dc0 = init.init_intrinsics_opencv(frames0, (640, 480), "radtan")
        intr1, dc1 = init.init_intrinsics_opencv(frames1, (640, 480), "radtan")
        poses0 = init.init_board_poses(frames0, intr0, dc0, "radtan")
        poses1 = init.init_board_poses(frames1, intr1, dc1, "radtan")
        ext = init.init_extrinsics({0: frames0, 1: frames1}, {0: poses0, 1: poses1}, 0)
        synced1 = frames1
        # then cam1's stamps trail by 3 ms (init pairs frames by exact stamp)
        frames1 = [dataclasses.replace(f, timestamp=f.timestamp + 0.003) for f in frames1]

        def solve(fr1, tol):
            opt = Optimizer(board, OptimizerOptions(max_iterations=100, snap_sync_tolerance_s=tol))
            return opt.solve(
                all_frames={0: frames0, 1: fr1}, all_intrinsics={0: intr0, 1: intr1},
                all_dist_coeffs={0: dc0, 1: dc1}, all_board_poses={0: poses0, 1: poses1},
                distortion_models={0: "radtan", 1: "radtan"}, reference_camera=0,
                image_sizes={0: (640, 480), 1: (640, 480)}, extrinsics=ext)
        ideal = solve(synced1, 0.0)                         # identical stamps
        snapped, interp = solve(frames1, 0.008), solve(frames1, 0.0)
        assert snapped.snap_sync_tolerance_s == 0.008 and interp.snap_sync_tolerance_s == 0.0
        assert snapped.overall_rms == pytest.approx(ideal.overall_rms, rel=1e-9)
        np.testing.assert_allclose(snapped.extrinsics[0].translation,
                                   ideal.extrinsics[0].translation, atol=1e-12)
        assert interp.overall_rms > 1.3 * snapped.overall_rms
        t_err = np.linalg.norm(interp.extrinsics[0].translation - ideal.extrinsics[0].translation)
        assert t_err > 1e-3                                  # the jitter biases the extrinsic
        # outlier rejection re-projects with the result's rule: same cam1 residuals
        from mvis.calibration.outlier_rejection import compute_per_observation_residuals
        res = compute_per_observation_residuals(
            snapped, {0: frames0, 1: frames1}, board, {0: (640, 480), 1: (640, 480)})
        r1 = np.array([r for fr in res[1] for r in fr if r is not None])
        assert np.median(r1) == pytest.approx(snapped.reprojection_errors[1].median, rel=1e-6)
        # beyond the tolerance (e.g. unsynced cameras) nothing is snapped
        assert solve(frames1, 0.002).overall_rms == pytest.approx(interp.overall_rms, rel=1e-9)

    def test_stereo_ref_only_poses(self, stereo_setup):
        """Verify board poses come only from ref camera timestamps."""
        board, K0, dist0, K1, dist1, T_gt = stereo_setup
        frames0, frames1 = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_gt, num_frames=15, noise_sigma=0.3)

        initializer = Initializer(board)
        intr0, dc0 = initializer.init_intrinsics_opencv(frames0, (640, 480), "radtan")
        intr1, dc1 = initializer.init_intrinsics_opencv(frames1, (640, 480), "radtan")
        poses0 = initializer.init_board_poses(frames0, intr0, dc0, "radtan")
        poses1 = initializer.init_board_poses(frames1, intr1, dc1, "radtan")
        extrinsics = initializer.init_extrinsics(
            {0: frames0, 1: frames1}, {0: poses0, 1: poses1}, 0)

        opts = OptimizerOptions(max_iterations=50)
        optimizer = Optimizer(board, opts)

        result = optimizer.solve(
            all_frames={0: frames0, 1: frames1},
            all_intrinsics={0: intr0, 1: intr1},
            all_dist_coeffs={0: dc0, 1: dc1},
            all_board_poses={0: poses0, 1: poses1},
            distortion_models={0: "radtan", 1: "radtan"},
            reference_camera=0,
            image_sizes={0: (640, 480), 1: (640, 480)},
            extrinsics=extrinsics,
        )

        # Board poses should equal valid ref camera frames, not total frames
        n_ref_valid = sum(1 for p in poses0 if p is not None)
        assert len(result.board_poses) == n_ref_valid

    def test_backward_compat_solve_multi_camera(self, stereo_setup):
        """Test that solve_multi_camera wrapper still works."""
        board, K0, dist0, K1, dist1, T_gt = stereo_setup
        frames0, frames1 = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_gt, num_frames=15, noise_sigma=0.3)

        initializer = Initializer(board)
        intr0, dc0 = initializer.init_intrinsics_opencv(frames0, (640, 480), "radtan")
        intr1, dc1 = initializer.init_intrinsics_opencv(frames1, (640, 480), "radtan")
        poses0 = initializer.init_board_poses(frames0, intr0, dc0, "radtan")
        poses1 = initializer.init_board_poses(frames1, intr1, dc1, "radtan")
        extrinsics = initializer.init_extrinsics(
            {0: frames0, 1: frames1}, {0: poses0, 1: poses1}, 0)

        opts = OptimizerOptions(max_iterations=50)
        optimizer = Optimizer(board, opts)

        result = optimizer.solve_multi_camera(
            all_frames={0: frames0, 1: frames1},
            all_intrinsics={0: intr0, 1: intr1},
            all_dist_coeffs={0: dc0, 1: dc1},
            all_board_poses={0: poses0, 1: poses1},
            extrinsics=extrinsics,
            distortion_models={0: "radtan", 1: "radtan"},
            reference_camera=0,
            image_sizes={0: (640, 480), 1: (640, 480)},
        )

        assert len(result.cameras) == 2
        assert result.overall_rms < 3.0


# ======================================================================
# Test: CalibrationPipeline
# ======================================================================

class TestCalibrationPipeline:
    def test_single_camera_pipeline(self):
        """Test CalibrationPipeline with one camera."""
        from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig

        board = _make_board()
        K = _make_K()
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        frames = _generate_single_cam_data(board, K, dist, num_frames=20, noise_sigma=0.5)

        config = PipelineConfig(
            max_iterations=50, max_iterations_single=30,
            distortion_models={0: "radtan"},
            reference_camera=0, verbose=False,
        )
        pipeline = CalibrationPipeline(board, config)

        result = pipeline.run(
            all_frames={0: frames},
            image_sizes={0: (640, 480)},
        )

        assert result is not None
        assert len(result.cameras) == 1
        assert result.overall_rms < 2.0

    def test_stereo_pipeline(self):
        """Test CalibrationPipeline with stereo pair."""
        from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig

        board = _make_board()
        K0 = _make_K(500, 500, 320, 240)
        K1 = _make_K(480, 480, 310, 235)
        dist0 = np.array([-0.05, 0.005, 0.001, -0.001, 0.0])
        dist1 = np.array([-0.08, 0.01, 0.0, 0.0, 0.0])
        T = np.eye(4); T[0, 3] = 0.1

        frames0, frames1 = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T, num_frames=20, noise_sigma=0.3)

        config = PipelineConfig(
            max_iterations=50, max_iterations_single=30,
            distortion_models={0: "radtan", 1: "radtan"},
            reference_camera=0,
            verbose=False,
        )
        pipeline = CalibrationPipeline(board, config)

        result = pipeline.run(
            all_frames={0: frames0, 1: frames1},
            image_sizes={0: (640, 480), 1: (640, 480)},
        )

        assert result is not None
        assert len(result.cameras) == 2
        assert result.overall_rms < 3.0

    def test_stage2_rate_subsample_preserves_full_timeline(self):
        """Stage 2 may optimize at 5 Hz without thinning Stage 4's data.

        The optimized subset poses must be mapped back onto the original
        10 Hz frame timeline.  A low final RMS alone is insufficient here:
        silently pairing subset poses with full-rate frames can discard half
        the observations while still reporting a plausible residual.
        """
        from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig

        board = _make_board()
        K0 = _make_K(500, 500, 320, 240)
        K1 = _make_K(480, 480, 310, 235)
        dist0 = np.array([-0.05, 0.005, 0.001, -0.001, 0.0])
        dist1 = np.array([-0.08, 0.01, 0.0, 0.0, 0.0])
        T_cam1_cam0 = np.eye(4)
        T_cam1_cam0[0, 3] = 0.1

        frames0, frames1 = _generate_stereo_data(
            board, K0, dist0, K1, dist1, T_cam1_cam0,
            num_frames=20, noise_sigma=0.3)
        assert len(frames0) == len(frames1) == 20

        config = PipelineConfig(
            max_iterations=50,
            max_iterations_single=30,
            distortion_models={0: "radtan", 1: "radtan"},
            reference_camera=0,
            stage2_target_hz=5.0,
            stage2_pre_filter=False,
            corner_subsample="all",
            adaptive_subsample=False,
            rerun_after_outlier_rejection=False,
            verbose=False,
        )
        result = CalibrationPipeline(board, config).run(
            all_frames={0: frames0, 1: frames1},
            image_sizes={0: (640, 480), 1: (640, 480)},
        )

        expected_times = [frame.timestamp for frame in frames0]
        assert result is not None
        assert len(result.board_poses) == 20
        assert len(result.board_pose_times) == 20
        assert result.board_pose_times == pytest.approx(expected_times)

        errors_by_camera = {
            error.camera_id: error for error in result.reprojection_errors}
        assert set(errors_by_camera) == {0, 1}
        assert errors_by_camera[0].num_observations == 700
        assert errors_by_camera[1].num_observations == 700

        cameras_by_id = {camera.camera_id: camera for camera in result.cameras}
        assert cameras_by_id[0].fx == pytest.approx(500.0, abs=15.0)
        assert cameras_by_id[0].fy == pytest.approx(500.0, abs=15.0)
        assert cameras_by_id[1].fx == pytest.approx(480.0, abs=15.0)
        assert cameras_by_id[1].fy == pytest.approx(480.0, abs=15.0)

    def test_stage2_prefilter_uses_pose_direction_and_filters_atomically(self):
        """The radtan prefilter scores T_board_cam correctly and keeps tracks aligned."""
        from mvis.calibration.pipeline_cam import (
            CalibrationPipeline,
            PipelineConfig,
            PipelineState,
        )

        board = _make_board()
        K = _make_K(500, 500, 320, 240)
        intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
        dist = np.array([-0.1, 0.01, 0.001, -0.001])
        frames = _generate_single_cam_data(
            board, K, dist, num_frames=20, noise_sigma=0.1)
        poses = Initializer(board).init_board_poses(
            frames, intrinsics, dist, "radtan")
        assert len(frames) == 20
        assert all(pose is not None for pose in poses)

        bad_index = 7
        bad_timestamp = frames[bad_index].timestamp
        bad_frame = frames[bad_index]
        bad_observations = [
            Observation(
                point_id=observation.point_id,
                pixel=observation.pixel + np.array([40.0, -30.0]),
                timestamp=observation.timestamp,
                row=observation.row,
            )
            for observation in bad_frame.observations
        ]
        frames[bad_index] = FrameObservations(
            camera_id=bad_frame.camera_id,
            timestamp=bad_timestamp,
            observations=bad_observations,
            image_size=bad_frame.image_size,
        )

        expected_timestamps = [
            frame.timestamp for i, frame in enumerate(frames)
            if i != bad_index
        ]
        all_frames = {0: list(frames)}
        state = PipelineState(
            intrinsics={0: intrinsics.copy()},
            dist_coeffs={0: dist.copy()},
            board_poses={0: list(poses)},
            distortion_models={0: "radtan"},
        )
        config = PipelineConfig(
            distortion_models={0: "radtan"},
            reference_camera=0,
            stage2_pre_filter=True,
            stage2_pre_filter_min_threshold_px=2.0,
            stage2_pre_filter_max_drop_fraction=0.60,
            verbose=False,
        )
        pipeline = CalibrationPipeline(board, config)
        pipeline._stage2_apply_pre_filter(all_frames, [0], state)

        kept_timestamps = [frame.timestamp for frame in all_frames[0]]
        assert bad_timestamp not in kept_timestamps
        assert kept_timestamps == pytest.approx(expected_timestamps)
        assert len(all_frames[0]) == len(state.board_poses[0]) == 19

        board_points = board.all_points_3d().astype(np.float64)
        for frame, pose in zip(all_frames[0], state.board_poses[0]):
            assert pose is not None
            T_cam_board = pose.inverse()
            rvec, _ = cv2.Rodrigues(T_cam_board.rotation().matrix())
            tvec = np.asarray(T_cam_board.translation()).reshape(3, 1)
            point_ids = np.array(
                [observation.point_id for observation in frame.observations])
            measured = np.array(
                [observation.pixel for observation in frame.observations])
            projected, _ = cv2.projectPoints(
                board_points[point_ids], rvec, tvec, K, dist)
            residuals = np.linalg.norm(
                projected.reshape(-1, 2) - measured, axis=1)
            assert np.median(residuals) < 1.0

    def test_stage2_prefilter_passes_through_unscored_frames(self):
        """Frames without a Stage-1 pose cannot be scored and must be kept.

        The pre-filter drops only scored frames it judges bad; a frame whose
        PnP failed (pose None) stays in all_frames with its None pose at the
        same position, so Stage 3/4 and the ledger still see it.
        """
        from mvis.calibration.pipeline_cam import (
            CalibrationPipeline,
            PipelineConfig,
            PipelineState,
        )

        board = _make_board()
        K = _make_K(500, 500, 320, 240)
        intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
        dist = np.array([-0.1, 0.01, 0.001, -0.001])
        frames = _generate_single_cam_data(
            board, K, dist, num_frames=20, noise_sigma=0.1)
        poses = Initializer(board).init_board_poses(
            frames, intrinsics, dist, "radtan")
        assert all(pose is not None for pose in poses)

        bad_index, unscored_index = 7, 3
        bad_frame = frames[bad_index]
        frames[bad_index] = FrameObservations(
            camera_id=bad_frame.camera_id,
            timestamp=bad_frame.timestamp,
            observations=[
                Observation(
                    point_id=o.point_id,
                    pixel=o.pixel + np.array([40.0, -30.0]),
                    timestamp=o.timestamp,
                    row=o.row,
                )
                for o in bad_frame.observations
            ],
            image_size=bad_frame.image_size,
        )
        poses[unscored_index] = None
        unscored_timestamp = frames[unscored_index].timestamp

        all_frames = {0: list(frames)}
        state = PipelineState(
            intrinsics={0: intrinsics.copy()},
            dist_coeffs={0: dist.copy()},
            board_poses={0: list(poses)},
            distortion_models={0: "radtan"},
        )
        config = PipelineConfig(
            distortion_models={0: "radtan"},
            reference_camera=0,
            stage2_pre_filter=True,
            stage2_pre_filter_min_threshold_px=2.0,
            stage2_pre_filter_max_drop_fraction=0.60,
            verbose=False,
        )
        CalibrationPipeline(board, config)._stage2_apply_pre_filter(
            all_frames, [0], state)

        kept_timestamps = [frame.timestamp for frame in all_frames[0]]
        assert bad_frame.timestamp not in kept_timestamps
        assert unscored_timestamp in kept_timestamps
        assert len(all_frames[0]) == len(state.board_poses[0]) == 19
        position = kept_timestamps.index(unscored_timestamp)
        assert state.board_poses[0][position] is None
        assert all(pose is not None
                   for i, pose in enumerate(state.board_poses[0])
                   if i != position)
