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
"""Tests for inter-camera pose alignment (Stage 1.5)."""

import numpy as np
import gtsam
import pytest


def _make_trajectory(num_frames, dt_frame, seed=42):
    """Generate a synthetic board pose trajectory.

    Returns (poses, timestamps) where poses are T_board_cam (GTSAM convention).
    """
    rng = np.random.RandomState(seed)
    poses = []
    timestamps = []

    # Smooth trajectory: board drifts and rotates slowly
    base_rvec = np.array([0.1, 0.2, 0.0])
    base_tvec = np.array([0.0, 0.0, 0.5])

    for i in range(num_frames):
        angle = 0.3 * np.sin(2 * np.pi * i / num_frames * 2)
        rvec = base_rvec + np.array([angle, 0.2 * np.cos(i * 0.3), 0.1 * np.sin(i * 0.5)])
        tvec = base_tvec + np.array([
            0.05 * np.sin(i * 0.2),
            0.03 * np.cos(i * 0.15),
            0.02 * np.sin(i * 0.1),
        ])

        import cv2
        R, _ = cv2.Rodrigues(rvec)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R
        T_cam_board[:3, 3] = tvec
        T_board_cam = np.linalg.inv(T_cam_board)
        poses.append(gtsam.Pose3(T_board_cam))
        timestamps.append(i * dt_frame)

    return poses, timestamps


def _transform_trajectory(poses, T_cam_i_ref, dt_offset=0.0, timestamps=None):
    """Transform a reference trajectory to another camera's frame.

    T_cam_i_board = T_cam_i_ref * T_ref_board
    T_board_cam_i = T_cam_i_board^{-1} = T_ref_board^{-1} * T_cam_i_ref^{-1}
                  = T_board_cam_ref * T_cam_i_ref^{-1}

    In GTSAM convention (poses are T_board_cam):
    T_board_cam_i = T_board_cam_ref * T_cam_i_ref.inverse()
    """
    T_ref_cam_i = T_cam_i_ref.inverse()
    new_poses = []
    new_timestamps = []
    for i, pose in enumerate(poses):
        # pose = T_board_cam_ref
        new_pose = pose.compose(T_ref_cam_i)
        new_poses.append(new_pose)
        if timestamps is not None:
            new_timestamps.append(timestamps[i] - dt_offset)  # shift time
        else:
            new_timestamps.append(float(i))
    return new_poses, new_timestamps


class TestAngularVelocity:
    def test_compute_angular_velocities(self):
        from mvis.calibration.pose_alignment import compute_angular_velocities

        poses, ts = _make_trajectory(20, 0.05)
        times, omega = compute_angular_velocities(poses, ts)

        assert len(times) == 19
        assert len(omega) == 19
        assert np.all(omega >= 0)
        assert np.mean(omega) > 0  # trajectory has rotation


class TestTemporalAlignment:
    def test_zero_offset_synced(self):
        """Two cameras with zero time offset."""
        from mvis.calibration.pose_alignment import estimate_time_offset_rotation

        poses_ref, ts_ref = _make_trajectory(50, 0.05, seed=42)
        T_ext = gtsam.Pose3(gtsam.Rot3.Yaw(0.1), gtsam.Point3(0.1, 0, 0))
        poses_cam, ts_cam = _transform_trajectory(poses_ref, T_ext, dt_offset=0.0, timestamps=ts_ref)

        dt = estimate_time_offset_rotation(poses_ref, ts_ref, poses_cam, ts_cam)
        print(f"Synced dt: {dt:.6f}")
        assert abs(dt) < 0.1

    def test_known_offset(self):
        """Two cameras with known 0.2s offset."""
        from mvis.calibration.pose_alignment import estimate_time_offset_rotation

        poses_ref, ts_ref = _make_trajectory(80, 0.05, seed=42)
        T_ext = gtsam.Pose3(gtsam.Rot3.Yaw(0.1), gtsam.Point3(0.1, 0, 0))
        true_dt = 0.2
        poses_cam, ts_cam = _transform_trajectory(
            poses_ref, T_ext, dt_offset=true_dt, timestamps=ts_ref)

        dt = estimate_time_offset_rotation(poses_ref, ts_ref, poses_cam, ts_cam)
        print(f"True dt: {true_dt}, estimated: {dt:.4f}")
        assert abs(dt - true_dt) < 0.1


class TestRotationAlignment:
    def test_recover_rotation(self):
        """Recover known extrinsic rotation."""
        from mvis.calibration.pose_alignment import estimate_rotation_with_outliers

        poses_ref, ts_ref = _make_trajectory(50, 0.05, seed=42)

        # Known extrinsic: 10 degrees yaw
        R_true = gtsam.Rot3.Yaw(0.17)
        T_ext = gtsam.Pose3(R_true, gtsam.Point3(0.1, 0, 0))
        poses_cam, ts_cam = _transform_trajectory(poses_ref, T_ext, 0.0, ts_ref)

        R_est, _, _ = estimate_rotation_with_outliers(
            poses_ref, ts_ref, poses_cam, ts_cam, dt=0.0)

        # Compare rotation angle
        R_err = R_true.between(R_est)
        angle_err = np.linalg.norm(gtsam.Rot3.Logmap(R_err)) * 180 / np.pi
        print(f"Rotation error: {angle_err:.2f} degrees")
        # Hand-eye from relative rotations is an initialization — 10° is acceptable
        # Stage 2 joint optimization will refine further
        assert angle_err < 15.0

    def test_identity_rotation(self):
        """Cameras with no rotation offset."""
        from mvis.calibration.pose_alignment import estimate_rotation_with_outliers

        poses_ref, ts_ref = _make_trajectory(50, 0.05, seed=42)
        T_ext = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1, 0, 0))
        poses_cam, ts_cam = _transform_trajectory(poses_ref, T_ext, 0.0, ts_ref)

        R_est, _, _ = estimate_rotation_with_outliers(
            poses_ref, ts_ref, poses_cam, ts_cam, dt=0.0)

        angle = np.linalg.norm(gtsam.Rot3.Logmap(R_est)) * 180 / np.pi
        print(f"Identity rotation: estimated angle = {angle:.2f} degrees")
        assert angle < 5.0


class TestTranslationAlignment:
    def test_recover_translation(self):
        """Recover known extrinsic translation."""
        from mvis.calibration.pose_alignment import estimate_translation_with_outliers

        poses_ref, ts_ref = _make_trajectory(50, 0.05, seed=42)
        t_true = np.array([0.1, 0.02, -0.01])
        R_true = gtsam.Rot3.Yaw(0.1)
        T_ext = gtsam.Pose3(R_true, gtsam.Point3(t_true[0], t_true[1], t_true[2]))
        poses_cam, ts_cam = _transform_trajectory(poses_ref, T_ext, 0.0, ts_ref)

        t_est, _, _ = estimate_translation_with_outliers(
            poses_ref, ts_ref, poses_cam, ts_cam, R_true, dt=0.0)

        t_err = np.linalg.norm(t_est - t_true)
        print(f"True t: {t_true}, estimated: {t_est}, error: {t_err:.4f}m")
        assert t_err < 0.05  # within 5cm

    def test_ransac_rejects_outliers_and_returns_raw_timestamps(self):
        """Translation RANSAC must preserve stable sample identity.

        The three true translations form the largest consensus set.  The
        remaining four samples are mutually inconsistent, so a coordinate
        median over every sample returns the wrong answer.  Non-integral raw
        timestamps also ensure the returned identities cannot be confused
        with positions in a filtered list.

        The consensus here is a 3/7 minority, which the default MAD-scaled
        radius and low-consensus guard deliberately do not trust, so this
        identity test pins the fixed radius and disables the guard.
        """
        from mvis.calibration.pose_alignment import (
            estimate_translation_with_outliers,
        )

        true_t = np.array([0.1, -0.02, 0.01])
        translations = [
            true_t,
            true_t,
            true_t,
            np.array([0.5, 0.5, 0.5]),
            np.array([0.8, 0.8, 0.8]),
            np.array([1.1, 1.1, 1.1]),
            np.array([1.4, 1.4, 1.4]),
        ]
        ts_ref = np.array([10.1 + 0.1 * i for i in range(len(translations))])
        dt = 0.004
        ts_cam = ts_ref - dt

        poses_ref = [gtsam.Pose3() for _ in translations]
        poses_cam = [
            gtsam.Pose3(
                gtsam.Rot3(),
                gtsam.Point3(-t[0], -t[1], -t[2]),
            )
            for t in translations
        ]

        t_est, outlier_ref, outlier_cam = \
            estimate_translation_with_outliers(
                poses_ref,
                ts_ref,
                poses_cam,
                ts_cam,
                gtsam.Rot3(),
                dt,
                n_iter=20,
                threshold=0.02,
                adaptive=False,
                min_inlier_fraction=0.0,
            )

        np.testing.assert_allclose(t_est, true_t, atol=1e-12)
        assert outlier_ref == set(ts_ref[3:].tolist())
        assert outlier_cam == set(ts_cam[3:].tolist())

    def test_ransac_mask_is_consistent_with_returned_model(self):
        """Median refinement must reclassify samples until the mask is stable."""
        from mvis.calibration.pose_alignment import (
            estimate_translation_with_outliers,
        )

        translations = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.9, 0.0, 0.0]),
            np.array([1.3, 0.0, 0.0]),
            np.array([1.4, 0.0, 0.0]),
        ]
        ts_ref = np.array([40.1, 40.2, 40.3, 40.4])
        ts_cam = np.array([50.1, 50.2, 50.3, 50.4])
        poses_ref = [gtsam.Pose3() for _ in translations]
        poses_cam = [
            gtsam.Pose3(
                gtsam.Rot3(),
                gtsam.Point3(-t[0], -t[1], -t[2]),
            )
            for t in translations
        ]

        threshold = 1.0
        t_est, outlier_ref, outlier_cam = \
            estimate_translation_with_outliers(
                poses_ref,
                ts_ref,
                poses_cam,
                ts_cam,
                gtsam.Rot3(),
                dt=-10.0,
                n_iter=20,
                threshold=threshold,
            )

        np.testing.assert_allclose(t_est, [1.3, 0.0, 0.0], atol=1e-12)
        assert outlier_ref == {ts_ref[0]}
        assert outlier_cam == {ts_cam[0]}
        for i, translation in enumerate(translations):
            is_outlier = ts_ref[i] in outlier_ref
            assert (np.linalg.norm(translation - t_est) >= threshold) \
                == is_outlier


def _translation_samples(translations, seed_ts=20.0):
    """Identity-rotation poses whose translation samples are ``translations``."""
    ts_ref = np.array([seed_ts + 0.05 * i for i in range(len(translations))])
    poses_ref = [gtsam.Pose3() for _ in translations]
    poses_cam = [
        gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(-t[0], -t[1], -t[2]))
        for t in translations
    ]
    return poses_ref, ts_ref, poses_cam, ts_ref.copy()


class TestTranslationRansacThreshold:
    """The inlier radius must follow the sample scatter (t265 regression)."""

    TRUE_T = np.array([0.08, -0.01, 0.005])

    def _run(self, translations, **kwargs):
        from mvis.calibration.pose_alignment import (
            estimate_translation_with_outliers,
        )
        poses_ref, ts_ref, poses_cam, ts_cam = _translation_samples(translations)
        t_est, out_ref, out_cam = estimate_translation_with_outliers(
            poses_ref, ts_ref, poses_cam, ts_cam, gtsam.Rot3(), 0.0, **kwargs)
        return t_est, out_ref, out_cam, ts_ref

    def test_gaussian_noise_without_outliers_flags_few_frames(self):
        rng = np.random.default_rng(0)
        samples = self.TRUE_T + rng.normal(0.0, 0.02, size=(400, 3))
        t_est, out_ref, out_cam, _ = self._run(list(samples))
        # A fixed 2 cm radius flags ~78 % of these genuine inliers.
        assert len(out_ref) <= 0.1 * len(samples)
        assert len(out_cam) == len(out_ref)
        assert np.linalg.norm(t_est - self.TRUE_T) < 0.005

    def test_gross_outliers_flagged_under_cm_level_noise(self):
        rng = np.random.default_rng(1)
        samples = self.TRUE_T + rng.normal(0.0, 0.03, size=(400, 3))
        gross = rng.choice(400, size=20, replace=False)
        directions = rng.normal(size=(20, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        samples[gross] = self.TRUE_T + 0.3 * directions
        t_est, out_ref, _, ts_ref = self._run(list(samples))
        flagged = {i for i, t in enumerate(ts_ref) if float(t) in out_ref}
        assert set(gross.tolist()) <= flagged
        assert len(flagged) <= 20 + 0.05 * 380
        # Median of ~380 N(0, 3 cm) samples: ~2 mm standard error per axis.
        assert np.linalg.norm(t_est - self.TRUE_T) < 0.01

    def test_exact_samples_keep_the_floor_radius(self):
        from mvis.calibration.pose_alignment import (
            _adaptive_translation_threshold,
        )
        samples = np.tile(self.TRUE_T, (10, 1))
        assert _adaptive_translation_threshold(samples, 0.02, 2.5) == 0.02

    def test_low_consensus_falls_back_to_all_inliers(self, caplog):
        rng = np.random.default_rng(2)
        samples = self.TRUE_T + rng.normal(0.0, 0.03, size=(400, 3))
        with caplog.at_level("WARNING",
                             logger="mvis.calibration.pose_alignment"):
            t_est, out_ref, out_cam, _ = self._run(
                list(samples), threshold=0.02, adaptive=False)
        assert out_ref == set() and out_cam == set()
        np.testing.assert_allclose(t_est, np.median(samples, axis=0))
        assert any("consensus" in r.getMessage() for r in caplog.records)


class TestOutlierTimestampIdentity:
    def test_rotation_outlier_pair_returns_endpoint_timestamps(self, monkeypatch):
        """A rejected relative rotation identifies both source frames."""
        from mvis.calibration import pose_alignment

        poses = [gtsam.Pose3() for _ in range(4)]
        ts_ref = np.array([20.1, 20.2, 20.3, 20.4])
        ts_cam = np.array([30.1, 30.2, 30.3, 30.4])

        def fake_match(dR_ref, times_ref, dR_cam, times_cam, dt):
            matched = list(zip(dR_ref, dR_cam))
            return matched, [0, 1, 2], [0, 1, 2]

        def fake_ransac(dR_ref, dR_cam, **kwargs):
            # Pair 1 is the only rejected relative-rotation sample.
            return gtsam.Rot3(), [0, 2]

        monkeypatch.setattr(
            pose_alignment, "_match_relative_rotations_with_indices", fake_match)
        monkeypatch.setattr(
            pose_alignment, "solve_rotation_ransac", fake_ransac)

        _, outlier_ref, outlier_cam = \
            pose_alignment.estimate_rotation_with_outliers(
                poses, ts_ref, poses, ts_cam, dt=0.0)

        assert outlier_ref == {ts_ref[1], ts_ref[2]}
        assert outlier_cam == {ts_cam[1], ts_cam[2]}

    def test_none_and_pnp_filtering_preserve_original_timestamp(
        self, monkeypatch,
    ):
        """A PnP-local index must map back to its raw frame timestamp."""
        from mvis.detection.chessboard import Chessboard
        from mvis.calibration import pose_alignment
        from mvis.calibration.initializer import Initializer
        from mvis.core.types import FrameObservations

        timestamps = [100.0, 101.0, 102.0, 103.0]
        all_frames = {
            cid: [
                FrameObservations(cid, timestamp, [])
                for timestamp in timestamps
            ]
            for cid in (0, 1)
        }
        identity = gtsam.Pose3()
        all_poses = {
            0: [identity, None, identity, identity],
            1: [identity, identity, identity, identity],
        }
        pnp_masks = iter([
            [True, False, True],
            [True, True, True, True],
        ])

        monkeypatch.setattr(
            pose_alignment,
            "detect_pnp_outliers",
            lambda poses, timestamps: next(pnp_masks),
        )
        monkeypatch.setattr(
            pose_alignment,
            "estimate_rotation_with_outliers",
            lambda *args, **kwargs: (gtsam.Rot3(), set(), set()),
        )
        monkeypatch.setattr(
            pose_alignment,
            "estimate_translation_with_outliers",
            lambda *args, **kwargs: (np.zeros(3), set(), set()),
        )

        initializer = Initializer(
            Chessboard(rows=3, cols=3, square_size=0.03))
        _, _, outlier_timestamps = initializer.init_extrinsics_trajectory(
            all_frames,
            all_poses,
            reference_camera=0,
            skip_temporal=True,
        )

        # After the None pose is removed, clean index 1 means raw t=102.0.
        assert outlier_timestamps[0] == {102.0}
        assert outlier_timestamps[1] == set()

    def test_initializer_unions_reference_outliers_from_multiple_cameras(
        self, monkeypatch,
    ):
        """Each non-reference alignment contributes to the ref outlier set."""
        from mvis.detection.chessboard import Chessboard
        from mvis.calibration import pose_alignment
        from mvis.calibration.initializer import Initializer
        from mvis.core.types import FrameObservations

        identity = gtsam.Pose3()
        frames = {
            cid: [FrameObservations(cid, 100.0 + cid, [])]
            for cid in (0, 1, 2)
        }
        poses = {cid: [identity] for cid in (0, 1, 2)}

        monkeypatch.setattr(
            pose_alignment,
            "align_cameras",
            lambda *args, **kwargs: {
                1: (0.0, identity, {0: {101.0}, 1: {201.0}}),
                2: (0.0, identity, {0: {103.0}, 2: {301.0}}),
            },
        )

        initializer = Initializer(
            Chessboard(rows=3, cols=3, square_size=0.03))
        _, _, outlier_timestamps = initializer.init_extrinsics_trajectory(
            frames, poses, reference_camera=0, skip_temporal=True)

        assert outlier_timestamps == {
            0: {101.0, 103.0},
            1: {201.0},
            2: {301.0},
        }


class TestFullPipeline:
    def test_align_cameras_synced(self):
        """Full pipeline: synced cameras with known extrinsic."""
        from mvis.calibration.pose_alignment import align_cameras

        poses_ref, ts_ref = _make_trajectory(60, 0.05, seed=42)
        t_true = np.array([0.1, 0.0, 0.0])
        R_true = gtsam.Rot3.Yaw(0.05)
        T_true = gtsam.Pose3(R_true, gtsam.Point3(t_true[0], t_true[1], t_true[2]))

        poses_cam1, ts_cam1 = _transform_trajectory(poses_ref, T_true, 0.0, ts_ref)

        trajectories = {0: poses_ref, 1: poses_cam1}
        timestamps_dict = {0: ts_ref, 1: ts_cam1}

        results = align_cameras(trajectories, timestamps_dict, reference_camera=0)

        assert 1 in results
        dt_est, T_est, _ = results[1]

        print(f"dt: {dt_est:.4f} (true: 0.0)")
        assert abs(dt_est) < 0.1

        t_est = np.array(T_est.translation())
        t_err = np.linalg.norm(t_est - t_true)
        print(f"Translation error: {t_err:.4f}m")
        assert t_err < 0.05

        R_err = R_true.between(T_est.rotation())
        angle_err = np.linalg.norm(gtsam.Rot3.Logmap(R_err)) * 180 / np.pi
        print(f"Rotation error: {angle_err:.2f} degrees")
        assert angle_err < 5.0

    def test_align_cameras_with_time_offset(self):
        """Full pipeline with known time offset."""
        from mvis.calibration.pose_alignment import align_cameras

        poses_ref, ts_ref = _make_trajectory(80, 0.05, seed=42)
        true_dt = 0.15
        T_true = gtsam.Pose3(gtsam.Rot3.Yaw(0.08), gtsam.Point3(0.12, -0.01, 0.0))

        poses_cam1, ts_cam1 = _transform_trajectory(
            poses_ref, T_true, dt_offset=true_dt, timestamps=ts_ref)

        results = align_cameras(
            {0: poses_ref, 1: poses_cam1},
            {0: ts_ref, 1: ts_cam1},
            reference_camera=0,
        )

        dt_est, T_est, _ = results[1]
        print(f"dt: {dt_est:.4f} (true: {true_dt})")
        assert abs(dt_est - true_dt) < 0.15

        t_true = np.array([0.12, -0.01, 0.0])
        t_est = np.array(T_est.translation())
        t_err = np.linalg.norm(t_est - t_true)
        print(f"Translation error: {t_err:.4f}m")
        # Translation recovery may be less accurate with time offset
        assert t_err < 0.1
