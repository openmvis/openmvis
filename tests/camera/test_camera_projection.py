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
"""Tests for the shared camera projection used by both the unified factor
and the HTML report.

Key property: both call `project_observation`, so their per-observation
residuals must agree by construction. These tests pin that contract plus
a few basic sanity checks on `find_bracketing_poses`.
"""

import numpy as np
import pytest
import gtsam

from mvis.calibration.factors.camera_projection import (
    project_observation, find_bracketing_poses,
)


# ---------------------------------------------------------------------
# find_bracketing_poses
# ---------------------------------------------------------------------

class TestFindBracketingPoses:
    def _tl(self):
        ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        idxs = [10, 11, 12, 13, 14]  # arbitrary B-variable indices
        return ts, idxs

    def test_inside(self):
        ts, idxs = self._tl()
        bracket = find_bracketing_poses(0.15, ts, idxs)
        assert bracket is not None
        bp0, bp1, t0, t1 = bracket
        assert (t0, t1) == (0.1, 0.2)
        assert (bp0, bp1) == (11, 12)

    def test_before_window(self):
        ts, idxs = self._tl()
        assert find_bracketing_poses(-0.01, ts, idxs) is None

    def test_after_window(self):
        ts, idxs = self._tl()
        assert find_bracketing_poses(0.41, ts, idxs) is None

    def test_exact_match_uses_neighbor(self):
        ts, idxs = self._tl()
        bracket = find_bracketing_poses(0.2, ts, idxs)
        assert bracket is not None
        bp0, bp1, t0, t1 = bracket
        # exact match at index 2 → pairs with index 3
        assert (t0, t1) == (0.2, 0.3)
        assert (bp0, bp1) == (12, 13)

    def test_exact_match_last(self):
        ts, idxs = self._tl()
        bracket = find_bracketing_poses(0.4, ts, idxs)
        assert bracket is not None
        bp0, bp1, t0, t1 = bracket
        # exact match at last element → pairs with previous one
        assert (t0, t1) == (0.3, 0.4)
        assert (bp0, bp1) == (13, 14)


# ---------------------------------------------------------------------
# project_observation — numerical correctness
# ---------------------------------------------------------------------

def _identity_pose_behind_board(z=0.5):
    """T_board_cam with camera at z=-z in board frame → point at (+z) in cam frame."""
    return gtsam.Pose3(gtsam.Rot3(), np.array([0.0, 0.0, -z]))


class TestProjectObservation:
    def test_centered_point_ref_camera_radtan(self):
        pose = _identity_pose_behind_board(0.5)
        cal = np.array([500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])
        pix = project_observation(
            board_point=np.array([0.0, 0.0, 0.0]),
            bp0=pose, bp1=pose,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=False,
        )
        assert pix is not None
        assert pix[0] == pytest.approx(320.0, abs=1e-6)
        assert pix[1] == pytest.approx(240.0, abs=1e-6)

    def test_offcenter_point_ref_camera_radtan(self):
        pose = _identity_pose_behind_board(1.0)
        cal = np.array([400.0, 400.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])
        pix = project_observation(
            board_point=np.array([0.1, 0.0, 0.0]),
            bp0=pose, bp1=pose,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=False,
        )
        # f * X/Z + cx = 400 * 0.1/1.0 + 320 = 360
        assert pix[0] == pytest.approx(360.0, abs=1e-6)
        assert pix[1] == pytest.approx(240.0, abs=1e-6)

    def test_point_behind_camera_returns_none(self):
        # Camera at z=+0.5 in board frame → board point at z=0 is at z=-0.5 in cam
        pose = gtsam.Pose3(gtsam.Rot3(), np.array([0.0, 0.0, 0.5]))
        cal = np.array([500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])
        pix = project_observation(
            board_point=np.array([0.0, 0.0, 0.0]),
            bp0=pose, bp1=pose,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=False,
        )
        assert pix is None

    def test_extrinsic_composes_correctly(self):
        # Ref pose: camera at z=-0.5 in board frame.
        # T_ext = identity → result must match ref-camera projection.
        pose = _identity_pose_behind_board(0.5)
        cal = np.array([500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])
        pix_ref = project_observation(
            board_point=np.array([0.05, 0.0, 0.0]),
            bp0=pose, bp1=pose,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=False,
        )
        pix_ext = project_observation(
            board_point=np.array([0.05, 0.0, 0.0]),
            bp0=pose, bp1=pose,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=False,
            T_ext=gtsam.Pose3(),  # identity
        )
        np.testing.assert_allclose(pix_ref, pix_ext, atol=1e-9)

    def test_interpolation_at_lambda_zero_equals_bp0(self):
        bp0 = _identity_pose_behind_board(0.5)
        bp1 = _identity_pose_behind_board(1.0)   # further away
        cal = np.array([500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])
        pix0 = project_observation(
            board_point=np.array([0.05, 0.0, 0.0]),
            bp0=bp0, bp1=bp1,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=False,
        )
        # Compare with direct projection from bp0 (no interpolation, no extrinsic)
        pix_direct = project_observation(
            board_point=np.array([0.05, 0.0, 0.0]),
            bp0=bp0, bp1=bp0,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=False,
        )
        np.testing.assert_allclose(pix0, pix_direct, atol=1e-9)

    def test_fisheye_mode_runs(self):
        pose = _identity_pose_behind_board(0.5)
        cal = np.array([300.0, 300.0, 424.0, 400.0, 0.0, 0.0, 0.0, 0.0])
        pix = project_observation(
            board_point=np.array([0.05, 0.02, 0.0]),
            bp0=pose, bp1=pose,
            time_meas=0.0, time_0=0.0, time_1=0.1,
            cal_vec=cal, is_fisheye=True,
        )
        assert pix is not None
        assert np.all(np.isfinite(pix))


# ---------------------------------------------------------------------
# Factor residual matches shared projection
# ---------------------------------------------------------------------

def test_factor_residual_matches_shared_projection():
    """The unified factor's residual must equal `project_observation(...) - meas`.

    This pins the contract that the HTML report (which will call
    `project_observation` directly) sees the same per-observation error
    the optimizer minimized.
    """
    from mvis.calibration.optimizer import _make_unified_factor

    pose = _identity_pose_behind_board(0.5)
    bp0_key = gtsam.symbol("B", 0)
    bp1_key = gtsam.symbol("B", 1)
    lm_key = gtsam.symbol("L", 0)
    cal_key = gtsam.symbol("K", 0)

    cal = np.array([500.0, 500.0, 320.0, 240.0, -0.01, 0.001, 0.0, 0.0])
    board_pt = np.array([0.05, 0.03, 0.0])

    # Predicted via shared projection
    predicted = project_observation(
        board_point=board_pt,
        bp0=pose, bp1=pose,
        time_meas=0.5, time_0=0.5, time_1=0.6,
        cal_vec=cal, is_fisheye=False,
    )
    # Use a measurement that is NOT the predicted pixel so the residual is non-zero
    meas = predicted + np.array([1.5, -0.7])

    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    factor, _fn = _make_unified_factor(
        measured=meas, board_point=board_pt,
        pixel_row=0.0, image_height=480.0,
        time_meas=0.5, time_0=0.5, time_1=0.6,
        noise_model=noise,
        bp0_key=bp0_key, bp1_key=bp1_key,
        lm_key=lm_key, cal_key=cal_key,
        is_fisheye=False,
    )

    values = gtsam.Values()
    values.insert(bp0_key, pose)
    values.insert(bp1_key, pose)
    values.insert(lm_key, gtsam.Point3(board_pt[0], board_pt[1], board_pt[2]))
    values.insert(cal_key, cal)

    # factor.error is 0.5 * ||whitened_residual||^2; we just check residual norm
    # matches |predicted - meas| via the unwhitened noise (sigma=1).
    expected_residual = predicted - meas  # shape (2,)
    expected_err_half_sq = 0.5 * float(expected_residual @ expected_residual)
    assert factor.error(values) == pytest.approx(expected_err_half_sq, abs=1e-9)
