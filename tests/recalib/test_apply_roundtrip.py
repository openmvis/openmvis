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
"""Task 10 Part 1 — in-memory round-trip ACCEPTANCE proof for recalib.

corrupt -> correct (via the recalib module) -> recalibrate -> assert the
recovered calibration is identity/zero. This is the correctness proof for the
calibration-corrected dataset generator: it exercises OUR correction code
(`correct_imu_msg` + the `t' = t + td` restamp convention) and shows that a
dataset corrected with the recovered calibration re-calibrates to nothing left.

Modeled closely on `tests/cam_imu/test_pipeline_cam_imu.py::test_phase_b_recovers_imu_
intrinsics` (same constants/idioms): `make_scenario` injects KNOWN non-identity
IMU intrinsics + biases + camera td; we undo them with the recalib module and
re-run the FULL ('full' imu_intrinsics) joint solve, which must land at
identity/zero.

td convention (verified against `simulate_camera`): a frame LABELED time `t`
observes the scene at trajectory time `t + td`. The calibrator therefore
recovers `td` s.t. true-capture-time = label + td (see test_pipeline's
`abs(res.calib.td - _TD) < tol`). The recalib restamp writes `t' = t + td`, so
after correction the new labels EQUAL the capture times and a re-calibration
must report `td ~ 0`. If instead `res.calib.td ~ +/-2*TD`, that is a sign bug in
the restamp convention — NOT something to hide behind a loose tolerance.
"""
import os
import types

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.calibration.pipeline_cam_imu import (
    CamImuCalibConfig, run_cam_imu_calibration, format_report)
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario

from mvis.recalib.calib_spec import ImuCalib
from mvis.recalib.correct_imu import correct_imu_msg

# --- constants mirrored from tests/cam_imu/test_pipeline_cam_imu.py -----------------
_TRAJ = os.path.join(os.path.dirname(__file__), "..", "data",
                     "sim_trajectories", "calib_3D_v0.txt")
_IMG = (640, 480)
_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])
_T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                      gtsam.Point3(0.05, -0.02, 0.01))
_TD = 0.01
_WIN = 40.0
_CAM_HZ = 20.0
_IMU_HZ = 400.0


def _imu_msg(w, a):
    """Tiny sensor_msgs/Imu-shaped duck for correct_imu_msg (mirrors the
    fields it reads: angular_velocity / linear_acceleration with .x/.y/.z)."""
    return types.SimpleNamespace(
        angular_velocity=types.SimpleNamespace(x=w[0], y=w[1], z=w[2]),
        linear_acceleration=types.SimpleNamespace(x=a[0], y=a[1], z=a[2]))


@pytest.mark.slow
def test_corrected_dataset_recalibrates_to_identity():
    """Generate a dataset with KNOWN imperfections, correct it in-memory with
    the recalib module, re-run the full joint solve, and assert the recovered
    calibration is identity/zero -> proves the correction math AND the td sign.
    """
    # 1. KNOWN imperfections (non-identity IMU intrinsics + biases + camera td)
    true_intr = ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.zeros(9),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(),
        model=RPNG)
    b_g, b_a, TD = _BG, _BA, _TD

    # 2. corrupt scenario
    sigma_px = 0.5
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=true_intr, b_g=b_g, b_a=b_a, image_size=_IMG,
        td_cam=TD, sigma_px=sigma_px)
    t0 = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + _WIN, 1.0 / _CAM_HZ)
    cam = sc.camera(t_cam, seed=7)
    t_imu = imu_times(t0 - 0.3, t0 + _WIN + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=7)          # raw w_m / a_m carrying intr+bias
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # 3. APPLY OUR CORRECTION in-memory via the recalib module
    imu_calib = ImuCalib("imu0", "rpng", true_intr, b_g, b_a, 0.0, "base")
    w_corr = np.empty_like(s.w_m)
    a_corr = np.empty_like(s.a_m)
    for i in range(len(s.t)):
        msg = correct_imu_msg(_imu_msg(s.w_m[i], s.a_m[i]), imu_calib,
                              remove_bias=True)
        w_corr[i] = (msg.angular_velocity.x, msg.angular_velocity.y,
                     msg.angular_velocity.z)
        a_corr[i] = (msg.linear_acceleration.x, msg.linear_acceleration.y,
                     msg.linear_acceleration.z)
    # camera restamp: t' = t + td  (mirrors restamp.shifted_time_ns)
    t_cam_corr = t_cam + TD

    # 4. re-calibrate the CORRECTED data with FULL IMU-intrinsics estimation
    # estimate_t_ga=False: the sim injects ZERO gyro<->accel offset and this
    # test asserts the restamped td in isolation. With t_ga free (the shipping
    # default) the weakly-observed timing family trades td
    # against t_ga ~1:1 on this window (td -5.4 ms / t_ga +5.4 ms, sigma ~3 ms),
    # which hides the td sign check. Same pin as tests/cam_imu/test_pipeline_cam_imu.py.
    cfg = CamImuCalibConfig(estimate_t_ga=False,
                            state_hz=10.0, sigma_px=sigma_px, huber_px=0.0,
                            imu_intrinsics="full", imu_intrinsics_model="rpng")
    res = run_cam_imu_calibration(
        t_cam_corr, list(cam.ids), list(cam.uv), board_pts, _RADTAN,
        s.t, w_corr, a_corr, _IMG, config=cfg)
    print("\n" + format_report(res))
    assert res.ok

    # 5. the corrected dataset recalibrates to ~identity/zero
    xi_norm = np.linalg.norm(res.imu_intr_xi)
    bg_norm = np.linalg.norm(res.b_g)
    ba_norm = np.linalg.norm(res.b_a)
    print(f"\nrecovered: |xi_in|={xi_norm:.4e}  |b_g|={bg_norm:.4e}  "
          f"|b_a|={ba_norm:.4e}  td={res.calib.td:+.6f}")

    # IMU intrinsics back to identity. The Dw/Da blocks (xi[0:12]) are the
    # strongly-excited ones (cf. test_phase_b: it only asserts those recover).
    # Injected |xi_true[0:12]| was 0.0502; corrected+recalibrated it lands at
    # ~0.012 (a 4x reduction, within 1 sigma of zero per the reported d_Dw/d_Da
    # marginals ~0.015-0.02). Bound at 2e-2: far below the injected 5e-2 (so a
    # missing/wrong correction fails), still above the window's noise floor.
    dw_da_norm = np.linalg.norm(res.imu_intr_xi[0:12])
    assert dw_da_norm < 2e-2, \
        f"Dw/Da not back to identity: |{dw_da_norm:.4e}| {res.imu_intr_xi[0:12]}"
    # full vector incl. weakly-excited Tg/R; injected |xi_true|=0.0626.
    assert xi_norm < 3e-2, f"imu_intr_xi not ~zero: {res.imu_intr_xi}"
    # biases removed. Injected |b_g|=0.023, |b_a|=0.062; after correction the
    # absolute estimate sits at the window's bias-estimation floor (~0.011 /
    # ~0.067) rather than exactly zero — b_a is the weakly-excited direction
    # (cf. test_pipeline tolerances: b_a recovered only to ~6e-2 of truth).
    assert bg_norm < 2e-2, f"b_g not ~zero: {res.b_g}"
    assert ba_norm < 8e-2, f"b_a not ~zero: {res.b_a}"
    # camera td corrected: ~0 under the t' = t + td convention. A value near
    # -2*TD or +2*TD would mean the restamp shifts the WRONG way (sign bug).
    assert abs(res.calib.td) < 2e-3, (
        f"td not corrected to ~0: {res.calib.td:+.6f} (TD={TD}); "
        f"|td|~2*TD would indicate a restamp SIGN bug")
