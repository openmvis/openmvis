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
"""Calibration scenario assembly (simulator Phase 3c-i).

A scenario bundles a gravity-aligned trajectory + ground-truth calibration + a
board into a coherent multi-sensor dataset. Tests check internal consistency:
board visibility, camera reprojection-at-truth, and base/aux IMU preintegration
round-trips reconstruct the (aux) trajectory endpoints."""

import os

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.lie_groups import skew
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario, AuxImu

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tum_room_segment.txt")
_IMG = (640, 480)
_T0OFF = 3.0          # board placed 3 s into the segment
_DEPTH = 3.0          # at ~3 m the fixed board stays in the FOV across the window


def _cal():
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
    return _cal_from_vector(
        np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004]), False)


def _intr(seed_off=0):
    return ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.array([0.01, -0.02, 0.0, 0.005, 0.0, -0.01, 0.0, 0.008, 0.0]),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(), model=RPNG)


_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])


def _scenario(**over):
    traj = load_trajectory(_TRAJ, dt=0.05)
    board = Chessboard(7, 7, 0.10)
    T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                         gtsam.Point3(0.05, -0.02, 0.01))
    aux = AuxImu(
        T_IstoI=gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.05, -0.03, 0.08])),
                            gtsam.Point3(0.10, -0.05, 0.02)),
        intr=_intr(), b_g=_BG.copy(), b_a=_BA.copy(), td=0.0)
    kw = dict(t0=traj.start_time + _T0OFF, depth=_DEPTH, board=board, cal=_cal(),
              T_CtoI=T_CtoI, imu_intr=_intr(), b_g=_BG, b_a=_BA,
              image_size=_IMG, aux=aux)
    kw.update(over)
    return traj, make_scenario(traj, **kw)


def test_base_imu_measurement_noise():
    """CalibScenario.base_imu must inject IMU measurement noise when asked (it
    previously dropped it silently). Noise std should match sigma/sqrt(dt)."""
    _t0, sc0 = _scenario()                       # noise-free
    traj, scn = _scenario(sigma_w=0.01, sigma_a=0.05)
    t = imu_times(traj.start_time, traj.start_time + 10.0, 200.0)
    clean = sc0.base_imu(t, seed=0)
    noisy = scn.base_imu(t, seed=1)
    dt = 0.005
    g_std = float(np.std(noisy.w_m - clean.w_m))
    a_std = float(np.std(noisy.a_m - clean.a_m))
    assert g_std > 1e-3                           # noise actually present
    assert abs(g_std - 0.01 / np.sqrt(dt)) / (0.01 / np.sqrt(dt)) < 0.2
    assert abs(a_std - 0.05 / np.sqrt(dt)) / (0.05 / np.sqrt(dt)) < 0.2


def _check_preint(pre, R0, p0, v0, R1, p1, v1, g):
    Dt = pre.Dt
    assert np.allclose(R0 @ pre.D_R, R1, atol=3e-4)
    assert np.allclose(v1, v0 + g * Dt + R0 @ pre.D_v, atol=6e-3)
    assert np.allclose(p1, p0 + v0 * Dt + 0.5 * g * Dt ** 2 + R0 @ pre.D_p, atol=2e-3)


# --------------------------------------------------------------------------
def test_board_well_visible():
    _traj, sc = _scenario()
    times = imu_times(sc.traj.start_time + _T0OFF, sc.traj.start_time + _T0OFF + 4.0, 20.0)
    cam = sc.camera(times)
    N = sc.board.num_points()
    frac = np.mean([len(ids) >= N // 2 for ids in cam.ids])
    assert frac > 0.9, f"only {frac*100:.0f}% of frames see >= half the corners"


def test_camera_reprojection_at_truth():
    _traj, sc = _scenario()
    times = np.linspace(sc.traj.start_time + _T0OFF, sc.traj.start_time + _T0OFF + 4.0, 5)
    cam = sc.camera(times)
    pts = sc.points_inW
    for f, t in enumerate(times):
        R_GtoI, p = sc.traj.get_pose(t)              # td_cam = 0
        T_CtoW = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(sc.T_CtoI)
        gcam = gtsam.PinholeCameraCal3DS2(T_CtoW, sc.cal)
        assert len(cam.ids[f]) > 0
        for i, uv in zip(cam.ids[f], cam.uv[f]):
            assert np.allclose(uv, gcam.project(gtsam.Point3(*pts[i])), atol=1e-9)


def test_base_imu_roundtrip():
    _traj, sc = _scenario()
    t0 = sc.traj.start_time + 1.0
    times = imu_times(t0, t0 + 0.5, 800.0)
    s = sc.base_imu(times)
    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=RPNG, avg_meas=True)
    pre.reset(sc.b_g, sc.b_a); pre.set_intrinsics(sc.imu_intr)
    pre.feed_imu_interval(s.t, s.w_m, s.a_m)
    R0g, p0, _w, v0 = sc.traj.get_velocity(times[0])
    R1g, p1, _w1, v1 = sc.traj.get_velocity(times[-1])
    _check_preint(pre, R0g.T, p0, v0, R1g.T, p1, v1, sc.g_inW)


def test_aux_imu_roundtrip():
    _traj, sc = _scenario()
    t0 = sc.traj.start_time + 1.0
    times = imu_times(t0, t0 + 0.5, 800.0)
    s = sc.aux_imu(times)
    a = sc.aux
    R_ext = a.T_IstoI.rotation().matrix()
    p_ext = np.asarray(a.T_IstoI.translation(), float)
    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=RPNG, avg_meas=True)
    pre.reset(a.b_g, a.b_a); pre.set_intrinsics(a.intr)
    pre.feed_imu_interval(s.t, s.w_m, s.a_m)

    def aux_state(t):
        R_GtoI, pos, w, v = sc.traj.get_velocity(t)
        R_ItoG = R_GtoI.T
        return R_ItoG @ R_ext, pos + R_ItoG @ p_ext, v + R_ItoG @ skew(w) @ p_ext
    R0, p0, v0 = aux_state(times[0])
    R1, p1, v1 = aux_state(times[-1])
    _check_preint(pre, R0, p0, v0, R1, p1, v1, sc.g_inW)


def test_camera_td_shift():
    _traj, sc = _scenario(td_cam=0.01)
    t = sc.traj.start_time + 2.0
    cam = sc.camera([t])
    R_GtoI, p = sc.traj.get_pose(t + 0.01)
    T_CtoW = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(sc.T_CtoI)
    gcam = gtsam.PinholeCameraCal3DS2(T_CtoW, sc.cal)
    for i, uv in zip(cam.ids[0], cam.uv[0]):
        assert np.allclose(uv, gcam.project(gtsam.Point3(*sc.points_inW[i])), atol=1e-9)
