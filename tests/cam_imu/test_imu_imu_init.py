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
"""I6 IMU-IMU initialization tests (C6a): direct gyro-to-gyro td + rotation
+ aux gyro bias against the simulator's AuxImu ground truth."""

import os

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.calibration.factors.lie_groups import log_so3
from mvis.calibration.cam_imu_init import initialize_imu_imu
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario, AuxImu

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                     "sim_trajectories", "calib_3D_v0.txt")
_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                       gtsam.Point3(0.06, -0.03, 0.02))
_TD_S = 0.004
_BG_S = np.array([-0.008, 0.015, -0.01])
_WIN = 40.0


def _aux_scenario():
    traj = load_trajectory(_TRAJ, dt=0.05)
    aux = AuxImu(T_IstoI=_T_IstoI, intr=ImuModel(model=RPNG),
                 b_g=_BG_S, b_a=np.array([0.02, -0.04, 0.03]), td=_TD_S)
    return make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=gtsam.Pose3(),
        imu_intr=ImuModel(model=RPNG),
        b_g=np.array([0.01, -0.02, 0.005]), b_a=np.array([0.05, -0.03, 0.02]),
        image_size=(640, 480), aux=aux)


@pytest.mark.parametrize("noisy", [False, True])
def test_imu_imu_init_recovers(noisy):
    """td / R_IstoI / b_g_s recovered from raw base+aux gyro streams; the aux
    stream runs at a DIFFERENT rate (real rigs mix 100-500 Hz IMUs)."""
    sc = _aux_scenario()
    t0 = sc.traj.start_time + 1.0
    t_b = imu_times(t0, t0 + _WIN, 400.0)
    t_a = imu_times(t0, t0 + _WIN, 250.0)        # different rate on purpose
    sb = sc.base_imu(t_b, seed=1)
    sa = sc.aux_imu(t_a, seed=2)
    w_b, w_a = sb.w_m, sa.w_m
    if noisy:
        rng = np.random.RandomState(3)
        w_b = w_b + 0.005 * rng.standard_normal(w_b.shape)
        w_a = w_a + 0.005 * rng.standard_normal(w_a.shape)

    seed = initialize_imu_imu(sb.t, w_b, sa.t, w_a,
                              b_g_base=np.array([0.01, -0.02, 0.005]))
    print(seed.report)
    assert seed.report["ok"], seed.report

    tol_td = (2e-4 if not noisy else 5e-4)
    tol_rot = np.deg2rad(0.1 if not noisy else 0.3)
    tol_bg = (1e-3 if not noisy else 4e-3)
    assert abs(seed.td - _TD_S) < tol_td, f"td {seed.td:+.5f} vs {_TD_S}"
    d_rot = np.linalg.norm(log_so3(
        _T_IstoI.rotation().matrix().T @ seed.T_IstoI.rotation().matrix()))
    assert d_rot < tol_rot, f"rot err {np.rad2deg(d_rot):.3f} deg"
    assert np.linalg.norm(seed.b_g_s - _BG_S) < tol_bg, \
        f"b_g_s {seed.b_g_s} vs {_BG_S}"
    # translation deliberately NOT estimated
    assert np.allclose(np.asarray(seed.T_IstoI.translation(), float), 0.0)
    assert seed.report["valley_ratio"] > 1.5
