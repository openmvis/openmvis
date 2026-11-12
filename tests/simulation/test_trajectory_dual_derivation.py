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
"""Multi-IMU dual-derivation integration test on a realistic trajectory.

Validates the aux-IMU generation by computing the aux readings two unrelated ways
on a real recorded trajectory (TUM-VI room handheld) and comparing:

  Method A (lever-arm): aux readings from the base-spline derivatives + the
    lever-arm algebra (aux_imu_true).
  Method B (independent aux trajectory): build the aux SE(3) trajectory directly
    and differentiate IT. For a cumulative B-spline, right-multiplying every
    control pose by a constant G gives EXACTLY spline(t)·G, so the aux trajectory
    T_ItoG(t)·T_IstoI is the spline with control poses P_i·T_IstoI; running the
    plain base-IMU generator on it derives w/a from the aux path, never touching
    the lever-arm formula.

The two derivations are algebraically identical (the cumulative-spline
right-translation property is exact), so they agree to a tight numerical
tolerance — the only difference is second-derivative round-off through the two
different control-pose sets, not a finite-difference step.
"""

import os

import numpy as np
import gtsam
import pytest

from mvis.imu import KALIBR, RPNG
from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.imu_measurements import simulate_imu, imu_times
from mvis.simulation.trajectory_io import load_trajectory
from tests.helpers.imu_fixtures import intr_full as _intr

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tum_room_segment.txt")
_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])


def _extrinsic():
    R = gtsam.Rot3.Expmap(np.array([0.05, -0.03, 0.08])).matrix()
    return R, np.array([0.10, -0.05, 0.02])


def _G(R, p):
    T = np.eye(4); T[0:3, 0:3] = R; T[0:3, 3] = p
    return T


def test_load_trajectory_smoke():
    traj = load_trajectory(_TRAJ)
    assert traj.end_time - traj.start_time > 10.0
    R_GtoI, p = traj.get_pose(traj.start_time + 5.0)
    assert R_GtoI.shape == (3, 3)
    assert abs(np.linalg.det(R_GtoI) - 1.0) < 1e-9      # valid rotation


@pytest.mark.parametrize("model", [KALIBR, RPNG])
@pytest.mark.parametrize("td", [0.0, 0.011])
def test_aux_lever_arm_matches_independent_trajectory(model, td):
    base = load_trajectory(_TRAJ, dt=0.02)        # smooth control spacing
    R_ext, p_ext = _extrinsic()
    intr = _intr(model)
    t0 = base.start_time + 3.0
    times = imu_times(t0, t0 + 1.0, 200.0)

    # Method A: lever-arm on the base spline
    A = simulate_imu(base, times, intr, R_IstoI=R_ext, p_IsinI=p_ext, td=td,
                     b_g0=_BG, b_a0=_BA)
    # Method B: independent aux spline (control poses · T_IstoI), plain generation
    aux = BsplineSE3(base.t0, base.dt, [P @ _G(R_ext, p_ext) for P in base.ctrl])
    B = simulate_imu(aux, times, intr, td=td, b_g0=_BG, b_a0=_BA)

    assert np.allclose(A.w_m, B.w_m, rtol=1e-3, atol=1e-4), np.max(np.abs(A.w_m - B.w_m))
    assert np.allclose(A.a_m, B.a_m, rtol=1e-3, atol=2e-3), np.max(np.abs(A.a_m - B.a_m))
