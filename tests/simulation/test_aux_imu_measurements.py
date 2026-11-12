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
"""Auxiliary-IMU measurement generation (multi-IMU sim, Phase 2b).

Validates the lever-arm coupling + extrinsic + time-offset independently of the
factors: (1) identity extrinsic reduces to the base IMU; (2) aux preintegration
(true intrinsics + biases) reconstructs the aux trajectory endpoints; (3) a pure
time offset shifts the sampled trajectory by +td.
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import KALIBR, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.lie_groups import skew
from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.imu_measurements import (
    simulate_imu, true_imu, aux_imu_true, imu_times, GRAVITY,
)
from tests.helpers.imu_fixtures import (wiggle as _wiggle, intr_full as _intr)

_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])


def _traj(dt=0.05):
    return BsplineSE3.from_pose_fn(_wiggle, t0=0.0, t1=4.0, dt=dt)


def _extrinsic():
    R = gtsam.Rot3.Expmap(np.array([0.05, -0.03, 0.08])).matrix()
    p = np.array([0.10, -0.05, 0.02])
    return R, p


def _aux_state(traj, t, R_IstoI, p_IsinI):
    """Ground-truth aux IMU (R_IstoG, p_IsinG, v_IsinG) at time t via coupling."""
    R_GtoI, pos, w, v = traj.get_velocity(t)
    R_ItoG = R_GtoI.T
    R_IstoG = R_ItoG @ R_IstoI
    p_IsinG = pos + R_ItoG @ p_IsinI
    v_IsinG = v + R_ItoG @ skew(w) @ p_IsinI
    return R_IstoG, p_IsinG, v_IsinG


def _check_preint_against(pre, R0, p0, v0, R1, p1, v1, g):
    Dt = pre.Dt
    assert np.allclose(R0 @ pre.D_R, R1, atol=2e-4)
    assert np.allclose(v1, v0 + g * Dt + R0 @ pre.D_v, atol=5e-3)
    assert np.allclose(p1, p0 + v0 * Dt + 0.5 * g * Dt ** 2 + R0 @ pre.D_p, atol=1e-3)


# --------------------------------------------------------------------------
def test_aux_true_reduces_to_base_at_identity():
    traj = _traj()
    for t in (1.0, 2.0, 3.0):
        wb, ab = true_imu(traj, t)
        wa, aa = aux_imu_true(traj, t, np.eye(3), np.zeros(3))
        assert np.allclose(wb, wa, atol=1e-12)
        assert np.allclose(ab, aa, atol=1e-12)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_identity_extrinsic_stream_equals_base(model):
    traj = _traj(); intr = _intr(model)
    times = imu_times(1.0, 1.4, 400.0)
    base = simulate_imu(traj, times, intr, b_g0=_BG, b_a0=_BA)
    aux = simulate_imu(traj, times, intr, R_IstoI=np.eye(3), p_IsinI=np.zeros(3),
                       b_g0=_BG, b_a0=_BA)
    assert np.allclose(base.w_m, aux.w_m, atol=1e-12)
    assert np.allclose(base.a_m, aux.a_m, atol=1e-12)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_aux_preintegration_roundtrip(model):
    """Aux stream (extrinsic, lever arm) preintegrates to the aux endpoints."""
    traj = _traj(); intr = _intr(model)
    R_ext, p_ext = _extrinsic()
    times = imu_times(1.0, 1.5, 800.0)
    s = simulate_imu(traj, times, intr, R_IstoI=R_ext, p_IsinI=p_ext,
                     b_g0=_BG, b_a0=_BA)
    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
    pre.reset(_BG, _BA); pre.set_intrinsics(intr)
    pre.feed_imu_interval(s.t, s.w_m, s.a_m)

    R0, p0, v0 = _aux_state(traj, times[0], R_ext, p_ext)
    R1, p1, v1 = _aux_state(traj, times[-1], R_ext, p_ext)
    _check_preint_against(pre, R0, p0, v0, R1, p1, v1, GRAVITY)


@pytest.mark.parametrize("model", [RPNG])
def test_time_offset_shifts_trajectory(model):
    """Identity extrinsic + td: the aux stream samples the base trajectory at
    t + td, so its preintegration reconstructs the base states at [t0+td, t1+td]."""
    traj = _traj(); intr = _intr(model)
    td = 0.013
    times = imu_times(1.0, 1.5, 800.0)
    s = simulate_imu(traj, times, intr, td=td, b_g0=_BG, b_a0=_BA)
    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
    pre.reset(_BG, _BA); pre.set_intrinsics(intr)
    pre.feed_imu_interval(s.t, s.w_m, s.a_m)

    R0g, p0, _w0, v0 = traj.get_velocity(times[0] + td)
    R1g, p1, _w1, v1 = traj.get_velocity(times[-1] + td)
    _check_preint_against(pre, R0g.T, p0, v0, R1g.T, p1, v1, GRAVITY)
