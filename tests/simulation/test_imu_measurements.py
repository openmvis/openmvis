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
"""Synthetic IMU generation: exact inverse of the ACI correction, and a
preintegration round-trip that reconstructs the trajectory endpoints."""

import numpy as np
import pytest

from mvis.imu import KALIBR, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.imu_measurements import (
    simulate_imu, measure_imu, true_imu, imu_times, GRAVITY,
)
from tests.helpers.imu_fixtures import (wiggle as _wiggle, intr_full as _intr)


def _traj(dt=0.05):
    return BsplineSE3.from_pose_fn(_wiggle, t0=0.0, t1=4.0, dt=dt)


# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_measurement_inverts_aci_correction(model):
    """measure_imu then the ACI correction recovers (w_inI, a_inI) exactly."""
    intr = _intr(model)
    b_g = np.array([0.01, -0.02, 0.005])
    b_a = np.array([0.05, -0.03, 0.02])
    Da, Dw, Tg = intr.Da(), intr.Dw(), intr.Tg()
    rng = np.random.RandomState(0)
    for _ in range(20):
        w_inI = rng.randn(3) * 0.5
        a_inI = rng.randn(3) * 3.0
        w_m, a_m = measure_imu(w_inI, a_inI, intr, b_g, b_a)
        # ACI correction (factors/aci.py)
        a_inI_rec = intr.R_atoI @ Da @ (a_m - b_a)
        w_inw = w_m - b_g - Tg @ a_inI_rec
        w_inI_rec = intr.R_wtoI @ Dw @ w_inw
        assert np.allclose(a_inI_rec, a_inI, atol=1e-12)
        assert np.allclose(w_inI_rec, w_inI, atol=1e-12)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_preintegration_roundtrip(model):
    """Feed generated measurements (true intrinsics, true biases) to the ACI
    preintegrator; D_R/D_v/D_p must reconstruct the trajectory endpoints."""
    traj = _traj()
    intr = _intr(model)
    b_g = np.array([0.01, -0.02, 0.005])
    b_a = np.array([0.05, -0.03, 0.02])
    g = GRAVITY

    t0, t1 = 1.0, 1.5
    times = imu_times(t0, t1, rate=800.0)
    stream = simulate_imu(traj, times, intr, g_inG=g, b_g0=b_g, b_a0=b_a)

    pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
    pre.reset(b_g, b_a)
    pre.set_intrinsics(intr)
    pre.feed_imu_interval(stream.t, stream.w_m, stream.a_m)

    R0_GtoI, p0, _w0, v0 = traj.get_velocity(times[0])
    R1_GtoI, p1, _w1, v1 = traj.get_velocity(times[-1])
    R0 = R0_GtoI.T            # R_ItoG at t0
    R1 = R1_GtoI.T
    Dt = pre.Dt
    assert Dt == pytest.approx(times[-1] - times[0], abs=1e-9)

    # rotation: R_ItoG(t1) = R_ItoG(t0) @ D_R   (corrections vanish at truth)
    assert np.allclose(R0 @ pre.D_R, R1, atol=2e-4)
    # velocity / position with gravity
    v1_pred = v0 + g * Dt + R0 @ pre.D_v
    p1_pred = p0 + v0 * Dt + 0.5 * g * Dt ** 2 + R0 @ pre.D_p
    assert np.allclose(v1, v1_pred, atol=5e-3)
    assert np.allclose(p1, p1_pred, atol=1e-3)


def test_noise_and_bias_walk_shapes():
    traj = _traj()
    intr = _intr(RPNG)
    times = imu_times(0.5, 1.0, rate=200.0)
    s = simulate_imu(traj, times, intr, b_g0=np.zeros(3), b_a0=np.zeros(3),
                     sigma_w=1e-3, sigma_a=1e-2, sigma_bg=1e-4, sigma_ba=1e-3,
                     seed=1)
    n = len(times)
    assert s.w_m.shape == (n, 3) and s.a_m.shape == (n, 3)
    assert s.b_g.shape == (n, 3) and s.b_a.shape == (n, 3)
    # bias random walk actually moved
    assert not np.allclose(s.b_g[0], s.b_g[-1])
