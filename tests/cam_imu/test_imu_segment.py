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
"""ImuSegment (IMU-propagated vision factor, elp-stereo-imu-investigation E8):
the live propagation must reproduce the ACI preintegrator exactly on the same
slice, be continuous across sample checkpoints, be exactly reversible, and
carry first-derivative-exact bias / intrinsic Jacobians."""
import numpy as np
import pytest

from mvis.calibration.cam_imu_init import _slice_imu
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_segment import ImuSegment
from mvis.calibration.factors.lie_groups import log_so3
from mvis.imu import CALYX, KALIBR
from tests.helpers.imu_fixtures import intr_layout as _intr

_HZ = 200.0


def _signal(T=0.4, hz=_HZ):
    """Smooth, non-trivial body rate + specific force (rad/s, m/s^2)."""
    t = np.arange(0.0, T, 1.0 / hz) + 1000.0          # realistic big stamps
    s = t - t[0]
    w = np.column_stack([1.2 * np.sin(7 * s), -0.8 * np.cos(5 * s) + 0.3,
                         0.5 * np.sin(3 * s + 1)])
    a = np.column_stack([6.0 * np.sin(4 * s), 9.81 + 2 * np.cos(6 * s),
                         -3.0 * np.sin(5 * s + 0.4)])
    return t, w, a


_BG = np.array([0.003, -0.002, 0.001])
_BA = np.array([-0.13, 0.11, -0.3])


def _aci_reference(t, w, a, t0, t1, model, intr):
    pre = ACIPreintegrator(0.01, 1e-3, 0.02, 2e-3, model=model, avg_meas=True)
    pre.reset(_BG, _BA)
    pre.set_intrinsics(intr)
    ts, ws, as_ = _slice_imu(t, w, a, t0, t1)
    pre.feed_imu_interval(ts, ws, as_)
    return pre


@pytest.mark.parametrize("model", [CALYX, KALIBR])
@pytest.mark.parametrize("delta", [0.0374, 0.045, 0.0500])  # off-sample, off-sample, on-sample
def test_matches_aci_preintegrator(model, delta):
    t, w, a = _signal()
    intr = _intr(model)
    seg = ImuSegment(t, w, a, model)
    t0 = t[10] + 0.0021                              # anchor between samples
    P = seg.propagate(t0, [delta], _BG, _BA, intr)[0]
    pre = _aci_reference(t, w, a, t0, t0 + delta, model, intr)
    assert np.abs(P.D_R - pre.D_R).max() < 1e-12
    assert np.abs(P.D_v - pre.D_v).max() < 1e-12
    assert np.abs(P.D_p - pre.D_p).max() < 1e-12
    assert np.abs(P.J_R_b[:, 0:3] - pre.J_R_bg).max() < 1e-12
    assert np.abs(P.J_R_b[:, 3:6] - pre.J_R_ba).max() < 1e-12
    assert np.abs(P.J_p_b[:, 0:3] - pre.J_p_bg).max() < 1e-12
    assert np.abs(P.J_p_b[:, 3:6] - pre.J_p_ba).max() < 1e-12
    assert np.abs(P.J_R_in - pre.J_R_in).max() < 1e-12
    assert np.abs(P.J_p_in - pre.J_p_in).max() < 1e-12


def test_continuous_across_checkpoint():
    t, w, a = _signal()
    seg = ImuSegment(t, w, a, CALYX)
    t0 = t[10] + 0.0021
    d_cp = t[18] - t0                                 # a raw sample inside
    Pm, Pp = seg.propagate(t0, [d_cp - 1e-9, d_cp + 1e-9], _BG, _BA, _intr(CALYX))
    assert np.abs(log_so3(Pm.D_R.T @ Pp.D_R)).max() < 1e-8
    assert np.abs(Pm.D_p - Pp.D_p).max() < 1e-8        # ~v * 2e-9 s of real motion
    # the midpoint model's time derivative has a kink of O(|w̄ x w| dt) at a
    # checkpoint (Jr(w̄ dt) w vs w) — ~1e-4 relative at 200 Hz
    assert np.abs(Pm.dtheta_dDelta - Pp.dtheta_dDelta).max() < 1e-3 * np.abs(Pp.dtheta_dDelta).max()


def test_mixed_sign_batch_matches_single():
    t, w, a = _signal()
    seg = ImuSegment(t, w, a, CALYX)
    intr = _intr(CALYX)
    t0 = t[30] + 0.001
    ds = [-0.012, 0.03, -0.004, 0.0, 0.041]
    batch = seg.propagate(t0, ds, _BG, _BA, intr)
    for d, P in zip(ds, batch):
        Q = seg.propagate(t0, [d], _BG, _BA, intr)[0]
        assert np.abs(P.D_R - Q.D_R).max() < 1e-14
        assert np.abs(P.D_p - Q.D_p).max() < 1e-14
        assert np.abs(P.J_p_in - Q.J_p_in).max() < 1e-14
    assert np.allclose(batch[3].D_R, np.eye(3)) and np.allclose(batch[3].D_p, 0)


@pytest.mark.parametrize("model", [CALYX, KALIBR])
def test_time_reversal_recovers_state(model):
    """Propagate forward by Δ, then backward by −Δ from the result: the
    midpoint recurrence is exactly time-symmetric."""
    t, w, a = _signal()
    intr = _intr(model)
    seg = ImuSegment(t, w, a, model)
    g = np.array([0.0, 0.0, -9.81])
    R0 = np.eye(3); p0 = np.array([0.1, -0.2, 0.3]); v0 = np.array([0.4, 0.1, -0.2])
    t0 = t[20] + 0.0013
    d = 0.0377
    F = seg.propagate(t0, [d], _BG, _BA, intr)[0]
    R1 = R0 @ F.D_R
    p1 = p0 + v0 * d + 0.5 * g * d * d + R0 @ F.D_p
    v1 = v0 + g * d + R0 @ F.D_v
    B = seg.propagate(t0 + d, [-d], _BG, _BA, intr)[0]
    R0b = R1 @ B.D_R
    p0b = p1 + v1 * (-d) + 0.5 * g * d * d + R1 @ B.D_p
    v0b = v1 + g * (-d) + R1 @ B.D_v
    assert np.abs(log_so3(R0.T @ R0b)).max() < 1e-9
    assert np.abs(p0b - p0).max() < 1e-9
    assert np.abs(v0b - v0).max() < 1e-9


@pytest.mark.parametrize("model", [CALYX, KALIBR])
@pytest.mark.parametrize("delta", [0.0377, -0.0213])
def test_bias_and_intrinsic_jacobians_fd(model, delta):
    t, w, a = _signal()
    intr = _intr(model)
    seg = ImuSegment(t, w, a, model)
    t0 = t[25] + 0.0017
    P0 = seg.propagate(t0, [delta], _BG, _BA, intr)[0]
    eps = 1e-6

    def run(bg, ba, xi):
        P = seg.propagate(t0, [delta], bg, ba, intr.retract(xi))[0]
        return P.D_R, P.D_p
    # biases
    for col in range(6):
        d = np.zeros(6); d[col] = eps
        Rp, pp = run(_BG + d[0:3], _BA + d[3:6], np.zeros(24))
        Rm, pm = run(_BG - d[0:3], _BA - d[3:6], np.zeros(24))
        dth = log_so3(Rm.T @ Rp) / (2 * eps)          # right tangent
        dp = (pp - pm) / (2 * eps)
        assert np.abs(dth - P0.J_R_b[:, col]).max() < 1e-6, f"J_R_b col {col}"
        assert np.abs(dp - P0.J_p_b[:, col]).max() < 1e-6, f"J_p_b col {col}"
    # intrinsics (all 24 tangent coordinates; unused ones must be zero)
    for col in range(24):
        d = np.zeros(24); d[col] = eps
        Rp, pp = run(_BG, _BA, d)
        Rm, pm = run(_BG, _BA, -d)
        dth = log_so3(Rm.T @ Rp) / (2 * eps)
        dp = (pp - pm) / (2 * eps)
        assert np.abs(dth - P0.J_R_in[:, col]).max() < 1e-6, f"J_R_in col {col}"
        assert np.abs(dp - P0.J_p_in[:, col]).max() < 1e-6, f"J_p_in col {col}"


def test_covers():
    t, w, a = _signal()
    seg = ImuSegment(t, w, a, CALYX)
    assert seg.covers(t[0], t[-1])
    assert not seg.covers(t[0] - 1e-3, t[5])
    assert not seg.covers(t[5], t[-1] + 1e-3)
