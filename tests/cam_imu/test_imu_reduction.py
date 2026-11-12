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
"""Independent cross-check: the multi-IMU factor must reduce to the single-IMU
factor when the extrinsic is identity, td = 0, and the aux IMU sees the same data.

This validates the multi factor's kinematic coupling + Jacobian wiring against the
single-IMU factor (which is itself FD-verified to be exact at consistent states),
without re-using either's finite differences — a non-circular check.
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import KALIBR, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.imu_factor_multi import ImuFactorACI_Multi_In_V
from tests.helpers.imu_fixtures import intr_full as _intr
from mvis.calibration.factors.lie_groups import exp_so3 as _exp

G = np.array([0.0, 0.0, -9.81])


def _data(seed=3):
    rng = np.random.RandomState(seed); n, dt = 15, 0.005; t = np.arange(n + 1) * dt
    w = np.array([[0.3 * np.sin(2 * x), 0.2 * np.cos(3 * x), 0.1 * x] for x in t]) + rng.randn(n + 1, 3) * 0.01
    a = np.array([[0.5 * np.cos(x), 9.81 + 0.3 * np.sin(x), 0.2 * x] for x in t]) + rng.randn(n + 1, 3) * 0.01
    return t, w, a


def _build_single(model):
    p = ACIPreintegrator(.01, .001, .02, .002, model=model, avg_meas=True)
    p.reset(np.array([.01, -.02, .005]), np.array([.05, -.03, .02]))
    p.set_intrinsics(_intr(model)); p.feed_imu_interval(*_data())
    return ImuFactorACI_In.from_preintegrator(p, G)


def _consistent(f, seed=222):
    rng = np.random.RandomState(seed)
    R0 = _exp(rng.randn(3) * 0.1); p0 = rng.randn(3) * 0.1; v0 = rng.randn(3) * 0.1
    bg0 = f.bg_init + rng.randn(3) * 0.002; ba0 = f.ba_init + rng.randn(3) * 0.01; xi = rng.randn(24) * 0.01
    dbg, dba = bg0 - f.bg_init, ba0 - f.ba_init
    dth = f.J_R_bg @ dbg + f.J_R_ba @ dba + f.J_R_in @ xi; Rb = _exp(dth); Dt = f.Dt
    R1 = R0 @ f.D_R @ Rb
    v1 = v0 + G * Dt + R0 @ (f.D_v + f.J_v_bg @ dbg + f.J_v_ba @ dba + f.J_v_in @ xi)
    p1 = p0 + v0 * Dt + 0.5 * G * Dt ** 2 + R0 @ (f.D_p + f.J_p_bg @ dbg + f.J_p_ba @ dba + f.J_p_in @ xi)
    return (gtsam.Pose3(gtsam.Rot3(R0), gtsam.Point3(*p0)),
            gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(*p1)),
            np.concatenate([v0, bg0, ba0]), np.concatenate([v1, bg0, ba0]), xi)


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_multi_reduces_to_single(model):
    single = _build_single(model)
    pm = ACIPreintegrator(.01, .001, .02, .002, model=model, avg_meas=True)
    pm.reset(np.array([.01, -.02, .005]), np.array([.05, -.03, .02]))
    pm.set_intrinsics(_intr(model)); pm.feed_imu_interval(*_data())
    multi = ImuFactorACI_Multi_In_V.from_preintegrator(pm, G, gtsam.Pose3(), 0.0)

    pose0, pose1, vb0, vb1, xi = _consistent(single)
    v0, bg0, ba0 = vb0[0:3], vb0[3:6], vb0[6:9]; v1 = vb1[0:3]
    # map to multi: base vb (only v used), aux vbs=[bg,ba,v], calib tangent=[0(7),xi]
    vb0_m = np.concatenate([v0, np.zeros(6)]); vb1_m = np.concatenate([v1, np.zeros(6)])
    vbs0 = np.concatenate([bg0, ba0, v0]); vbs1 = np.concatenate([bg0, ba0, v1])
    xic = np.concatenate([np.zeros(7), xi])

    e_s = single.whitened_error(pose0, pose1, vb0, vb1, xi)
    e_m = multi.whitened_error(pose0, pose1, vb0_m, vb1_m, vbs0, vbs1, xic)
    assert np.allclose(e_m, e_s, atol=1e-10)         # residual reduces exactly

    Hp0_s, Hp1_s, Hvb0_s, Hvb1_s, Hintr_s = single.error_and_jacobians(pose0, pose1, vb0, vb1, xi)[1]
    Hm = multi.error_and_jacobians(pose0, pose1, vb0_m, vb1_m, vbs0, vbs1, xic)[1]
    Hp0_m, Hp1_m, Hvb0_m, Hvb1_m, Hvbs0_m, Hvbs1_m, Hcal_m = Hm

    def rel(A, B):
        return np.max(np.abs(A - B)) / max(1.0, np.max(np.abs(A)), np.max(np.abs(B)))

    # EVERY shared block maps single -> multi (the multi factor generalizes single):
    assert rel(Hp0_m, Hp0_s) < 1e-7                  # base pose0
    assert rel(Hp1_m, Hp1_s) < 1e-7                  # base pose1
    assert rel(Hcal_m[:, 7:31], Hintr_s) < 1e-7      # calib intrinsic block == single intrinsics
    # aux vbs layout [bg,ba,v] is single vb [v,bg,ba] reordered:
    for Hvbs_m, Hvb_s in [(Hvbs0_m, Hvb0_s), (Hvbs1_m, Hvb1_s)]:
        ref = np.column_stack([Hvb_s[:, 3:6], Hvb_s[:, 6:9], Hvb_s[:, 0:3]])  # [bg,ba,v]
        assert rel(Hvbs_m, ref) < 1e-7
    # base-IMU velocity drops out at td=0 (enters only via v*td):
    assert np.max(np.abs(Hvb0_m)) < 1e-6
    assert np.max(np.abs(Hvb1_m)) < 1e-6
