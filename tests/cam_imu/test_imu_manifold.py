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
"""ImuModel / ImuCalib as Python manifolds: axioms + gtsam optimization.

Validates the two manifold contracts:
  local(a, retract(a, ξ)) == ξ          (retract/local are inverse)
  retract(a, local(a, b)) == b          (round-trip through tangent)
and that an ImuCalib variable, stored as its 31-dim tangent Vector around a base,
optimizes to a target inside a gtsam factor graph.
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import ImuModel, KALIBR, RPNG, CALYX
from mvis.calibration.factors.imu_calib import ImuCalib


def _rand_intr(rng, model):
    if model == CALYX:
        # full 3x3 gyro correction (9), no rotation DOF (R ≡ I)
        x_dw = (np.eye(3) + rng.randn(3, 3) * 0.03).flatten()
        R = None
    else:
        x_dw = 1 + rng.randn(6) * 0.05
        R = gtsam.Rot3.Expmap(rng.randn(3) * 0.1).matrix()
    return ImuModel(
        x_dw=x_dw, x_da=1 + rng.randn(6) * 0.05,
        x_tg=rng.randn(9) * 0.02, R=R, model=model)


def _rand_calib(rng, model):
    T = gtsam.Pose3(gtsam.Rot3.Expmap(rng.randn(3) * 0.1), gtsam.Point3(*rng.randn(3) * 0.1))
    return ImuCalib(T, rng.randn() * 0.01, _rand_intr(rng, model))


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_intrinsics_manifold_axioms(model):
    rng = np.random.RandomState(0)
    a = _rand_intr(rng, model)
    xi = rng.randn(24) * 0.05
    assert np.allclose(a.local_coordinates(a.retract(xi)), xi, atol=1e-9)
    b = _rand_intr(rng, model)
    assert np.allclose(a.retract(a.local_coordinates(b)).x21, b.x21, atol=1e-9)
    assert np.allclose(a.retract(a.local_coordinates(b)).R, b.R, atol=1e-9)


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_calib_manifold_axioms(model):
    rng = np.random.RandomState(1)
    a = _rand_calib(rng, model)
    xi = rng.randn(31) * 0.05
    assert np.allclose(a.local_coordinates(a.retract(xi)), xi, atol=1e-9)
    b = _rand_calib(rng, model)
    rt = a.retract(a.local_coordinates(b))
    assert np.allclose(rt.T.matrix(), b.T.matrix(), atol=1e-9)
    assert abs(rt.td - b.td) < 1e-9
    assert np.allclose(rt.imu_in.x21, b.imu_in.x21, atol=1e-9)


def test_calib_optimizes_in_graph():
    """An ImuCalib stored as a tangent Vector around a base recovers a target."""
    rng = np.random.RandomState(2)
    base = _rand_calib(rng, RPNG)
    target = _rand_calib(rng, RPNG)
    K = gtsam.symbol('Y', 0)
    noise = gtsam.noiseModel.Unit.Create(31)

    def err(this, values, H):
        xi = values.atVector(K)
        cur = base.retract(xi)
        e = cur.local_coordinates(target)   # 0 when cur == target
        if H is not None:
            J = np.zeros((31, 31)); eps = 1e-6
            for j in range(31):
                d = np.zeros(31); d[j] = eps
                ep = base.retract(xi + d).local_coordinates(target)
                em = base.retract(xi - d).local_coordinates(target)
                J[:, j] = (ep - em) / (2 * eps)
            H[0] = J
        return e

    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.CustomFactor(noise, gtsam.KeyVector([K]), err))
    v = gtsam.Values(); v.insert(K, np.zeros(31))
    res = gtsam.LevenbergMarquardtOptimizer(g, v).optimize()
    cur = base.retract(res.atVector(K))
    assert np.allclose(cur.T.matrix(), target.T.matrix(), atol=1e-6)
    assert abs(cur.td - target.td) < 1e-6
    assert np.allclose(cur.imu_in.x21, target.imu_in.x21, atol=1e-6)
    assert np.allclose(cur.imu_in.R, target.imu_in.R, atol=1e-6)
