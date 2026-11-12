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
"""CameraCalib as a Python manifold: axioms + gtsam optimization.

The camera analogue of test_imu_manifold.py. Validates the manifold contract:
  local(a, retract(a, ξ)) == ξ          (retract/local are inverse)
  retract(a, local(a, b)) == b          (round-trip through tangent)
and that a CameraCalib variable, stored as its 16-dim tangent Vector around a
base, optimizes to a target inside a gtsam factor graph.
"""

import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib


def _rand_calib(rng, is_fisheye):
    T = gtsam.Pose3(gtsam.Rot3.Expmap(rng.randn(3) * 0.1),
                    gtsam.Point3(*rng.randn(3) * 0.1))
    intr = np.array([458.0, 457.0, 367.0, 248.0, 0.0, 0.0, 0.0, 0.0]) \
        + rng.randn(8) * np.array([2, 2, 2, 2, 0.02, 0.01, 0.005, 0.005])
    return CameraCalib(T, rng.randn() * 0.01, rng.randn() * 0.005, intr, is_fisheye)


@pytest.mark.parametrize("is_fisheye", [False, True])
def test_camera_manifold_axioms(is_fisheye):
    rng = np.random.RandomState(0)
    a = _rand_calib(rng, is_fisheye)
    xi = rng.randn(16) * 0.05
    assert np.allclose(a.local_coordinates(a.retract(xi)), xi, atol=1e-9)
    b = _rand_calib(rng, is_fisheye)
    rt = a.retract(a.local_coordinates(b))
    assert np.allclose(rt.T.matrix(), b.T.matrix(), atol=1e-9)
    assert abs(rt.td - b.td) < 1e-9
    assert abs(rt.tr - b.tr) < 1e-9
    assert np.allclose(rt.intr, b.intr, atol=1e-9)
    assert rt.is_fisheye == b.is_fisheye


@pytest.mark.parametrize("is_fisheye", [False, True])
def test_cal_accessor_roundtrips(is_fisheye):
    """`.cal` reconstructs a gtsam Cal3 whose params match the stored 8-vector."""
    rng = np.random.RandomState(3)
    c = _rand_calib(rng, is_fisheye)
    cal = c.cal
    expect = gtsam.Cal3Fisheye if is_fisheye else gtsam.Cal3DS2
    assert isinstance(cal, expect)
    assert np.allclose([cal.fx(), cal.fy(), cal.px(), cal.py()], c.intr[0:4], atol=1e-9)
    # from_cal is the inverse of .cal
    c2 = CameraCalib.from_cal(c.T, c.td, c.tr, cal)
    assert np.allclose(c2.intr, c.intr, atol=1e-9)
    assert c2.is_fisheye == is_fisheye


def test_calib_optimizes_in_graph():
    """A CameraCalib stored as a tangent Vector around a base recovers a target."""
    rng = np.random.RandomState(2)
    base = _rand_calib(rng, False)
    target = _rand_calib(rng, False)
    K = gtsam.symbol('C', 0)
    noise = gtsam.noiseModel.Unit.Create(16)

    def err(this, values, H):
        xi = values.atVector(K)
        cur = base.retract(xi)
        e = cur.local_coordinates(target)   # 0 when cur == target
        if H is not None:
            J = np.zeros((16, 16)); eps = 1e-6
            for j in range(16):
                d = np.zeros(16); d[j] = eps
                ep = base.retract(xi + d).local_coordinates(target)
                em = base.retract(xi - d).local_coordinates(target)
                J[:, j] = (ep - em) / (2 * eps)
            H[0] = J
        return e

    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.CustomFactor(noise, gtsam.KeyVector([K]), err))
    v = gtsam.Values(); v.insert(K, np.zeros(16))
    res = gtsam.LevenbergMarquardtOptimizer(g, v).optimize()
    cur = base.retract(res.atVector(K))
    assert np.allclose(cur.T.matrix(), target.T.matrix(), atol=1e-6)
    assert abs(cur.td - target.td) < 1e-6
    assert abs(cur.tr - target.tr) < 1e-6
    assert np.allclose(cur.intr, target.intr, atol=1e-6)
