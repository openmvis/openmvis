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
"""Multi-IMU & gyro factor optimization-recovery tests (simulator Phase 2c).

Integration test for the auxiliary-IMU factors: a simulated base trajectory + an
auxiliary IMU stream (lever-arm coupling) -> ACI preintegration per interval ->
the Multi_In_V (7-key) / Multi_In_SO3 (gyro, 5-key) factors -> LM, exercising
their analytic calib (extrinsic+td+intrinsics) Jacobians inside the real solver.

That the factor's residual is ~0 at ground truth (graph error ~7e-7) is itself a
cross-check: the simulator's lever-arm generation matches the factor's kinematic
coupling.

True td = 0 here (aux synchronized) to isolate the sim-vs-factor td conventions;
the calib's td DOF is still exercised by perturbing it and fitting back. A
true-td != 0 round-trip against the factor is a follow-up (see plan Phase 2c).
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import KALIBR, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator, ACIGyroPreintegrator
from mvis.calibration.factors.imu_factor_multi import ImuFactorACI_Multi_In_V
from mvis.calibration.factors.imu_factor_gyro import ImuFactorACI_Multi_In_SO3
from mvis.calibration.factors.lie_groups import skew
from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.imu_measurements import simulate_imu, imu_times, GRAVITY
from tests.helpers.imu_fixtures import (wiggle as _wiggle, intr_full as _intr)

X = gtsam.symbol_shorthand.X


def _Vk(k):       # base velAndBias
    return gtsam.symbol('v', k)


def _VS(k):       # aux velAndBias [bg, ba, v]
    return gtsam.symbol('s', k)


def _BG(k):       # aux gyro bias (gyro factor)
    return gtsam.symbol('g', k)


C0 = gtsam.symbol('C', 0)
_BG_S = np.array([0.01, -0.02, 0.005])
_BA_S = np.array([0.05, -0.03, 0.02])


def _extrinsic():
    R = gtsam.Rot3.Expmap(np.array([0.05, -0.03, 0.08])).matrix()
    return R, np.array([0.10, -0.05, 0.02])


def _aux_state(traj, t, R_ext, p_ext):
    R_GtoI, pos, w, v = traj.get_velocity(t)
    R_ItoG = R_GtoI.T
    return R_ItoG @ R_ext, pos + R_ItoG @ p_ext, v + R_ItoG @ skew(w) @ p_ext


TIGHT6 = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
TIGHT9 = gtsam.noiseModel.Isotropic.Sigma(9, 1e-6)
TIGHT3 = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)


def _multi_chain(model, K=8, dt=0.1, rate=600.0, t0=1.0):
    traj = BsplineSE3.from_pose_fn(_wiggle, 0.0, t0 + K * dt + 0.5, 0.05)
    intr = _intr(model)
    R_ext, p_ext = _extrinsic()
    T_ext = gtsam.Pose3(gtsam.Rot3(R_ext), gtsam.Point3(*p_ext))
    bounds = [t0 + k * dt for k in range(K + 1)]

    base_pose, base_vb, aux_vbs = [], [], []
    for t in bounds:
        R_GtoI, pos, _w, v = traj.get_velocity(t)
        base_pose.append(gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*pos)))
        base_vb.append(np.concatenate([v, np.zeros(6)]))   # base bias unused
        _, _, v_s = _aux_state(traj, t, R_ext, p_ext)
        aux_vbs.append(np.concatenate([_BG_S, _BA_S, v_s]))  # [bg, ba, v]

    factors = []
    for k in range(K):
        ts = imu_times(bounds[k], bounds[k + 1], rate)
        s = simulate_imu(traj, ts, intr, R_IstoI=R_ext, p_IsinI=p_ext, td=0.0,
                         b_g0=_BG_S, b_a0=_BA_S)
        pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
        pre.reset(_BG_S, _BA_S); pre.set_intrinsics(intr)
        pre.feed_imu_interval(s.t, s.w_m, s.a_m)
        factors.append(ImuFactorACI_Multi_In_V.from_preintegrator(pre, GRAVITY, T_ext, 0.0))
    return traj, base_pose, base_vb, aux_vbs, factors, K


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_multi_calib_fit_to_floor(model):
    """Pin base trajectory + aux velAndBias; perturb the calib (extrinsic+td+
    intrinsics); the analytic calib Jacobian drives the data-fit to the GT floor."""
    _traj, base_pose, base_vb, aux_vbs, factors, K = _multi_chain(model)
    g = gtsam.NonlinearFactorGraph()
    for k, f in enumerate(factors):
        g.add(f.make_factor(X(k), X(k + 1), _Vk(k), _Vk(k + 1), _VS(k), _VS(k + 1), C0))
    for k in range(K + 1):
        g.add(gtsam.PriorFactorPose3(X(k), base_pose[k], TIGHT6))
        g.add(gtsam.PriorFactorVector(_Vk(k), base_vb[k], TIGHT9))
        g.add(gtsam.PriorFactorVector(_VS(k), aux_vbs[k], TIGHT9))

    floor = gtsam.Values(); floor.insert(C0, np.zeros(31))
    for k in range(K + 1):
        floor.insert(X(k), base_pose[k]); floor.insert(_Vk(k), base_vb[k])
        floor.insert(_VS(k), aux_vbs[k])
    err_floor = g.error(floor)

    rng = np.random.RandomState(2)
    vals = gtsam.Values(); vals.insert(C0, rng.standard_normal(31) * 0.01)
    for k in range(K + 1):
        vals.insert(X(k), base_pose[k]); vals.insert(_Vk(k), base_vb[k])
        vals.insert(_VS(k), aux_vbs[k])
    err_init = g.error(vals)

    res = gtsam.LevenbergMarquardtOptimizer(g, vals).optimize()
    err_opt = g.error(res)
    assert err_init > 1.0
    assert err_opt < 10 * err_floor + 1e-9
    assert err_opt < 1e-3 * err_init


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_multi_recover_aux_states(model):
    """Anchor base trajectory + calib + aux state 0; recover the aux velAndBias
    chain from a perturbed init (validates the coupling drives propagation)."""
    _traj, base_pose, base_vb, aux_vbs, factors, K = _multi_chain(model)
    g = gtsam.NonlinearFactorGraph()
    for k, f in enumerate(factors):
        g.add(f.make_factor(X(k), X(k + 1), _Vk(k), _Vk(k + 1), _VS(k), _VS(k + 1), C0))
    for k in range(K + 1):
        g.add(gtsam.PriorFactorPose3(X(k), base_pose[k], TIGHT6))
        g.add(gtsam.PriorFactorVector(_Vk(k), base_vb[k], TIGHT9))
    g.add(gtsam.PriorFactorVector(C0, np.zeros(31), gtsam.noiseModel.Isotropic.Sigma(31, 1e-8)))
    g.add(gtsam.PriorFactorVector(_VS(0), aux_vbs[0], TIGHT9))

    rng = np.random.RandomState(3)
    vals = gtsam.Values(); vals.insert(C0, np.zeros(31))
    for k in range(K + 1):
        vals.insert(X(k), base_pose[k]); vals.insert(_Vk(k), base_vb[k])
    vals.insert(_VS(0), aux_vbs[0])
    perturb = np.concatenate([[0.005] * 6, [0.05] * 3])   # bias, then velocity
    for k in range(1, K + 1):
        vals.insert(_VS(k), aux_vbs[k] + rng.standard_normal(9) * perturb)

    res = gtsam.LevenbergMarquardtOptimizer(g, vals).optimize()
    for k in range(K + 1):
        assert np.allclose(res.atVector(_VS(k)), aux_vbs[k], atol=1e-3), f"aux vbs {k}"


def _gyro_chain(model, K=8, dt=0.1, rate=600.0, t0=1.0):
    traj = BsplineSE3.from_pose_fn(_wiggle, 0.0, t0 + K * dt + 0.5, 0.05)
    intr = _intr(model)
    R_ext, p_ext = _extrinsic()
    T_ext = gtsam.Pose3(gtsam.Rot3(R_ext), gtsam.Point3(*p_ext))
    bounds = [t0 + k * dt for k in range(K + 1)]
    base_pose = [gtsam.Pose3(gtsam.Rot3(traj.get_pose(t)[0].T), gtsam.Point3(*traj.get_pose(t)[1]))
                 for t in bounds]
    factors = []
    for k in range(K):
        ts = imu_times(bounds[k], bounds[k + 1], rate)
        s = simulate_imu(traj, ts, intr, R_IstoI=R_ext, p_IsinI=p_ext, td=0.0, b_g0=_BG_S, b_a0=_BA_S)
        pre = ACIGyroPreintegrator(0.01, 0.001, model=model, avg_meas=True)
        pre.reset(_BG_S); pre.set_intrinsics(intr)
        pre.feed_imu_interval(s.t, s.w_m)
        factors.append(ImuFactorACI_Multi_In_SO3.from_preintegrator(pre, T_ext, 0.0))
    return base_pose, factors, K


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_gyro_calib_fit_to_floor(model):
    """Gyro-only: pin base orientations + aux gyro bias; perturb the calib and
    fit the rotation-only data-fit error back to the GT floor."""
    base_pose, factors, K = _gyro_chain(model)
    g = gtsam.NonlinearFactorGraph()
    for k, f in enumerate(factors):
        g.add(f.make_factor(X(k), X(k + 1), _BG(k), _BG(k + 1), C0))
    for k in range(K + 1):
        g.add(gtsam.PriorFactorPose3(X(k), base_pose[k], TIGHT6))
        g.add(gtsam.PriorFactorVector(_BG(k), _BG_S, TIGHT3))

    floor = gtsam.Values(); floor.insert(C0, np.zeros(31))
    for k in range(K + 1):
        floor.insert(X(k), base_pose[k]); floor.insert(_BG(k), _BG_S)
    err_floor = g.error(floor)

    rng = np.random.RandomState(4)
    vals = gtsam.Values(); vals.insert(C0, rng.standard_normal(31) * 0.01)
    for k in range(K + 1):
        vals.insert(X(k), base_pose[k]); vals.insert(_BG(k), _BG_S)
    err_init = g.error(vals)

    res = gtsam.LevenbergMarquardtOptimizer(g, vals).optimize()
    err_opt = g.error(res)
    assert err_init > 1.0
    assert err_opt < 10 * err_floor + 1e-9
    assert err_opt < 1e-2 * err_init
