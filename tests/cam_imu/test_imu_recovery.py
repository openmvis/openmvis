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
"""IMU-factor optimization-recovery test (integration level).

The per-factor unit tests check residuals/Jacobians (FD, analytic==numerical,
reduction, Taylor). This test closes the loop: a simulated ground-truth
trajectory -> ACI preintegration over consecutive intervals -> a chain of
single-IMU factors -> LM optimization recovers the ground-truth states /
intrinsics from a perturbed initialization. It exercises the analytic Jacobians
inside the actual gtsam solver, which is what the simulator (M0) is for.

Two well-posed scenarios:
  A. anchor the first state + intrinsics, recover the rest of the trajectory.
  B. pin the trajectory (poses+velAndBias) via priors, recover perturbed
     intrinsics (ξ around the true intr_init, truth = 0).
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import KALIBR, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.imu_measurements import simulate_imu, imu_times, GRAVITY
from tests.helpers.imu_fixtures import (wiggle as _wiggle, intr_full as _intr)

X = gtsam.symbol_shorthand.X


def _Vk(k):
    return gtsam.symbol('v', k)


I0 = gtsam.symbol('I', 0)


def _chain(model, K=8, dt_seg=0.1, rate=600.0, t0=1.0):
    """Build a ground-truth chain: trajectory, GT states at K+1 boundaries, and
    a single-IMU factor per interval (true intrinsics + true biases)."""
    traj = BsplineSE3.from_pose_fn(_wiggle, 0.0, t0 + K * dt_seg + 0.5, dt=0.05)
    intr = _intr(model)
    bg = np.array([0.01, -0.02, 0.005])
    ba = np.array([0.05, -0.03, 0.02])
    bounds = [t0 + k * dt_seg for k in range(K + 1)]

    gt_pose, gt_vb = [], []
    for t in bounds:
        R_GtoI, p, _w, v = traj.get_velocity(t)
        gt_pose.append(gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)))
        gt_vb.append(np.concatenate([v, bg, ba]))

    factors = []
    for k in range(K):
        ts = imu_times(bounds[k], bounds[k + 1], rate)
        s = simulate_imu(traj, ts, intr, g_inG=GRAVITY, b_g0=bg, b_a0=ba)
        pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
        pre.reset(bg, ba); pre.set_intrinsics(intr)
        pre.feed_imu_interval(s.t, s.w_m, s.a_m)
        factors.append(ImuFactorACI_In.from_preintegrator(pre, GRAVITY))
    return traj, intr, gt_pose, gt_vb, factors, K


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_recover_trajectory_states(model):
    """Anchor state 0 + intrinsics at truth; recover the rest from a perturbed
    init (validates the IMU factors propagate correctly under LM)."""
    _traj, _intr_, gt_pose, gt_vb, factors, K = _chain(model)
    graph = gtsam.NonlinearFactorGraph()
    for k, f in enumerate(factors):
        graph.add(f.make_factor(X(k), X(k + 1), _Vk(k), _Vk(k + 1), I0))
    tight6 = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
    tight9 = gtsam.noiseModel.Isotropic.Sigma(9, 1e-6)
    tight24 = gtsam.noiseModel.Isotropic.Sigma(24, 1e-8)
    graph.add(gtsam.PriorFactorPose3(X(0), gt_pose[0], tight6))
    graph.add(gtsam.PriorFactorVector(_Vk(0), gt_vb[0], tight9))
    graph.add(gtsam.PriorFactorVector(I0, np.zeros(24), tight24))

    rng = np.random.RandomState(0)
    vals = gtsam.Values()
    vals.insert(I0, np.zeros(24))
    vals.insert(X(0), gt_pose[0]); vals.insert(_Vk(0), gt_vb[0])
    vstep = np.concatenate([[0.05] * 3, [0.01] * 6])
    for k in range(1, K + 1):
        vals.insert(X(k), gt_pose[k].retract(rng.standard_normal(6) * 0.05))
        vals.insert(_Vk(k), gt_vb[k] + rng.standard_normal(9) * vstep)

    res = gtsam.LevenbergMarquardtOptimizer(graph, vals).optimize()
    for k in range(K + 1):
        assert gt_pose[k].equals(res.atPose3(X(k)), 1e-3), f"pose {k}"
        assert np.allclose(res.atVector(_Vk(k)), gt_vb[k], atol=1e-3), f"vb {k}"


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_intrinsics_fit_data_to_floor(model):
    """Pin the trajectory (poses + velAndBias); perturb the intrinsics and
    optimize. The analytic intrinsic Jacobian must drive the IMU-factor data-fit
    error back to the ground-truth floor.

    Note (observability): over this short 8-interval trajectory the intrinsic
    Jacobian is full-rank but has two near-null directions (σ≈4e-5), so the
    *parameters* are not fully recoverable — the optimizer fits the data (error →
    floor) while leaving the unobservable directions near their init. This is the
    honest, basis-independent assertion; full parameter recovery needs the richer
    excitation a full simulation scenario provides (later phase).

    Both models: the whitening congruence fix (Jl·S·Jl^T) keeps the covariance PD
    far from consistency, so KALIBR no longer crashes LM trial steps."""
    _traj, _intr_, gt_pose, gt_vb, factors, K = _chain(model)
    graph = gtsam.NonlinearFactorGraph()
    for k, f in enumerate(factors):
        graph.add(f.make_factor(X(k), X(k + 1), _Vk(k), _Vk(k + 1), I0))
    tight6 = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
    tight9 = gtsam.noiseModel.Isotropic.Sigma(9, 1e-6)
    for k in range(K + 1):
        graph.add(gtsam.PriorFactorPose3(X(k), gt_pose[k], tight6))
        graph.add(gtsam.PriorFactorVector(_Vk(k), gt_vb[k], tight9))

    # error floor: states + true intrinsics (ξ = 0)
    floor = gtsam.Values()
    floor.insert(I0, np.zeros(24))
    for k in range(K + 1):
        floor.insert(X(k), gt_pose[k]); floor.insert(_Vk(k), gt_vb[k])
    err_floor = graph.error(floor)

    rng = np.random.RandomState(1)
    vals = gtsam.Values()
    vals.insert(I0, rng.standard_normal(24) * 0.02)   # perturbed intrinsics
    for k in range(K + 1):
        vals.insert(X(k), gt_pose[k]); vals.insert(_Vk(k), gt_vb[k])
    err_init = graph.error(vals)

    res = gtsam.LevenbergMarquardtOptimizer(graph, vals).optimize()
    err_opt = graph.error(res)

    assert err_init > 1.0                       # perturbation actually hurt the fit
    assert err_opt < 10 * err_floor + 1e-9      # optimizer fit back to the floor
    assert err_opt < 1e-4 * err_init            # huge reduction
