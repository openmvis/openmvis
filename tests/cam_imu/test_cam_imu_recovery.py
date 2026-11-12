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
"""Joint camera-IMU optimization-recovery test (simulator Phase 3c-ii).

The integration payoff of the whole port: ONE factor graph holding both the
camera reprojection factors (CameraCalibFactor over the CameraCalib bundle) and
the single-IMU ACI factors, sharing the trajectory states — i.e. the core of
`pipeline_cam_imu` in miniature, validated against the simulator's known ground
truth:

  states:   X(k) = T_ItoW, vb(k) = [v, bg, ba] at each camera frame time
  unknowns: C = CameraCalib ξ16 (T_CtoI, td, tr, intrinsics), B = board T_BinW
  factors:  ACI IMU factor per interval + camera factor per frame
            + priors: X(0), vb(0) anchored, IMU intrinsics I0 pinned

Asserts: (a) graph error at ground truth is at the data floor — cross-validates
every sim-vs-factor convention through the full joint chain (camera analogue of
the IMU side's 7e-7 GT-residual check); (b) LM recovers the camera-IMU
spatial-temporal calibration (T_CtoI, td) and the board pose from a perturbed
initialization; camera intrinsics are asserted error-to-floor (short-window
observability is partial — same finding as the IMU intrinsics).

Observability finding (0.5 px noise): on a tum_room handheld window the
extrinsic is information-limited (σ_rot ≈ 0.02 rad, σ_trans_z ≈ 0.3 m — the
classic focal/depth/extrinsic-z correlation; estimates land within their gtsam
marginals). The noisy test therefore runs on the purpose-built calibration
trajectory `calib_3D_v28` (rich rotation sweep, full board visibility) and
asserts 3σ-consistency against the marginals — the honest statistical claim."""

import os

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.camera_calib import CameraCalib
from tests.helpers.camera_calib_factor import CameraCalibFactor
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tum_room_segment.txt")
_TRAJ_CALIB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                           "sim_trajectories", "calib_3D_v28.txt")
_IMG = (640, 480)
_T0OFF = 3.0
_DEPTH = 3.0
_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])
_T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                      gtsam.Point3(0.05, -0.02, 0.01))

X = gtsam.symbol_shorthand.X
_Vk = lambda k: gtsam.symbol('v', k)
B0 = gtsam.symbol('b', 0)
C0 = gtsam.symbol('c', 0)
I0 = gtsam.symbol('I', 0)


def _imu_intr():
    return ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.zeros(9),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(), model=RPNG)


def _build(K=12, cam_rate=10.0, imu_rate=400.0, traj_path=_TRAJ, t0_off=_T0OFF,
           depth=_DEPTH, sigma_px=0.0, cam_seed=0):
    """Joint graph at ground truth: K camera frames, K IMU intervals, plus
    the ground-truth Values. Truth td_cam = 0, global shutter (the td/RS
    conventions are pinned by the factor unit tests; here truth must be exact
    so the GT floor is the discretization floor only)."""
    traj = load_trajectory(traj_path, dt=0.05)
    board = Chessboard(7, 7, 0.10)
    sc = make_scenario(
        traj, t0=traj.start_time + t0_off, depth=depth, board=board,
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=_imu_intr(), b_g=_BG, b_a=_BA, image_size=_IMG,
        sigma_px=sigma_px)

    t0 = traj.start_time + t0_off
    frames = np.array([t0 + k / cam_rate for k in range(K + 1)])
    calib_base = CameraCalib(sc.T_CtoI, 0.0, 0.0, _RADTAN, False)
    pts_B = np.asarray(board.all_points_3d(), float)
    cam = sc.camera(frames, seed=cam_seed)

    graph = gtsam.NonlinearFactorGraph()
    gt = gtsam.Values()
    gt.insert(B0, sc.T_BinW)
    gt.insert(C0, np.zeros(16))
    gt.insert(I0, np.zeros(24))

    for k, t in enumerate(frames):
        R_GtoI, p, w_IinI, v = traj.get_velocity(t)
        gt.insert(X(k), gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)))
        gt.insert(_Vk(k), np.concatenate([v, _BG, _BA]))
        # camera factor for this frame
        ids, uv = cam.ids[k], cam.uv[k]
        assert len(ids) > 10, f"frame {k} barely sees the board"
        fac = CameraCalibFactor(uv, pts_B[ids], w_IinI, calib_base, _IMG)
        graph.add(fac.make_factor(X(k), _Vk(k), B0, C0,
                                  sigma_px=sigma_px if sigma_px > 0 else 1.0))

    for k in range(K):
        ts = imu_times(frames[k], frames[k + 1], imu_rate)
        s = sc.base_imu(ts)
        pre = ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=RPNG, avg_meas=True)
        pre.reset(_BG, _BA); pre.set_intrinsics(sc.imu_intr)
        pre.feed_imu_interval(s.t, s.w_m, s.a_m)
        imu_fac = ImuFactorACI_In.from_preintegrator(pre, sc.g_inW)
        graph.add(imu_fac.make_factor(X(k), X(k + 1), _Vk(k), _Vk(k + 1), I0))

    # gauge + pinned IMU intrinsics
    graph.add(gtsam.PriorFactorPose3(X(0), gt.atPose3(X(0)),
                                     gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)))
    graph.add(gtsam.PriorFactorVector(_Vk(0), gt.atVector(_Vk(0)),
                                      gtsam.noiseModel.Isotropic.Sigma(9, 1e-6)))
    graph.add(gtsam.PriorFactorVector(I0, np.zeros(24),
                                      gtsam.noiseModel.Isotropic.Sigma(24, 1e-8)))
    return sc, graph, gt, calib_base, K


def _perturbed_init(gt, K, seed=0):
    """Perturbed initialization: calib ξ (rot/trans/td), board pose, trajectory."""
    rng = np.random.RandomState(seed)
    vals = gtsam.Values()
    xi0 = np.zeros(16)
    xi0[0:3] = rng.standard_normal(3) * 0.02            # extrinsic rotation
    xi0[3:6] = rng.standard_normal(3) * 0.02            # extrinsic translation
    xi0[6] = 0.005                                      # td
    vals.insert(C0, xi0)
    vals.insert(B0, gt.atPose3(B0).retract(rng.standard_normal(6) * 0.02))
    vals.insert(I0, np.zeros(24))
    vals.insert(X(0), gt.atPose3(X(0))); vals.insert(_Vk(0), gt.atVector(_Vk(0)))
    for k in range(1, K + 1):
        vals.insert(X(k), gt.atPose3(X(k)).retract(rng.standard_normal(6) * 0.02))
        vals.insert(_Vk(k), gt.atVector(_Vk(k)) + rng.standard_normal(9) * 0.02)
    return vals


def test_error_at_truth_is_floor():
    """The full joint graph evaluated at ground truth: camera residuals are
    exact (1e-9 px) and the IMU residuals sit at the discrete-integration
    floor — cross-validates every sim-vs-factor convention jointly."""
    _sc, graph, gt, _base, _K = _build()
    err = graph.error(gt)
    assert err < 1e-3, f"graph error at truth {err:.3e}"


def test_joint_recovery_spatial_temporal():
    """Perturb the camera-IMU calibration (rotation, translation, td), the
    board pose, and the trajectory init; LM on the joint graph must recover
    T_CtoI and td — the first working camera-IMU calibration of the port."""
    _sc, graph, gt, base, K = _build()
    err_floor = graph.error(gt)
    vals = _perturbed_init(gt, K, seed=0)
    err_init = graph.error(vals)
    assert err_init > 100 * err_floor                   # perturbation hurts

    res = gtsam.LevenbergMarquardtOptimizer(graph, vals).optimize()
    err_opt = graph.error(res)
    assert err_opt < 2 * err_floor + 1e-9               # fit back to the floor

    # calibration recovery: retract the solved ξ and compare to ground truth
    sol = base.retract(res.atVector(C0))
    d_ext = _T_CtoI.localCoordinates(sol.T)
    assert np.linalg.norm(d_ext[0:3]) < 5e-4, f"extrinsic rot {d_ext[0:3]}"
    assert np.linalg.norm(d_ext[3:6]) < 1e-3, f"extrinsic trans {d_ext[3:6]}"
    assert abs(sol.td) < 2e-4, f"td {sol.td:.2e}"
    # board pose recovered too
    d_board = gt.atPose3(B0).localCoordinates(res.atPose3(B0))
    assert np.linalg.norm(d_board) < 2e-3, f"board {d_board}"
    # intrinsics: error-to-floor is the honest claim on a short window
    # (parameters may slide along near-null directions; the fit must not)


def test_joint_recovery_with_pixel_noise():
    """0.5 px noise on the purpose-built calibration trajectory (calib_3D_v28,
    rich rotation sweep): the recovered spatial-temporal calibration must be
    consistent with its own gtsam marginals (within 3σ), and the marginals
    themselves must show the calibration is actually observed (σ caps).

    (On a tum_room handheld window the same estimator is information-limited —
    σ_rot ≈ 0.02 rad, σ_trans_z ≈ 0.3 m from the focal/depth/extrinsic-z
    correlation — verified consistent with its marginals; not asserted here.)"""
    _sc, graph, gt, base, K = _build(
        K=30, traj_path=_TRAJ_CALIB, t0_off=1.0, depth=1.5,
        sigma_px=0.5, cam_seed=4)
    vals = _perturbed_init(gt, K, seed=1)

    res = gtsam.LevenbergMarquardtOptimizer(graph, vals).optimize()
    sol = base.retract(res.atVector(C0))
    d_ext = _T_CtoI.localCoordinates(sol.T)

    marg = gtsam.Marginals(graph, res)
    sig = np.sqrt(np.diag(marg.marginalCovariance(C0)))
    # estimates statistically consistent with the marginals (3σ per component)
    assert np.all(np.abs(d_ext[0:3]) < 3 * sig[0:3]), \
        f"rot {d_ext[0:3]} vs 3σ {3 * sig[0:3]}"
    assert np.all(np.abs(d_ext[3:6]) < 3 * sig[3:6]), \
        f"trans {d_ext[3:6]} vs 3σ {3 * sig[3:6]}"
    assert abs(sol.td) < 3 * sig[6], f"td {sol.td:.2e} vs 3σ {3 * sig[6]:.2e}"
    # ... and the information content is there (regression guard on excitation)
    assert np.max(sig[0:3]) < 0.02, f"σ_rot {sig[0:3]}"
    assert np.max(sig[3:6]) < 0.06, f"σ_trans {sig[3:6]}"
    assert sig[6] < 5e-3, f"σ_td {sig[6]:.2e}"
