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
"""CameraCalibFactor unit tests (simulator Phase 3c-ii).

Validates the camera reprojection factor over the CameraCalib bundle against
the 3c-i simulator: residual≈0 at ground truth (pins the projection chain and
the td / rolling-shutter conventions — the factor's Exp(ω·t) propagation vs the
sim's exact pose sampling), the ξ16 manifold wiring, and the ANALYTIC Jacobians
(per-block vs central differences, at truth and at a perturbed state, plus a
full Taylor check). The MVIS C++ camera factor is fully numerical, so FD is the
ground truth here.
"""

import os

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.camera_calib import CameraCalib
from tests.helpers.camera_calib_factor import CameraCalibFactor
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.scenario import make_scenario

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tum_room_segment.txt")
_IMG = (640, 480)
_T0OFF = 3.0
_DEPTH = 3.0

_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_FISHEYE = np.array([400.0, 400.0, 320.0, 240.0, -0.01, 0.005, -0.002, 0.001])

_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])

_T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                      gtsam.Point3(0.05, -0.02, 0.01))


def _imu_intr():
    return ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.zeros(9),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(), model=RPNG)


def _scenario(fisheye=False, td_cam=0.0, readout=0.0, rolling_mode=0):
    traj = load_trajectory(_TRAJ, dt=0.05)
    board = Chessboard(7, 7, 0.10)
    cal_vec = _FISHEYE if fisheye else _RADTAN
    sc = make_scenario(
        traj, t0=traj.start_time + _T0OFF, depth=_DEPTH, board=board,
        cal=_cal_from_vector(cal_vec, fisheye), T_CtoI=_T_CtoI,
        imu_intr=_imu_intr(), b_g=_BG, b_a=_BA, image_size=_IMG,
        fisheye=fisheye, td_cam=td_cam, readout=readout,
        rolling_mode=rolling_mode)
    return sc, cal_vec


def _truth_inputs(sc, t):
    """Ground-truth (pose, vb, w_m) at frame time t."""
    R_GtoI, p, w_IinI, v = sc.traj.get_velocity(t)
    pose = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p))
    vb = np.concatenate([v, _BG, _BA])
    return pose, vb, w_IinI


def _make_factor(sc, cal_vec, t, fisheye=False):
    """Factor for the sim frame at time t, base calib = ground truth (ξ=0)."""
    cam = sc.camera([t])
    ids, uv = cam.ids[0], cam.uv[0]
    assert len(ids) > 4, "frame must see the board"
    p_FinB = np.asarray(sc.board.all_points_3d(), float)[ids]
    pose, vb, w_m = _truth_inputs(sc, t)
    base = CameraCalib(sc.T_CtoI, sc.td_cam, sc.readout, cal_vec, fisheye)
    fac = CameraCalibFactor(uv, p_FinB, w_m, base, _IMG,
                            rolling_mode=sc.rolling_mode)
    return fac, pose, vb


# --------------------------------------------------------------------------
@pytest.mark.parametrize("fisheye", [False, True])
def test_residual_zero_at_truth(fisheye):
    """td=tr=0: the factor chain must reproduce the sim projection exactly."""
    sc, cal_vec = _scenario(fisheye=fisheye)
    t = sc.traj.start_time + _T0OFF + 1.0
    fac, pose, vb = _make_factor(sc, cal_vec, t, fisheye)
    err = fac.error(pose, vb, sc.T_BinW, np.zeros(16))
    assert np.max(np.abs(err)) < 1e-9


def test_residual_small_with_td():
    """td≠0 pins the sign convention: the factor's Exp(ω·td)/v·td propagation
    must agree with the sim's exact pose at t+td to O(td²) (sub-pixel)."""
    td = 0.01
    sc, cal_vec = _scenario(td_cam=td)
    t = sc.traj.start_time + _T0OFF + 1.0
    fac, pose, vb = _make_factor(sc, cal_vec, t)
    err = fac.error(pose, vb, sc.T_BinW, np.zeros(16))
    assert np.max(np.abs(err)) < 0.5      # px; O(td²) propagation error only
    # sign check: flipping the td in the calib must be MUCH worse
    xi_flip = np.zeros(16); xi_flip[6] = -2 * td
    err_flip = fac.error(pose, vb, sc.T_BinW, xi_flip)
    assert np.max(np.abs(err_flip)) > 20 * max(np.max(np.abs(err)), 1e-3)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_residual_small_rolling_shutter(mode):
    """tr≠0: row-of-measurement timing must match the sim's fixed-point RS to
    first order."""
    tr = 0.02
    sc, cal_vec = _scenario(readout=tr, rolling_mode=mode)
    t = sc.traj.start_time + _T0OFF + 1.0
    fac, pose, vb = _make_factor(sc, cal_vec, t)
    err = fac.error(pose, vb, sc.T_BinW, np.zeros(16))
    assert np.max(np.abs(err)) < 1.0      # px


def test_xi_wiring():
    """error(ξ) must equal rebuilding the factor at base.retract(ξ) with ξ=0."""
    sc, cal_vec = _scenario()
    t = sc.traj.start_time + _T0OFF + 1.0
    fac, pose, vb = _make_factor(sc, cal_vec, t)
    rng = np.random.RandomState(3)
    xi = rng.standard_normal(16) * np.concatenate(
        [[0.02] * 6, [0.005, 0.005], [1.0, 1.0, 0.5, 0.5], [0.005] * 4])
    err = fac.error(pose, vb, sc.T_BinW, xi)
    retracted = fac.base.retract(xi)
    fac2 = CameraCalibFactor(fac.uv, fac.p_B, fac.w_m, retracted, _IMG,
                             rolling_mode=fac.rolling_mode)
    err2 = fac2.error(pose, vb, sc.T_BinW, np.zeros(16))
    assert np.allclose(err, err2, atol=1e-9)


# --------------------------------------------------------------------------
def _perturbed_state(sc, cal_vec, t, fisheye=False, seed=7):
    """A consistent-but-perturbed evaluation point: ξ_T≠0 (exercises the Jr_se3
    retract chain), td/tr≠0 (exercises the time/velocity couplings)."""
    fac, pose, vb = _make_factor(sc, cal_vec, t, fisheye)
    rng = np.random.RandomState(seed)
    pose_p = pose.retract(rng.standard_normal(6) * 0.02)
    vb_p = vb + rng.standard_normal(9) * 0.05
    board_p = sc.T_BinW.retract(rng.standard_normal(6) * 0.02)
    xi = rng.standard_normal(16) * np.concatenate(
        [[0.02] * 6, [0.008, 0.008], [0.5, 0.5, 0.2, 0.2], [0.002] * 4])
    return fac, pose_p, vb_p, board_p, xi


@pytest.mark.parametrize("fisheye", [False, True])
@pytest.mark.parametrize("where", ["truth", "perturbed"])
def test_analytic_matches_numerical(fisheye, where):
    """Every analytic block == central differences (FD is the ground truth —
    the C++ camera factor has no analytic to compare against)."""
    sc, cal_vec = _scenario(fisheye=fisheye)
    t = sc.traj.start_time + _T0OFF + 1.0
    if where == "truth":
        fac, pose, vb = _make_factor(sc, cal_vec, t, fisheye)
        board, xi = sc.T_BinW, np.zeros(16)
    else:
        fac, pose, vb, board, xi = _perturbed_state(sc, cal_vec, t, fisheye)
    err, Ja = fac.error_and_jacobians(pose, vb, board, xi)
    assert np.allclose(err, fac.error(pose, vb, board, xi), atol=1e-12)
    Jn = fac._numerical_jacobians(pose, vb, board, xi, eps=1e-6)
    names = ["H_pose", "H_vb", "H_board", "H_calib"]
    for a, n, name in zip(Ja, Jn, names):
        scale = max(1.0, np.max(np.abs(n)))
        assert np.max(np.abs(a - n)) / scale < 2e-5, \
            f"{name} ({where}): max dev {np.max(np.abs(a - n)):.3e}"


def test_taylor():
    """e(x⊕δ) ≈ e + H·δ across all 4 blocks (catches coupling terms)."""
    sc, cal_vec = _scenario()
    t = sc.traj.start_time + _T0OFF + 1.0
    fac, pose, vb, board, xi = _perturbed_state(sc, cal_vec, t)
    e0, J = fac.error_and_jacobians(pose, vb, board, xi)
    rng = np.random.RandomState(11)
    for scale in [1e-4, 1e-5]:
        d_pose = rng.standard_normal(6) * scale
        d_vb = rng.standard_normal(9) * scale
        d_board = rng.standard_normal(6) * scale
        d_xi = rng.standard_normal(16) * scale
        e1 = fac.error(pose.retract(d_pose), vb + d_vb,
                       board.retract(d_board), xi + d_xi)
        pred = e0 + J[0] @ d_pose + J[1] @ d_vb + J[2] @ d_board + J[3] @ d_xi
        # remainder must be second order: ‖e1 − pred‖ « ‖e1 − e0‖
        assert np.linalg.norm(e1 - pred) < 1e-2 * max(np.linalg.norm(e1 - e0), 1e-12)


def test_graph_build_and_linearize():
    """CustomFactor wiring: build a graph, evaluate, linearize."""
    sc, cal_vec = _scenario()
    t = sc.traj.start_time + _T0OFF + 1.0
    fac, pose, vb = _make_factor(sc, cal_vec, t)
    X, V = gtsam.symbol('x', 0), gtsam.symbol('v', 0)
    B, C = gtsam.symbol('b', 0), gtsam.symbol('c', 0)
    graph = gtsam.NonlinearFactorGraph()
    graph.add(fac.make_factor(X, V, B, C, sigma_px=1.0))
    vals = gtsam.Values()
    vals.insert(X, pose); vals.insert(V, vb)
    vals.insert(B, sc.T_BinW); vals.insert(C, np.zeros(16))
    assert graph.error(vals) < 1e-15          # at truth
    lin = graph.linearize(vals)
    assert lin.size() == 1
    vals2 = gtsam.Values()
    vals2.insert(X, pose.retract(0.01 * np.ones(6))); vals2.insert(V, vb)
    vals2.insert(B, sc.T_BinW); vals2.insert(C, np.zeros(16))
    assert graph.error(vals2) > 1.0           # perturbation visible
