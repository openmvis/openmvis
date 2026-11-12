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
"""Feature-only (anonymous-board) mono SfM + init tests (F-stages).

The board's simulated detections are fed WITHOUT the 3-D geometry: SfM must
reconstruct the pose track up to an arbitrary similarity (rotation exact,
translation up to global scale), and the scale-unknown init must recover the
METRIC quantities from the IMU — ŝ converting structure units to meters, with
‖g‖ as the anchor. Scored against the same simulator ground truth as board
mode, on the same scenarios.
"""

import os

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.calibration.factors.lie_groups import log_so3
from mvis.calibration.mono_sfm import normalize_detections, planar_sfm
from mvis.calibration.cam_imu_init import (
    initialize_cam_imu_anonymous, GRAVITY_MAG)
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                     "sim_trajectories", "calib_3D_v0.txt")
_IMG = (640, 480)
_RADTAN = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_BG = np.array([0.01, -0.02, 0.005])
_BA = np.array([0.05, -0.03, 0.02])
_T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                      gtsam.Point3(0.05, -0.02, 0.01))
_WIN = 40.0
_CAM_HZ = 20.0
_IMU_HZ = 400.0


def _scenario(td_cam=0.0, sigma_px=0.0):
    # depth 1.5 m: a small planar target at large depth is the worst case for
    # monocular structure (per-pose depth ill-conditioned → bent trajectory →
    # attenuated scale). 1.5 m keeps the board large in the image; natural
    # 3-D scenes (the real feature-only use case) are better conditioned.
    traj = load_trajectory(_TRAJ, dt=0.05)
    return make_scenario(
        traj, t0=traj.start_time + 1.0, depth=1.5, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=td_cam, sigma_px=sigma_px)


def _data(sc, win=_WIN, seed=0):
    t0 = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + win, 1.0 / _CAM_HZ)
    cam = sc.camera(t_cam, seed=seed)
    t_imu = imu_times(t0 - 0.3, t0 + win + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=seed)
    return t_cam, cam, s


def _gt_T_CtoB(sc, t, td):
    R_GtoI, p = sc.traj.get_pose(t + td)
    T_CtoW = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(sc.T_CtoI)
    return sc.T_BinW.inverse().compose(T_CtoW)


# --------------------------------------------------------------------------
def test_sfm_reconstructs_up_to_similarity():
    """F1 on clean anonymous detections: rotations exact, translations exact
    up to ONE global scale, reconstructed points coplanar (geometry withheld
    yet recovered)."""
    sc = _scenario()
    t_cam, cam, _s = _data(sc)
    tracks = normalize_detections(cam.ids, cam.uv, _RADTAN)
    sfm = planar_sfm(t_cam, [t[0] for t in tracks], [t[1] for t in tracks])

    assert len(sfm.t) > 100          # keyframe track (parallax-gated)
    assert sfm.reproj_rms < 0.1, f"BA reproj {sfm.reproj_rms:.3f} px"
    # planarity recovered (relative to map extent ~0.6 in structure units)
    assert sfm.planarity_rms < 1e-3, f"planarity {sfm.planarity_rms:.2e}"

    # compare vs GT: T_CtoS = T_BtoS · T_CtoB with one fixed similarity
    # (T_BtoS rigid + global scale). Estimate the scale from one pair, then
    # every relative pose must match.
    gt = [_gt_T_CtoB(sc, t, 0.0) for t in sfm.t]
    # relative transforms between far-apart frames
    k0, k1 = 0, len(sfm.t) // 2
    rel_sfm = sfm.poses[k0].inverse().compose(sfm.poses[k1])
    rel_gt = gt[k0].inverse().compose(gt[k1])
    scale = (np.linalg.norm(np.asarray(rel_gt.translation(), float))
             / max(np.linalg.norm(np.asarray(rel_sfm.translation(), float)), 1e-12))
    for ka, kb in [(0, len(sfm.t) - 1), (len(sfm.t) // 4, 3 * len(sfm.t) // 4)]:
        r_s = sfm.poses[ka].inverse().compose(sfm.poses[kb])
        r_g = gt[ka].inverse().compose(gt[kb])
        d_rot = np.linalg.norm(log_so3(
            r_g.rotation().matrix().T @ r_s.rotation().matrix()))
        assert d_rot < 5e-3, f"rel rot err {d_rot:.2e}"   # ~0.3 deg
        t_s = scale * np.asarray(r_s.translation(), float)
        t_g = np.asarray(r_g.translation(), float)
        assert np.linalg.norm(t_s - t_g) < 5e-3 * max(np.linalg.norm(t_g), 1.0), \
            f"rel trans err {t_s - t_g}"


@pytest.mark.parametrize("sigma_px", [0.0, 0.5])
def test_feature_mode_init_recovers_metric(sigma_px):
    """Feature-only init end-to-end on sim: td / R_CtoI / gravity (scale-free)
    AND the metric quantities via ŝ — seed trajectory distances and velocities
    must come out in true meters even though the pose track had no scale."""
    td_true = 0.01
    sc = _scenario(td_cam=td_true, sigma_px=sigma_px)
    t_cam, cam, s = _data(sc, seed=2)
    seed = initialize_cam_imu_anonymous(
        t_cam, cam.ids, cam.uv, _RADTAN, s.t, s.w_m, s.a_m)

    # tolerances: clean = tight; noisy = mono-VI-init grade (the scale comes
    # from a small planar target — 5-10% scale error is the literature norm)
    tol_td = 1e-3 if sigma_px == 0 else 15e-3
    tol_rot = np.deg2rad(0.5 if sigma_px == 0 else 3.0)
    assert abs(seed.td - td_true) < tol_td, f"td {seed.td:+.4f}"
    d_rot = np.linalg.norm(_T_CtoI.localCoordinates(seed.T_CtoI)[0:3])
    assert d_rot < tol_rot, f"rot err {np.rad2deg(d_rot):.2f} deg"
    assert np.linalg.norm(seed.b_g - _BG) < (2e-3 if sigma_px == 0 else 8e-3)
    # gravity magnitude gate (the metric anchor itself)
    assert abs(seed.report["g_norm_ls"] - GRAVITY_MAG) / GRAVITY_MAG < \
        (0.01 if sigma_px == 0 else 0.05)
    assert seed.report["sfm"]["reproj_rms_px"] < (0.1 if sigma_px == 0 else 1.5)

    # METRIC check: inter-keyframe distances of the (ŝ-scaled) seed states
    # must match the GT trajectory in meters — the pose track had NO scale
    k0, k1 = 0, len(seed.t_key) - 1
    p0 = np.asarray(seed.T_ItoW[k0].translation(), float)
    p1 = np.asarray(seed.T_ItoW[k1].translation(), float)
    _R, q0 = sc.traj.get_pose(seed.t_key[k0])
    _R, q1 = sc.traj.get_pose(seed.t_key[k1])
    d_seed, d_gt = np.linalg.norm(p1 - p0), np.linalg.norm(q1 - q0)
    rel = abs(d_seed - d_gt) / max(d_gt, 1e-9)
    assert rel < (0.02 if sigma_px == 0 else 0.15), \
        f"metric distance err {100 * rel:.1f}% (seed {d_seed:.3f} vs gt {d_gt:.3f})"
    # velocities in true meters per second (median over keyframes — single
    # boundary keyframes are less constrained)
    errs = []
    for i in range(min(20, len(seed.t_key))):
        _R, _p, _w, v_gt = sc.traj.get_velocity(seed.t_key[i])
        errs.append(abs(np.linalg.norm(seed.v_inW[i]) - np.linalg.norm(v_gt)))
    med = float(np.median(errs))
    assert med < (0.05 if sigma_px == 0 else 0.15), f"median |v| err {med:.3f}"
