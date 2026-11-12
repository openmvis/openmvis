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
"""Synthetic camera projection (simulator Phase 3a, global shutter).

Validates board projection against gtsam's PinholeCamera, FOV/behind-camera
culling, reprojection-at-truth on a trajectory, and the camera time offset."""

import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.camera_measurements import (
    project_points, simulate_camera, board_points_in_global,
    place_board_facing_camera,
)

_CALVEC = np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004])
_IMG = (640, 480)


def _cal():
    return _cal_from_vector(_CALVEC, is_fisheye=False)


def _board_centered():
    """A 6x6 chessboard centered at the world origin, in the z=0 plane."""
    board = Chessboard(6, 6, 0.05)
    c = board.all_points_3d().mean(0)
    T_BinG = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*(-c)))
    return board_points_in_global(board, T_BinG)


def _lookdown(p=(0.0, 0.0, 2.0)):
    """Camera 2 m above the board looking straight down (+z_cam = -z_world)."""
    return gtsam.Pose3(gtsam.Rot3(np.diag([1.0, -1.0, -1.0])), gtsam.Point3(*p))


def _traj_lookdown(dt=0.1, n=8):
    ctrl = []
    for k in range(n):
        T = np.eye(4)
        T[0:3, 0:3] = np.diag([1.0, -1.0, -1.0])
        T[0:3, 3] = [0.05 * np.sin(0.3 * k), 0.05 * np.cos(0.2 * k), 2.0]
        ctrl.append(T)
    return BsplineSE3(0.0, dt, ctrl)


# --------------------------------------------------------------------------
def test_project_matches_gtsam_pinhole():
    cal, pts, T = _cal(), _board_centered(), _lookdown()
    ids, uv = project_points(T, cal, pts, _IMG)
    assert len(ids) > 0
    cam = gtsam.PinholeCameraCal3DS2(T, cal)
    for i, p in zip(ids, uv):
        assert np.allclose(p, cam.project(gtsam.Point3(*pts[i])), atol=1e-9)


def test_culls_behind_camera_and_out_of_fov():
    cal = _cal()
    T = gtsam.Pose3()                       # identity: camera at origin, +z forward
    pts = np.array([[0.0, 0.0, -2.0],       # behind camera -> culled
                    [0.0, 0.0, 5.0],        # in front, projects to principal point
                    [50.0, 0.0, 5.0]])      # in front but way outside the image
    ids, _uv = project_points(T, cal, pts, _IMG)
    assert 1 in ids and 0 not in ids and 2 not in ids


def test_simulate_camera_reprojection_at_truth():
    cal, pts = _cal(), _board_centered()
    traj = _traj_lookdown()
    T_CtoI = gtsam.Pose3()                   # camera coincident with IMU
    times = np.linspace(traj.start_time + 0.1, traj.end_time - 0.1, 5)
    s = simulate_camera(traj, times, cal, T_CtoI, pts, _IMG)  # noise-free
    for f, t in enumerate(times):
        R_GtoI, p = traj.get_pose(t)
        T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p))
        cam = gtsam.PinholeCameraCal3DS2(T_CtoG, cal)
        assert len(s.ids[f]) > 0
        for i, uv in zip(s.ids[f], s.uv[f]):
            assert np.allclose(uv, cam.project(gtsam.Point3(*pts[i])), atol=1e-9)


def test_td_samples_shifted_pose():
    cal, pts = _cal(), _board_centered()
    traj = _traj_lookdown()
    T_CtoI = gtsam.Pose3()
    td, t = 0.02, traj.start_time + 0.5
    s = simulate_camera(traj, [t], cal, T_CtoI, pts, _IMG, td=td)
    R_GtoI, p = traj.get_pose(t + td)
    T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p))
    ids, uv = project_points(T_CtoG, cal, pts, _IMG)
    assert np.array_equal(s.ids[0], ids)
    assert np.allclose(s.uv[0], uv, atol=1e-12)


def test_pixel_noise_applied():
    cal, pts = _cal(), _board_centered()
    traj = _traj_lookdown()
    times = np.linspace(traj.start_time + 0.1, traj.end_time - 0.1, 3)
    clean = simulate_camera(traj, times, cal, gtsam.Pose3(), pts, _IMG)
    noisy = simulate_camera(traj, times, cal, gtsam.Pose3(), pts, _IMG,
                            sigma_px=0.5, seed=1)
    assert not np.allclose(clean.uv[0], noisy.uv[0])
    # same visibility set, modest perturbation
    assert np.array_equal(clean.ids[0], noisy.ids[0])
    assert np.max(np.abs(clean.uv[0] - noisy.uv[0])) < 5.0


def test_project_fisheye_matches_gtsam():
    """Fisheye (Cal3Fisheye) path matches gtsam's PinholeCameraCal3Fisheye."""
    cal_vec = np.array([350.0, 350.0, 320.0, 240.0, 0.05, -0.01, 0.002, -0.001])
    cal = _cal_from_vector(cal_vec, is_fisheye=True)
    pts, T = _board_centered(), _lookdown()
    ids, uv = project_points(T, cal, pts, _IMG)
    assert len(ids) > 0
    cam = gtsam.PinholeCameraCal3Fisheye(T, cal)
    for i, p in zip(ids, uv):
        assert np.allclose(p, cam.project(gtsam.Point3(*pts[i])), atol=1e-9)


def test_nonidentity_extrinsic_chain():
    """A real camera-IMU extrinsic T_CtoI is composed correctly into T_CtoG."""
    cal, pts = _cal(), _board_centered()
    traj = _traj_lookdown()
    T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.03, -0.02, 0.05])),
                         gtsam.Point3(0.04, -0.02, 0.01))
    t = traj.start_time + 0.5
    s = simulate_camera(traj, [t], cal, T_CtoI, pts, _IMG)
    # manual chain: T_CtoG = T_ItoG(t) · T_CtoI
    R_GtoI, p = traj.get_pose(t)
    T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(T_CtoI)
    ids, uv = project_points(T_CtoG, cal, pts, _IMG)
    assert len(ids) > 0
    assert np.array_equal(s.ids[0], ids)
    assert np.allclose(s.uv[0], uv, atol=1e-12)


def test_board_points_in_global():
    board = Chessboard(4, 5, 0.03)
    pts_B = board.all_points_3d()
    # identity placement -> unchanged
    same = board_points_in_global(board, gtsam.Pose3())
    assert np.allclose(same, pts_B, atol=1e-12)
    # pure translation -> shifted by the same offset
    off = np.array([1.0, -2.0, 0.5])
    shifted = board_points_in_global(board, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*off)))
    assert np.allclose(shifted, pts_B + off, atol=1e-12)


def test_no_visibility_returns_empty():
    """Camera looking away from the board -> empty frame with correct shapes."""
    cal, pts = _cal(), _board_centered()
    # identity rotation at z=2: +z_cam = +z_world, board (z=0) is behind -> nothing
    T = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 2.0))
    ids, uv = project_points(T, cal, pts, _IMG)
    assert ids.shape == (0,)
    assert uv.shape == (0, 2)


def test_depth_gating_boundaries():
    cal = _cal()
    T = gtsam.Pose3()                         # identity, +z forward, looks at +z
    pts = np.array([[0.0, 0.0, 0.30],         # inside [z_near, z_far]
                    [0.0, 0.0, 0.02],         # nearer than z_near
                    [0.0, 0.0, 20.0]])        # farther than z_far
    ids, _uv = project_points(T, cal, pts, _IMG, z_near=0.05, z_far=15.0)
    assert list(ids) == [0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_place_board_facing_camera(seed):
    """Auto-placed board is centred on the optical axis at the requested depth,
    faces the camera, and is fully visible."""
    rng = np.random.RandomState(seed)
    cal = _cal()
    board = Chessboard(6, 6, 0.04)
    # arbitrary camera pose (random orientation + position)
    T_ItoG = gtsam.Pose3(gtsam.Rot3.Expmap(rng.standard_normal(3)),
                         gtsam.Point3(*rng.standard_normal(3)))
    ctrl = [T_ItoG.matrix()] * 6
    traj = BsplineSE3(0.0, 0.1, ctrl)
    t0 = traj.start_time + 0.2
    depth = 1.2

    T_BinG = place_board_facing_camera(traj, t0, depth, board)
    pts = board_points_in_global(board, T_BinG)

    R_GtoI, p = traj.get_pose(t0)
    T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p))   # T_CtoI = I
    z_cam = T_CtoG.rotation().matrix()[:, 2]
    p_cam = np.asarray(T_CtoG.translation(), float)

    # board centroid is `depth` along the optical axis, projects to the centre
    centroid_G = pts.mean(0)
    assert np.allclose(centroid_G - p_cam, depth * z_cam, atol=1e-9)
    cam = gtsam.PinholeCameraCal3DS2(T_CtoG, cal)
    assert np.allclose(cam.project(gtsam.Point3(*centroid_G)),
                       [_CALVEC[2], _CALVEC[3]], atol=1e-6)   # principal point

    # board normal faces the camera (points opposite the optical axis)
    n_board = T_BinG.rotation().matrix()[:, 2]
    assert n_board @ z_cam < -0.99

    # fully visible
    ids, _uv = project_points(T_CtoG, cal, pts, _IMG)
    assert len(ids) == board.num_points()
