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
"""Board-pose PnP init for wide-FOV models (eucm, ucm) — regression for two
defects found on real wide-FOV (~180 deg) data:

* the RANSAC threshold (8.0, meant in px) was applied on the normalized plane,
  so it rejected nothing and one bad corner could wreck the pose;
* corners that failed to unproject were passed on as RAW PIXELS into the
  normalized-plane solve, mixing units.
"""
import cv2
import gtsam
import numpy as np
import pytest

from mvis.camera.camera_model import build_camera_model
from mvis.calibration.initializer import Initializer
from mvis.core.types import FrameObservations, Observation
from mvis.detection.apriltag_board import AprilTagBoard

# a synthetic wide-FOV eucm lens (about 180 deg) on a 1280 x 1024 sensor
CAL = np.array([455.0, 455.5, 630.0, 470.0, 0.66, 1.05, 0.0, 0.0])
K = np.array([[CAL[0], 0, CAL[2]], [0, CAL[1], CAL[3]], [0, 0, 1.0]])
DIST = CAL[4:]


def project_model(model, p, cal):
    """The model's pixel of camera-frame point `p`, or None outside its range."""
    uv = build_camera_model(model, cal).project(np.asarray(p, float))
    return uv if np.isfinite(uv).all() else None


# the same lens as a UCM (alpha form), for the ucm case of the peripheral test
UCM_CAL = np.array([455.0, 455.5, 630.0, 470.0, 0.64])


def _frame(board, R, t, outliers=0, seed=0, model="eucm", cal=CAL):
    """Project every board corner through the true model with pose (R, t)
    (board -> camera); optionally corrupt a few corners by 40-80 px."""
    rng = np.random.default_rng(seed)
    obs = []
    for pid in range(board.num_points()):
        p = R @ board.point_3d(pid) + t
        uv = project_model(model, p, cal) if p[2] > 0 else None
        if uv is None or not (0 <= uv[0] < 1280 and 0 <= uv[1] < 1024):
            continue
        obs.append(Observation(point_id=pid, pixel=np.asarray(uv, float)))
    for o in rng.choice(len(obs), size=outliers, replace=False):
        obs[o].pixel = obs[o].pixel + rng.uniform(40, 80, 2) * rng.choice([-1, 1], 2)
    return FrameObservations(camera_id=0, timestamp=0.0, observations=obs, image_size=(1280, 1024))


def _pose_err(pose, R, t):
    T = np.linalg.inv(pose.matrix())                     # back to board -> camera
    dr = np.linalg.norm(cv2.Rodrigues(T[:3, :3] @ R.T)[0])
    return dr, np.linalg.norm(T[:3, 3] - t)


def _oblique(board, yaw_deg, dist=0.9):
    """Board centre `yaw_deg` off the optical axis at `dist` m."""
    c = board.point_3d(board.num_points() // 2)
    a = np.deg2rad(yaw_deg)
    R = cv2.Rodrigues(np.array([0.2, a, 0.1]))[0]
    return R, np.array([np.sin(a) * dist, 0.0, np.cos(a) * dist]) - R @ c


@pytest.mark.parametrize("model,cal", [("eucm", CAL), ("ucm", UCM_CAL)])
@pytest.mark.parametrize("yaw_deg,dist", [(55, 0.5), (55, 0.9), (65, 0.9), (75, 0.9)])
def test_eucm_peripheral_views_recover_pose(yaw_deg, dist, model, cal):
    """Edge-of-FOV views (corners up to ~90 deg off-axis): the old normalized-
    plane solve returned poses with depths of 1e17-1e33 m here."""
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    R, t = _oblique(board, yaw_deg, dist)
    pose = Initializer(board).init_board_pose(_frame(board, R, t, model=model, cal=cal),
                                              K, cal[4:], distortion_model=model)
    assert pose is not None
    dr, dt = _pose_err(pose, R, t)
    assert dr < 0.01 and dt < 0.01, f"rot err {dr:.4f} rad, trans err {dt:.4f} m"


def test_eucm_central_view_rejects_outlier_corners():
    """10 corners off by 40-80 px: the 8 px RANSAC threshold must apply in px
    (it was 8 normalized units, ~3700 px, i.e. no rejection)."""
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    R, t = _oblique(board, 0.0)
    pose = Initializer(board).init_board_pose(_frame(board, R, t, outliers=10), K, DIST,
                                              distortion_model="eucm")
    dr, dt = _pose_err(pose, R, t)
    assert dr < 1e-3 and dt < 1e-3, f"rot err {dr:.5f} rad, trans err {dt:.5f} m"


def test_too_few_usable_rays_returns_none():
    """At 85 deg only a handful of corners survive the 80 deg ray cut: return
    None rather than a confident wrong pose."""
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    R, t = _oblique(board, 85.0)
    assert Initializer(board).init_board_pose(_frame(board, R, t), K, DIST,
                                              distortion_model="eucm") is None


def test_unprojectable_corners_are_dropped_not_used_as_pixels(monkeypatch):
    """If some corners fail to unproject, the solve must use only the others."""
    import mvis.camera.eucm as eucm_mod
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    R = cv2.Rodrigues(np.array([0.1, 0.3, 0.0]))[0]
    t = np.array([0.1, 0.0, 0.8]) - R @ board.point_3d(board.num_points() // 2)
    f = _frame(board, R, t)
    bad = {tuple(np.round(o.pixel, 6)) for o in f.observations[::7]}
    orig = eucm_mod.unproject_eucm

    def flaky(pixel, cal_vec):
        return None if tuple(np.round(np.asarray(pixel), 6)) in bad else orig(pixel, cal_vec)

    monkeypatch.setattr(eucm_mod, "unproject_eucm", flaky)
    pose = Initializer(board).init_board_pose(f, K, DIST, distortion_model="eucm")
    assert pose is not None
    dr, dt = _pose_err(pose, R, t)
    assert dr < 0.01 and dt < 0.01, f"rot err {dr:.4f} rad, trans err {dt:.4f} m"


@pytest.mark.parametrize("yaw_deg", [0.0, 45.0, 70.0])
def test_omni_radtan_uses_bearing_pnp(yaw_deg):
    """omni_radtan was missing from the wide-FOV branch, so its [xi, k1, k2, p1]
    went to OpenCV's radtan PnP as [k1, k2, p1, p2] (k1 = xi ~ 1.9): every board
    pose was wrong and the cam-only solve could not start."""
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    cal = np.array([1320.0, 1320.5, 630.0, 470.0, 1.85, -0.05, 0.01, 0.0])   # the same lens as Mei
    k = np.array([[cal[0], 0, cal[2]], [0, cal[1], cal[3]], [0, 0, 1.0]])
    R, t = _oblique(board, yaw_deg)
    obs = []
    for pid in range(board.num_points()):
        uv = project_model("omni_radtan", R @ board.point_3d(pid) + t, cal)
        if uv is not None and 0 <= uv[0] < 1280 and 0 <= uv[1] < 1024:
            obs.append(Observation(point_id=pid, pixel=np.asarray(uv, float)))
    f = FrameObservations(camera_id=0, timestamp=0.0, observations=obs, image_size=(1280, 1024))
    pose = Initializer(board).init_board_pose(f, k, cal[4:], distortion_model="omni_radtan")
    assert pose is not None
    dr, dt = _pose_err(pose, R, t)
    assert dr < 0.01 and dt < 0.01, f"rot err {dr:.4f} rad, trans err {dt:.4f} m"


# ── subprocess worker + cv2.error fallback (custom models never go to cv2 as radtan) ──

def _run_worker(tmp_path, board, frames, cal, model):
    """Drive `_pnp_worker` in-process through its pickle in / pickle out."""
    import pickle
    from mvis.calibration.initializer import _pnp_worker
    inp, out = tmp_path / "in.pkl", tmp_path / "out.pkl"
    with open(inp, "wb") as f:
        pickle.dump({
            "board_pts": {pid: board.point_3d(pid).tolist() for pid in range(board.num_points())},
            "frame_data": [[(o.point_id, o.pixel.tolist()) for o in fr.observations]
                           for fr in frames],
            "intrinsics": list(cal[:4]), "dist_coeffs": list(cal[4:]),
            "distortion_model": model, "pnp_method": "ippe",
            "min_points_for_pose": board.min_points_for_pose(),
        }, f)
    _pnp_worker(str(inp), str(out))
    with open(out, "rb") as f:
        return [None if m is None else gtsam.Pose3(np.asarray(m, float))
                for m in pickle.load(f)]


def test_pnp_worker_custom_model_uses_bearings(tmp_path):
    """The subprocess path solves eucm on bearing rays (not eucm coeffs as cv2
    radtan) and applies the in-process ray floor: the 85 deg view -> None."""
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    views = [_oblique(board, y) for y in (0.0, 55.0, 85.0)]
    poses = _run_worker(tmp_path, board, [_frame(board, R, t) for R, t in views], CAL, "eucm")
    for (R, t), pose in zip(views[:2], poses[:2]):
        dr, dt = _pose_err(pose, R, t)
        assert dr < 0.01 and dt < 0.01, f"rot err {dr:.4f} rad, trans err {dt:.4f} m"
    assert poses[2] is None


def test_pnp_worker_unknown_model_raises(tmp_path):
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    with pytest.raises(ValueError, match="Unknown distortion model"):
        _run_worker(tmp_path, board, [], CAL, "bogus")


def test_cv2_error_fallback_stays_on_bearings(monkeypatch):
    """If RANSAC throws cv2.error, the iterative retry must use the eucm
    bearing rays, not the raw pixels with eucm coeffs as radtan."""
    import mvis.calibration.initializer as ini
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    R, t = _oblique(board, 30.0)

    def boom(*a, **k):
        raise cv2.error("forced")

    monkeypatch.setattr(ini.cv2, "solvePnPRansac", boom)
    pose = Initializer(board).init_board_pose(_frame(board, R, t), K, DIST, distortion_model="eucm")
    assert pose is not None
    dr, dt = _pose_err(pose, R, t)
    assert dr < 0.01 and dt < 0.01, f"rot err {dr:.4f} rad, trans err {dt:.4f} m"


def test_init_board_pose_unknown_model_raises():
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    R, t = _oblique(board, 0.0)
    with pytest.raises(ValueError, match="Unknown distortion model"):
        Initializer(board).init_board_pose(_frame(board, R, t), K, DIST, distortion_model="bogus")
