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
"""Equidistant's valid domain: the forward guard at z <= 0 and the unique inverse branch, and
the planar board-pose candidates.

The inverse cases are edge pixels from Monado MGC fits whose polynomial turns
before 90 deg: pixels that have a first-branch ray which cv2.fisheye and gtsam miss or get wrong,
and a pixel past the turn that has no ray. Written against the behaviour, not the implementation.
"""
import logging
import pickle

import cv2
import gtsam
import numpy as np
import pytest

from mvis.calibration.cam_imu_init import pnp_pose_track
from mvis.calibration.corner_accounting import CornerTally, accounting_text, warn_past_90
from mvis.calibration.factors.custom_projections import _kb_fold, _kb_poly
from mvis.calibration.factors.projection import make_projector
from mvis.calibration.initializer import (
    Initializer, board_pose_pnp, _best_planar_pose, _bearing_points, _model_project, _pnp_worker)
from mvis.calibration.mono_sfm import normalize_detections
from mvis.camera.pinhole_equidistant import PinholeEquidistant
from mvis.core.types import FrameObservations, Observation
from mvis.detection.apriltag_board import AprilTagBoard
from mvis.detection.chessboard import Chessboard

MGC0 = np.array([269.70088, 269.96685, 322.53804, 229.26493, 0.08185, -0.03516, 0.21853, -0.12494])
MGC1 = np.array([268.59891, 268.52486, 323.93628, 237.2758, 0.07572, -0.02449, 0.20449, -0.11745])
FOLDING = np.array([300.0, 300.0, 320.0, 240.0, -0.05, -0.05, 0.0, 0.0])     # turns before 90 deg
OUTSIDE = [0.0, 0.0]               # past MGC0's first branch
LATER_BRANCH = [64.01004, 20.073845]   # gtsam 4.2 reports success on a later root here


def _cam(cal):
    return PinholeEquidistant(*cal[:4], np.array(cal[4:]))


def _limit(cal):
    return min(_kb_fold(tuple(float(k) for k in cal[4:])), np.pi / 2)


def _K(cal):
    return np.array([[cal[0], 0, cal[2]], [0, cal[1], cal[3]], [0, 0, 1.0]])


# --- forward domain (#25 acceptance 3) ------------------------------------------------------------

@pytest.mark.parametrize("deg", [90.0, 91.0, 95.0, 120.0, 179.0])
def test_project_is_invalid_at_and_past_90_with_and_without_jacobians(deg):
    a = np.radians(deg)
    p = np.array([np.sin(a), 0.0, np.cos(a) if deg != 90.0 else 0.0])
    cam = _cam(MGC0)
    assert np.isnan(cam.project(p)).all()
    uv, J_pt, J_intr = cam.project(p, jac=True)
    assert np.isnan(uv).all() and J_pt is None and J_intr is None


def test_project_keeps_its_value_just_inside_90():
    a = np.radians(89.9)
    p = np.array([np.sin(a), 0.0, np.cos(a)])
    cam = _cam(MGC0)
    np.testing.assert_array_equal(cam.project(p), cam.project(p, jac=True)[0])
    assert np.isfinite(cam.project(p)).all()


def test_projector_raises_past_90_with_and_without_jacobians():
    """The cost and its linearisation must agree: jac=False mirrored such a corner while jac=True
    dropped it."""
    proj = make_projector("equidistant", MGC0)
    p = gtsam.Point3(1.0, 0.0, -0.1)
    with pytest.raises(RuntimeError):
        proj.project(gtsam.Pose3(), p, jac=False)
    with pytest.raises(RuntimeError):
        proj.project(gtsam.Pose3(), p, jac=True)


def test_past_90_count_warns_to_switch_to_kb4(caplog):
    tally = CornerTally(0, "equidistant", MGC0)
    tally.add(np.array([0.1, 0.0, 1.0]), True)
    tally.add(np.array([1.0, 0.0, -0.1]), False)
    acc = tally.to_dict()
    assert acc["other_invalid"] == 1
    with caplog.at_level(logging.WARNING):
        warn_past_90(acc, "Camera 0", logging.getLogger("t"))
    assert "kb4" in caplog.text and "1 " in caplog.text
    assert "kb4" in accounting_text({**acc, "detected": 2, "not_in_solve": 0})


@pytest.mark.parametrize("model,cal,invalid", [
    ("equidistant", MGC0, 0),                     # nothing to report
    ("radtan", [300.0, 300.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0], 3),   # kb4 is no answer there
])
def test_no_warning_without_a_past_90_count(caplog, model, cal, invalid):
    tally = CornerTally(0, model, cal)
    for _ in range(invalid):
        tally.add(np.array([1.0, 0.0, -0.1]), False)
    with caplog.at_level(logging.WARNING):
        warn_past_90(tally.to_dict(), "Camera 0", logging.getLogger("t"))
    assert caplog.text == ""


def test_warning_tolerates_a_missing_count(caplog):
    with caplog.at_level(logging.WARNING):
        warn_past_90(None, "Camera 0", logging.getLogger("t"))
    assert caplog.text == ""


# --- unique inverse branch --------------------------------------------------------------------------

@pytest.mark.parametrize("cal,pixel", [
    (MGC0, [51.09850311279297, 33.369964599609375]),
    (MGC1, [48.79353713989258, 426.57611083984375]),
    (MGC0, LATER_BRANCH),
])
def test_observed_edge_pixels_get_a_first_branch_ray_that_round_trips(cal, pixel):
    cam = _cam(cal)
    ray = cam.unproject(np.array(pixel))
    assert np.isfinite(ray).all()
    theta = np.arctan(np.hypot(ray[0] / ray[2], ray[1] / ray[2]))
    assert theta <= _limit(cal) + 1e-10
    np.testing.assert_allclose(cam.project(ray), pixel, atol=1e-8)


def test_pixel_past_the_first_branch_is_invalid():
    assert np.isnan(_cam(MGC0).unproject(np.array(OUTSIDE))).all()


def test_later_branch_pixel_is_the_first_root_not_a_later_one():
    """The ray at this pixel has to be on the rising part of theta_d(theta)."""
    cam = _cam(MGC0)
    ray = cam.unproject(np.array(LATER_BRANCH))
    theta = np.arctan(np.hypot(ray[0] / ray[2], ray[1] / ray[2]))
    assert _kb_poly(theta, MGC0[4:])[1] > 0.0


@pytest.mark.parametrize("cal", [MGC0, MGC1, FOLDING, [400.0, 400.0, 300.0, 200.0, 0.0, 0.0, 0.0, 0.0],
                                 [285.0, 285.0, 424.0, 400.0, -0.007, 0.04, -0.037, 0.006]])
def test_round_trip_over_the_whole_first_branch(cal):
    cam = _cam(cal)
    limit = _limit(cal)
    for theta in np.linspace(1e-4, limit * (1.0 - 1e-6), 60):
        for phi in np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False):
            p = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            uv = cam.project(p)
            ray = cam.unproject(uv)
            np.testing.assert_allclose(cam.project(ray), uv, atol=1e-7)
            back = np.arctan(np.hypot(ray[0] / ray[2], ray[1] / ray[2]))
            assert abs(back - theta) < 2e-7


def test_no_ray_past_a_fold_inside_90():
    fold = _kb_fold(tuple(FOLDING[4:]))
    assert fold < np.pi / 2
    r_fold = FOLDING[0] * fold * _kb_poly(fold, FOLDING[4:])[0]
    cam = _cam(FOLDING)
    assert np.isnan(cam.unproject(FOLDING[2:4] + [r_fold + 1e-3, 0.0])).all()
    assert np.isfinite(cam.unproject(FOLDING[2:4] + [r_fold - 1e-3, 0.0])).all()


def test_same_root_as_kb4_on_the_shared_branch():
    from mvis.calibration.factors.custom_projections import unproject_kb
    cam = _cam(MGC0)
    for uv in ([320.0, 240.0], [100.0, 100.0], [51.09850311279297, 33.369964599609375], LATER_BRANCH):
        ray, unit = cam.unproject(np.array(uv)), unproject_kb(np.array(uv), MGC0)
        np.testing.assert_allclose(ray / np.linalg.norm(ray), unit, atol=1e-12)


def test_principal_point_and_depth():
    cam = _cam(MGC0)
    np.testing.assert_allclose(cam.unproject(MGC0[2:4], depth=2.5), [0.0, 0.0, 2.5], atol=1e-12)
    ray = cam.unproject(np.array([300.0, 200.0]), depth=2.5)
    assert ray[2] == 2.5


def test_non_finite_coefficients_give_no_ray_and_no_exception():
    cam = PinholeEquidistant(300.0, 300.0, 320.0, 240.0, np.array([np.nan, 0.0, 0.0, 0.0]))
    assert np.isnan(cam.unproject(np.array([300.0, 200.0]))).all()


# --- the four init paths: invalid pixels leave in lockstep with their ids and board points --------

def _mgc_frame(board, *, invalid=True, later_branch=False, noise_px=0.0):
    """An MGC-like frame. Returns (frame, K, dist, T_board_to_cam)."""
    cam = _cam(MGC0)
    R, _ = cv2.Rodrigues(np.array([0.08, -0.04, 0.03]))
    t = np.array([-0.06, -0.04, 0.65])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    pixels = np.array([cam.project(R @ p + t) for p in board.all_points_3d()])
    if noise_px:
        phase = np.arange(len(pixels), dtype=float)
        pixels += noise_px * np.column_stack((np.sin(phase * 1.7), np.cos(phase * 1.3)))
    if invalid:
        pixels[0] = OUTSIDE
    if later_branch:
        pixels[1] = LATER_BRANCH    # a mismatched board point: RANSAC's ordinary outlier
    obs = [Observation(point_id=i, pixel=p, row=int(p[1])) for i, p in enumerate(pixels)]
    frame = FrameObservations(camera_id=0, timestamp=0.0, observations=obs, image_size=(640, 480))
    return frame, _K(MGC0), MGC0[4:], T


@pytest.fixture
def board():
    return Chessboard(rows=7, cols=5, square_size=0.03)


def _assert_pose(pose, T, rot=np.deg2rad(0.05), trans=1e-4):
    est = pose.inverse().matrix()
    dr, _ = cv2.Rodrigues(est[:3, :3] @ T[:3, :3].T)
    assert np.linalg.norm(dr) < rot
    assert np.linalg.norm(est[:3, 3] - T[:3, 3]) < trans


def test_bearing_points_drops_the_invalid_pixel_in_lockstep(board):
    frame, K, dist, _ = _mgc_frame(board)
    img = np.array([o.pixel for o in frame.observations])
    pts, keep = _bearing_points(img, K, dist, "equidistant")
    assert not keep[0] and keep[1:].all()
    assert len(pts) == keep.sum() and np.isfinite(pts).all()


def test_direct_path_filters_and_ignores_a_later_branch_outlier(board):
    frame, K, dist, T = _mgc_frame(board, later_branch=True)
    pose = Initializer(board).init_board_pose(frame, K, dist, "equidistant", max_reprojection_px=0.1)
    assert pose is not None
    _assert_pose(pose, T)


def test_direct_path_refuses_a_frame_with_too_few_valid_rays(board):
    obs = [Observation(point_id=i, pixel=np.array(OUTSIDE), row=0)
           for i in range(board.min_points_for_pose())]
    frame = FrameObservations(camera_id=0, timestamp=0.0, observations=obs, image_size=(640, 480))
    assert Initializer(board).init_board_pose(frame, _K(MGC0), MGC0[4:], "equidistant") is None


def _worker_payload(board, frame, K, dist, **extra):
    return {"board_pts": {i: board.point_3d(i).tolist() for i in range(board.num_points())},
            "frame_data": [[(o.point_id, o.pixel.tolist()) for o in frame.observations]],
            "intrinsics": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], "dist_coeffs": dist.tolist(),
            "distortion_model": "equidistant", **extra}


def test_worker_filters_the_invalid_pixel(board, tmp_path):
    frame, K, dist, T = _mgc_frame(board)
    src, dst = tmp_path / "in.pkl", tmp_path / "out.pkl"
    src.write_bytes(pickle.dumps(_worker_payload(board, frame, K, dist)))
    _pnp_worker(str(src), str(dst))
    result = pickle.loads(dst.read_bytes())[0]
    assert result is not None
    assert np.linalg.norm(np.linalg.inv(np.asarray(result))[:3, 3] - T[:3, 3]) < 1e-4


def test_spawned_worker_filters_the_invalid_pixel(board):
    frame, K, dist, T = _mgc_frame(board)
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    pose = Initializer(board).init_board_poses([frame], intr, dist, "equidistant", use_subprocess=True)[0]
    assert pose is not None
    _assert_pose(pose, T)


def test_cam_imu_track_drops_the_invalid_corner_in_lockstep():
    gx, gy = np.meshgrid(np.linspace(-0.18, 0.18, 4), np.linspace(-0.18, 0.18, 4))
    p_B = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    ids = np.arange(len(p_B))
    T_BtoC = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.08, -0.04, 0.02), gtsam.Point3(0.02, -0.01, 1.2))
    proj = make_projector("equidistant", MGC0)
    uv = np.array([proj.project(gtsam.Pose3(), T_BtoC.transformFrom(gtsam.Point3(*p)), jac=False)[0]
                   for p in p_B])
    uv[-1] = OUTSIDE
    _t, poses, counts = pnp_pose_track([0.0], [ids], [uv], p_B, MGC0, model="equidistant")
    assert counts.tolist() == [len(ids) - 1]
    want = T_BtoC.inverse()
    assert np.linalg.norm(np.asarray(poses[0].translation()) - np.asarray(want.translation())) < 1e-4
    assert np.linalg.norm(gtsam.Rot3.Logmap(poses[0].rotation().between(want.rotation()))) < 1e-4


def test_cam_imu_track_skips_a_frame_with_too_few_valid_rays():
    p_B = np.array([[x, y, 0.0] for x in (0, 0.1, 0.2) for y in (0, 0.1, 0.2)])
    ids = np.arange(len(p_B))
    good = np.array([[320.0 + 40 * x, 240.0 + 40 * y] for x in range(3) for y in range(3)])
    bad = np.tile(OUTSIDE, (len(p_B), 1))
    t, _poses, _n = pnp_pose_track([0.0, 0.1], [ids, ids], [bad, good], p_B, MGC0, model="equidistant")
    assert t.tolist() == [0.1]


def test_mono_sfm_keeps_the_valid_and_drops_the_invalid_with_its_id():
    ids, xy = normalize_detections([np.array([10, 11])], [np.array([LATER_BRANCH, OUTSIDE])], MGC0,
                                   model="equidistant")[0]
    np.testing.assert_array_equal(ids, [10])
    assert xy.shape == (1, 2) and np.isfinite(xy).all()
    assert np.arctan(np.linalg.norm(xy[0])) < np.deg2rad(72.0)


# --- planar pose candidates and the pixel gate ----------------------------------------------------

def _branch_frame(n_outliers):
    """The view where ITERATIVE RANSAC selects a wrong rotation branch from this seed."""
    board = AprilTagBoard(rows=6, cols=6, tag_size=0.03, tag_spacing=0.009, tag_family="t36h11")
    cam = PinholeEquidistant(270.0, 270.0, 323.0, 230.0, np.array([0.085, -0.044, 0.226, -0.127]))
    R, _ = cv2.Rodrigues(np.array([-1.904, 1.753, -1.109]))
    t = np.array([0.049, 0.074, 0.343])
    pixels = np.array([cam.project(R @ p + t) for p in board.all_points_3d()])
    if n_outliers:
        pixels[np.linspace(0, len(pixels) - 1, n_outliers, dtype=int)] += np.array([300.0, -250.0])
    obs = [Observation(point_id=i, pixel=p, row=int(p[1])) for i, p in enumerate(pixels)]
    frame = FrameObservations(camera_id=0, timestamp=0.0, observations=obs, image_size=(640, 480))
    return board, frame, R


SEED_INTR = np.array([413.0, 413.0, 319.5, 239.5])


def test_the_branch_case_really_is_one_ransac_alone_gets_wrong():
    """Guards the test below: without the candidates this view must fail."""
    board, frame, R = _branch_frame(0)
    obj = np.array([board.point_3d(o.point_id) for o in frame.observations])
    img = np.array([o.pixel for o in frame.observations])
    K = _K(SEED_INTR)
    pts, keep = _bearing_points(img, K, np.zeros(4), "equidistant")
    ok, rv, _tv, _in = cv2.solvePnPRansac(obj[keep], pts, np.eye(3), None, reprojectionError=8.0 / 413.0,
                                          iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    dr, _ = cv2.Rodrigues(cv2.Rodrigues(rv)[0] @ R.T)
    assert np.linalg.norm(dr) > np.deg2rad(5.0)


@pytest.mark.parametrize("use_subprocess", [False, True])
@pytest.mark.parametrize("n_outliers", [0, 4])
def test_candidates_recover_the_planar_branch_with_and_without_outliers(use_subprocess, n_outliers):
    board, frame, R = _branch_frame(n_outliers)
    pose = Initializer(board).init_board_poses([frame], SEED_INTR, np.zeros(4), "equidistant",
                                               use_subprocess=use_subprocess, max_reprojection_px=10.0)[0]
    assert pose is not None
    T = pose.inverse().matrix()
    dr, _ = cv2.Rodrigues(T[:3, :3] @ R.T)
    assert np.linalg.norm(dr) < np.deg2rad(5.0)
    assert (T[:3, :3] @ board.all_points_3d().T + T[:3, 3:4]).T[:, 2].min() > 0.0


@pytest.mark.parametrize("use_subprocess", [False, True])
def test_reprojection_gate_is_in_pixels_on_both_paths(board, use_subprocess):
    frame, K, dist, _ = _mgc_frame(board, invalid=False, noise_px=0.2)
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    init = Initializer(board)
    assert init.init_board_poses([frame], intr, dist, "equidistant", use_subprocess=use_subprocess,
                                 max_reprojection_px=1.0)[0] is not None
    assert init.init_board_poses([frame], intr, dist, "equidistant", use_subprocess=use_subprocess,
                                 max_reprojection_px=0.01)[0] is None


@pytest.mark.parametrize("n_outliers", [0, 4])
@pytest.mark.parametrize("model,cal", [("equidistant", None), ("kb4", None), ("eucm", [0.6, 1.1])])
def test_direct_and_worker_give_the_same_pose(tmp_path, model, cal, n_outliers):
    board, frame, _R = _branch_frame(n_outliers)
    dist = np.asarray(cal, float) if cal else np.zeros(4)
    K = _K(SEED_INTR)
    direct = Initializer(board).init_board_pose(frame, K, dist, model, max_reprojection_px=10.0)
    src, dst = tmp_path / "in.pkl", tmp_path / "out.pkl"
    pay = _worker_payload(board, frame, K, dist, max_reprojection_px=10.0, distortion_model=model,
                          min_points_for_pose=int(board.min_points_for_pose()))
    src.write_bytes(pickle.dumps(pay))
    _pnp_worker(str(src), str(dst))
    worker = pickle.loads(dst.read_bytes())[0]
    assert (direct is None) == (worker is None)
    if direct is not None:
        np.testing.assert_allclose(direct.matrix(), np.asarray(worker), atol=1e-9)


def _count_ransac(monkeypatch):
    calls = []
    real = cv2.solvePnPRansac

    def spy(*a, **k):
        calls.append(k["flags"])
        return real(*a, **k)
    monkeypatch.setattr(cv2, "solvePnPRansac", spy)
    return calls


def test_robust_planar_candidate_stays_out_of_a_frame_that_fits(board, monkeypatch):
    frame, K, dist, T = _mgc_frame(board, invalid=False, noise_px=0.2)
    calls = _count_ransac(monkeypatch)
    pose = Initializer(board).init_board_pose(frame, K, dist, "equidistant", max_reprojection_px=10.0)
    assert pose is not None
    assert calls == [cv2.SOLVEPNP_ITERATIVE]


def test_robust_planar_candidate_is_tried_when_nothing_fits(monkeypatch):
    board, frame, _R = _branch_frame(4)
    calls = _count_ransac(monkeypatch)
    Initializer(board).init_board_pose(frame, _K(SEED_INTR), np.zeros(4), "equidistant")
    assert calls == [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE]


def test_model_project_marks_z_behind_invalid_for_equidistant_only_by_the_model():
    K = _K(MGC0)
    p = np.array([[0.1, 0.0, 1.0], [1.0, 0.0, -0.1]])
    uv = _model_project(p, K, MGC0[4:], "equidistant")
    assert np.isfinite(uv[0]).all() and np.isnan(uv[1]).all()


def test_model_project_lets_kb4_score_a_valid_ray_past_90():
    """A correct past-90 pose has z <= 0 corners: 'valid forward projection under the camera's
    model' is the rule, not positive depth."""
    cal = np.array([300.0, 300.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])
    a = np.radians(95.0)
    uv = _model_project(np.array([[np.sin(a), 0.0, np.cos(a)]]), _K(cal), cal[4:], "kb4")
    assert np.isfinite(uv).all()


def test_candidate_with_too_few_valid_projections_is_rejected():
    """A pose that loses most of the board can't win on the few corners it still projects, however
    small their error."""
    obj = np.array([[x, y, 0.0] for x in np.linspace(-0.2, 0.2, 5) for y in np.linspace(-0.2, 0.2, 5)])
    K, dist = _K(MGC0), MGC0[4:]
    R = cv2.Rodrigues(np.array([0.0, np.pi / 2, 0.0]))[0]
    rv, tv = cv2.Rodrigues(R)[0], np.array([[0.0], [0.0], [0.0]])    # half the board at z <= 0
    p_cam = obj @ R.T + tv.reshape(1, 3)
    img = _model_project(p_cam, K, dist, "equidistant")
    assert np.isnan(img).mean() > 0.1
    img = np.nan_to_num(img, nan=320.0)                # so the z > 0 half has zero error
    r, t, err = _best_planar_pose(obj, img, np.zeros((len(obj), 2)), [(rv, tv)], K, dist, "equidistant")
    assert r is None or not np.allclose(r, rv)


def test_non_planar_points_keep_their_solve():
    rng = np.random.default_rng(0)
    obj = rng.uniform(-0.2, 0.2, (40, 3))
    R = cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))[0]
    t = np.array([0.02, -0.01, 1.0])
    K, dist = _K(MGC0), MGC0[4:]
    img = _model_project(obj @ R.T + t, K, dist, "equidistant")
    pose = board_pose_pnp(obj, img, K, dist, "equidistant", cv2.SOLVEPNP_ITERATIVE, 8, 1.0)
    assert pose is not None
    dr, _ = cv2.Rodrigues(cv2.Rodrigues(pose[0])[0] @ R.T)
    assert np.linalg.norm(dr) < 1e-4
