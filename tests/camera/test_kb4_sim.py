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
"""kb4 on synthetic truth (plan kb4-kb8, gate G3): seeds 0-4, sigma 0.05 px.

Frames come from `tests/_reference/ref_kb4.py`, a frozen formula written apart from `project_kb`,
so a wrong kernel cannot pass by generating its own data. Tolerances are the plan's proposals.
"""
import math

import gtsam
import numpy as np
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.calibration.factors.custom_projections import _kb_fold, project_kb
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
from mvis.core.types import FrameObservations, Observation
from mvis.detection.chessboard import Chessboard
from tests._reference import ref_kb4

_IMG = (1000, 1000)
# monotonic well past 110 deg, fold above 130 deg; fx 250 keeps ~105 deg inside the image
GT = np.array([250.0, 251.0, 501.0, 498.0, 0.02, -0.005, 5e-4, 1e-5])
SEEDS = [0, 1, 2, 3, 4]
SIGMA = 0.05


def _look_at(c, rng):
    z = -c / np.linalg.norm(c)
    x = np.cross([0.0, 1.0, 0.0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    R = gtsam.Rot3(np.column_stack([x, np.cross(z, x), z]))
    return R.compose(gtsam.Rot3.Rodrigues(*rng.uniform(-0.4, 0.4, 3)))


def _frames(board, seed, n=40, sigma=SIGMA, max_dir_deg=80.0, cal=GT, reach_deg=100.0):
    """Views all around the lens, 0.3-0.6 m. The first five must reach past `reach_deg` (views are
    rejected until they do), so th_max > 100 deg holds on every seed. Returns (frames, th_max,
    corners with z <= 0, poses)."""
    rng = np.random.RandomState(seed)
    pts = board.all_points_3d().astype(float)
    ctr = pts.mean(0)
    frames, poses, t, th_max, n_behind = [], [], 0.0, 0.0, 0
    for _ in range(n * 200):
        if len(frames) == n:
            break
        a, ph = np.deg2rad(rng.uniform(0, max_dir_deg)), rng.uniform(0, 2 * np.pi)
        c = rng.uniform(0.3, 0.6) * np.array([np.sin(a) * np.cos(ph), np.sin(a) * np.sin(ph), np.cos(a)])
        R = _look_at(c, rng)
        T = gtsam.Pose3(R, gtsam.Point3(*(c - R.matrix() @ ctr)))
        obs, th, behind = [], 0.0, 0
        for i, p in enumerate(pts):
            P = np.asarray(T.transformFrom(gtsam.Point3(*p)))
            uv = np.array(ref_kb4.project(P, cal))
            if not (5 <= uv[0] < _IMG[0] - 5 and 5 <= uv[1] < _IMG[1] - 5):
                continue
            th = max(th, math.atan2(math.hypot(P[0], P[1]), P[2]))
            behind += P[2] <= 0
            obs.append(Observation(point_id=i, pixel=uv + rng.standard_normal(2) * sigma, timestamp=t))
        if len(obs) < 0.8 * len(pts):
            continue
        if len(frames) < 5 and math.degrees(th) <= reach_deg:
            continue
        frames.append(FrameObservations(camera_id=0, timestamp=t, observations=obs, image_size=_IMG))
        poses.append(T)
        th_max, n_behind = max(th_max, th), n_behind + behind
        t += 0.1
    assert len(frames) == n
    return frames, th_max, n_behind, poses


def _solve(board, frames, model):
    pc = PipelineConfig(max_iterations=30, max_iterations_single=30, pixel_sigma=1.0,
                        huber_threshold=1.345, corner_subsample="all", optimize_intrinsics=True,
                        optimize_time_offset=False, max_init_frames=100,
                        init_frame_selection="uniform", reference_camera=None,
                        rerun_after_outlier_rejection=False,
                        distortion_models={0: model}, verbose=False)
    return CalibrationPipeline(board, pc).run({0: frames}, {0: _IMG})


@pytest.fixture(scope="module")
def board():
    return Chessboard(7, 9, 0.04)


@pytest.fixture(scope="module")
def past90(board):
    out = {}
    for s in SEEDS:
        fr, th, nb, _ = _frames(board, s)
        out[s] = (fr, th, nb, _solve(board, fr, "kb4"))
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_past_90_recovery(past90, seed):
    frames, th_max, n_behind, r = past90[seed]
    assert math.degrees(th_max) > 100.0, "the frames must reach past 100 deg"
    assert n_behind > 0
    cam = r.cameras[0]
    assert cam.distortion_model == "kb4" and len(cam.distortion) == 4
    assert np.abs(cam.intrinsics - GT[:4]).max() < 0.1
    est = np.r_[cam.intrinsics, cam.distortion]
    th = np.linspace(0.01, th_max, 200)
    gap = np.array([abs(ref_kb4.theta_d_px(est, t) - ref_kb4.theta_d_px(GT, t)) for t in th])
    assert gap.max() < 0.5
    assert r.overall_rms < 0.08


def _seeded_solve(board, frames, poses, model):
    """One solve from the true poses and intrinsics, so no frame is lost to the init: the count is
    then over exactly the generated corners. Poses are T_board_cam, the optimizer's form."""
    opts = OptimizerOptions(max_iterations=5, pixel_sigma=1.0, huber_threshold=1.345, verbose=False)
    return Optimizer(board, opts).solve_single_camera(
        frames, GT[:4].copy(), GT[4:].copy(), [T.inverse() for T in poses], model, _IMG, camera_id=0)


@pytest.mark.parametrize("seed", SEEDS)
def test_past_90_accounting(board, seed):
    """kb4 uses every corner, past 90 deg included; equidistant has no pixel for those with z <= 0,
    and its count equals what was generated. No RMS ratio: both share the polynomial."""
    frames, th_max, n_behind, poses = _frames(board, seed)
    assert n_behind > 0
    n_all = sum(len(f.observations) for f in frames)
    a = _seeded_solve(board, frames, poses, "kb4").corner_accounting[0]
    assert a["used"] == a["n_corners"] == n_all
    assert a["past_fold"] == a["other_invalid"] == a["no_pose_frames"] == 0
    assert a["max_theta_deg"] == pytest.approx(math.degrees(th_max), abs=0.5)
    eq = _seeded_solve(board, frames, poses, "equidistant").corner_accounting[0]
    assert eq["other_invalid"] == n_behind
    assert eq["used"] + eq["other_invalid"] == n_all and eq["past_fold"] == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_pipeline_count_against_detections(past90, seed):
    frames, _th, _nb, r = past90[seed]
    a = r.corner_accounting[0]
    assert a["used"] == r.reprojection_errors[0].num_observations
    assert a["past_fold"] == a["other_invalid"] == 0 and a["fold_level"] in ("none", "warn")
    assert a["detected"] == sum(len(f.observations) for f in frames)
    assert 0 <= a["not_in_solve"]
    assert a["not_in_solve"] <= 0.1 * a["detected"]      # the pipeline may drop a diverged view
    assert a["max_theta_deg"] > 95.0


@pytest.mark.parametrize("seed", SEEDS)
def test_sub_90_matches_equidistant(board, seed):
    """Below 90 deg the two are one polynomial: same mapping, same error."""
    frames, th_max, n_behind, _ = _frames(board, seed, max_dir_deg=60.0, reach_deg=0.0)
    assert n_behind == 0
    r4, re = _solve(board, frames, "kb4"), _solve(board, frames, "equidistant")
    c4, ce = r4.cameras[0], re.cameras[0]
    assert abs(r4.overall_rms - re.overall_rms) < 1e-3
    assert np.abs(c4.intrinsics - ce.intrinsics).max() < 0.05
    th = np.linspace(0.01, th_max, 100)
    e4, ee = np.r_[c4.intrinsics, c4.distortion], np.r_[ce.intrinsics, ce.distortion]
    gap = [abs(ref_kb4.theta_d_px(e4, t) - ref_kb4.theta_d_px(ee, t)) for t in th]
    assert max(gap) < 0.05


def _frozen_solve(board, frames, poses, cal):
    """One solve with the intrinsics pinned at `cal`: what the count says of a fixed polynomial."""
    opts = OptimizerOptions(max_iterations=2, pixel_sigma=1.0, huber_threshold=1.345,
                            frozen_intrinsics_cameras={0}, verbose=False)
    return Optimizer(board, opts).solve_single_camera(
        frames, np.array(cal[:4]), np.array(cal[4:]), [T.inverse() for T in poses], "kb4", _IMG,
        camera_id=0)


def test_corners_past_the_fold_are_counted_and_alarm_without_stopping(board, caplog):
    """Coefficients perturbed so corners cross the fold: they are counted, the level is alarm, the
    solve still returns, and RMS drops only because those corners left it, so it is compared on
    the same corner set."""
    from mvis.calibration.pipeline_cam import _log_kb_valid_domain
    frames, th_max, _nb, poses = _frames(board, 0, n=12, sigma=0.0)
    cal = GT.copy()
    cal[4:] = [-0.125, 0.0, 0.0, 0.0]         # folds at 93.6 deg, inside these views
    fold = _kb_fold(tuple(cal[4:]))
    assert math.radians(90) < fold < th_max
    want = 0
    for f, T in zip(frames, poses):
        for o in f.observations:
            P = np.asarray(T.transformFrom(gtsam.Point3(*board.all_points_3d()[o.point_id])))
            want += math.atan2(math.hypot(P[0], P[1]), P[2]) >= fold
    assert want > 0
    r = _frozen_solve(board, frames, poses, cal)
    a = r.corner_accounting[0]
    assert a["past_fold"] > 0
    assert abs(a["past_fold"] - want) <= 0.1 * want       # the poses move a little in two iterations
    assert a["fold_level"] == "alarm"
    assert a["used"] + a["past_fold"] + a["other_invalid"] == a["n_corners"]
    assert a["fold_deg"] == pytest.approx(math.degrees(fold))
    assert a["max_theta_deg"] > a["fold_deg"]
    assert a["max_theta_used_deg"] < a["fold_deg"]
    with caplog.at_level("WARNING"):
        _log_kb_valid_domain(r, {0: _IMG})
    assert "ALARM" in caplog.text


def _interp_factor(cal):
    """Hand-built cam-IMU board factor (plan G3): corners at 30, 95 and, for `cal`, past the fold."""
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, cal, model="kb4")
    pose = gtsam.Pose3()
    fold = _kb_fold(tuple(cal[4:]))
    degs = [30.0, 95.0, 110.0] + ([math.degrees(fold) + 3.0] if fold < math.pi else [])
    p_B = np.array([[0.8 * math.sin(math.radians(d)), 0.6 * math.sin(math.radians(d)),
                     math.cos(math.radians(d))] for d in degs])
    uv = np.array([(0.0, 0.0)] * len(degs))
    for i in range(len(degs)):
        q = project_kb(p_B[i], np.asarray(cal))[0]
        uv[i] = q if q is not None else (100.0, 100.0)
    f = CameraCalibInterpFactor(uv, p_B, 0.0, 0.1, base, (1000, 1000), rolling_mode=0)
    return f, pose, base, degs


def test_cam_imu_factor_past_90_jacobians_and_zero_rows():
    cal = np.array([250.0, 251.0, 501.0, 498.0, 0.02, -0.005, 5e-4, 1e-5])
    f, pose, base, degs = _interp_factor(cal)
    args = (pose, pose, gtsam.Pose3(), np.zeros(base.DIM))
    err, Hs = f.error_and_jacobians(*args)
    Hn = f._numerical_jacobians(*args)
    assert np.abs(err).max() < 1e-9
    for a, n in zip(Hs, Hn):
        np.testing.assert_allclose(a, n, rtol=1e-4, atol=1e-3)
    # a corner past the fold: zero residual, zero rows, NaN in the stats. This is the silent drop.
    fold_cal = np.array([250.0, 251.0, 501.0, 498.0, -0.125, 0.0, 0.0, 0.0])
    f, pose, base, degs = _interp_factor(fold_cal)
    args = (pose, pose, gtsam.Pose3(), np.zeros(base.DIM))
    err, Hs = f.error_and_jacobians(*args)
    i = len(degs) - 1
    assert (err[2 * i:2 * i + 2] == 0).all() and all((H[2 * i:2 * i + 2] == 0).all() for H in Hs)
    assert np.isnan(f.corner_norms(*args)[i])


def test_cam_imu_accounting_counts_the_silent_drop():
    from mvis.calibration.pipeline_cam_imu import _corner_accounting
    cal = np.array([250.0, 251.0, 501.0, 498.0, -0.125, 0.0, 0.0, 0.0])
    f, pose, base, degs = _interp_factor(cal)
    vals = gtsam.Values()
    k = [gtsam.symbol("x", 0), gtsam.symbol("x", 1), gtsam.symbol("b", 0), gtsam.symbol("c", 0)]
    vals.insert(k[0], pose)
    vals.insert(k[1], pose)
    vals.insert(k[2], gtsam.Pose3())
    vals.insert(k[3], np.zeros(base.DIM))
    f.make_factor(*k, sigma_px=1.0)
    (a,) = _corner_accounting([(0, 0, f)], vals, [base], [(1000, 1000)],
                                [{"no_pose_frames": 2, "no_pose_corners": 40, "detected": 50}])
    n_past = sum(math.radians(d) >= _kb_fold(tuple(cal[4:])) for d in degs)
    assert n_past == 3
    assert a["past_fold"] == n_past and a["used"] == len(degs) - n_past and a["other_invalid"] == 0
    assert a["no_pose_frames"] == 2 and a["fold_level"] == "alarm"
    assert a["max_theta_deg"] == pytest.approx(max(degs), abs=1e-6)
