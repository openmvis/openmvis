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
"""LM termination semantics the mvis solvers rely on (cam-imu-speedup L1).

Pins, against the installed GTSAM wrapper:

* `optimize()` stops after the first accepted iteration whose relative OR
  absolute cost decrease is below the tolerance, and returns exactly that
  iterate (the manual `iterate()` trajectory, cut at that point).
* With the absolute tolerance disabled the stop is the first iteration whose
  relative decrease is below 1e-5.
* The lambda upper bound caps the retry loop of an iteration that cannot
  decrease the cost (7 tries at 1e2 vs 10 at GTSAM's 1e5), and the values
  returned from such a stall are the last accepted iterate.
* The lambda lower bound floors the damping of a converging run.
"""

import numpy as np
import gtsam
import pytest

from mvis.calibration.lm_params import (
    LM_LAMBDA_LOWER_DEFAULT, LM_LAMBDA_UPPER_DEFAULT, make_lm_params)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _pose_graph(seed=0, n=8, noise=0.05, init_noise=0.3):
    """Noisy Pose2 loop (prior + n odometry + loop closure) with a perturbed
    init: converges in ~5 LM iterations to a non-zero cost, so the relative
    decrease crosses 1e-5 gradually."""
    rng = np.random.RandomState(seed)
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
    odo = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
    g.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior))
    gt = [gtsam.Pose2(0, 0, 0)]
    step = gtsam.Pose2(1.0, 0.0, 2 * np.pi / n)
    for i in range(1, n):
        gt.append(gt[-1].compose(step))
        meas = step.compose(gtsam.Pose2(*(rng.randn(3) * noise)))
        g.add(gtsam.BetweenFactorPose2(i - 1, i, meas, odo))
    loop = gt[n - 1].between(gt[0]).compose(gtsam.Pose2(*(rng.randn(3) * noise)))
    g.add(gtsam.BetweenFactorPose2(n - 1, 0, loop, odo))
    for i in range(n):
        v.insert(i, gt[i].compose(gtsam.Pose2(*(rng.randn(3) * init_noise))))
    return g, v


def _values_arrays(values):
    return [values.atPose2(k).matrix() for k in sorted(values.keys())]


def _trajectory(graph, init, params, n_iter):
    """Cost after 0..n_iter manual `iterate()` calls, plus the values at each."""
    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    errs = [opt.error()]
    vals = [_values_arrays(opt.values())]
    for _ in range(n_iter):
        opt.iterate()
        errs.append(opt.error())
        vals.append(_values_arrays(opt.values()))
    return np.array(errs), vals


def _first_converged(errs, rel_tol, abs_tol):
    """GTSAM `checkConvergence`: first k with rel OR abs decrease <= tol."""
    for k in range(1, len(errs)):
        dec = errs[k - 1] - errs[k]
        if dec / errs[k - 1] <= rel_tol or dec <= abs_tol:
            return k
    raise AssertionError("trajectory never converged")


def _assert_same_values(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------------------
# make_lm_params
# ---------------------------------------------------------------------------

def test_make_lm_params_defaults_and_overrides():
    p = make_lm_params(30)
    assert p.getMaxIterations() == 30
    # lower bound = GTSAM's 0 (bitwise-preserving), upper 1e2 (retry cap)
    assert p.getlambdaLowerBound() == LM_LAMBDA_LOWER_DEFAULT == 0.0
    assert p.getlambdaUpperBound() == LM_LAMBDA_UPPER_DEFAULT == 1e2
    # tolerances untouched = GTSAM defaults (what the cam-IMU solve uses)
    assert p.getRelativeErrorTol() == 1e-5
    assert p.getAbsoluteErrorTol() == 1e-5
    assert p.getlambdaInitial() == 1e-5

    q = make_lm_params(7, absolute_error_tol=1e-9, relative_error_tol=1e-8,
                       lambda_lower=0.0, lambda_upper=1e5, lambda_initial=1e-3)
    assert q.getMaxIterations() == 7
    assert q.getAbsoluteErrorTol() == 1e-9 and q.getRelativeErrorTol() == 1e-8
    assert q.getlambdaLowerBound() == 0.0 and q.getlambdaUpperBound() == 1e5
    assert q.getlambdaInitial() == 1e-3
    # None keeps GTSAM's bounds; lambda_initial <= 0 is "unset"
    r = make_lm_params(1, lambda_lower=None, lambda_upper=None, lambda_initial=0.0)
    assert r.getlambdaLowerBound() == 0.0 and r.getlambdaUpperBound() == 1e5
    assert r.getlambdaInitial() == 1e-5


# ---------------------------------------------------------------------------
# production defaults + plumbing (the actual L1 fix; a regression of the
# cam-only stall would start by one of these drifting back)
# ---------------------------------------------------------------------------

def test_production_defaults_pin_the_l1_fix():
    from mvis.core.config import CamImuCalibConfig, OptimizerConfig
    from mvis.calibration.optimizer import OptimizerOptions
    from mvis.calibration.pipeline_cam import PipelineConfig

    oc = OptimizerConfig()                      # `mvis cam` CLI path
    assert oc.relative_error_tol == 1e-5        # was 1e-10: the stall
    assert oc.absolute_error_tol == 1e-5
    assert (oc.lm_lambda_lower, oc.lm_lambda_upper) == (0.0, 1e2)
    cc = CamImuCalibConfig()                    # joint cam-IMU solve
    assert (cc.lm_lambda_lower, cc.lm_lambda_upper) == (0.0, 1e2)
    oo = OptimizerOptions()                     # what Optimizer actually reads
    assert (oo.lm_lambda_lower, oo.lm_lambda_upper) == (0.0, 1e2)
    assert oo.relative_error_tol == 1e-5
    pc = PipelineConfig()                       # run_calibration / regression path
    assert (pc.lm_lambda_lower, pc.lm_lambda_upper) == (0.0, 1e2)


def test_lambda_bounds_are_forwarded_from_config():
    from types import SimpleNamespace
    from mvis.core.config import OptimizerConfig
    from mvis.calibration.pipeline_cam import PipelineConfig
    from mvis.calibration.pipeline_cam_imu import _lm_params

    # cam-IMU: cfg.lm_lambda_* -> LM params (and duck-typed cfg -> defaults)
    p = _lm_params(SimpleNamespace(max_iterations=12, lm_lambda_lower=1e-6,
                                   lm_lambda_upper=1e3))
    assert p.getMaxIterations() == 12
    assert p.getlambdaLowerBound() == 1e-6 and p.getlambdaUpperBound() == 1e3
    assert p.getRelativeErrorTol() == 1e-5 and p.getAbsoluteErrorTol() == 1e-5
    q = _lm_params(SimpleNamespace(max_iterations=3))
    assert q.getlambdaLowerBound() == LM_LAMBDA_LOWER_DEFAULT
    assert q.getlambdaUpperBound() == LM_LAMBDA_UPPER_DEFAULT
    # cam-only pipeline: config.optimizer.lm_lambda_* -> PipelineConfig
    cfg = SimpleNamespace(
        optimizer=OptimizerConfig(lm_lambda_lower=1e-6, lm_lambda_upper=1e3),
        cameras=[SimpleNamespace(camera_id=0, distortion_model="radtan")],
        optimize_time_offset=False, rerun_outliers=False,
        corner_subsample="all", reference_camera=0)
    pc = PipelineConfig.from_calibration_config(cfg)
    assert (pc.lm_lambda_lower, pc.lm_lambda_upper) == (1e-6, 1e3)


# ---------------------------------------------------------------------------
# termination = first iteration under tolerance; result = that iterate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_tol,abs_tol", [
    (1e-5, 1e-5),   # shipped defaults (GTSAM's): OR of the two
    (1e-5, 0.0),    # relative only: stop at the first rel decrease < 1e-5
    (0.0, 1e-5),    # absolute only
])
def test_optimize_stops_at_first_converged_iterate(rel_tol, abs_tol):
    graph, init = _pose_graph()
    params = make_lm_params(100, absolute_error_tol=abs_tol,
                            relative_error_tol=rel_tol,
                            lambda_lower=0.0, lambda_upper=1e5)
    errs, vals = _trajectory(graph, init, params, n_iter=8)
    # the fixture really converges and the criterion is crossed, not skipped
    assert errs[-1] < errs[0] * 1e-3
    k = _first_converged(errs, rel_tol, abs_tol)
    assert 2 <= k < 8
    if abs_tol == 0.0:
        rel = (errs[k - 1] - errs[k]) / errs[k - 1]
        rel_prev = (errs[k - 2] - errs[k - 1]) / errs[k - 2]
        assert rel <= 1e-5 < rel_prev

    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    res = opt.optimize()
    assert opt.iterations() == k
    assert opt.error() == errs[k]
    assert graph.error(res) == errs[k]
    _assert_same_values(_values_arrays(res), vals[k])


def test_tighter_tolerance_only_appends_iterations():
    """1e-10 (the old cam-only setting) walks the same prefix as 1e-5 and
    then keeps going — so moving to 1e-5 returns an earlier iterate of the
    same trajectory, never a different one."""
    graph, init = _pose_graph()
    kw = dict(lambda_lower=0.0, lambda_upper=1e5)
    loose = gtsam.LevenbergMarquardtOptimizer(
        graph, init, make_lm_params(100, absolute_error_tol=1e-5,
                                    relative_error_tol=1e-5, **kw))
    loose.optimize()
    k = loose.iterations()
    tight_params = make_lm_params(100, absolute_error_tol=1e-10,
                                  relative_error_tol=1e-10, **kw)
    tight = gtsam.LevenbergMarquardtOptimizer(graph, init, tight_params)
    tight.optimize()
    assert tight.iterations() > k
    _errs, vals = _trajectory(graph, init, tight_params, n_iter=k)
    _assert_same_values(_values_arrays(loose.values()), vals[k])


# ---------------------------------------------------------------------------
# lambda bounds
# ---------------------------------------------------------------------------

class _Stall:
    """Scalar CustomFactor whose Jacobian has the wrong sign: every LM step
    increases the cost, so each iteration rejects every lambda until the
    upper bound. Counts error-only evaluations (= one per lambda try + the
    constructor's initial error)."""

    def __init__(self):
        self.n_error_only = 0

    def __call__(self, this, values, H):
        x = values.atVector(this.keys()[0])
        if H is not None:
            H[0] = np.array([[1.0]])        # true derivative is -1
        else:
            self.n_error_only += 1
        return np.array([2.0 - x[0]])


def _stall_graph():
    f = _Stall()
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(1, 1.0),
                             gtsam.KeyVector([0]), f))
    v = gtsam.Values()
    v.insert(0, np.array([0.0]))
    return g, v, f


@pytest.mark.parametrize("upper,n_tries", [(1e2, 7), (1e5, 10)])
def test_lambda_upper_bound_caps_retry_loop(upper, n_tries):
    """lambda starts at 1e-5 and is multiplied by 10 per rejected try until
    it reaches the bound: 1e-5..1e1 = 7 tries at 1e2, 1e-5..1e4 = 10 at 1e5
    (the 15-try stalls in the board_rep logs started from lambda 1e-10)."""
    graph, init, f = _stall_graph()
    params = make_lm_params(30, lambda_lower=0.0, lambda_upper=upper)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    e0 = opt.error()
    assert f.n_error_only == 1
    res = opt.optimize()
    assert f.n_error_only == 1 + n_tries
    # nothing accepted: no iteration counted, values = initial (last accepted)
    assert opt.iterations() == 0
    assert opt.error() == e0
    assert np.array_equal(res.atVector(0), init.atVector(0))


def test_stall_after_convergence_stops_on_relative_tol():
    """The L1 case: once the attempted cost change is below
    relative_error_tol * cost, LM stops searching lambda at the first
    rejected try instead of walking to the upper bound."""
    graph, init, f = _stall_graph()
    # cost 0.5 * 2^2 = 2; the wrong-sign step changes it by ~O(1) relative,
    # so a huge relative tol stops the search after one try
    params = make_lm_params(30, relative_error_tol=10.0,
                            lambda_lower=0.0, lambda_upper=1e5)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    res = opt.optimize()
    assert f.n_error_only == 1 + 1
    assert opt.iterations() == 0
    assert np.array_equal(res.atVector(0), init.atVector(0))


def test_lambda_lower_bound_floors_damping():
    """Each accepted iteration divides lambda by 10, floored at the lower
    bound; the floor changes the damping of later steps (so it is a
    rounding-level change on a converging run, documented in L1)."""
    graph, init = _pose_graph()
    for lower, expect in [(0.0, 1e-5 / 10 ** 3), (1e-6, 1e-6)]:
        params = make_lm_params(3, lambda_lower=lower, lambda_upper=1e5,
                                absolute_error_tol=0.0, relative_error_tol=0.0)
        opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
        opt.optimize()
        assert opt.iterations() == 3
        assert opt.lambda_() == pytest.approx(expect, rel=1e-12)


# ---------------------------------------------------------------------------
# production call sites: Optimizer, CalibrationPipeline (single-camera and
# joint stages), run_cam_calibration_multi -> make_lm_params with the
# configured bounds / tolerances (spy on the symbol the optimizer module
# imported; every LM params object on the cam-only path is built there)
# ---------------------------------------------------------------------------

def _chessboard_frames(board, K, n=16, seed=0, img=(640, 480)):
    import cv2
    from mvis.core.types import FrameObservations, Observation
    rng = np.random.RandomState(seed)
    pts = board.all_points_3d().astype(np.float64)
    frames, t, attempts = [], 0.0, 0
    while len(frames) < n and attempts < n * 20:
        attempts += 1
        rvec = rng.uniform(-0.3, 0.3, 3)
        tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1),
                         rng.uniform(0.6, 0.9)])
        uv, _ = cv2.projectPoints(pts, rvec, tvec, K, np.zeros(5))
        uv = uv.reshape(-1, 2)
        inb = ((uv[:, 0] >= 0) & (uv[:, 0] < img[0])
               & (uv[:, 1] >= 0) & (uv[:, 1] < img[1]))
        if inb.sum() < 0.85 * len(pts):
            continue
        obs = [Observation(point_id=int(i), pixel=uv[i] + rng.standard_normal(2) * 0.2,
                           row=int(uv[i, 1]), timestamp=t)
               for i in np.where(inb)[0]]
        frames.append(FrameObservations(camera_id=0, timestamp=t,
                                        observations=obs, image_size=img))
        t += 0.1
    return frames


@pytest.fixture
def lm_params_spy(monkeypatch):
    """Records every make_lm_params call made through the optimizer module."""
    import mvis.calibration.optimizer as optmod
    calls = []
    real = optmod.make_lm_params

    def spy(max_iterations, **kw):
        calls.append(dict(max_iterations=max_iterations, **kw))
        return real(max_iterations, **kw)

    monkeypatch.setattr(optmod, "make_lm_params", spy)
    return calls


_BOUNDS = dict(lm_lambda_lower=1e-6, lm_lambda_upper=1e3)   # not the defaults


def test_optimizer_hands_options_to_make_lm_params(lm_params_spy):
    from mvis.detection.chessboard import Chessboard
    from mvis.calibration.initializer import Initializer
    from mvis.calibration.optimizer import Optimizer, OptimizerOptions

    board = Chessboard(7, 7, 0.05)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    frames = _chessboard_frames(board, K)
    init = Initializer(board)
    intr, dist = init.init_intrinsics_opencv(frames, (640, 480), "radtan")
    poses = init.init_board_poses(frames, intr, dist, "radtan")
    opts = OptimizerOptions(max_iterations=7, absolute_error_tol=1e-7,
                            relative_error_tol=1e-6, **_BOUNDS)
    Optimizer(board, opts).solve_single_camera(
        frames, intr, dist, poses, distortion_model="radtan", image_size=(640, 480))
    assert len(lm_params_spy) == 1
    c = lm_params_spy[0]
    assert c["max_iterations"] == 7
    assert (c["absolute_error_tol"], c["relative_error_tol"]) == (1e-7, 1e-6)
    assert (c["lambda_lower"], c["lambda_upper"]) == (1e-6, 1e3)


def test_pipeline_and_wrapper_hand_bounds_to_make_lm_params(lm_params_spy):
    from mvis.detection.chessboard import Chessboard
    from mvis.core.config import CamCalibConfig
    from mvis.calibration.optimizer import OptimizerOptions
    from mvis.calibration.pipeline_cam import (
        CalibrationPipeline, PipelineConfig, run_cam_calibration_multi)

    board = Chessboard(7, 7, 0.05)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    frames = _chessboard_frames(board, K)
    img = {0: (640, 480)}

    # CalibrationPipeline: the single-camera stage and the joint stage(s)
    pc = PipelineConfig(max_iterations=10, max_iterations_single=10,
                        corner_subsample="top_left", optimize_time_offset=True,
                        max_init_frames=100, init_frame_selection="uniform",
                        reference_camera=None, rerun_after_outlier_rejection=False,
                        distortion_models={0: "radtan"}, verbose=False, **_BOUNDS)
    CalibrationPipeline(board, pc).run({0: frames}, img)
    n_pipeline = len(lm_params_spy)
    assert n_pipeline >= 1                      # one camera: the single-camera stage
    # the tolerances on this path are OptimizerOptions' own (documented in
    # PipelineConfig), the bounds are the configured ones, on every call
    oo = OptimizerOptions()
    for c in lm_params_spy:
        assert (c["lambda_lower"], c["lambda_upper"]) == (1e-6, 1e3)
        assert c["absolute_error_tol"] == oo.absolute_error_tol
        assert c["relative_error_tol"] == oo.relative_error_tol

    # run_cam_calibration_multi: cfg.lm_lambda_* (optional attributes) -> PipelineConfig
    cfg = CamCalibConfig(max_iterations=10)
    for k, v in _BOUNDS.items():
        setattr(cfg, k, v)
    cams = [dict(camera_id=0, t_cam=[f.timestamp for f in frames],
                 ids_list=[[o.point_id for o in f.observations] for f in frames],
                 uv_list=[[o.pixel for o in f.observations] for f in frames],
                 image_size=(640, 480), model="radtan")]
    run_cam_calibration_multi(cams, board, config=cfg)
    assert len(lm_params_spy) > n_pipeline
    for c in lm_params_spy[n_pipeline:]:
        assert (c["lambda_lower"], c["lambda_upper"]) == (1e-6, 1e3)
    # and a config without the attributes falls back to the shipped defaults
    del lm_params_spy[:]
    run_cam_calibration_multi(cams, board, config=CamCalibConfig(max_iterations=10))
    assert lm_params_spy and all(
        (c["lambda_lower"], c["lambda_upper"]) == (LM_LAMBDA_LOWER_DEFAULT,
                                                   LM_LAMBDA_UPPER_DEFAULT)
        for c in lm_params_spy)
