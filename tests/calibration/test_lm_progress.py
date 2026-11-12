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
"""optimize_logged: GTSAM's LM loop run iterate by iterate, with progress lines.
It must return bitwise what `optimize()` returns, on every way the loop can stop."""
import logging

import gtsam
import numpy as np
import pytest

from mvis.calibration.lm_progress import IterationLog, optimize_logged


def _problem(n=12, seed=0):
    """Pose2 chain with a loop closure, started far off: a few LM steps to converge."""
    rng = np.random.default_rng(seed)
    g = gtsam.NonlinearFactorGraph()
    prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
    odo = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    g.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior))
    for i in range(n - 1):
        g.add(gtsam.BetweenFactorPose2(i, i + 1, gtsam.Pose2(1.0, 0.0, 0.5), odo))
    g.add(gtsam.BetweenFactorPose2(n - 1, 0, gtsam.Pose2(1.0, 0.0, 0.5), odo))
    v = gtsam.Values()
    for i in range(n):
        x, y, th = rng.normal(0, 3, 3)
        v.insert(i, gtsam.Pose2(x, y, th))
    return g, v


def _params(**kw):
    p = gtsam.LevenbergMarquardtParams()
    for k, val in kw.items():
        getattr(p, k)(val)
    return p


CASES = {
    "converges": dict(),
    "iteration cap": dict(setMaxIterations=2),
    # zero tolerances: LM keeps rejecting steps at the minimum until lambda passes the bound
    "damping cap": dict(setRelativeErrorTol=0.0, setAbsoluteErrorTol=0.0,
                        setlambdaUpperBound=1e7, setMaxIterations=200),
    "error tol early out": dict(setErrorTol=1e12),
    "tight tolerances": dict(setRelativeErrorTol=1e-12, setAbsoluteErrorTol=1e-12,
                             setMaxIterations=50),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bitwise_equal_to_optimize(case, seed):
    g, v = _problem(seed=seed)
    ref = gtsam.LevenbergMarquardtOptimizer(g, v, _params(**CASES[case]))
    ref_vals = ref.optimize()
    opt = gtsam.LevenbergMarquardtOptimizer(g, v, _params(**CASES[case]))
    vals = optimize_logged(opt, _params(**CASES[case]))
    assert opt.iterations() == ref.iterations()
    assert opt.error() == ref.error()
    assert opt.lambda_() == ref.lambda_()
    for k in range(12):
        np.testing.assert_array_equal(vals.atPose2(k).matrix(), ref_vals.atPose2(k).matrix())
    if case == "damping cap":
        assert ref.lambda_() >= 1e7 and ref.iterations() < 200     # really stopped at the cap


def test_bitwise_equal_after_a_step_taken_outside():
    """An optimizer already stepped to the cap: optimize_logged's `iterations >= max` early out."""
    g, v = _problem()
    ref = gtsam.LevenbergMarquardtOptimizer(g, v, _params(setMaxIterations=1))
    ref.iterate()
    ref_vals = ref.optimize()
    opt = gtsam.LevenbergMarquardtOptimizer(g, v, _params(setMaxIterations=1))
    opt.iterate()
    vals = optimize_logged(opt, _params(setMaxIterations=1))
    assert opt.iterations() == ref.iterations() == 1
    assert opt.error() == ref.error() and opt.lambda_() == ref.lambda_()
    for k in range(12):
        np.testing.assert_array_equal(vals.atPose2(k).matrix(), ref_vals.atPose2(k).matrix())


def _opposed_priors():
    """Two opposed priors: non-zero error, zero gradient, so LM takes no step."""
    k = gtsam.symbol("z", 2)
    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorVector(k, np.ones(2), noise))
    g.add(gtsam.PriorFactorVector(k, -np.ones(2), noise))
    v = gtsam.Values()
    v.insert(k, np.zeros(2))
    return g, v


@pytest.mark.parametrize("graph", ["failed cholesky", "zero gradient"])
def test_bitwise_equal_on_no_step_solves(graph):
    """The 0-iteration results _lm_no_step_report reads right after optimize_logged."""
    from mvis.calibration.lm_params import make_lm_params
    from tests.cam_imu.test_lm_no_step import _tiny_coupling
    g, v = _tiny_coupling() if graph == "failed cholesky" else _opposed_priors()
    ref = gtsam.LevenbergMarquardtOptimizer(g, v, make_lm_params(20))
    ref_vals = ref.optimize()
    opt = gtsam.LevenbergMarquardtOptimizer(g, v, make_lm_params(20))
    vals = optimize_logged(opt, make_lm_params(20))
    assert opt.iterations() == ref.iterations() == 0
    assert opt.error() == ref.error() and opt.lambda_() == ref.lambda_()
    for k in v.keys():
        np.testing.assert_array_equal(vals.atVector(k), ref_vals.atVector(k))


def test_iteration_lines_are_throttled(caplog):
    now = [0.0]
    log = IterationLog(40, "LM", interval_s=10.0, clock=lambda: now[0])
    with caplog.at_level(logging.INFO, logger="mvis.calibration.lm_progress"):
        log(1, 100.0, 50.0, 1e-5)            # 0 s: silent
        now[0] = 12.0
        log(2, 50.0, 40.0, 1e-6)             # 12 s: logged
        now[0] = 15.0
        log(3, 40.0, 39.0, 1e-7)             # 3 s after: silent
    lines = [r.getMessage() for r in caplog.records]
    assert lines == ["  LM it 2/40: error 50 -> 40, lambda 1e-06, 6.0 s/it"]


def test_logs_each_iteration_of_a_long_solve(caplog, monkeypatch):
    monkeypatch.setenv("MVIS_LM_PROGRESS_S", "1e-9")
    g, v = _problem()
    p = _params()
    opt = gtsam.LevenbergMarquardtOptimizer(g, v, p)
    with caplog.at_level(logging.INFO, logger="mvis.calibration.lm_progress"):
        optimize_logged(opt, p, "cam-IMU LM")
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) >= opt.iterations() > 1            # one line per iterate() call
    assert lines[0].startswith("  cam-IMU LM it 1/100: error ")
    assert f" it {opt.iterations()}/100: " in lines[-1]


def test_disabled_and_other_optimizers(caplog, monkeypatch):
    monkeypatch.setenv("MVIS_LM_PROGRESS_S", "0")
    g, v = _problem()
    with caplog.at_level(logging.INFO, logger="mvis.calibration.lm_progress"):
        optimize_logged(gtsam.LevenbergMarquardtOptimizer(g, v, _params()), _params())
        d = gtsam.DoglegOptimizer(g, v, gtsam.DoglegParams())
        out = optimize_logged(d, _params())                    # passes through to optimize()
    assert not caplog.records
    assert out.size() == 12


class _SelfLogging:
    """Stands in for TgaRelinOptimizer: steps and logs itself, given a label."""
    logs_progress = True

    def __init__(self):
        self.labels = []

    def optimize(self, label="cam-IMU LM"):
        self.labels.append(label)
        return "values"


def test_self_logging_optimizer_gets_the_label():
    opt = _SelfLogging()
    assert optimize_logged(opt, None, "LM rescue") == "values"
    assert opt.labels == ["LM rescue"]


def test_pipeline_stage_lines_at_info(caplog, monkeypatch):
    """The stage lines format (a per-frame RMS array once broke the 'optimising' line, and
    caplog re-raises logging errors) and LM iteration lines carry their label."""
    from mvis.detection.chessboard import Chessboard
    from mvis.calibration.pipeline_cam_imu import CamImuCalibConfig, run_cam_imu_calibration
    from tests.helpers.sim_scenarios import (scenario_for_pipeline as _scenario,
                                             data_for_pipeline as _data, _RADTAN, _IMG)
    monkeypatch.setenv("MVIS_LM_PROGRESS_S", "1e-9")
    sc = _scenario(sigma_px=0.5)
    t_cam, cam, s = _data(sc, win=12.0, seed=3)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)
    cfg = CamImuCalibConfig(estimate_t_ga=False, state_hz=10.0, sigma_px=0.5, vision_pose="interp",
                            huber_px=2.0, imu_intrinsics="pinned", max_iterations=3)
    with caplog.at_level(logging.INFO, logger="mvis"):
        r = run_cam_imu_calibration(t_cam, list(cam.ids), list(cam.uv), board_pts,
                                    _RADTAN, s.t, s.w_m, s.a_m, _IMG, config=cfg)
    assert r.ok
    lines = [rec.getMessage() for rec in caplog.records]
    for stage in ("cam-IMU: initialising ", "cam-IMU: init done ", "cam-IMU: graph built ",
                  "cam-IMU: solve done "):
        assert any(m.startswith(stage) for m in lines), stage
    opt_line = next(m for m in lines if m.startswith("cam-IMU: graph built "))
    assert "median per-frame reprojection RMS " in opt_line and "max 3 iterations" in opt_line
    assert any(m.startswith("  cam-IMU LM it 1/3: ") for m in lines)
