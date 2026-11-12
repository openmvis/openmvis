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
"""A solve LM gives up on far from a minimum is rescued, and reported as failed if it stays stuck.

The hard starts are phase-B scenarios whose seed is poor: before the rescue, seed 3 without a robust
kernel stopped at 15 px and seed 9 with one at 2102 px, both with ok=True. `interp` fails too."""
import dataclasses
from types import SimpleNamespace as NS

import gtsam
import numpy as np
import pytest

from mvis.calibration import lm_rescue as R
from mvis.calibration import pipeline_cam_imu as P

A = gtsam.symbol("z", 0)


def _graph(target=1.0, sigma=1.0):
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorVector(A, np.array([target]), gtsam.noiseModel.Isotropic.Sigma(1, sigma)))
    return g


def _values(x):
    v = gtsam.Values()
    v.insert(A, np.array([float(x)]))
    return v


def _params(upper=100.0):
    p = gtsam.LevenbergMarquardtParams()
    p.setlambdaUpperBound(upper)
    return p


def _opt(lam, err=1.0, iterations=3, values=None):
    return NS(lambda_=lambda: lam, error=lambda: err, iterations=lambda: iterations, values=lambda: values)


def test_predicted_decrease_is_large_far_from_the_minimum_and_zero_at_it():
    assert R.predicted_decrease_ratio(_graph(), _values(5.0)) == pytest.approx(1.0)
    assert R.predicted_decrease_ratio(_graph(), _values(1.0)) == 0.0


def test_unsolvable_linear_system_is_not_called_stuck():
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorVector(A, np.zeros(2), gtsam.noiseModel.Isotropic.Sigma(2, 1.0)))
    v = gtsam.Values()
    v.insert(A, np.ones(2))
    v.insert(gtsam.symbol("z", 1), np.ones(1))
    g.add(gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(1, 1.0), gtsam.KeyVector([gtsam.symbol("z", 1)]),
                             lambda _f, _v, H: (H.__setitem__(0, np.zeros((1, 1))) if H is not None else None,
                                                np.ones(1))[1]))
    assert R.predicted_decrease_ratio(g, v) == 0.0


def test_lm_stopping_below_the_cap_is_never_rescued():
    """A normal solve keeps its result and pays for no extra linear solve."""
    values = _values(5.0)
    opt = _opt(lam=1e-3, values=values)
    got_values, got_opt, report = R.rescue_if_stuck(_graph(), values, values, opt, _params(), make_opt=None)
    assert got_values is values and got_opt is opt and report is None


def test_converged_at_the_cap_is_not_stuck():
    values = _values(1.0)
    opt = _opt(lam=100.0, values=values)
    got_values, got_opt, report = R.rescue_if_stuck(_graph(), values, values, opt, _params(), make_opt=None)
    assert got_values is values and got_opt is opt and report is None


def test_stuck_solve_gets_diagonal_damping_then_normal_damping():
    used = []

    def make_opt(values, params):
        used.append(params.getDiagonalDamping())
        return gtsam.LevenbergMarquardtOptimizer(_graph(), values, params)

    start, stuck = _values(3.0), _values(5.0)
    starts = []
    inner = make_opt

    def make_opt(values, params):
        starts.append(values)
        return inner(values, params)
    values, opt, report = R.rescue_if_stuck(_graph(), start, stuck, _opt(lam=100.0, values=stuck, iterations=2),
                                            _params(), make_opt)
    assert starts[0] is start                        # not the stuck point
    assert used == [True, False]
    assert values.atVector(A)[0] == pytest.approx(1.0)
    assert report["ratio_before"] == pytest.approx(1.0) and report["failed"] is False
    assert report["ratio_after"] < R.STUCK_RATIO
    assert opt.iterations() >= 3                     # counts the steps before the rescue


def test_rescue_that_ends_worse_keeps_the_original_result():
    def make_opt(values, params):
        return NS(optimize=lambda: _values(50.0), error=lambda: 1e6, iterations=lambda: 4)

    stuck = _values(5.0)
    opt = _opt(lam=100.0, err=8.0, iterations=2, values=stuck)
    values, got, report = R.rescue_if_stuck(_graph(), _values(3.0), stuck, opt, _params(), make_opt)
    assert values is stuck and got.error() == 8.0
    assert report["kept"] is False and report["failed"] is True


def test_still_stuck_after_the_rescue_is_reported_as_failed():
    def make_opt(values, params):
        return NS(optimize=lambda: values, error=lambda: 0.5, iterations=lambda: 0)

    stuck = _values(5.0)
    _values_out, _opt_out, report = R.rescue_if_stuck(_graph(), stuck, stuck, _opt(lam=100.0, values=stuck),
                                                      _params(), make_opt)
    assert report["failed"] is True


@pytest.mark.slow
@pytest.mark.parametrize("seed,huber", [(3, 0.0), (9, 2.0)])
def test_hard_start_reaches_the_minimum(seed, huber, monkeypatch):
    import tests.cam_imu.test_pipeline_cam_imu as M
    last = {}
    orig = P._run_cam_imu_calibration_multi_once

    def spy(*a, **k):
        k["config"] = dataclasses.replace(k["config"], vision_pose="imu", huber_px=huber)
        last["r"] = orig(*a, **k)
        return last["r"]
    monkeypatch.setattr(P, "_run_cam_imu_calibration_multi_once", spy)
    data = M._data
    monkeypatch.setattr(M, "_data", lambda sc, *a, **k: data(sc, *a, **{**k, "seed": seed}))
    try:
        M.test_phase_b_recovers_imu_intrinsics()
    except AssertionError:
        pass                                       # that test's own limits are not the point here
    r = last["r"]
    # Whether LM sticks from this start depends on the machine's rounding; the outcome must not.
    rescue = r.diagnostics.get("lm_rescue")
    assert r.ok and not (rescue and rescue["failed"])
    assert r.diagnostics["reproj_rms_px"] < 0.8    # sigma_px 0.5: a 2D RMS of about 0.7
    sigma = np.linalg.norm(np.asarray(r.calib_sigma)[:3])
    assert np.linalg.norm(M._T_CtoI.localCoordinates(r.calib.T)[:3]) < 3 * sigma
