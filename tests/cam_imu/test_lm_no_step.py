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
"""The silent LM stop: GTSAM swallows a failed linear solve, so Cholesky failing at every lambda looks
like a converged start (0 iterations, err_final == err_init). `_lm_no_step_report` tells them apart."""
import logging
from types import SimpleNamespace as NS

import gtsam
import numpy as np
import pytest

from mvis.calibration.lm_params import make_lm_params
from mvis.calibration.pipeline_cam_imu import _lm_no_step_report

A, B = gtsam.symbol("z", 0), gtsam.symbol("z", 1)
TINY = 1e-12


def _tiny_coupling(sigma=TINY):
    """A tiny-sigma row on a + b beside unit priors, as _TINY priors sit beside other keys: with
    1 / sigma^2 = 1e24 the unit diagonal and any lambda up to the bound are lost to round-off, so the
    second pivot of the normal equations is 0 at every lambda."""
    def err(_f, v, H):
        a, b = v.atVector(A), v.atVector(B)
        if H is not None:
            H[0], H[1] = np.eye(1), np.eye(1)
        return a + b - 1.0
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(1, sigma), gtsam.KeyVector([A, B]), err))
    for k in (A, B):
        g.add(gtsam.PriorFactorVector(k, np.zeros(1), gtsam.noiseModel.Isotropic.Sigma(1, 1.0)))
    v = gtsam.Values()
    v.insert(A, np.zeros(1))
    v.insert(B, np.zeros(1))
    return g, v


def _lm(g, v, max_iterations=20):
    opt = gtsam.LevenbergMarquardtOptimizer(g, v, make_lm_params(max_iterations))
    e0 = opt.error()
    r = opt.optimize()
    return opt, r, e0


def test_failed_cholesky_at_every_lambda_is_a_failure(caplog):
    g, v = _tiny_coupling()
    opt, r, e0 = _lm(g, v)
    assert opt.iterations() == 0 and opt.error() == e0           # the silent stop itself
    assert r.atVector(A)[0] == 0.0
    with caplog.at_level(logging.WARNING, logger="mvis"):
        rep = _lm_no_step_report(g, v, opt, NS(max_iterations=20))
    assert rep["failed"] is True and "Indeterminant" in rep["reason"]
    assert any("LM made no step" in m.getMessage() for m in caplog.records)


def test_a_start_already_at_the_minimum_is_not_a_failure(caplog):
    """Two opposed priors: non-zero error, zero gradient. LM takes 0 iterations and is right."""
    k = gtsam.symbol("z", 2)
    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorVector(k, np.ones(2), noise))
    g.add(gtsam.PriorFactorVector(k, -np.ones(2), noise))
    v = gtsam.Values()
    v.insert(k, np.zeros(2))
    opt, _r, e0 = _lm(g, v)
    assert opt.iterations() == 0 and e0 == 2.0
    with caplog.at_level(logging.WARNING, logger="mvis"):
        rep = _lm_no_step_report(g, v, opt, NS(max_iterations=20))
    assert rep["failed"] is False and rep["err"] == 2.0
    assert abs(rep["predicted_decrease"]) < 1e-12
    assert any("LM made no step" in m.getMessage() for m in caplog.records)


def test_zero_error_start_is_not_a_failure():
    k = gtsam.symbol("z", 2)
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorVector(k, np.zeros(2), gtsam.noiseModel.Isotropic.Sigma(2, 1.0)))
    v = gtsam.Values()
    v.insert(k, np.zeros(2))
    opt, _r, _e0 = _lm(g, v)
    assert opt.iterations() == 0
    assert _lm_no_step_report(g, v, opt, NS(max_iterations=20))["failed"] is False


def test_a_solve_that_iterated_is_never_probed(monkeypatch):
    g, v = _tiny_coupling(sigma=1e-2)
    opt, _r, _e0 = _lm(g, v)
    assert opt.iterations() > 0
    class NoLinearize:
        def linearize(self, _v):
            raise AssertionError("probed a solve that iterated")
    assert _lm_no_step_report(NoLinearize(), v, opt, NS(max_iterations=20)) is None


def test_no_iterations_allowed_is_not_probed():
    g, v = _tiny_coupling()
    opt = NS(iterations=lambda: 0, error=lambda: 1.0)
    assert _lm_no_step_report(g, v, opt, NS(max_iterations=0)) is None


@pytest.mark.parametrize("failed,status", [(True, "FAIL"), (False, "WARN")])
def test_verdict_reports_the_no_step(tmp_path, failed, status):
    import json
    from mvis.analysis.verdict import assess
    d = dict(ok=not failed, reproj_rms_px=0.3, cams=[], imus=[], diagnostics=dict(
        iterations=0, max_iterations=40, lm_no_step=dict(failed=failed, reason="r", err=1.0)))
    p = tmp_path / "cam_imu.json"
    p.write_text(json.dumps(d))
    row = next(c for c in assess(str(p)).checks if c.id == "solve.lm_step")
    assert row.status.upper() == status
