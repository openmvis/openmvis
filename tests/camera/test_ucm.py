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
"""UCM (unified camera model, alpha form) camera model: kernels, class, registry.

  * L1 equivalence: ucm == eucm(beta=1) == double_sphere(xi=0) == omni_radtan
    (gamma, xi, no radtan) == the raw Kalibr omni-none and Basalt UnifiedCamera
    formulas, to 1e-9 px, over alpha in {0, 0.3, 0.5, 0.64, 0.8};
  * analytic J_point / J_cal[:, :5] vs central differences up to 120 deg and at a
    point behind the camera; project -> unproject round trip;
  * J_cal[:, 5:] == 0 and cal_vec[5:] (zero or garbage padding) never read;
  * fold (Basalt validity): rays past the fold and pixels outside the valid disc
    give None;
  * class / factory / registry / sanity rule / mvisdb enum + schema CHECK.
"""
import math
import sqlite3

import numpy as np
import pytest

from mvis.calibration.factors.custom_projections import (
    project_double_sphere, project_eucm, project_omni_radtan,
    project_ucm, ucm_fold_w, unproject_ucm,
)
from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.camera import UCM, create_camera_model
from mvis.camera.camera_model import build_camera_model
from mvis.core import camera_models as cm

ALPHAS = [0.0, 0.3, 0.5, 0.64, 0.8]
K = (350.0, 352.0, 512.0, 510.0)


def _dir(theta_deg, phi_deg=30.0):
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def _fold_deg(alpha):
    """Incidence of the fold: Z = -w d  ->  cos(theta) = -w."""
    return math.degrees(math.acos(-ucm_fold_w(alpha)))


def _ucm(alpha, pad=(0.0, 0.0, 0.0)):
    return np.array([*K, alpha, *pad])


def _points(alpha, n=200, seed=0):
    """Directions up to 2 deg inside the fold (pinhole: up to 80 deg), at depth 0.5-4."""
    rng = np.random.default_rng(seed)
    th_max = 80.0 if alpha == 0.0 else min(_fold_deg(alpha) - 2.0, 175.0)
    th = rng.uniform(0.0, th_max, n)
    ph = rng.uniform(0.0, 360.0, n)
    return [_dir(t, p) * rng.uniform(0.5, 4.0) for t, p in zip(th, ph)]


# ── L1: same model, every form ──

def _kalibr_omni_none(X, gamma_u, gamma_v, cx, cy, xi):
    d = np.linalg.norm(X)
    return np.array([gamma_u * X[0] / (X[2] + xi * d) + cx, gamma_v * X[1] / (X[2] + xi * d) + cy])


def _basalt_ucm(X, fx, fy, cx, cy, alpha):
    d = np.linalg.norm(X)
    den = alpha * d + (1 - alpha) * X[2]
    return np.array([fx * X[0] / den + cx, fy * X[1] / den + cy])


@pytest.mark.parametrize("alpha", ALPHAS)
def test_equivalent_forms(alpha):
    fx, fy, cx, cy = K
    xi = alpha / (1 - alpha)
    for X in _points(alpha):
        uv, _, _ = project_ucm(X, _ucm(alpha))
        assert uv is not None
        ref = [project_eucm(X, np.array([*K, alpha, 1.0, 0, 0]))[0],
               project_double_sphere(X, np.array([*K, 0.0, alpha, 0, 0]))[0],
               project_omni_radtan(X, np.array([fx * (1 + xi), fy * (1 + xi), cx, cy, xi, 0, 0, 0, 0]))[0],
               _kalibr_omni_none(X, fx * (1 + xi), fy * (1 + xi), cx, cy, xi),
               _basalt_ucm(X, fx, fy, cx, cy, alpha)]
        for r in ref:
            assert r is not None
            np.testing.assert_allclose(uv, r, atol=1e-9, rtol=0)


# ── Jacobians, round trip ──

def _fd(f, x, h=1e-6):
    cols = []
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.array(cols).T


JAC_CASES = [(a, th) for a in (0.3, 0.5, 0.64, 0.8) for th in (60.0, 95.0, 110.0, 120.0)
             if th < _fold_deg(a) - 1.0]


@pytest.mark.parametrize("alpha,theta", JAC_CASES)
def test_jacobians_vs_finite_differences(alpha, theta):
    cal = _ucm(alpha)
    X = _dir(theta) * 1.7
    uv, J_point, J_cal = project_ucm(X, cal, True)
    assert J_cal.shape == (2, 8)
    np.testing.assert_allclose(J_point, _fd(lambda p: project_ucm(p, cal)[0], X), atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(J_cal[:, :5], _fd(lambda c: project_ucm(X, np.r_[c, 0, 0, 0])[0], cal[:5]),
                               atol=1e-4, rtol=1e-6)
    assert not J_cal[:, 5:].any()


@pytest.mark.parametrize("alpha", [0.64, 0.8])
def test_jacobian_behind_camera(alpha):
    cal = _ucm(alpha)
    X = np.array([0.9, 0.3, -0.15])          # z < 0 (99 deg), inside the fold for both
    assert np.degrees(np.arccos(X[2] / np.linalg.norm(X))) < _fold_deg(alpha)
    _, J_point, J_cal = project_ucm(X, cal, True)
    np.testing.assert_allclose(J_point, _fd(lambda p: project_ucm(p, cal)[0], X), atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(J_cal[:, :5], _fd(lambda c: project_ucm(X, np.r_[c, 0, 0, 0])[0], cal[:5]),
                               atol=1e-4, rtol=1e-6)


@pytest.mark.parametrize("alpha,theta", JAC_CASES + [(0.0, 60.0), (0.3, 5.0), (0.64, 0.0)])
def test_round_trip(alpha, theta):
    cal = _ucm(alpha)
    X = _dir(theta)
    uv, _, _ = project_ucm(X, cal)
    ray = unproject_ucm(uv, cal)
    np.testing.assert_allclose(ray, X, atol=1e-9)


def test_padding_never_read():
    X = _dir(80.0) * 2.0
    a = project_ucm(X, _ucm(0.64), True)
    for pad in ([5.0, 7.0, -3.0], [1.0, 0.0, 0.0]):           # garbage, and a would-be beta
        b = project_ucm(X, _ucm(0.64, pad), True)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
    _, _, J5 = project_ucm(X, np.array([*K, 0.64]), True)   # 5-wide cal_vec
    assert J5.shape == (2, 8) and not J5[:, 5:].any()


# ── fold ──

def test_fold_alpha_064():
    cal = _ucm(0.64)
    assert 123.0 < _fold_deg(0.64) < 125.0
    for th in (125.0, 130.0, 135.0):
        assert project_ucm(_dir(th), cal)[0] is None
    uv, _, _ = project_ucm(_dir(120.0), cal)
    np.testing.assert_allclose(unproject_ucm(uv, cal), _dir(120.0), atol=1e-9)


def test_fold_alpha_09_and_unproject_disc():
    cal = _ucm(0.9)
    assert project_ucm(np.array([0.9, 0.2, -0.2]), cal)[0] is None
    r = 1.0 / math.sqrt(2 * 0.9 - 1)                         # edge of the valid disc
    assert unproject_ucm(np.array([K[2] + K[0] * r * 1.001, K[3]]), cal) is None


# ── class, factory, registry ──

def test_class_and_factory():
    c = create_camera_model("ucm", *K, np.array([0.64]))
    assert isinstance(c, UCM)
    assert c.n_intr == 5 and c.num_params() == 5 and c.distortion_model_name() == "ucm"
    np.testing.assert_allclose(c.xi(), 0.64 / 0.36)
    np.testing.assert_allclose(c.kalibr_gamma(), (K[0] / 0.36, K[1] / 0.36))
    X = _dir(100.0)
    uv, _, J = c.project(X, jac=True)
    assert J.shape == (2, 5)
    assert np.isnan(c.unproject(uv)).all()      # past 90 deg: no z-depth (D12)
    np.testing.assert_allclose(unproject_ucm(uv, c.cal_vec), X, atol=1e-9)
    assert np.isnan(c.project(_dir(130.0))).all()
    b = build_camera_model("ucm", np.array([*K, 0.64]))
    assert isinstance(b, UCM) and b.n_intr == 5
    assert isinstance(create_camera_model("ucm", *K, np.array([0.64, 0.0, 0.0, 0.0])), UCM)
    with pytest.raises(ValueError):
        create_camera_model("ucm", *K, np.array([0.64, 1.0]))   # a beta is not UCM


def test_camera_calib_dim():
    import gtsam
    cc = CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.array([*K, 0.64]), model="ucm")
    assert cc.DIM == 13 and cc.n_intr == 5


def test_registry():
    assert "ucm" in cm.MODELS and cm.n_dist("ucm") == 1
    assert cm.DIST_NAMES["ucm"] == ("alpha",)
    assert cm.default_dist("ucm") == [0.6]
    assert cm.is_custom("ucm") and "ucm" in cm.KALIBR_EXPORTABLE
    assert cm.canonical_cal_vec("ucm", [*K, 0.64, 0, 0, 0]) == [*K, 0.64]
    for a in ("ucm", "UCM", "unified", "omni_none", "omni-none"):
        assert cm.resolve_model_alias(a) == "ucm"
    for a in ("omni", "mei", "eucm", "omni_radtan"):
        assert cm.resolve_model_alias(a) != "ucm"


def test_sanity_rule():
    assert cm.dist_sanity_issue("ucm", [0.64]) is None
    assert cm.dist_sanity_issue("ucm", [0.0]) is None
    for bad in (1.0, -0.1, float("nan")):
        assert cm.dist_sanity_issue("ucm", [bad]) is not None


def test_mvisdb_enum_and_schema():
    from importlib.resources import files
    from mvis.mvisdb.mvisdatabase import CameraModel
    assert CameraModel("ucm") is CameraModel.UCM
    schema = (files("mvis.mvisdb") / "schema.sql").read_text()
    con = sqlite3.connect(":memory:")
    con.executescript(schema)
    assert "'ucm'" in schema


def test_not_in_freeze_branches():
    """ucm never takes the omni xi-freeze schedule (alpha would
    freeze): that branch tests the model by equality, the optimizer never
    names ucm, and pipeline_cam names it only in its seed helpers."""
    import inspect
    from mvis.calibration import optimizer, pipeline_cam
    src = inspect.getsource(optimizer)
    assert 'distortion_models.get(cam_id) == "omni_radtan"' in src
    assert "ucm" not in src
    src = inspect.getsource(pipeline_cam)
    allowed = {"_fallback_dist", "_stage1_dist_extreme"}
    for name, fn in inspect.getmembers(pipeline_cam, inspect.isfunction):
        if fn.__module__ == pipeline_cam.__name__ and name not in allowed:
            assert '"ucm"' not in inspect.getsource(fn), name
    for name, cls in inspect.getmembers(pipeline_cam, inspect.isclass):
        if cls.__module__ == pipeline_cam.__name__:
            assert '"ucm"' not in inspect.getsource(cls), name
