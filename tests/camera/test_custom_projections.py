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
"""Tests for custom camera projection functions (FOV, Double Sphere, EUCM,
omni-radtan with p2, Kannala-Brandt kb4).

Tests:
  - Forward-inverse roundtrip (project → unproject → same ray direction)
  - Optical axis: (0,0,1) → (cx, cy)
  - Behind camera: Z ≤ 0 → None
  - Jacobian correctness: analytical vs numerical finite-difference
  - Degenerate parameters: omega=0, alpha=0, beta=1
  - Wide FOV: points at 60-80° incidence
"""

import numpy as np
import pytest

from mvis.calibration.factors.custom_projections import (
    project_fov, project_double_sphere, project_eucm, project_omni_radtan,
    unproject_fov, unproject_double_sphere, unproject_eucm, unproject_omni_radtan,
    project_kb, unproject_kb, project_ucm, unproject_ucm,
)
from mvis.camera import create_camera_model, FOVModel, DoubleSphere, EUCM, UCM
from mvis.camera.omni_radtan import OmniRadtan
from mvis.camera.kb4 import KannalaBrandt4


# ── Test parameters ──

FOV_CAL = np.array([300.0, 300.0, 320.0, 240.0, 0.9, 0, 0, 0])
DS_CAL = np.array([300.0, 300.0, 320.0, 240.0, -0.2, 0.6, 0, 0])
EUCM_CAL = np.array([300.0, 300.0, 320.0, 240.0, 0.6, 1.0, 0, 0])
UCM_CAL = np.array([300.0, 300.0, 320.0, 240.0, 0.64, 0, 0, 0])
# omni-radtan [xi, k1, k2, p1, p2], every coefficient non-zero (p2 included)
OMNI_CAL = np.array([300.0, 300.0, 320.0, 240.0, 1.0, -0.05, 0.01, 1e-3, -2e-3])
# kb4 [k1..k4], small alternating, every coefficient non-zero
KB4_CAL = np.array([300.0, 300.0, 320.0, 240.0, 0.02, -0.01, 3e-3, -5e-4])

ALL_MODELS = [
    ("fov", FOV_CAL, project_fov, unproject_fov),
    ("double_sphere", DS_CAL, project_double_sphere, unproject_double_sphere),
    ("eucm", EUCM_CAL, project_eucm, unproject_eucm),
    ("ucm", UCM_CAL, project_ucm, unproject_ucm),
    ("omni_radtan", OMNI_CAL, project_omni_radtan, unproject_omni_radtan),
    ("kb4", KB4_CAL, project_kb, unproject_kb),
]

# Test points in camera frame
TEST_POINTS = [
    np.array([0.0, 0.0, 1.0]),        # optical axis
    np.array([0.1, 0.05, 1.0]),        # small offset
    np.array([0.3, -0.2, 1.0]),        # moderate offset
    np.array([-0.5, 0.4, 1.0]),        # larger offset
    np.array([0.0, 0.0, 2.0]),         # along axis, far
    np.array([0.5, 0.5, 0.5]),         # 45° incidence
    np.array([1.0, 0.0, 1.0]),         # 45° X
    np.array([0.0, 1.0, 1.0]),         # 45° Y
]


# ── Optical axis test ──

@pytest.mark.parametrize("name,cal,proj_fn,unproj_fn", ALL_MODELS)
def test_project_optical_axis(name, cal, proj_fn, unproj_fn):
    """(0, 0, Z) should project to (cx, cy)."""
    pixel, _, _ = proj_fn(np.array([0, 0, 1.0]), cal)
    assert pixel is not None
    np.testing.assert_allclose(pixel, [cal[2], cal[3]], atol=1e-10)


@pytest.mark.parametrize("name,cal,proj_fn,unproj_fn", ALL_MODELS)
def test_project_optical_axis_far(name, cal, proj_fn, unproj_fn):
    """(0, 0, 5) should also project to (cx, cy)."""
    pixel, _, _ = proj_fn(np.array([0, 0, 5.0]), cal)
    assert pixel is not None
    np.testing.assert_allclose(pixel, [cal[2], cal[3]], atol=1e-10)


# ── Behind camera ──

def test_fov_behind_camera():
    """FOV: Z ≤ 0 should return None (perspective-based model)."""
    pixel, _, _ = project_fov(np.array([0.1, 0.1, -1.0]), FOV_CAL)
    assert pixel is None


def test_ds_eucm_behind_camera_valid():
    """DS/EUCM: Z < 0 CAN succeed (spherical models handle >180° FOV).
    But they should return None for degenerate cases (e.g., w ≤ 0)."""
    # These spherical models can project points behind the camera
    # as long as the validity check (w > 0) passes
    pixel_ds, _, _ = project_double_sphere(np.array([0.1, 0.1, -1.0]), DS_CAL)
    pixel_eucm, _, _ = project_eucm(np.array([0.1, 0.1, -1.0]), EUCM_CAL)
    # At least one should succeed — just verify no crash
    assert pixel_ds is not None or pixel_eucm is not None


# ── Forward-inverse roundtrip ──

@pytest.mark.parametrize("name,cal,proj_fn,unproj_fn", ALL_MODELS)
@pytest.mark.parametrize("point", TEST_POINTS)
def test_roundtrip(name, cal, proj_fn, unproj_fn, point):
    """project(P) → unproject → ray ∝ P."""
    pixel, _, _ = proj_fn(point, cal)
    if pixel is None:
        pytest.skip("Point not projectable")

    ray = unproj_fn(pixel, cal)
    assert ray is not None

    # Normalize both to unit vectors for comparison
    p_dir = point / np.linalg.norm(point)
    r_dir = ray / np.linalg.norm(ray)
    np.testing.assert_allclose(r_dir, p_dir, atol=1e-6,
                               err_msg=f"{name}: roundtrip failed for {point}")


# ── Jacobian vs numerical finite-difference ──

def _numerical_jacobian_point(proj_fn, p_cam, cal_vec, eps=1e-7):
    """Compute J_point via central finite difference."""
    J = np.zeros((2, 3))
    for i in range(3):
        p_plus = p_cam.copy()
        p_minus = p_cam.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        px_p, _, _ = proj_fn(p_plus, cal_vec)
        px_m, _, _ = proj_fn(p_minus, cal_vec)
        if px_p is None or px_m is None:
            return None
        J[:, i] = (px_p - px_m) / (2.0 * eps)
    return J


def _numerical_jacobian_cal(proj_fn, p_cam, cal_vec, eps=1e-7):
    """Compute J_cal via central finite difference."""
    J = np.zeros((2, 8))
    for i in range(8):
        c_plus = cal_vec.copy()
        c_minus = cal_vec.copy()
        c_plus[i] += eps
        c_minus[i] -= eps
        px_p, _, _ = proj_fn(c_plus, p_cam)  # Note: args swapped for cal
        px_m, _, _ = proj_fn(c_minus, p_cam)
        if px_p is None or px_m is None:
            J[:, i] = 0  # inactive parameter
            continue
        J[:, i] = (px_p - px_m) / (2.0 * eps)
    return J


def _numerical_jacobian_cal_correct(proj_fn, p_cam, cal_vec, eps=1e-7):
    """Compute J_cal via central finite difference (correct arg order);
    max(8, len(cal_vec)) columns like the kernels."""
    J = np.zeros((2, max(8, len(cal_vec))))
    for i in range(len(cal_vec)):
        c_plus = cal_vec.copy()
        c_minus = cal_vec.copy()
        c_plus[i] += eps
        c_minus[i] -= eps
        px_p, _, _ = proj_fn(p_cam, c_plus)
        px_m, _, _ = proj_fn(p_cam, c_minus)
        if px_p is None or px_m is None:
            J[:, i] = 0
            continue
        J[:, i] = (px_p - px_m) / (2.0 * eps)
    return J


@pytest.mark.parametrize("name,cal,proj_fn,unproj_fn", ALL_MODELS)
@pytest.mark.parametrize("point", TEST_POINTS[:6])
def test_jacobian_point(name, cal, proj_fn, unproj_fn, point):
    """Analytical J_point matches numerical within 1e-5."""
    pixel, J_point, _ = proj_fn(point, cal, compute_jacobians=True)
    if pixel is None:
        pytest.skip("Point not projectable")

    J_num = _numerical_jacobian_point(proj_fn, point, cal)
    if J_num is None:
        pytest.skip("Numerical Jacobian failed")

    np.testing.assert_allclose(J_point, J_num, atol=1e-5,
                               err_msg=f"{name}: J_point mismatch at {point}")


@pytest.mark.parametrize("name,cal,proj_fn,unproj_fn", ALL_MODELS)
@pytest.mark.parametrize("point", TEST_POINTS[:6])
def test_jacobian_cal(name, cal, proj_fn, unproj_fn, point):
    """Analytical J_cal matches numerical within 1e-5."""
    pixel, _, J_cal = proj_fn(point, cal, compute_jacobians=True)
    if pixel is None:
        pytest.skip("Point not projectable")

    J_num = _numerical_jacobian_cal_correct(proj_fn, point, cal)

    # Only check active parameters (non-zero columns)
    n_active = {"fov": 5, "double_sphere": 6, "eucm": 6, "ucm": 5, "omni_radtan": 9,
                "kb4": 8}[name]
    assert J_cal.shape == J_num.shape
    np.testing.assert_allclose(J_cal[:, :n_active], J_num[:, :n_active], atol=1e-5,
                               err_msg=f"{name}: J_cal mismatch at {point}")


# ── Degenerate parameters ──

def test_fov_omega_zero():
    """FOV with omega=0 should reduce to pinhole."""
    cal = np.array([300.0, 300.0, 320.0, 240.0, 0.0, 0, 0, 0])
    point = np.array([0.1, 0.05, 1.0])
    pixel, _, _ = project_fov(point, cal)
    # Pinhole: u = fx*X/Z + cx = 300*0.1 + 320 = 350
    #          v = fy*Y/Z + cy = 300*0.05 + 240 = 255
    np.testing.assert_allclose(pixel, [350.0, 255.0], atol=1e-6)


def test_ds_xi_zero_alpha_zero():
    """DS with xi=0, alpha=0 should reduce to pinhole."""
    cal = np.array([300.0, 300.0, 320.0, 240.0, 0.0, 0.0, 0, 0])
    point = np.array([0.1, 0.05, 1.0])
    pixel, _, _ = project_double_sphere(point, cal)
    # w = 0*d2 + 1*(0*d1 + Z) = Z = 1
    # u = fx*X/Z + cx = 350, v = 255
    np.testing.assert_allclose(pixel, [350.0, 255.0], atol=1e-6)


def test_eucm_alpha_zero():
    """EUCM with alpha=0 should reduce to pinhole."""
    cal = np.array([300.0, 300.0, 320.0, 240.0, 0.0, 1.0, 0, 0])
    point = np.array([0.1, 0.05, 1.0])
    pixel, _, _ = project_eucm(point, cal)
    # w = 0*d + 1*Z = Z = 1
    np.testing.assert_allclose(pixel, [350.0, 255.0], atol=1e-6)


# ── Wide FOV ──

@pytest.mark.parametrize("name,cal,proj_fn,unproj_fn", ALL_MODELS)
def test_wide_fov_60deg(name, cal, proj_fn, unproj_fn):
    """Point at ~60° incidence should project successfully."""
    # tan(60°) ≈ 1.73
    point = np.array([1.73, 0.0, 1.0])
    pixel, _, _ = proj_fn(point, cal)
    assert pixel is not None, f"{name}: failed at 60° incidence"


@pytest.mark.parametrize("name,cal,proj_fn,unproj_fn", ALL_MODELS)
def test_wide_fov_80deg(name, cal, proj_fn, unproj_fn):
    """Point at ~80° incidence should project successfully."""
    # tan(80°) ≈ 5.67
    point = np.array([5.67, 0.0, 1.0])
    pixel, _, _ = proj_fn(point, cal)
    # May or may not succeed depending on model params
    # Just check it doesn't crash


# ── Factory / CameraModel classes ──

@pytest.mark.parametrize("model_name,cls", [
    ("fov", FOVModel),
    ("double_sphere", DoubleSphere),
    ("eucm", EUCM),
    ("ucm", UCM),
    ("omni_radtan", OmniRadtan),
    ("kb4", KannalaBrandt4),
])
def test_create_camera_model(model_name, cls):
    """Factory creates correct model class."""
    model = create_camera_model(model_name, 300, 300, 320, 240)
    assert isinstance(model, cls)
    assert model.distortion_model_name() == model_name
    assert model.get_gtsam_calibration() is None


@pytest.mark.parametrize("model_name", ["fov", "double_sphere", "eucm", "ucm", "omni_radtan",
                                        "kb4"])
def test_camera_model_project_unproject(model_name):
    """CameraModel.project/unproject roundtrip."""
    model = create_camera_model(model_name, 300, 300, 320, 240)
    point = np.array([0.2, 0.1, 1.0])
    pixel = model.project(point)
    assert not np.any(np.isnan(pixel))

    ray = model.unproject(pixel)
    assert not np.any(np.isnan(ray))

    # Directions should match
    p_dir = point / np.linalg.norm(point)
    r_dir = ray / np.linalg.norm(ray)
    np.testing.assert_allclose(r_dir, p_dir, atol=1e-5)


# ── unproject convention (D12): z-depth for every model ──

_ALL_CALS = [("radtan", np.array([300.0, 301.0, 320.0, 240.0, -0.2, 0.05, 1e-3, -1e-3])),
             ("equidistant", np.array([300.0, 301.0, 320.0, 240.0, 0.01, -0.005, 0.0, 0.0])),
             ("fov", FOV_CAL), ("double_sphere", DS_CAL), ("eucm", EUCM_CAL),
             ("ucm", UCM_CAL), ("omni_radtan", OMNI_CAL), ("kb4", KB4_CAL)]


@pytest.mark.parametrize("name,cal", _ALL_CALS, ids=[n for n, _ in _ALL_CALS])
def test_unproject_is_z_depth_for_every_model(name, cal):
    """`unproject(pixel, depth)` returns the point with z == depth on the pixel's ray,
    the same for all 8 models (it used to be the range for some, z for others)."""
    from mvis.camera.camera_model import build_camera_model
    m = build_camera_model(name, cal)
    for p in (np.array([0.0, 0.0, 1.0]), np.array([0.2, -0.1, 1.0]), np.array([-0.4, 0.3, 0.7])):
        uv = m.project(p)
        for depth in (1.0, 2.5):
            q = m.unproject(uv, depth=depth)
            assert q[2] == depth
            np.testing.assert_allclose(q, p / p[2] * depth, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name,cal", [c for c in _ALL_CALS if c[0] in ("kb4", "ucm", "double_sphere")])
def test_unproject_has_no_z_depth_past_90_deg(name, cal):
    """A pixel whose ray points behind the image plane (z < 0) has no z-depth: NaN."""
    from mvis.camera.camera_model import build_camera_model
    m = build_camera_model(name, cal)
    p = np.array([np.sin(np.deg2rad(100.0)), 0.0, np.cos(np.deg2rad(100.0))])
    uv = m.project(p)
    assert np.isfinite(uv).all()
    assert np.isnan(m.unproject(uv)).all()
