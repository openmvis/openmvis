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
"""Lock-in regression test for the Double Sphere camera model.

Validates MVIS's `project_double_sphere` against the published values
from Usenko, Demmel, Cremers (IROS 2018) "The Double Sphere Camera Model"
Table 3, plus a forward/unproject round-trip.

This test is the canary for any future regression in DS projection /
parameterization (the bug discovered 2026-05-01 took the optimizer from
fx-correct to fx-2x-too-large because the optimizer escaped the
canonical xi/alpha basin without a Phase-1 anchor).
"""
from __future__ import annotations

import numpy as np
import pytest

from mvis.calibration.factors.custom_projections import (
    project_double_sphere, unproject_double_sphere,
)


# Usenko et al. 2018, Table 3 — per-lens DS parameters
# Format: name → (fx, fy, cx, cy, xi, alpha)
USENKO_TABLE_3 = {
    "BM2820":      (386.17, 386.23, 624.29, 512.49, -0.20, 0.55),
    "BF5M13720":   (208.36, 208.35, 637.45, 512.18, -0.20, 0.64),
    "GoPro":       (364.84, 364.86, 621.02, 513.26, -0.27, 0.57),
    "BM4018S118":  (565.58, 565.68, 635.52, 521.81, -0.23, 0.62),
    "BF2M2020S23": (313.21, 313.21, 638.65, 514.39, -0.18, 0.59),
}


@pytest.mark.parametrize("lens_name,params", USENKO_TABLE_3.items())
def test_on_axis_projects_to_principal_point(lens_name, params):
    """A 3D point on the optical axis must project to (cx, cy)."""
    fx, fy, cx, cy, xi, alpha = params
    cal = np.array([fx, fy, cx, cy, xi, alpha, 0, 0])
    pixel, _, _ = project_double_sphere(np.array([0.0, 0.0, 1.0]), cal)
    assert pixel is not None, f"{lens_name}: projection failed for on-axis point"
    np.testing.assert_allclose(pixel, [cx, cy], atol=1e-9,
                                err_msg=f"{lens_name}: on-axis projection")


@pytest.mark.parametrize("lens_name,params", USENKO_TABLE_3.items())
def test_round_trip_recovers_ray_direction(lens_name, params):
    """project → unproject of a 3D point should recover the ray direction."""
    fx, fy, cx, cy, xi, alpha = params
    cal = np.array([fx, fy, cx, cy, xi, alpha, 0, 0])

    # Test multiple points: on-axis, slight off-axis, larger off-axis
    test_points = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.1, 0.0, 1.0]),
        np.array([0.0, 0.1, 1.0]),
        np.array([0.2, 0.15, 1.0]),
        np.array([0.05, -0.05, 1.0]),
    ]
    for p in test_points:
        pixel, _, _ = project_double_sphere(p, cal)
        if pixel is None:
            continue  # outside FOV, skip
        ray = unproject_double_sphere(pixel, cal)
        assert ray is not None, f"{lens_name}: unproject failed for {p}"
        # Compare ray directions (normalized)
        p_norm = p / np.linalg.norm(p)
        ray_norm = ray / np.linalg.norm(ray)
        np.testing.assert_allclose(
            ray_norm, p_norm, atol=1e-6,
            err_msg=f"{lens_name}: round-trip failed for point {p}"
        )


@pytest.mark.parametrize("lens_name,params", USENKO_TABLE_3.items())
def test_jacobian_matches_numerical(lens_name, params):
    """Analytical Jacobians (J_point, J_cal) must match central-difference
    numerical Jacobians to ~1e-6 (limited by float64 precision)."""
    cal = np.array(list(params) + [0, 0])
    p = np.array([0.5, 0.3, 1.2])  # off-axis test point

    pix_an, J_pt_an, J_cal_an = project_double_sphere(p, cal, compute_jacobians=True)

    # Numerical J wrt point
    eps = 1e-6
    J_pt_num = np.zeros((2, 3))
    for i in range(3):
        pp = p.copy(); pp[i] += eps
        pm = p.copy(); pm[i] -= eps
        u_p, _, _ = project_double_sphere(pp, cal)
        u_m, _, _ = project_double_sphere(pm, cal)
        J_pt_num[:, i] = (u_p - u_m) / (2 * eps)

    np.testing.assert_allclose(J_pt_an, J_pt_num, atol=1e-5, rtol=1e-5,
                                err_msg=f"{lens_name}: J_point analytical vs numerical")

    # Numerical J wrt cal_vec (xi, alpha specifically — these are the
    # parameters most prone to bug)
    for col in (4, 5):  # xi, alpha
        cp = cal.copy(); cp[col] += eps
        cm = cal.copy(); cm[col] -= eps
        u_p, _, _ = project_double_sphere(p, cp)
        u_m, _, _ = project_double_sphere(p, cm)
        J_num_col = (u_p - u_m) / (2 * eps)
        np.testing.assert_allclose(
            J_cal_an[:, col], J_num_col, atol=1e-5, rtol=1e-5,
            err_msg=f"{lens_name}: J_cal[{col}] (xi/alpha) analytical vs numerical"
        )


def test_projection_consistent_at_paper_params():
    """Spot-check: project a known 3D point with each Usenko paper lens
    and verify the projection is consistent (within sane bounds for a
    fisheye camera)."""
    # 3D point at 0.5 m off-axis horizontally, 1 m forward
    p = np.array([0.5, 0.0, 1.0])

    for lens_name, params in USENKO_TABLE_3.items():
        fx, fy, cx, cy, xi, alpha = params
        cal = np.array(list(params) + [0, 0])
        pixel, _, _ = project_double_sphere(p, cal)
        assert pixel is not None, f"{lens_name}: projection should not fail"
        # Pixel should be to the right of cx (X > 0 → u > cx)
        assert pixel[0] > cx, (
            f"{lens_name}: X>0 should project right of cx, got u={pixel[0]:.2f} cx={cx}")
        # Pixel y should equal cy (Y=0 → v=cy)
        np.testing.assert_allclose(
            pixel[1], cy, atol=1e-9,
            err_msg=f"{lens_name}: Y=0 should project to cy"
        )


def test_xi_sign_convention_matches_paper():
    """Regression guard: xi MUST be negative for typical fisheye lenses
    in the Usenko convention. If a future change flips this sign, the
    optimizer will silently drift to a 2x-fx wrong basin."""
    # All 5 paper lenses have xi < 0
    for lens_name, params in USENKO_TABLE_3.items():
        xi = params[4]
        assert xi < 0, (
            f"{lens_name}: Usenko paper xi={xi} should be negative. "
            "If this assertion fires, the test data is wrong, not the code."
        )

    # Project with xi=-0.20 and xi=+0.20 and confirm they produce
    # different pixels (not symmetric — sign matters)
    cal_neg = np.array([386.17, 386.23, 624.29, 512.49, -0.20, 0.55, 0, 0])
    cal_pos = np.array([386.17, 386.23, 624.29, 512.49, +0.20, 0.55, 0, 0])
    p = np.array([0.3, 0.2, 1.0])
    pix_neg, _, _ = project_double_sphere(p, cal_neg)
    pix_pos, _, _ = project_double_sphere(p, cal_pos)
    diff = np.linalg.norm(pix_neg - pix_pos)
    assert diff > 1.0, (
        f"Projection should change meaningfully when xi flips sign "
        f"(got |Δpx|={diff:.4f}); xi has weak influence — possible bug"
    )
