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
"""PinholeRadtan.project(jac=False) returns NaN for a point behind the camera (z <= 0)
instead of mirroring it into the image (cleanup plan PR F, item F1). In front of the camera
the pixel is bitwise the old one, and the jac=True path (pixel and Jacobians) is unchanged."""
import warnings

import gtsam
import numpy as np
import pytest

from mvis.camera.camera_model import _gtsam_project, _in_front
from mvis.camera.pinhole_equidistant import PinholeEquidistant
from mvis.camera.pinhole_radtan import PinholeRadtan

DIST = np.array([-0.3153, 0.08503, -1.514e-4, -2.938e-4])


def _cam():
    return PinholeRadtan(448.44, 447.77, 372.15, 280.57, DIST)


def _old_project(cam, p):
    """The pre-F1 jac=False body: uncalibrate the z-divided point, no front check."""
    p = np.asarray(p, dtype=np.float64)
    p2d = cam.get_gtsam_calibration().uncalibrate(p[:2] / p[2])
    return np.array([p2d[0], p2d[1]])


def _front_points(n=500, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.8, 0.8, (n, 2))
    z = rng.uniform(0.05, 5.0, n)
    return np.c_[xy * z[:, None], z]


@pytest.mark.parametrize("p", [[0.1, 0.2, -1.0], [0.0, 0.0, -1.0], [0.3, -0.1, -1e-9],
                               [0.1, 0.2, 0.0], [0.0, 0.0, -0.0], [0.1, 0.2, np.nan]])
def test_behind_or_on_the_camera_plane_is_nan(p):
    cam = _cam()
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no divide-by-zero on z = 0 any more
        uv = cam.project(p)
    assert uv.shape == (2,) and np.isnan(uv).all()
    uv_j, J_pt, J_intr = cam.project(p, jac=True)
    assert np.isnan(uv_j).all()
    if not np.isnan(p[2]):              # gtsam's cheirality check rejects z <= 0
        assert J_pt is None and J_intr is None


def test_mirrored_point_was_a_valid_pixel_before():
    """The bug: (x, y, -z) mirrored to the pixel of (-x, -y, z)."""
    cam = _cam()
    p = np.array([0.1, 0.2, -1.0])
    assert np.isfinite(_old_project(cam, p)).all()
    assert np.isnan(cam.project(p)).all()


def test_in_front_pixels_are_bitwise_unchanged():
    cam = _cam()
    for p in _front_points():
        assert cam.project(p).tobytes() == _old_project(cam, p).tobytes()


def test_jacobian_path_is_unchanged():
    cam = _cam()
    cal = cam.get_gtsam_calibration()
    for p in _front_points(200, seed=5):
        uv, J_pt, J_intr = cam.project(p, jac=True)
        uv0, J_pt0, J_intr0 = _gtsam_project(cal, p, True)
        assert uv.tobytes() == uv0.tobytes()
        assert J_pt.tobytes() == J_pt0.tobytes() and J_intr.tobytes() == J_intr0.tobytes()
        # and the no-jac pixel still agrees with the jac pixel to rounding
        np.testing.assert_allclose(cam.project(p), uv, rtol=0, atol=1e-9)


def test_jacobian_matches_direct_gtsam_call():
    cam = _cam()
    p = np.array([0.2, -0.1, 1.3])
    Jpt = np.zeros((2, 3), order="F")
    Jpose = np.zeros((2, 6), order="F")
    Jc9 = np.zeros((2, 9), order="F")
    uv0 = gtsam.PinholeCameraCal3DS2(gtsam.Pose3(), cam.get_gtsam_calibration()).project(
        gtsam.Point3(*p), Jpose, Jpt, Jc9)
    uv, J_pt, J_intr = cam.project(p, jac=True)
    assert np.asarray(uv0).tobytes() == uv.tobytes()
    assert Jpt.tobytes() == J_pt.tobytes()
    assert Jc9[:, [0, 1, 3, 4, 5, 6, 7, 8]].tobytes() == J_intr.tobytes()


def test_in_front_helper():
    assert _in_front(np.array([0.0, 0.0, 1e-12]))
    for z in (0.0, -0.0, -1.0, np.nan):
        assert not _in_front(np.array([0.0, 0.0, z]))


def test_equidistant_uses_the_same_guard():
    cam = PinholeEquidistant(380.0, 380.0, 510.0, 514.0, np.array([0.005, -0.002, 4e-5, -2e-4]))
    assert np.isnan(cam.project([0.1, 0.2, -1.0])).all()
    assert np.isnan(cam.project([0.1, 0.2, 0.0])).all()
    assert np.isfinite(cam.project([0.1, 0.2, 1.0])).all()
