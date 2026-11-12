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
"""Tests for camera models."""

import numpy as np
import pytest

from mvis.camera.pinhole_radtan import PinholeRadtan
from mvis.camera.pinhole_equidistant import PinholeEquidistant
from mvis.camera.camera_model import create_camera_model


class TestPinholeRadtan:
    def test_creation(self):
        cam = PinholeRadtan(500.0, 500.0, 320.0, 240.0)
        intr = cam.intrinsics()
        np.testing.assert_allclose(intr, [500.0, 500.0, 320.0, 240.0])
        assert cam.distortion_model_name() == "radtan"
        assert cam.num_params() == 8

    def test_project_center(self):
        cam = PinholeRadtan(500.0, 500.0, 320.0, 240.0)
        # A point at [0, 0, 1] should project to the principal point
        px = cam.project(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(px, [320.0, 240.0], atol=1e-6)

    def test_project_off_center(self):
        cam = PinholeRadtan(500.0, 500.0, 320.0, 240.0)
        # [0.1, 0.2, 1.0] -> [500*0.1 + 320, 500*0.2 + 240] = [370, 340]
        px = cam.project(np.array([0.1, 0.2, 1.0]))
        np.testing.assert_allclose(px, [370.0, 340.0], atol=1.0)  # distortion may shift slightly

    def test_project_unproject_roundtrip(self):
        cam = PinholeRadtan(500.0, 500.0, 320.0, 240.0, np.array([0, 0, 0, 0]))
        # With zero distortion, project then unproject should be identity
        p3d = np.array([0.15, -0.1, 2.0])
        px = cam.project(p3d)
        p3d_back = cam.unproject(px, depth=2.0)
        np.testing.assert_allclose(p3d_back, p3d, atol=1e-6)

    def test_project_unproject_with_distortion(self):
        cam = PinholeRadtan(500.0, 500.0, 320.0, 240.0, np.array([-0.1, 0.01, 0.001, -0.001]))
        p3d = np.array([0.05, 0.03, 1.5])
        px = cam.project(p3d)
        # Should produce a valid pixel
        assert not np.any(np.isnan(px))
        # Roundtrip
        p3d_back = cam.unproject(px, depth=1.5)
        np.testing.assert_allclose(p3d_back, p3d, atol=0.01)

    def test_to_from_opencv(self):
        cam = PinholeRadtan(500.0, 500.0, 320.0, 240.0, np.array([-0.1, 0.01, 0.001, -0.001]))
        K, dist = cam.to_opencv()
        assert K.shape == (3, 3)
        assert K[0, 0] == 500.0
        assert K[1, 1] == 500.0
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0

        cam2 = PinholeRadtan.from_opencv(K, dist)
        np.testing.assert_allclose(cam2.intrinsics(), cam.intrinsics())
        np.testing.assert_allclose(cam2.distortion(), cam.distortion())

    def test_distortion_zero(self):
        cam = PinholeRadtan(500.0, 500.0, 320.0, 240.0)
        dist = cam.distortion()
        np.testing.assert_allclose(dist, [0, 0, 0, 0])


class TestPinholeEquidistant:
    def test_creation(self):
        cam = PinholeEquidistant(400.0, 400.0, 300.0, 200.0)
        intr = cam.intrinsics()
        np.testing.assert_allclose(intr, [400.0, 400.0, 300.0, 200.0])
        assert cam.distortion_model_name() == "equidistant"
        assert cam.num_params() == 8

    def test_project_center(self):
        cam = PinholeEquidistant(400.0, 400.0, 300.0, 200.0)
        px = cam.project(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(px, [300.0, 200.0], atol=1e-6)

    def test_project_unproject_roundtrip(self):
        cam = PinholeEquidistant(400.0, 400.0, 300.0, 200.0, np.zeros(4))
        p3d = np.array([0.1, -0.05, 1.5])
        px = cam.project(p3d)
        p3d_back = cam.unproject(px, depth=1.5)
        np.testing.assert_allclose(p3d_back, p3d, atol=0.01)

    def test_to_opencv(self):
        cam = PinholeEquidistant(400.0, 400.0, 300.0, 200.0, np.array([0.01, -0.001, 0.0001, 0]))
        K, dist = cam.to_opencv()
        assert K[0, 0] == 400.0
        assert len(dist) == 4


class TestCreateCameraModel:
    def test_create_radtan(self):
        cam = create_camera_model("radtan", 500, 500, 320, 240)
        assert cam.distortion_model_name() == "radtan"
        assert isinstance(cam, PinholeRadtan)

    def test_create_equidistant(self):
        cam = create_camera_model("equidistant", 400, 400, 300, 200)
        assert cam.distortion_model_name() == "equidistant"
        assert isinstance(cam, PinholeEquidistant)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError):
            create_camera_model("unknown_model", 500, 500, 320, 240)
