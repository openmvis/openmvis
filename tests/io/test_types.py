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
"""Tests for core types."""

import numpy as np
import pytest

from mvis.core.types import (
    Observation,
    FrameObservations,
    CameraParams,
    CalibrationResult,
    ExtrinsicTransform,
    ReprojectionErrorStats,
)


class TestObservation:
    def test_creation(self):
        obs = Observation(point_id=0, pixel=[100.5, 200.3])
        assert obs.point_id == 0
        np.testing.assert_allclose(obs.pixel, [100.5, 200.3])
        assert obs.pixel.dtype == np.float64

    def test_default_values(self):
        obs = Observation(point_id=5, pixel=[0, 0])
        assert obs.timestamp == 0.0
        assert obs.row == 0

    def test_pixel_conversion(self):
        obs = Observation(point_id=0, pixel=[1, 2])
        assert isinstance(obs.pixel, np.ndarray)


class TestFrameObservations:
    def test_creation(self):
        obs_list = [
            Observation(point_id=i, pixel=[i * 10, i * 20])
            for i in range(5)
        ]
        frame = FrameObservations(camera_id=0, timestamp=1.0, observations=obs_list)
        assert frame.camera_id == 0
        assert frame.timestamp == 1.0
        assert frame.num_points == 5

    def test_point_ids(self):
        obs_list = [
            Observation(point_id=i, pixel=[0, 0])
            for i in [3, 7, 12]
        ]
        frame = FrameObservations(camera_id=0, timestamp=0, observations=obs_list)
        np.testing.assert_array_equal(frame.point_ids, [3, 7, 12])

    def test_pixels(self):
        obs_list = [
            Observation(point_id=0, pixel=[10.0, 20.0]),
            Observation(point_id=1, pixel=[30.0, 40.0]),
        ]
        frame = FrameObservations(camera_id=0, timestamp=0, observations=obs_list)
        pixels = frame.pixels
        assert pixels.shape == (2, 2)
        np.testing.assert_allclose(pixels[0], [10.0, 20.0])
        np.testing.assert_allclose(pixels[1], [30.0, 40.0])


class TestCameraParams:
    def test_creation(self):
        cam = CameraParams(
            camera_id=0,
            intrinsics=[500, 500, 320, 240],
            distortion=[0.1, -0.05, 0.001, -0.001],
            distortion_model="radtan",
        )
        assert cam.fx == 500.0
        assert cam.fy == 500.0
        assert cam.cx == 320.0
        assert cam.cy == 240.0
        assert cam.distortion_model == "radtan"
        assert cam.intrinsics.dtype == np.float64

    def test_default_rolling_shutter(self):
        cam = CameraParams(
            camera_id=0, intrinsics=[500, 500, 320, 240],
            distortion=[0, 0, 0, 0], distortion_model="radtan",
        )
        assert cam.rolling_shutter_readout == 0.0


class TestExtrinsicTransform:
    def test_creation(self):
        T = np.eye(4)
        ext = ExtrinsicTransform(
            from_camera=1, to_camera=0,
            T_to_from=T, so3=[0, 0, 0], translation=[0.1, 0, 0],
        )
        assert ext.from_camera == 1
        assert ext.to_camera == 0
        np.testing.assert_array_equal(ext.T_to_from, np.eye(4))


class TestCalibrationResult:
    def test_creation(self):
        result = CalibrationResult(cameras=[])
        assert result.cameras == []
        assert result.extrinsics == []
        assert result.reference_camera == 0
        assert result.overall_rms == 0.0
