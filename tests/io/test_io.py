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
"""Tests for I/O modules."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest
import yaml

from mvis.core.types import (
    CalibrationResult,
    CameraParams,
    ExtrinsicTransform,
    ReprojectionErrorStats,
)
from mvis.io.image_source import ImageSource
from mvis.io.output_writer import OutputWriter


class TestImageSource:
    def test_load_images(self, tmp_path):
        """Test loading images from a directory."""
        # Create some test images
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), img)

        source = ImageSource(str(tmp_path))
        assert source.num_cameras() == 1

        frames = list(source)
        assert len(frames) == 3
        for img, ts, cam_id in frames:
            assert img.shape == (100, 100) and img.dtype == np.uint8   # grey since G6
            assert cam_id == 0

    def test_empty_directory(self, tmp_path):
        source = ImageSource(str(tmp_path))
        frames = list(source)
        assert len(frames) == 0

    def test_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            ImageSource("/nonexistent/path")

    def test_multi_camera(self, tmp_path):
        cam0 = tmp_path / "cam0"
        cam1 = tmp_path / "cam1"
        cam0.mkdir()
        cam1.mkdir()

        for i in range(2):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(cam0 / f"img_{i:03d}.png"), img)
            cv2.imwrite(str(cam1 / f"img_{i:03d}.png"), img)

        source = ImageSource([str(cam0), str(cam1)])
        assert source.num_cameras() == 2

        frames = list(source)
        assert len(frames) == 4  # 2 images * 2 cameras
        cam_ids = [f[2] for f in frames]
        assert 0 in cam_ids
        assert 1 in cam_ids


class TestOutputWriter:
    def _make_result(self):
        return CalibrationResult(
            cameras=[
                CameraParams(
                    camera_id=0,
                    intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
                    distortion=np.array([-0.1, 0.01, 0.001, -0.001]),
                    distortion_model="radtan",
                    image_size=(640, 480),
                ),
            ],
            extrinsics=[
                ExtrinsicTransform(
                    from_camera=1,
                    to_camera=0,
                    T_to_from=np.eye(4),
                    so3=np.array([0.01, -0.02, 0.03]),
                    translation=np.array([0.1, 0.0, 0.0]),
                ),
            ],
            reference_camera=0,
            reprojection_errors=[
                ReprojectionErrorStats(
                    camera_id=0, rms=0.25, mean=0.20, median=0.18, max=1.1,
                    num_observations=5000,
                ),
            ],
            overall_rms=0.25,
        )

    def test_to_dict(self):
        result = self._make_result()
        data = OutputWriter.to_dict(result)

        assert len(data["cameras"]) == 1
        assert data["cameras"][0]["intrinsics"]["fx"] == 500.0
        assert data["reprojection_errors"]["overall_rms"] == 0.25

    def test_write_yaml(self, tmp_path):
        result = self._make_result()
        path = str(tmp_path / "result.yaml")
        OutputWriter.write_yaml(result, path)

        with open(path) as f:
            data = yaml.safe_load(f)

        assert data["cameras"][0]["intrinsics"]["fx"] == 500.0
        assert data["reprojection_errors"]["overall_rms"] == 0.25

    def test_write_json(self, tmp_path):
        result = self._make_result()
        path = str(tmp_path / "result.json")
        OutputWriter.write_json(result, path)

        with open(path) as f:
            data = json.load(f)

        assert data["cameras"][0]["intrinsics"]["fx"] == 500.0
        assert len(data["extrinsics"]["transforms"]) == 1

    def test_write_auto_format(self, tmp_path):
        result = self._make_result()
        path_yaml = str(tmp_path / "result.yaml")
        path_json = str(tmp_path / "result.json")

        OutputWriter.write(result, path_yaml, "yaml")
        OutputWriter.write(result, path_json, "json")

        assert os.path.exists(path_yaml)
        assert os.path.exists(path_json)

    def test_write_unknown_format_raises(self, tmp_path):
        result = self._make_result()
        with pytest.raises(ValueError):
            OutputWriter.write(result, str(tmp_path / "result.txt"), "txt")
