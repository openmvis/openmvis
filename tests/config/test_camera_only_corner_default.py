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
"""The camera-only corner default is a diagonal pair: single corners shift the principal point."""
from dataclasses import fields

from mvis.calibration.pipeline_cam import PipelineConfig
from mvis.core.config import CalibrationConfig, CamCalibConfig, CamImuCalibConfig


def default_of(cls):
    return next(f.default for f in fields(cls) if f.name == "corner_subsample")


def test_camera_only_defaults_agree_on_two_anti():
    for cls in (CalibrationConfig, CamCalibConfig, PipelineConfig):
        assert default_of(cls) == "two_anti", cls.__name__


def test_camera_imu_keeps_all_corners():
    assert default_of(CamImuCalibConfig) == "all"
