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
"""Task 2: config-schema defaults — window_mode/segment_length, factor_hz 10, aux_imu_stride 3.

Tests (TDD, written BEFORE the fields exist):
  1. CamImuInputConfig defaults to window_mode="motion" and segment_length=80.0.
  2. CameraConfig defaults to factor_hz=10.0 (was 5.0).
  3. CamImuCalibConfig has aux_imu_stride=3 as default.
"""

from mvis.core.config import CamImuInputConfig, CameraConfig, CamImuCalibConfig


def test_motion_is_default():
    c = CamImuInputConfig()
    assert c.window_mode == "motion" and c.segment_length == 80.0


def test_camera_factor_hz_default_10():
    assert CameraConfig().factor_hz == 10.0


def test_aux_imu_stride_default_3():
    assert CamImuCalibConfig().aux_imu_stride == 3


def test_cam_imu_factor_hz_default_10():
    from mvis.core.config import CamImuCalibConfig
    assert CamImuCalibConfig().factor_hz == 10.0
