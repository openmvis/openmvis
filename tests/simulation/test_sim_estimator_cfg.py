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
"""Task 2 — shared matched-estimator-config helper (sim-cleanup plan)."""
from mvis.simulation.noise import MVIS_IMU_NOISE, matched_calib_cfg


def test_matched_calib_cfg_mirrors_sim_noise():
    c = matched_calib_cfg(MVIS_IMU_NOISE)
    assert c.sigma_g == MVIS_IMU_NOISE.sigma_w     # gyro density: sim sigma_w -> solve sigma_g
    assert c.sigma_a == MVIS_IMU_NOISE.sigma_a
    assert c.sigma_bg == MVIS_IMU_NOISE.sigma_bg
    assert c.sigma_ba == MVIS_IMU_NOISE.sigma_ba
    assert c.imu_intrinsics_model == "calyx"
    assert c.imu_intrinsics == "dw_da"


def test_matched_calib_cfg_overrides():
    c = matched_calib_cfg(MVIS_IMU_NOISE, imu_model="rpng", sigma_px=0.5,
                          state_hz=12.0, imu_intrinsics="pinned")
    assert c.imu_intrinsics_model == "rpng"
    assert c.sigma_px == 0.5
    assert c.state_hz == 12.0
    assert c.imu_intrinsics == "pinned"
