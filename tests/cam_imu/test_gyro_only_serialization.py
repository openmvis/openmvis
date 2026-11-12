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
"""P3 (gyro-only-imu): serialize a self-documenting `gyro_only` flag end-to-end.

A gyro-only aux IMU pins lever/Da/Tg/accel-bias and only estimates the gyro
rotation+td+Dw+gyro-bias. The numbers already serialize SAFELY (no-op accel
correction). P3 adds a `gyro_only: true` flag so a reader knows the lever/Da
were pinned, not estimated. These tests cover writer emission, calib_spec
reading, and round-trip preservation.
"""
import gtsam
import numpy as np

from mvis.io.cam_imu_output import _aux_imu_block
from mvis.recalib.calib_spec import _imu_from_block


def _aux(gyro_only):
    aux = dict(name="t265", td=0.01, intr_xi24=[0.0] * 24,
               T_IstoI=gtsam.Pose3(), b_g_s=np.zeros(3), b_a_s=np.zeros(3))
    if gyro_only:
        aux["gyro_only"] = True
    return aux


# --- calib_spec reading -----------------------------------------------------

def test_calib_spec_reads_gyro_only_true():
    b = {"name": "t265", "role": "aux", "model": "calyx",
         "imu_intr_xi": [0.0] * 24, "gyro_only": True}
    assert _imu_from_block(b).gyro_only is True


def test_calib_spec_gyro_only_defaults_false():
    b = {"name": "t265", "role": "aux", "model": "calyx",
         "imu_intr_xi": [0.0] * 24}
    assert _imu_from_block(b).gyro_only is False


# --- writer emission --------------------------------------------------------

def test_writer_emits_gyro_only_flag():
    a = _aux_imu_block(_aux(gyro_only=True), "calyx")
    assert a.get("gyro_only") is True


def test_writer_omits_gyro_only_when_false():
    a = _aux_imu_block(_aux(gyro_only=False), "calyx")
    assert "gyro_only" not in a


# --- round-trip -------------------------------------------------------------

def test_round_trip_preserves_gyro_only():
    block = _aux_imu_block(_aux(gyro_only=True), "calyx")
    block["role"] = "aux"
    assert _imu_from_block(block).gyro_only is True


def test_aux_t_ga_written_and_read_back_for_apply_calib():
    """An aux IMU's own t_ga reaches the calib YAML, so apply-calib shifts that IMU's accel."""
    aux = _aux(gyro_only=False)
    aux.update(t_ga=-0.0067, t_ga_sigma=2e-5)
    block = _aux_imu_block(aux, "calyx")
    assert block["t_ga"] == -0.0067 and block["t_ga_sigma"] == 2e-5
    assert _imu_from_block(block).t_ga == -0.0067


def test_aux_without_t_ga_writes_none():
    assert "t_ga" not in _aux_imu_block(_aux(gyro_only=True), "calyx")
