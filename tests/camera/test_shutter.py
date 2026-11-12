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
"""Tests for the shared rolling-shutter vocabulary (config-cleanup M3b).

One source of truth for the camera `shutter_type` (string) <-> cam-IMU `pin_tr`
(bool) correspondence, so both pipelines agree on what "rolling" means.
"""

from mvis.core import shutter


def test_is_rolling():
    assert shutter.is_rolling("rolling") is True
    assert shutter.is_rolling("global") is False
    assert shutter.is_rolling("auto") is False        # auto treated as global
    assert shutter.is_rolling("ROLLING") is True       # case-insensitive
    assert shutter.is_rolling(" rolling ") is True     # whitespace-tolerant
    assert shutter.is_rolling("") is False
    assert shutter.is_rolling(None) is False


def test_pin_tr_for():
    # cam-IMU pins the readout unless the camera is rolling
    assert shutter.pin_tr_for("rolling") is False
    assert shutter.pin_tr_for("global") is True
    assert shutter.pin_tr_for("auto") is True


def test_shutter_type_for():
    assert shutter.shutter_type_for(True) == "global"
    assert shutter.shutter_type_for(False) == "rolling"


def test_round_trip():
    for st in ("rolling", "global"):
        assert shutter.shutter_type_for(shutter.pin_tr_for(st)) == st
    for pin in (True, False):
        assert shutter.pin_tr_for(shutter.shutter_type_for(pin)) is pin


def test_default_rolling_mode_top():
    assert shutter.DEFAULT_ROLLING_MODE == 0


def test_row_offset_is_bitwise_the_frozen_camera_factor_copy():
    """`shutter.row_offset` moved from `camera_factor._rs_row_offset`: same bytes."""
    import numpy as np
    from tests._reference.ref_camera_factor import _rs_row_offset as ref
    rng = np.random.default_rng(20261107)
    rows = np.concatenate([rng.uniform(-50.0, 2100.0, 4000), [0.0, 0.5, 479.5, 480.0, 1e-300]])
    for h in (1, 7, 480, 720, 1080, 2048):
        for mode in (0, 1, 2, 3):
            got = np.array([shutter.row_offset(float(r), h, mode) for r in rows])
            want = np.array([ref(float(r), h, mode) for r in rows])
            assert got.tobytes() == want.tobytes()
