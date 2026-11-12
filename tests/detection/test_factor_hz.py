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
"""Per-camera factor_hz vs trajectory state_hz (config-cleanup M7).

`state_hz` is the cam-IMU TRAJECTORY pose-state rate; `factor_hz` is the per-camera
vision-FACTOR rate (same concept as the camera-only CameraConfig.factor_hz).
Decoupled: `factor_hz` defaults to `state_hz` (behavior-preserving), and a lower
`factor_hz` thins a camera's vision factors WITHOUT touching the shared trajectory.
"""

import numpy as np

from mvis.core.config import CamImuCalibConfig
from mvis.calibration.pipeline_cam_imu import _decimate, _cam_factor_hz


def test_cam_factor_hz_precedence():
    cfg = CamImuCalibConfig(state_hz=30.0)          # factor_hz default 10.0
    assert _cam_factor_hz({}, cfg) == 10.0          # default -> cfg.factor_hz (10)
    cfg2 = CamImuCalibConfig(state_hz=30.0, factor_hz=5.0)
    assert _cam_factor_hz({}, cfg2) == 5.0          # cfg.factor_hz
    assert _cam_factor_hz({"factor_hz": 2.0}, cfg2) == 2.0   # per-cam overrides cfg
    assert _cam_factor_hz({"factor_hz": 12.0}, cfg) == 12.0  # per-cam over default


def test_default_factor_hz_decoupled_from_state_hz():
    """factor_hz (vision-factor rate) is independent of state_hz (trajectory rate):
    with the modern default factor_hz=10, the config's state_hz does NOT change the
    per-camera factor rate."""
    t = np.linspace(0.0, 10.0, 300)                 # ~30 Hz
    cfg_a = CamImuCalibConfig(state_hz=8.0)         # factor_hz default 10
    cfg_b = CamImuCalibConfig(state_hz=30.0)        # factor_hz default 10
    assert _cam_factor_hz({}, cfg_a) == 10.0
    assert _cam_factor_hz({}, cfg_b) == 10.0
    # same factor rate -> identical factor decimation regardless of state_hz
    assert np.array_equal(_decimate(t, _cam_factor_hz({}, cfg_a)),
                          _decimate(t, _cam_factor_hz({}, cfg_b)))


def test_lower_factor_hz_keeps_fewer_factors():
    t = np.linspace(0.0, 10.0, 300)
    cfg = CamImuCalibConfig(state_hz=30.0)          # dense trajectory, factor_hz 10
    n_default = _decimate(t, _cam_factor_hz({}, cfg)).sum()        # == 10 Hz
    n_cfg_low = _decimate(t, _cam_factor_hz({}, CamImuCalibConfig(
        state_hz=30.0, factor_hz=5.0))).sum()
    n_percam = _decimate(t, _cam_factor_hz({"factor_hz": 2.0}, cfg)).sum()
    assert n_default == _decimate(t, 10.0).sum()    # default factor-rate factors
    assert n_cfg_low < n_default                     # cfg.factor_hz thins factors
    assert n_percam < n_cfg_low                      # per-cam override thins more
    # the trajectory (state_hz) is a SEPARATE knob — unaffected by factor_hz
    assert _cam_factor_hz({"factor_hz": 2.0}, cfg) != cfg.state_hz


def test_zero_factor_hz_keeps_all():
    t = np.linspace(0.0, 1.0, 50)
    cfg = CamImuCalibConfig(state_hz=0.0, factor_hz=0.0)   # 0 = keep all factors
    assert _decimate(t, _cam_factor_hz({}, cfg)).all()


def test_legacy_target_hz_cam_dict_alias():
    """Back-compat: the legacy cam-dict `target_hz` key still works; canonical
    `factor_hz` wins when both are present."""
    cfg = CamImuCalibConfig(state_hz=30.0)
    assert _cam_factor_hz({"target_hz": 4.0}, cfg) == 4.0
    assert _cam_factor_hz({"factor_hz": 6.0, "target_hz": 4.0}, cfg) == 6.0
