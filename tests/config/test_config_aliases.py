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
"""Field-name alias tests (config-cleanup M3a): pixel_sigma <-> sigma_px.

The same pixel-noise concept is spelled `pixel_sigma` in the camera-only optimizer
and `sigma_px` in the cam-IMU solve. The parser now accepts BOTH spellings in either
config (canonical wins on conflict), so a YAML written for one pipeline reads in the
other.
"""

import os
import tempfile

from mvis.core.config import CalibrationConfig, CamImuRunConfig


def _yaml(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        return f.name


def _parse(loader, text):
    path = _yaml(text)
    try:
        return loader(path)
    finally:
        os.unlink(path)


def test_camera_optimizer_accepts_sigma_px_alias():
    cfg = _parse(CalibrationConfig.from_yaml, "optimizer:\n  sigma_px: 0.7\n")
    assert cfg.optimizer.pixel_sigma == 0.7      # alias mapped to canonical


def test_camera_optimizer_canonical_still_works():
    cfg = _parse(CalibrationConfig.from_yaml, "optimizer:\n  pixel_sigma: 0.6\n")
    assert cfg.optimizer.pixel_sigma == 0.6


def test_camera_optimizer_canonical_wins_on_conflict():
    cfg = _parse(CalibrationConfig.from_yaml,
                 "optimizer:\n  pixel_sigma: 0.9\n  sigma_px: 0.1\n")
    assert cfg.optimizer.pixel_sigma == 0.9      # canonical wins; alias ignored


def test_cam_imu_solve_accepts_pixel_sigma_alias():
    cfg = _parse(CamImuRunConfig.from_yaml, "solve:\n  pixel_sigma: 0.7\n")
    assert cfg.solve.sigma_px == 0.7             # alias mapped to canonical


def test_cam_imu_solve_canonical_still_works():
    cfg = _parse(CamImuRunConfig.from_yaml, "solve:\n  sigma_px: 0.55\n")
    assert cfg.solve.sigma_px == 0.55


def test_cam_imu_solve_canonical_wins_on_conflict():
    cfg = _parse(CamImuRunConfig.from_yaml,
                 "solve:\n  sigma_px: 0.55\n  pixel_sigma: 0.1\n")
    assert cfg.solve.sigma_px == 0.55


# ---------------------------------------------------------------------------
# pin_tr -> estimate_readout rename (config-cleanup M3d). Canonical is
# estimate_readout (bool, default False = global). pin_tr is the legacy alias
# (inverts: pin_tr True == estimate_readout False) + a read-only property.
# ---------------------------------------------------------------------------
from mvis.calibration.pipeline_cam_imu import _cam_pin_tr
from mvis.core.config import CamImuCalibConfig


def test_solve_estimate_readout_canonical():
    cfg = _parse(CamImuRunConfig.from_yaml, "solve:\n  estimate_readout: true\n")
    assert cfg.solve.estimate_readout is True
    assert cfg.solve.pin_tr is False           # property derives from canonical


def test_solve_pin_tr_legacy_inverts():
    cfg = _parse(CamImuRunConfig.from_yaml, "solve:\n  pin_tr: true\n")
    assert cfg.solve.estimate_readout is False  # pin_tr True -> not estimating
    assert cfg.solve.pin_tr is True
    cfg2 = _parse(CamImuRunConfig.from_yaml, "solve:\n  pin_tr: false\n")
    assert cfg2.solve.estimate_readout is True   # pin_tr False -> estimating (rolling)


def test_camera_input_estimate_readout_and_legacy():
    cfg = _parse(CamImuRunConfig.from_yaml,
                 "camera_input:\n  estimate_readout: true\n")
    assert cfg.camera_input.estimate_readout is True
    cfg2 = _parse(CamImuRunConfig.from_yaml, "camera_input:\n  pin_tr: false\n")
    assert cfg2.camera_input.estimate_readout is True   # legacy inverted


def test_estimate_readout_canonical_wins_on_conflict():
    cfg = _parse(CamImuRunConfig.from_yaml,
                 "solve:\n  estimate_readout: true\n  pin_tr: true\n")
    assert cfg.solve.estimate_readout is True   # canonical wins; legacy ignored


def test_cam_pin_tr_helper_prefers_canonical():
    cfg = CamImuCalibConfig()  # estimate_readout False -> cfg.pin_tr True
    assert _cam_pin_tr({"estimate_readout": True}, cfg) is False   # rolling
    assert _cam_pin_tr({"estimate_readout": False}, cfg) is True   # global
    assert _cam_pin_tr({"pin_tr": False}, cfg) is False            # legacy key
    assert _cam_pin_tr({"pin_tr": True}, cfg) is True
    assert _cam_pin_tr({}, cfg) is True                            # cfg default
    # canonical key wins over legacy in the same dict
    assert _cam_pin_tr({"estimate_readout": True, "pin_tr": True}, cfg) is False
