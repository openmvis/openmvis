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
"""U2 — unified `estimate:` block reconciliation (camera-only).

The `estimate:` block is the SOURCE OF TRUTH: when present it writes through to the
legacy `optimize_*` knobs the camera-only pipeline consumes (estimate wins, warns on
conflict). Split by sensor kind (`estimate.camera` / `estimate.imu`), rig-wide
defaults + per-camera overrides. When absent, legacy fields are untouched (covered by
test_estimate_characterization). Cam-IMU wiring (`estimate.imu`, hard pins) is U3.
"""

import os
import tempfile

from mvis.core.config import (
    CalibrationConfig,
    CameraEstimateConfig,
    CamImuRunConfig,
    CamRunConfig,
    OptimizerConfig,
)


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


# ---------------------------------------------------------------------------
# rig-wide estimate.camera -> legacy optimize_*
# ---------------------------------------------------------------------------

def test_rigwide_camera_maps_to_optimize_flags():
    c = _parse(CalibrationConfig.from_yaml,
               "estimate:\n  camera: {intrinsics: false, td: true, tr: true}\n")
    assert c.optimizer.optimize_intrinsics is False
    assert c.optimize_time_offset is True
    assert c.optimizer.optimize_rolling_shutter is True
    # and the parsed estimate block is attached
    assert c.estimate.camera.intrinsics is False


def test_unspecified_keys_leave_legacy_defaults():
    """Tri-state: a partial estimate: only touches the keys it sets."""
    c = _parse(CalibrationConfig.from_yaml,
               "estimate:\n  camera: {tr: true}\n")
    assert c.optimizer.optimize_rolling_shutter is True
    # intrinsics unspecified -> legacy default (True), td -> default (False)
    assert c.optimizer.optimize_intrinsics is OptimizerConfig().optimize_intrinsics
    assert c.optimize_time_offset is False


def test_estimate_wins_over_legacy_on_conflict():
    """Both estimate.camera.intrinsics and optimizer.optimize_intrinsics set ->
    estimate wins (a warning is emitted)."""
    c = _parse(CalibrationConfig.from_yaml,
               "optimizer:\n  optimize_intrinsics: true\n"
               "estimate:\n  camera: {intrinsics: false}\n")
    assert c.optimizer.optimize_intrinsics is False


def test_absent_estimate_is_noop():
    c = _parse(CalibrationConfig.from_yaml, "optimizer:\n  max_iterations: 7\n")
    assert c.optimizer.optimize_intrinsics is True   # untouched default
    assert c.optimize_time_offset is False
    assert c.estimate.camera.intrinsics is None      # nothing specified


# ---------------------------------------------------------------------------
# plural time-offset wart folded into the singular
# ---------------------------------------------------------------------------

def test_plural_time_offsets_folds_into_singular():
    c = _parse(CalibrationConfig.from_yaml,
               "optimizer:\n  optimize_time_offsets: true\n")
    assert c.optimize_time_offset is True


def test_estimate_td_beats_plural():
    """estimate.camera.td is the source of truth even against the legacy plural."""
    c = _parse(CalibrationConfig.from_yaml,
               "optimizer:\n  optimize_time_offsets: true\n"
               "estimate:\n  camera: {td: false}\n")
    assert c.optimize_time_offset is False


# ---------------------------------------------------------------------------
# per-camera override
# ---------------------------------------------------------------------------

def test_per_camera_estimate_override_parsed():
    c = _parse(CalibrationConfig.from_yaml,
               "cameras:\n"
               "  - {camera_id: 0, estimate: {intrinsics: false, tr: true}}\n"
               "  - {camera_id: 1}\n")
    assert c.cameras[0].estimate == CameraEstimateConfig(intrinsics=False, tr=True)
    assert c.cameras[1].estimate == CameraEstimateConfig()   # all None (inherit)


# ---------------------------------------------------------------------------
# camera-only honesty: vestigial keys recorded; imu sub-block ignored
# ---------------------------------------------------------------------------

def test_vestigial_keys_recorded_on_optimizer():
    c = _parse(CalibrationConfig.from_yaml,
               "estimate:\n"
               "  camera: {distortion: false, extrinsics: false}\n"
               "  board_points: true\n")
    assert c.optimizer.optimize_distortion is False
    assert c.optimizer.optimize_extrinsics is False
    assert c.optimizer.optimize_board_points is True


def test_imu_subblock_ignored_for_camera_only():
    """estimate.imu parses (attached) but does not affect the camera-only solver."""
    c = _parse(CalibrationConfig.from_yaml,
               "estimate:\n  imu: {intrinsics: full, model: rpng}\n")
    assert c.estimate.imu.intrinsics == "full"
    assert c.estimate.imu.model == "rpng"
    # camera-only optimizer untouched
    assert c.optimizer.optimize_intrinsics is True


# ---------------------------------------------------------------------------
# sectioned CamRunConfig path
# ---------------------------------------------------------------------------

def test_sectioned_camrun_estimate():
    c = _parse(CamRunConfig.from_yaml,
               "input: {cameras: [{camera_id: 0}]}\n"
               "estimate: {camera: {intrinsics: false, td: true}}\n")
    assert c.solve.optimize_intrinsics is False
    assert c.optimize_time_offset is True


def test_flat_camrun_carries_estimate():
    """Legacy flat shape routed through CamRunConfig still applies estimate:."""
    c = _parse(CamRunConfig.from_yaml,
               "cameras: [{camera_id: 0}]\n"
               "estimate: {camera: {intrinsics: false}}\n")
    assert c.solve.optimize_intrinsics is False
    assert c.estimate.camera.intrinsics is False


# ---------------------------------------------------------------------------
# cam-IMU reconciliation (parse-level; the hard pins themselves are exercised
# by a real solve in test_estimate_pins.py)
# ---------------------------------------------------------------------------

def test_camimu_camera_estimate_maps_to_solve_and_camera_input():
    c = _parse(CamImuRunConfig.from_yaml,
               "estimate:\n"
               "  camera: {intrinsics: true, tr: true, td: false, extrinsics: false}\n")
    assert c.camera_input.free_intrinsics is True          # intrinsics estimated
    assert c.camera_input.estimate_readout is True         # tr estimated
    assert c.solve.estimate_readout is True
    assert c.solve.estimate_td is False                    # td pinned
    assert c.solve.estimate_extrinsics is False            # cam-IMU extrinsic pinned


def test_camimu_imu_estimate_maps_intrinsics_model_and_aux():
    c = _parse(CamImuRunConfig.from_yaml,
               "estimate:\n"
               "  imu: {intrinsics: full, model: rpng, extrinsics: false, td: false}\n")
    assert c.solve.imu_intrinsics == "full"
    assert c.solve.imu_intrinsics_model == "rpng"
    assert c.solve.estimate_aux_extrinsics is False
    assert c.solve.estimate_aux_td is False


def test_camimu_imu_intrinsics_bool_maps_to_staging():
    """bool sugar: True -> full, False -> pinned."""
    c_true = _parse(CamImuRunConfig.from_yaml,
                    "estimate:\n  imu: {intrinsics: true}\n")
    assert c_true.solve.imu_intrinsics == "full"
    c_false = _parse(CamImuRunConfig.from_yaml,
                     "estimate:\n  imu: {intrinsics: false}\n")
    assert c_false.solve.imu_intrinsics == "pinned"


def test_camimu_estimate_absent_preserves_defaults():
    c = _parse(CamImuRunConfig.from_yaml, "input:\n  path: x.bag\n")
    assert c.camera_input.free_intrinsics is True      # estimate-intrinsics default
    assert c.solve.imu_intrinsics == "dw_da"
    assert c.solve.imu_intrinsics_model == "calyx"
    assert c.solve.estimate_extrinsics is True
    assert c.solve.estimate_aux_td is True


def test_camimu_estimate_wins_over_legacy():
    """estimate.imu.intrinsics beats an explicit solve.imu_intrinsics (warns)."""
    c = _parse(CamImuRunConfig.from_yaml,
               "solve:\n  imu_intrinsics: pinned\n"
               "estimate:\n  imu: {intrinsics: full}\n")
    assert c.solve.imu_intrinsics == "full"
