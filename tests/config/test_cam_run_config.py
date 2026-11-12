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
"""Tests for the sectioned camera-only CamRunConfig (config-cleanup M6, Stage A).

CamRunConfig mirrors CamImuRunConfig (input/board/solve/output) and is
read-compatible with the legacy flat CalibrationConfig. from_yaml accepts BOTH the
new sectioned shape and the legacy flat shape (adapted).
"""

import os
import tempfile

from mvis.core.config import CamRunConfig, CalibrationConfig, OptimizerConfig


def _yaml(text):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        return f.name


def _parse(loader, text):
    p = _yaml(text)
    try:
        return loader(p)
    finally:
        os.unlink(p)


_SECTIONED = """
input:
  cameras:
    - {camera_id: 0, distortion_model: radtan, shutter_type: rolling}
    - {camera_id: 1, distortion_model: equidistant}
board: {board_type: apriltag, rows: 6, cols: 6, tag_size: 0.088, tag_spacing: 0.3}
solve:
  max_iterations: 30
  pixel_sigma: 0.8
  frame_quality: {min_corner_views: 7}
output: {path: out.yaml, format: yaml}
reference_camera: 1
corner_subsample: two_diag
"""

_LEGACY = """
cameras:
  - {camera_id: 0, distortion_model: radtan}
board: {board_type: apriltag, rows: 6, cols: 6, tag_size: 0.088, tag_spacing: 0.3}
optimizer: {max_iterations: 25}
output_file: legacy.yaml
corner_subsample: all
reference_camera: 0
"""


def test_sectioned_shape_parses():
    cfg = _parse(CamRunConfig.from_yaml, _SECTIONED)
    assert [c.camera_id for c in cfg.input.cameras] == [0, 1]
    assert cfg.input.cameras[0].distortion_model == "radtan"
    assert cfg.input.cameras[0].shutter_type == "rolling"
    assert cfg.solve.max_iterations == 30
    assert cfg.solve.pixel_sigma == 0.8
    assert cfg.solve.frame_quality.min_corner_views == 7   # nested sub-config recurses
    assert cfg.output.path == "out.yaml"
    assert cfg.reference_camera == 1
    assert cfg.corner_subsample == "two_diag"


def test_read_compat_properties():
    cfg = _parse(CamRunConfig.from_yaml, _SECTIONED)
    assert cfg.cameras is cfg.input.cameras            # legacy .cameras
    assert isinstance(cfg.optimizer, OptimizerConfig)  # legacy .optimizer == .solve
    assert cfg.optimizer is cfg.solve
    assert cfg.output_file == "out.yaml"
    assert cfg.output_format == "yaml"


def test_legacy_flat_shape_adapts():
    cfg = _parse(CamRunConfig.from_yaml, _LEGACY)
    assert [c.camera_id for c in cfg.cameras] == [0]
    assert cfg.optimizer.max_iterations == 25
    assert cfg.output_file == "legacy.yaml"
    assert cfg.corner_subsample == "all"


def test_legacy_matches_calibration_config():
    """The legacy flat shape via CamRunConfig must expose the same effective values
    as CalibrationConfig.from_yaml on the same file."""
    path = _yaml(_LEGACY)
    try:
        cr = CamRunConfig.from_yaml(path)
        cc = CalibrationConfig.from_yaml(path)
    finally:
        os.unlink(path)
    assert len(cr.cameras) == len(cc.cameras)
    assert cr.optimizer.max_iterations == cc.optimizer.max_iterations
    assert cr.corner_subsample == cc.corner_subsample
    assert cr.output_file == cc.output_file
    assert cr.reference_camera == cc.reference_camera


def test_legacy_flat_fixture_via_cam_run():
    """The legacy flat shape (tests/data/configs/legacy_flat_cam.yaml, the old flat
    examples/configs/example_config.yaml) loads through both parsers."""
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
    path = os.path.join(root, "tests", "data", "configs", "legacy_flat_cam.yaml")
    cr = CamRunConfig.from_yaml(path)
    cc = CalibrationConfig.from_yaml(path)
    assert len(cr.cameras) == len(cc.cameras) >= 1
    assert cr.optimizer.max_iterations == cc.optimizer.max_iterations
    assert cr.corner_subsample == cc.corner_subsample


def test_sectioned_example_config_equals_legacy_flat():
    """examples/configs/example_config.yaml is the sectioned rewrite of the flat
    fixture: both shapes give the same CamRunConfig."""
    import dataclasses
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
    flat = CamRunConfig.from_yaml(os.path.join(root, "tests", "data", "configs",
                                               "legacy_flat_cam.yaml"))
    sect = CamRunConfig.from_yaml(os.path.join(root, "examples", "configs",
                                               "example_config.yaml"))
    assert dataclasses.asdict(sect) == dataclasses.asdict(flat)


def test_sectioned_global_factor_hz_fills_cameras():
    cfg = _parse(CamRunConfig.from_yaml,
                 "input:\n  cameras:\n    - {camera_id: 0}\n"
                 "    - {camera_id: 1, factor_hz: 12.0}\n"
                 "solve: {factor_hz: 7.0}\n")
    assert cfg.input.cameras[0].factor_hz == 7.0    # inherited solve.factor_hz
    assert cfg.input.cameras[1].factor_hz == 12.0   # per-camera override


def test_shipped_sectioned_example_cam_run():
    """The shipped sectioned example (examples/configs/example_cam_run.yaml) loads as
    CamRunConfig with the canonical factor_hz (per-camera override + global inherit)."""
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
    path = os.path.join(root, "examples", "configs", "example_cam_run.yaml")
    cfg = CamRunConfig.from_yaml(path)
    assert [c.camera_id for c in cfg.input.cameras] == [0, 1]
    assert cfg.input.cameras[0].factor_hz == 5.0                       # per-camera override
    assert cfg.input.cameras[1].factor_hz == cfg.solve.factor_hz       # inherited global
    assert cfg.solve.max_iterations == 15
    assert cfg.output.path == "mvis.yaml"
    # read-compat with the legacy CalibrationConfig surface
    assert cfg.cameras is cfg.input.cameras
    assert cfg.optimizer is cfg.solve


_ALIAS = """
input:
  cameras:
    - {camera_id: 0, distortion_model: radtan}
board: {board_type: apriltag, rows: 6, cols: 6, tag_size: 0.088, tag_spacing: 0.3}
solve: {max_iterations: 30, corner_subsample: top_left}
output: {path: out.yaml, format: yaml}
"""


def test_solve_corner_subsample_is_an_alias(caplog):
    """solve.corner_subsample (the cam-IMU spelling) sets the camera-only top-level key,
    and is no longer an unknown, ignored key (F14)."""
    with caplog.at_level("WARNING"):
        cfg = _parse(CamRunConfig.from_yaml, _ALIAS)
    assert cfg.corner_subsample == "top_left"
    assert cfg.solve.max_iterations == 30
    assert "unknown key" not in caplog.text


def test_top_level_corner_subsample_wins_over_the_solve_alias():
    cfg = _parse(CamRunConfig.from_yaml, _ALIAS + "corner_subsample: two_diag\n")
    assert cfg.corner_subsample == "two_diag"


def test_no_corner_subsample_keeps_the_two_anti_default():
    cfg = _parse(CamRunConfig.from_yaml, _ALIAS.replace(", corner_subsample: top_left", ""))
    assert cfg.corner_subsample == "two_anti"


def test_solve_alias_value_is_validated():
    from mvis.core.config_validator import validate_config
    cfg = _parse(CamRunConfig.from_yaml, _ALIAS.replace("top_left", "bogus"))
    assert cfg.corner_subsample == "bogus"
    assert any("corner_subsample" in e for e in validate_config(cfg))
