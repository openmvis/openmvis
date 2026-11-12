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
"""Characterization tests for the camera-pipeline config mapping (config-cleanup M2).

Locks CURRENT behavior BEFORE deduplicating the camera-pipeline config, so the
refactor can't silently change effective settings. Two divergent camera paths
exist today:

  * cli/main.py:calibrate  -> OptimizerOptions, reads config.optimizer.* (honors YAML)
  * scripts/run_calibration.py -> PipelineConfig, HARDCODES max_iterations=15,
    pixel_sigma=1.0, huber_threshold=1.345 (ignores the YAML optimizer block)

M2 will route both through one converter that reads config.optimizer.*. These tests
pin (1) the YAML->CalibrationConfig parse, (2) the PipelineConfig schema/defaults,
(3) the CLI OptimizerOptions contract, and (4) document the run_calibration.py
divergence the refactor removes.
"""

import os
import tempfile

from mvis.core.config import CalibrationConfig
from mvis.calibration.pipeline_cam import PipelineConfig
from mvis.calibration.optimizer import OptimizerOptions

_YAML = """
board:
  board_type: apriltag
  rows: 6
  cols: 6
  tag_size: 0.088
  tag_spacing: 0.3
cameras:
  - camera_id: 0
    name: cam0
    distortion_model: radtan
    shutter_type: rolling
  - camera_id: 1
    name: cam1
    distortion_model: equidistant
    shutter_type: global
optimizer:
  max_iterations: 42
  max_init_frames: 80
  init_frame_selection: diverse
  pixel_sigma: 1.5
  huber_threshold: 2.0
  absolute_error_tol: 1.0e-9
  relative_error_tol: 1.0e-8
  optimize_rolling_shutter: true
reference_camera: 1
corner_subsample: two_diag
rerun_outliers: true
"""


def _parsed() -> CalibrationConfig:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(_YAML)
        path = f.name
    try:
        return CalibrationConfig.from_yaml(path)
    finally:
        os.unlink(path)


def test_calibration_config_parses_optimizer_block():
    o = _parsed().optimizer
    assert o.max_iterations == 42
    assert o.max_init_frames == 80
    assert o.init_frame_selection == "diverse"
    assert o.pixel_sigma == 1.5
    assert o.huber_threshold == 2.0
    assert o.absolute_error_tol == 1.0e-9
    assert o.relative_error_tol == 1.0e-8
    assert o.optimize_rolling_shutter is True


def test_calibration_config_parses_top_level_and_cameras():
    c = _parsed()
    assert c.reference_camera == 1
    assert c.corner_subsample == "two_diag"
    assert c.rerun_outliers is True
    by_id = {cam.camera_id: cam for cam in c.cameras}
    assert by_id[0].shutter_type == "rolling"
    assert by_id[0].distortion_model == "radtan"
    assert by_id[1].shutter_type == "global"
    assert by_id[1].distortion_model == "equidistant"


def test_pipeline_config_schema_and_defaults():
    """Lock the PipelineConfig fields M2 touches so the refactor can't drop/rename.
    The first three are exactly what run_calibration.py hardcodes."""
    p = PipelineConfig()
    assert p.max_iterations == 15
    assert p.pixel_sigma == 1.0
    assert p.huber_threshold == 1.345
    assert p.max_init_frames == 100
    assert p.init_frame_selection == "uniform"
    assert p.optimize_intrinsics is True
    assert p.optimize_time_offset is True
    assert p.corner_subsample == "two_anti"
    assert p.rerun_after_outlier_rejection is False
    assert p.reference_camera is None
    assert p.stage2_target_hz is None
    assert p.frozen_intrinsics_cameras is None
    assert p.freeze_rs_cameras is None


def test_cli_optimizer_options_honor_config():
    """cli/main.py builds OptimizerOptions from config.optimizer.* — lock that
    contract (the CLI mapping M2 must preserve)."""
    c = _parsed()
    opts = OptimizerOptions(
        max_iterations=c.optimizer.max_iterations,
        pixel_sigma=c.optimizer.pixel_sigma,
        huber_threshold=c.optimizer.huber_threshold,
        absolute_error_tol=c.optimizer.absolute_error_tol,
        relative_error_tol=c.optimizer.relative_error_tol,
    )
    assert opts.max_iterations == 42
    assert opts.pixel_sigma == 1.5
    assert opts.huber_threshold == 2.0
    assert opts.absolute_error_tol == 1.0e-9
    assert opts.relative_error_tol == 1.0e-8


def test_pipeline_config_now_honors_optimizer_via_converter():
    """M2 LANDED (flip of the old divergence characterization): the camera pipeline
    now builds its PipelineConfig via `PipelineConfig.from_calibration_config`, which
    reads `config.optimizer.*` instead of the old hardcoded 15/1.0/1.345. The config
    here asks for 42/1.5/2.0 and the PipelineConfig must carry exactly that — the
    run_calibration.py divergence is gone. (Full converter coverage:
    test_pipeline_config_converter.py.)"""
    from mvis.calibration.pipeline_cam import PipelineConfig
    c = _parsed()
    pc = PipelineConfig.from_calibration_config(c)
    assert pc.max_iterations == c.optimizer.max_iterations == 42
    assert pc.pixel_sigma == c.optimizer.pixel_sigma == 1.5
    assert pc.huber_threshold == c.optimizer.huber_threshold == 2.0
