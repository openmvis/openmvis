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
"""Unit tests for PipelineConfig.from_calibration_config (config-cleanup M2).

The single converter that replaces the hardcoded PipelineConfig construction in
scripts/run_calibration.py — optimizer numerics now come from config.optimizer.*
(no more 15/1.0/1.345 literals), runtime/derived fields are keyword overrides.
"""

from mvis.core.config import CalibrationConfig, CameraConfig, OptimizerConfig
from mvis.calibration.pipeline_cam import PipelineConfig


def _cfg(**opt_over):
    opt = dict(max_iterations=42, max_init_frames=80, init_frame_selection="diverse",
               pixel_sigma=1.5, huber_threshold=2.0, optimize_intrinsics=False)
    opt.update(opt_over)
    return CalibrationConfig(
        cameras=[CameraConfig(camera_id=0, distortion_model="radtan"),
                 CameraConfig(camera_id=1, distortion_model="equidistant")],
        optimizer=OptimizerConfig(**opt),
        reference_camera=1, corner_subsample="two_diag", rerun_outliers=True,
        optimize_time_offset=False)


def test_optimizer_numerics_from_config_not_hardcoded():
    pc = PipelineConfig.from_calibration_config(_cfg())
    assert pc.max_iterations == 42                # was hardcoded 15
    assert pc.max_iterations_single == 42         # both map to config.optimizer
    assert pc.pixel_sigma == 1.5                  # was hardcoded 1.0
    assert pc.huber_threshold == 2.0              # was hardcoded 1.345
    assert pc.max_init_frames == 80
    assert pc.init_frame_selection == "diverse"
    assert pc.optimize_intrinsics is False


def test_distortion_models_built_from_cameras():
    pc = PipelineConfig.from_calibration_config(_cfg())
    assert pc.distortion_models == {0: "radtan", 1: "equidistant"}
    # explicit override wins
    pc2 = PipelineConfig.from_calibration_config(_cfg(), distortion_models={0: "eucm"})
    assert pc2.distortion_models == {0: "eucm"}


def test_config_values_used_when_no_override():
    pc = PipelineConfig.from_calibration_config(_cfg())
    assert pc.reference_camera == 1               # config.reference_camera
    assert pc.corner_subsample == "two_diag"      # config.corner_subsample
    assert pc.rerun_after_outlier_rejection is True   # config.rerun_outliers
    assert pc.optimize_time_offset is False       # config.optimize_time_offset
    assert [c.camera_id for c in pc.camera_configs] == [0, 1]


def test_overrides_win():
    pc = PipelineConfig.from_calibration_config(
        _cfg(), reference_camera=0, corner_subsample="all",
        optimize_time_offset=True, rerun_after_outlier_rejection=False)
    assert pc.reference_camera == 0
    assert pc.corner_subsample == "all"
    assert pc.optimize_time_offset is True
    assert pc.rerun_after_outlier_rejection is False


def test_runtime_fields_default_none():
    pc = PipelineConfig.from_calibration_config(_cfg())
    assert pc.frozen_intrinsics_cameras is None
    assert pc.freeze_rs_cameras is None
    assert pc.stage2_target_hz is None
    assert pc.rerun_visualization is False
    pc2 = PipelineConfig.from_calibration_config(
        _cfg(), frozen_intrinsics_cameras={0}, stage2_target_hz={0: 10.0})
    assert pc2.frozen_intrinsics_cameras == {0}
    assert pc2.stage2_target_hz == {0: 10.0}


def test_behavior_preserving_for_regression_values():
    """Regression YAMLs all specify 15/1.0/1.345 — the converter must reproduce
    exactly what run_calibration.py used to hardcode."""
    cfg = _cfg(max_iterations=15, pixel_sigma=1.0, huber_threshold=1.345,
               optimize_intrinsics=True)
    pc = PipelineConfig.from_calibration_config(cfg)
    assert (pc.max_iterations, pc.max_iterations_single) == (15, 15)
    assert pc.pixel_sigma == 1.0
    assert pc.huber_threshold == 1.345
    assert pc.optimize_intrinsics is True
