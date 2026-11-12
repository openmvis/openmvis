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
"""U1 characterization — pin the CURRENT 'what to calibrate' control mapping.

This is the regression gate for the unified `estimate:` block (unified-estimate
U2-U3): when `estimate:` is absent, the parser must keep producing byte-identical
low-level field values, so the solve is unchanged. These tests document — and lock
— how each quantity (cam intrinsics, IMU intrinsics, extrinsics, td, tr) is
controlled TODAY, including the camera-only knobs that are currently VESTIGIAL
(defined on the dataclass but never consumed by the pipeline).

Findings audited 2026-06-19 (grep over src/mvis):
  camera-only, REAL:      optimize_intrinsics, optimize_time_offset,
                          optimize_rolling_shutter (CLI, + per-cam shutter_type)
  camera-only, VESTIGIAL: optimize_distortion, optimize_extrinsics,
                          optimize_board_points, optimize_time_offsets (plural)
  cam-IMU:                camera_input.free_intrinsics -> per-cam pin_intrinsics;
                          camera_input.estimate_readout -> per-cam pin_tr (inverted);
                          solve.imu_intrinsics (staging string); solve.estimate_readout;
                          solve.sigma_td_prior (td anchor; 0 = free); extrinsic ALWAYS free.
"""

import os
import tempfile

from mvis.calibration.pipeline_cam import PipelineConfig
from mvis.calibration.pipeline_cam_imu import _cam_pin_tr
from mvis.core.config import (
    CalibrationConfig,
    CamImuRunConfig,
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
# Camera-only: which optimize_* flags are REAL (carried into the pipeline)
# ---------------------------------------------------------------------------

def test_camera_optimize_intrinsics_reaches_pipeline():
    """cam_intrinsics control == OptimizerConfig.optimize_intrinsics, carried
    by from_calibration_config into the PipelineConfig the solver reads."""
    cfg = _parse(CalibrationConfig.from_yaml,
                 "optimizer:\n  optimize_intrinsics: false\n")
    assert cfg.optimizer.optimize_intrinsics is False
    pc = PipelineConfig.from_calibration_config(cfg)
    assert pc.optimize_intrinsics is False


def test_camera_optimize_time_offset_reaches_pipeline():
    """td control (camera-only) == CalibrationConfig.optimize_time_offset."""
    cfg = _parse(CalibrationConfig.from_yaml, "optimize_time_offset: true\n")
    assert cfg.optimize_time_offset is True
    pc = PipelineConfig.from_calibration_config(cfg)
    assert pc.optimize_time_offset is True


def test_camera_rolling_shutter_flag_default_off():
    """tr control (camera-only) == OptimizerConfig.optimize_rolling_shutter,
    consumed in the CLI together with per-camera shutter_type."""
    cfg = _parse(CalibrationConfig.from_yaml,
                 "optimizer:\n  optimize_rolling_shutter: true\n")
    assert cfg.optimizer.optimize_rolling_shutter is True
    # default stays off
    assert OptimizerConfig().optimize_rolling_shutter is False


def test_camera_vestigial_flags_parse_but_are_not_pipelined():
    """optimize_distortion / optimize_extrinsics / optimize_board_points are
    currently NOT consumed by the camera-only pipeline (vestigial). They parse
    and round-trip onto the dataclass, but from_calibration_config does NOT carry
    them. This test locks that reality so U2 mapping treats them honestly (no-op +
    warn for camera-only) rather than silently pretending to control them."""
    cfg = _parse(CalibrationConfig.from_yaml,
                 "optimizer:\n"
                 "  optimize_distortion: false\n"
                 "  optimize_extrinsics: false\n"
                 "  optimize_board_points: true\n")
    assert cfg.optimizer.optimize_distortion is False
    assert cfg.optimizer.optimize_extrinsics is False
    assert cfg.optimizer.optimize_board_points is True
    # ... but the PipelineConfig the solver reads has no such fields.
    pc = PipelineConfig.from_calibration_config(cfg)
    assert not hasattr(pc, "optimize_distortion")
    assert not hasattr(pc, "optimize_extrinsics")
    assert not hasattr(pc, "optimize_board_points")


def test_camera_optimize_time_offsets_plural_folds_to_singular():
    """OptimizerConfig.optimize_time_offsets (plural) was historically vestigial;
    U2 folds it into the live SINGULAR top-level optimize_time_offset (deprecation
    warning). The singular / estimate.camera.td still win over it."""
    assert hasattr(OptimizerConfig(), "optimize_time_offsets")
    cfg = _parse(CalibrationConfig.from_yaml,
                 "optimizer:\n  optimize_time_offsets: true\n")
    assert cfg.optimizer.optimize_time_offsets is True
    assert cfg.optimize_time_offset is True   # folded into the live knob (U2)


# ---------------------------------------------------------------------------
# Cam-IMU: free-by-default, pin-to-control
# ---------------------------------------------------------------------------

def test_camimu_free_intrinsics_default_estimates():
    """cam intrinsics: camera_input.free_intrinsics defaults True == estimated
    jointly (the provided file seeds them). The run path sets per-cam
    pin_intrinsics = not free_intrinsics. Set False to PIN (omni/fisheye gauge)."""
    cfg = _parse(CamImuRunConfig.from_yaml, "input:\n  path: x.bag\n")
    assert cfg.camera_input.free_intrinsics is True
    cfg2 = _parse(CamImuRunConfig.from_yaml,
                  "camera_input:\n  free_intrinsics: false\n")
    assert cfg2.camera_input.free_intrinsics is False


def test_camimu_tr_default_global_and_estimate_readout():
    """tr: solve.estimate_readout defaults False (global shutter). Per-camera
    estimate_readout overrides via the cams dict (_cam_pin_tr; pin_tr == not est)."""
    cfg = _parse(CamImuRunConfig.from_yaml, "input:\n  path: x.bag\n")
    assert cfg.solve.estimate_readout is False
    assert cfg.solve.pin_tr is True
    # per-cam override semantics
    assert _cam_pin_tr({"estimate_readout": True}, cfg.solve) is False
    assert _cam_pin_tr({"estimate_readout": False}, cfg.solve) is True
    # legacy pin_tr key still honored, config default falls through
    assert _cam_pin_tr({"pin_tr": False}, cfg.solve) is False
    assert _cam_pin_tr({}, cfg.solve) is True


def test_camimu_imu_intrinsics_staging_default_dwda():
    """IMU intrinsics: solve.imu_intrinsics staging string; default 'dw_da'
    (gyro/accel scale + misalignment freed) — the stage every shipped cam-IMU
    config already selects."""
    cfg = _parse(CamImuRunConfig.from_yaml, "input:\n  path: x.bag\n")
    assert cfg.solve.imu_intrinsics == "dw_da"
    for stage in ("pinned", "dw_da_r", "full"):
        c = _parse(CamImuRunConfig.from_yaml,
                   f"solve:\n  imu_intrinsics: {stage}\n")
        assert c.solve.imu_intrinsics == stage


def test_camimu_model_default_calyx():
    """IMU model FAMILY: solve.imu_intrinsics_model; default 'calyx'. Distinct
    from the staging knob above (family vs how-much)."""
    cfg = _parse(CamImuRunConfig.from_yaml, "input:\n  path: x.bag\n")
    assert cfg.solve.imu_intrinsics_model == "calyx"
    for model in ("kalibr", "rpng", "calyx"):
        c = _parse(CamImuRunConfig.from_yaml,
                   f"solve:\n  imu_intrinsics_model: {model}\n")
        assert c.solve.imu_intrinsics_model == model


def test_camimu_td_anchor_default_free():
    """td: solve.sigma_td_prior defaults 0.0 == td free. A >0 value anchors it
    (and is what U3's estimate.td=false hard pin will set to a _TINY sigma)."""
    cfg = _parse(CamImuRunConfig.from_yaml, "input:\n  path: x.bag\n")
    assert cfg.solve.sigma_td_prior == 0.0
    # NB: YAML 1.1 only resolves a float with a decimal point + sign on the
    # exponent (`1.0e-6`); bare `1e-6` would arrive as the string "1e-6" (the
    # parser passes values through without coercion). U3 sets the hard pin
    # programmatically as a float, so this affects hand-written YAML only.
    c = _parse(CamImuRunConfig.from_yaml, "solve:\n  sigma_td_prior: 1.0e-6\n")
    assert c.solve.sigma_td_prior == 1.0e-6


def test_camimu_extrinsic_controls_default_to_estimate():
    """U3 added explicit on/off controls for the cam-IMU extrinsic + the inter-IMU
    extrinsic/td (the gaps this characterization originally flagged). All default
    True = estimated (the historical always-free behavior is preserved)."""
    cfg = _parse(CamImuRunConfig.from_yaml, "input:\n  path: x.bag\n")
    assert cfg.solve.estimate_extrinsics is True       # cam-IMU T_C->I
    assert cfg.solve.estimate_td is True               # camera <-> base-IMU td
    assert cfg.solve.estimate_aux_extrinsics is True   # aux-IMU T_IstoI
    assert cfg.solve.estimate_aux_td is True           # inter-IMU td
