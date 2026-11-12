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
"""Tests for the config module: parsing, validation, bridge, backward compat."""

import tempfile
from pathlib import Path

import pytest
import yaml

from mvis.core.config import (
    BoardConfig,
    CalibrationConfig,
    CameraConfig,
    OptimizerConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, data: dict, name: str = "test.yaml") -> str:
    p = tmp_path / name
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return str(p)


MINIMAL_CONFIG = {
    "board": {
        "board_type": "apriltag",
        "rows": 6,
        "cols": 6,
        "tag_size": 0.088,
        "tag_spacing": 0.3,
    },
    "cameras": [
        {"camera_id": 0, "name": "cam0", "distortion_model": "equidistant"},
        {"camera_id": 1, "name": "cam1", "distortion_model": "equidistant"},
    ],
    "reference_camera": 0,
}


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_minimal_config_loads(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_CONFIG)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.board.board_type == "apriltag"
    assert cfg.board.rows == 6
    assert len(cfg.cameras) == 2
    assert cfg.cameras[0].camera_id == 0
    assert cfg.cameras[1].distortion_model == "equidistant"
    assert cfg.reference_camera == 0


def test_missing_sections_use_defaults(tmp_path):
    """A config with only cameras should still parse — everything else defaults."""
    path = _write_yaml(tmp_path, {
        "cameras": [{"camera_id": 0}],
    })
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.board.board_type == "chessboard"  # default
    assert cfg.optimizer.max_iterations == 200   # default
    assert cfg.reference_camera == 0             # default


def test_empty_file_uses_all_defaults(tmp_path):
    path = _write_yaml(tmp_path, {})
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.board.board_type == "chessboard"
    assert len(cfg.cameras) == 0
    assert cfg.reference_camera == 0


# ---------------------------------------------------------------------------
# Per-camera fields
# ---------------------------------------------------------------------------

def test_camera_shutter_type_parsed(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["cameras"] = [
        {"camera_id": 0, "shutter_type": "global"},
        {"camera_id": 1, "shutter_type": "rolling"},
    ]
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.cameras[0].shutter_type == "global"
    assert cfg.cameras[1].shutter_type == "rolling"


def test_camera_shutter_type_defaults_auto(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["cameras"] = [{"camera_id": 0}]
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.cameras[0].shutter_type == "auto"


def test_camera_factor_hz_parsed(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["cameras"] = [
        {"camera_id": 0, "factor_hz": 20.0},
        {"camera_id": 1, "factor_hz": 10.0},
    ]
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.cameras[0].factor_hz == 20.0
    assert cfg.cameras[1].factor_hz == 10.0


def test_camera_factor_hz_defaults_5(tmp_path):
    data = dict(MINIMAL_CONFIG)
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.cameras[0].factor_hz == 5.0


# ---------------------------------------------------------------------------
# Board config
# ---------------------------------------------------------------------------

def test_board_corner_subsample_removed_is_ignored(tmp_path):
    """config-cleanup M3c: BoardConfig.corner_subsample was removed (dead — the
    precedence never reached it). A YAML still setting board.corner_subsample parses
    (key ignored); the top-level corner_subsample is the camera-only canonical."""
    data = dict(MINIMAL_CONFIG)
    data["board"]["corner_subsample"] = "two_diag"    # dead key -> ignored
    data["corner_subsample"] = "top_left"             # canonical
    cfg = CalibrationConfig.from_yaml(_write_yaml(tmp_path, data))
    assert not hasattr(cfg.board, "corner_subsample")
    assert cfg.corner_subsample == "top_left"


def test_board_refine_subpixel_default_false(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_CONFIG)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.board.refine_subpixel is False


def test_board_refine_subpixel_explicit(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["board"]["refine_subpixel"] = True
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.board.refine_subpixel is True


# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------

def test_optimizer_defaults(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_CONFIG)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.optimizer.max_iterations == 200
    assert cfg.optimizer.max_init_frames == 100  # default: 100 frames for init
    assert cfg.optimizer.pixel_sigma == 1.0
    assert cfg.optimizer.huber_threshold == 1.345


def test_optimizer_custom_values(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["optimizer"] = {
        "max_iterations": 50,
        "pixel_sigma": 0.5,
        "huber_threshold": 2.0,
    }
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.optimizer.max_iterations == 50
    assert cfg.optimizer.pixel_sigma == 0.5
    assert cfg.optimizer.huber_threshold == 2.0


def test_max_init_frames_zero_means_no_cap(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["optimizer"] = {"max_init_frames": 0}
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.optimizer.max_init_frames == 0


# ---------------------------------------------------------------------------
# Round-trip: load → save → load
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path):
    path1 = _write_yaml(tmp_path, MINIMAL_CONFIG)
    cfg1 = CalibrationConfig.from_yaml(path1)

    path2 = str(tmp_path / "roundtrip.yaml")
    cfg1.to_yaml(path2)
    cfg2 = CalibrationConfig.from_yaml(path2)

    assert cfg2.board.board_type == cfg1.board.board_type
    assert cfg2.board.rows == cfg1.board.rows
    assert len(cfg2.cameras) == len(cfg1.cameras)
    assert cfg2.reference_camera == cfg1.reference_camera


# ---------------------------------------------------------------------------
# Backward compatibility — unknown keys should not crash
# ---------------------------------------------------------------------------

def test_unknown_board_key_ignored(tmp_path):
    """Extra keys in board section should not crash (forward compat)."""
    data = dict(MINIMAL_CONFIG)
    data["board"]["some_future_field"] = 42
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.board.rows == 6  # other fields still parsed


def test_unknown_camera_key_ignored(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["cameras"] = [{"camera_id": 0, "future_sensor_type": "lidar"}]
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.cameras[0].camera_id == 0


# ---------------------------------------------------------------------------
# Nested config sections (future — config_parser.py)
# ---------------------------------------------------------------------------

def test_nested_board_detection_config(tmp_path):
    """Board with a nested 'detection' section."""
    data = dict(MINIMAL_CONFIG)
    data["board"]["detection"] = {
        "min_tag_px": 15.0,
        "max_scale": 8.0,
    }
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.board.detection.min_tag_px == 15.0
    assert cfg.board.detection.max_scale == 8.0
    # Unset fields keep defaults
    assert cfg.board.detection.module_px_target == 25.0


def test_nested_optimizer_sections(tmp_path):
    """Optimizer with nested frame_quality, trajectory_alignment, guardrail."""
    data = dict(MINIMAL_CONFIG)
    data["optimizer"] = {
        "max_iterations": 15,
        "frame_quality": {
            "pose_filter": True,
            "min_corner_views": 3,
        },
        "trajectory_alignment": {
            "ransac_iterations": 500,
            "max_expected_dt_s": 1.0,
        },
        "guardrail": {
            "fx_min": 10.0,
            "p95_regression_factor": 2.0,
        },
    }
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.optimizer.max_iterations == 15
    assert cfg.optimizer.frame_quality.min_corner_views == 3
    assert cfg.optimizer.frame_quality.pose_filter is True
    assert cfg.optimizer.trajectory_alignment.ransac_iterations == 500
    assert cfg.optimizer.trajectory_alignment.max_expected_dt_s == 1.0
    assert cfg.optimizer.guardrail.fx_min == 10.0
    assert cfg.optimizer.guardrail.p95_regression_factor == 2.0
    # Unset nested fields keep defaults
    assert cfg.optimizer.outlier_rejection.threshold_px == 2.0


def test_frame_quality_new_fields(tmp_path):
    """New frame_quality fields (min_tags_per_frame, etc.) parse correctly."""
    data = dict(MINIMAL_CONFIG)
    data["optimizer"] = {
        "frame_quality": {
            "min_tags_per_frame": 8,
            "min_corner_views_per_camera": 5,
            "max_pnp_reprojection_px": 15.0,
            "stage1_intrinsic_validation": False,
        },
    }
    path = _write_yaml(tmp_path, data)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.optimizer.frame_quality.min_tags_per_frame == 8
    assert cfg.optimizer.frame_quality.min_corner_views_per_camera == 5
    assert cfg.optimizer.frame_quality.max_pnp_reprojection_px == 15.0
    assert cfg.optimizer.frame_quality.stage1_intrinsic_validation is False


def test_frame_quality_new_fields_defaults(tmp_path):
    """New frame_quality fields default correctly when not specified."""
    path = _write_yaml(tmp_path, MINIMAL_CONFIG)
    cfg = CalibrationConfig.from_yaml(path)
    assert cfg.optimizer.frame_quality.min_tags_per_frame == 6
    assert cfg.optimizer.frame_quality.min_corner_views_per_camera == 3
    assert cfg.optimizer.frame_quality.max_pnp_reprojection_px == 10.0
    assert cfg.optimizer.frame_quality.stage1_intrinsic_validation is True


# ---------------------------------------------------------------------------
# Validation (future — config_validator.py)
# ---------------------------------------------------------------------------

def test_validate_missing_cameras():
    from mvis.core.config_validator import validate_config
    cfg = CalibrationConfig()
    errors = validate_config(cfg)
    assert any("No cameras" in e for e in errors)


def test_validate_duplicate_camera_ids():
    from mvis.core.config_validator import validate_config
    cfg = CalibrationConfig(cameras=[
        CameraConfig(camera_id=0),
        CameraConfig(camera_id=0),
    ])
    errors = validate_config(cfg)
    assert any("Duplicate" in e for e in errors)


def test_validate_ref_camera_not_in_list():
    from mvis.core.config_validator import validate_config
    cfg = CalibrationConfig(
        cameras=[CameraConfig(camera_id=0)],
        reference_camera=5,
    )
    errors = validate_config(cfg)
    assert any("reference_camera" in e for e in errors)


def test_validate_negative_factor_hz():
    from mvis.core.config_validator import validate_config
    cfg = CalibrationConfig(cameras=[
        CameraConfig(camera_id=0, factor_hz=-1.0),
    ])
    errors = validate_config(cfg)
    assert any("factor_hz" in e for e in errors)


def test_validate_invalid_shutter_type():
    from mvis.core.config_validator import validate_config
    cfg = CalibrationConfig(cameras=[
        CameraConfig(camera_id=0, shutter_type="strobe"),
    ])
    errors = validate_config(cfg)
    assert any("shutter_type" in e for e in errors)


def test_validate_single_cam_rs_warning():
    from mvis.core.config_validator import validate_config
    cfg = CalibrationConfig(cameras=[
        CameraConfig(camera_id=0, shutter_type="rolling"),
    ])
    errors = validate_config(cfg)
    assert any("degenerate" in e for e in errors)


def test_validate_rs_ref_warning():
    from mvis.core.config_validator import validate_config
    cfg = CalibrationConfig(
        cameras=[
            CameraConfig(camera_id=0, shutter_type="rolling"),
            CameraConfig(camera_id=1, shutter_type="global"),
        ],
        reference_camera=0,
    )
    errors = validate_config(cfg)
    assert any("rolling shutter" in e.lower() and "reference" in e.lower()
               for e in errors)


# ---------------------------------------------------------------------------
# Config bridge (future — config_bridge.py)
# ---------------------------------------------------------------------------

def test_bridge_maps_corner_subsample():
    """CalibrationConfig.corner_subsample → PipelineConfig.corner_subsample (M2 converter)."""
    from mvis.calibration.pipeline_cam import PipelineConfig
    cfg = CalibrationConfig(corner_subsample="two_diag",
                            cameras=[CameraConfig(camera_id=0)])
    pc = PipelineConfig.from_calibration_config(cfg)
    assert pc.corner_subsample == "two_diag"


def test_bridge_maps_distortion_models():
    """Per-camera distortion_model → PipelineConfig.distortion_models dict."""
    cfg = CalibrationConfig(cameras=[
        CameraConfig(camera_id=0, distortion_model="equidistant"),
        CameraConfig(camera_id=1, distortion_model="radtan"),
    ])
    # pipeline_cfg = to_pipeline_config(cfg)
    # assert pipeline_cfg.distortion_models == {0: "equidistant", 1: "radtan"}


def test_bridge_cli_overrides_config():
    """CLI override beats the config value via the M2 converter."""
    from mvis.calibration.pipeline_cam import PipelineConfig
    cfg = CalibrationConfig(corner_subsample="all",
                            cameras=[CameraConfig(camera_id=0)])
    pc = PipelineConfig.from_calibration_config(cfg, corner_subsample="top_left")
    assert pc.corner_subsample == "top_left"   # override wins



def test_from_calibration_config_forwards_board_points(tmp_path):
    """estimate.board_points on a camera-only run config reaches
    PipelineConfig.board_heights through the run_calibration.py converter."""
    import yaml
    from mvis.calibration.pipeline_cam import PipelineConfig
    from mvis.core.config import CamRunConfig
    base = {
        "input": {"cameras": [{"camera_id": 0, "name": "c0", "source_type": "images",
                               "source_path": str(tmp_path), "distortion_model": "radtan"}]},
        "board": {"board_type": "apriltag", "rows": 6, "cols": 6, "tag_size": 0.088,
                  "tag_spacing": 0.3, "tag_family": "t36h11"},
        "solve": {"max_iterations": 15},
        "output": {"path": str(tmp_path / "out.yaml"), "format": "yaml"},
    }
    for flag in (True, False):
        p = tmp_path / f"cfg_{flag}.yaml"
        d = dict(base)
        if flag:
            d["estimate"] = {"board_points": True}
        yaml.safe_dump(d, open(p, "w"))
        cfg = CamRunConfig.from_yaml(str(p))
        pc = PipelineConfig.from_calibration_config(cfg)
        assert pc.board_heights is flag

def test_bridge_camera_configs_threaded():
    """CameraConfig list passed through for Stage 3 shutter_type lookup."""
    cams = [
        CameraConfig(camera_id=0, shutter_type="global"),
        CameraConfig(camera_id=1, shutter_type="rolling"),
    ]
    cfg = CalibrationConfig(cameras=cams)
    # pipeline_cfg = to_pipeline_config(cfg)
    # assert pipeline_cfg.camera_configs == cams


def test_removed_dead_config_keys_still_parse(tmp_path):
    """config-cleanup M5: SubsampleConfig and OptimizerConfig.rs_detection were
    removed (dead — never read by the pipeline). YAML that still carries the old
    `subsample:` / `optimizer.rs_detection:` sections must still parse (keys
    ignored: top-level dropped, nested warned), not crash."""
    data = dict(MINIMAL_CONFIG)
    data["subsample"] = {"min_centroid_shift_px": 99.0, "min_time_gap_s": 0.5}
    data["optimizer"] = {
        "max_iterations": 20,
        "rs_detection": {"correlation_threshold": 0.5, "min_frequency_hz": 7.0},
    }
    cfg = CalibrationConfig.from_yaml(_write_yaml(tmp_path, data))
    # parsed fine; the live field still applied, the dead sections ignored
    assert cfg.optimizer.max_iterations == 20
    assert not hasattr(cfg, "subsample")
    assert not hasattr(cfg.optimizer, "rs_detection")


def test_global_optimizer_factor_hz_fills_omitting_cameras(tmp_path):
    """config-cleanup M7c: optimizer.factor_hz is the global per-camera default
    (mirrors cam-IMU). Cameras omitting factor_hz inherit it; per-camera wins."""
    data = dict(MINIMAL_CONFIG)
    data["cameras"] = [
        {"camera_id": 0},                       # omits -> inherits the global
        {"camera_id": 1, "factor_hz": 20.0},    # explicit -> wins
    ]
    data["optimizer"] = {"factor_hz": 8.0}
    cfg = CalibrationConfig.from_yaml(_write_yaml(tmp_path, data))
    assert cfg.cameras[0].factor_hz == 8.0
    assert cfg.cameras[1].factor_hz == 20.0


def test_no_global_factor_hz_keeps_5(tmp_path):
    """Default global (5.0) preserves the historical per-camera default."""
    data = dict(MINIMAL_CONFIG)
    data["cameras"] = [{"camera_id": 0}]        # omits, no optimizer.factor_hz
    cfg = CalibrationConfig.from_yaml(_write_yaml(tmp_path, data))
    assert cfg.cameras[0].factor_hz == 5.0


def test_legacy_target_hz_yaml_alias(tmp_path):
    """Back-compat (config-cleanup M7 rename): the legacy `target_hz` YAML key still
    sets the canonical factor_hz, per-camera and global."""
    data = dict(MINIMAL_CONFIG)
    data["cameras"] = [{"camera_id": 0, "target_hz": 14.0}]    # legacy per-camera key
    data["optimizer"] = {"target_hz": 9.0}                     # legacy global key
    cfg = CalibrationConfig.from_yaml(_write_yaml(tmp_path, data))
    assert cfg.cameras[0].factor_hz == 14.0
    assert cfg.optimizer.factor_hz == 9.0


# ---------------------------------------------------------------------------
# Defaults and the file API (from_yaml / to_yaml)
# ---------------------------------------------------------------------------

def test_default_config():
    config = CalibrationConfig()
    assert config.board.board_type == "chessboard"
    assert config.reference_camera == 0
    assert config.output_format == "yaml"


def test_from_yaml_reads_every_section(tmp_path):
    data = {
        "board": {"board_type": "chessboard", "rows": 7, "cols": 5, "square_size": 0.03},
        "cameras": [{"camera_id": 0, "source_type": "images", "source_path": "/tmp/test",
                     "distortion_model": "radtan"}],
        "optimizer": {"max_iterations": 100, "pixel_sigma": 0.5},
        "reference_camera": 0,
        "output_file": "result.json",
        "output_format": "json",
    }
    config = CalibrationConfig.from_yaml(_write_yaml(tmp_path, data))
    assert config.board.rows == 7
    assert config.board.cols == 5
    assert config.board.square_size == 0.03
    assert len(config.cameras) == 1
    assert config.cameras[0].source_type == "images"
    assert config.optimizer.max_iterations == 100
    assert config.optimizer.pixel_sigma == 0.5
    assert config.output_format == "json"


def test_to_yaml_round_trips_the_board(tmp_path):
    config = CalibrationConfig()
    config.board = BoardConfig(board_type="apriltag", rows=4, cols=3)
    path = str(tmp_path / "out.yaml")
    config.to_yaml(path)
    loaded = CalibrationConfig.from_yaml(path)
    assert loaded.board.board_type == "apriltag"
    assert loaded.board.rows == 4
    assert loaded.board.cols == 3
