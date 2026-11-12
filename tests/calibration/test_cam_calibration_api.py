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
"""Tests for the free-function camera-only API (config-cleanup M8).

`run_cam_calibration[_multi]` + `CamCalibConfig` mirror the cam-IMU
`run_cam_imu_calibration[_multi]` + `CamImuCalibConfig`. The wrapper must be a
faithful pass-through to `CalibrationPipeline` (same `cams`-dict input the cam-IMU
API uses)."""

import numpy as np

from mvis.detection.chessboard import Chessboard
from mvis.core.config import CamCalibConfig
from mvis.calibration.pipeline_cam import (
    CalibrationPipeline, PipelineConfig,
    run_cam_calibration, run_cam_calibration_multi,
)
from tests.helpers.cam_frames import (_IMG, _gen_frames)


def _cams_from_frames(frames):
    return [dict(camera_id=0,
                 t_cam=[f.timestamp for f in frames],
                 ids_list=[[o.point_id for o in f.observations] for f in frames],
                 uv_list=[[o.pixel for o in f.observations] for f in frames],
                 image_size=_IMG, model="radtan")]


def test_cam_calib_config_defaults():
    c = CamCalibConfig()
    assert c.max_iterations == 15
    assert c.sigma_px == 1.0
    assert c.huber_px == 1.345
    assert c.corner_subsample == "two_anti"


def test_wrapper_matches_direct_pipeline():
    """run_cam_calibration_multi must produce the same result as calling
    CalibrationPipeline directly with the equivalent PipelineConfig."""
    board = Chessboard(7, 7, 0.05)
    frames = _gen_frames(board)
    cfg = CamCalibConfig(max_iterations=10)

    r_wrap = run_cam_calibration_multi(_cams_from_frames(frames), board, config=cfg)

    pc = PipelineConfig(
        max_iterations=10, max_iterations_single=10, pixel_sigma=1.0,
        huber_threshold=1.345, corner_subsample="two_anti",
        optimize_intrinsics=True, optimize_time_offset=True, max_init_frames=100,
        init_frame_selection="uniform", reference_camera=None,
        rerun_after_outlier_rejection=False, distortion_models={0: "radtan"},
        verbose=False)
    r_dir = CalibrationPipeline(board, pc).run({0: frames}, {0: _IMG})

    assert r_wrap is not None and r_dir is not None
    np.testing.assert_allclose(np.asarray(r_wrap.cameras[0].intrinsics, float),
                               np.asarray(r_dir.cameras[0].intrinsics, float),
                               rtol=1e-6, atol=1e-6)


def test_single_cam_wrapper_runs():
    """run_cam_calibration (single-cam convenience) recovers ~the true focal."""
    board = Chessboard(7, 7, 0.05)
    frames = _gen_frames(board)
    cams = _cams_from_frames(frames)[0]
    r = run_cam_calibration(cams["t_cam"], cams["ids_list"], cams["uv_list"],
                            board, _IMG, model="radtan",
                            config=CamCalibConfig(max_iterations=10))
    assert r is not None
    fx = float(np.asarray(r.cameras[0].intrinsics, float)[0])
    assert abs(fx - 500.0) < 25.0          # near the true fx=500
