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
"""I0 guards + fix tests (database plan §11 / task I0).

Four items: detection-cache emission-order round-trip; _calib.yaml round-trip
through load_camera_intrinsics (untested before); is_fisheye = model != radtan
(the camera_input fix); ArUco boards take the tag-grid branch of the VP init
(the _is_tag_grid fix). regenerate_html's stored-board_poses fix is tested via
its helper.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from mvis.core.types import CalibrationResult, CameraParams, FrameObservations, Observation
from mvis.io.detection_cache import DetectionCache
from mvis.io.camera_input import load_camera_intrinsics
from mvis.io.output_writer import OutputWriter
from mvis.calibration.initializer import _is_tag_grid
from mvis.analysis.regenerate_html import _apply_stored_board_poses


# ── I0.1: cache round-trip preserves emission order + bits ──────────────────

def test_detection_cache_round_trip_bit_identical(tmp_path):
    frames = {0: [FrameObservations(
        camera_id=0, timestamp=12.5,
        observations=[Observation(point_id=9, pixel=np.array([3.5, 7.25]), row=7),
                      Observation(point_id=2, pixel=np.array([np.pi, np.e]), row=2),
                      Observation(point_id=5, pixel=np.array([0.1, 0.2]), row=0)],
        image_size=(640, 480))]}
    cache = DetectionCache(str(tmp_path / "detections.json"))
    cache.save(frames, "src", {"k": 1})
    loaded, _, _ = cache.load("src", {"k": 1})
    a, b = frames[0][0], loaded[0][0]
    assert [o.point_id for o in b.observations] == [9, 2, 5]   # emission order kept
    for oa, ob in zip(a.observations, b.observations):
        assert oa.pixel.tobytes() == ob.pixel.tobytes()        # exact float64 bits
        assert oa.row == ob.row
    assert b.timestamp == a.timestamp and b.image_size == a.image_size


# ── I0.2: calib yaml round-trips through load_camera_intrinsics ─────────────

def test_calib_yaml_round_trip(tmp_path):
    cams = [
        CameraParams(0, np.array([500.0, 501.0, 320.0, 240.0]),
                     np.array([-0.1, 0.01, 0.001, -0.001]), "radtan", (640, 480)),
        CameraParams(1, np.array([280.0, 281.0, 424.0, 400.0]),
                     np.array([0.01, -0.002, 0.0005, -0.0001]), "equidistant",
                     (848, 800), rolling_shutter_readout=0.03),
        CameraParams(2, np.array([180.0, 181.0, 320.0, 240.0]),
                     np.array([-0.2, 0.6, 0.0, 0.0]), "double_sphere", (640, 480)),
        CameraParams(3, np.array([380.0, 381.0, 512.0, 510.0]),
                     np.array([0.64]), "ucm", (1024, 1024)),
    ]
    result = CalibrationResult(cameras=cams)
    path = str(tmp_path / "x_calib.yaml")
    OutputWriter.write_calibration_yaml(result, path)
    loaded = load_camera_intrinsics(path)
    assert len(loaded) == 4
    for cp, lc in zip(cams, loaded):
        np.testing.assert_allclose(lc["intrinsics"], cp.intrinsics)
        got = np.asarray(lc["distortion_coeffs"], float)
        k = min(len(got), len(cp.distortion))   # loader trims ds/eucm to their
        np.testing.assert_allclose(got[:k], cp.distortion[:k])  # real param count
        assert np.allclose(cp.distortion[k:], 0.0)              # nothing lost
        assert lc["distortion_model"] == cp.distortion_model
        assert tuple(lc["resolution"]) == cp.image_size
        assert lc["rolling_shutter_readout"] == pytest.approx(cp.rolling_shutter_readout)


# ── I0.4: is_fisheye covers all non-radtan models ───────────────────────────

def test_is_fisheye_all_models(tmp_path):
    expected = {"radtan": False, "equidistant": True, "fov": True,
                "double_sphere": True, "eucm": True, "omni_radtan": True,
                "kb4": True, "ucm": True}
    dist_len = {"radtan": 4, "equidistant": 4, "fov": 1,
                "double_sphere": 2, "eucm": 2, "omni_radtan": 5, "kb4": 4, "ucm": 1}
    cams = [CameraParams(i, np.array([300.0, 300.0, 320.0, 240.0]),
                         np.linspace(0.01, 0.04, dist_len[m]), m, (640, 480))
            for i, m in enumerate(expected)]
    path = str(tmp_path / "m_calib.yaml")
    OutputWriter.write_calibration_yaml(CalibrationResult(cameras=cams), path)
    for lc in load_camera_intrinsics(path):
        assert lc["is_fisheye"] is expected[lc["distortion_model"]], lc["distortion_model"]


# ── I0.5: ArUco grids take the tag-grid branch of the VP init ───────────────

def test_is_tag_grid_predicate():
    class AprilLike:  _tag_size = 0.08
    class ArucoLike:  _marker_size = 0.05
    class ChessLike:  _square_size = 0.03
    assert _is_tag_grid(AprilLike()) is True
    assert _is_tag_grid(ArucoLike()) is True      # the I0 bug: was False
    assert _is_tag_grid(ChessLike()) is False
    # the real Chessboard aliases _tag_size = square_size; it is still no tag grid
    from mvis.detection.chessboard import Chessboard
    assert hasattr(Chessboard(7, 9, 0.04), "_tag_size")
    assert _is_tag_grid(Chessboard(7, 9, 0.04)) is False


# ── I0.3: regenerate_html uses stored board_poses ───────────────────────────

def test_apply_stored_board_poses():
    result = CalibrationResult(cameras=[CameraParams(
        0, np.array([500.0, 500.0, 320.0, 240.0]), np.zeros(4), "radtan", (640, 480))])
    T = np.eye(4); T[:3, 3] = (0.1, 0.2, 0.7)
    doc = {"board_poses": [{"frame_index": 0, "timestamp": 3.25, "T_board_cam": T.tolist()},
                           {"frame_index": 2, "timestamp": 3.45, "T_board_cam": np.eye(4).tolist()}]}
    assert _apply_stored_board_poses(result, doc) is True
    assert len(result.board_poses) == 2
    np.testing.assert_array_equal(result.board_poses[0], T)
    assert result.board_pose_times == [3.25, 3.45]
    np.testing.assert_allclose(result.board_poses_gtsam[0].matrix(), T)
    # calib-only yaml (no board_poses) -> not applied, PnP fallback stays live
    empty = CalibrationResult(cameras=result.cameras)
    assert _apply_stored_board_poses(empty, {"cameras": []}) is False
    assert empty.board_poses == []
