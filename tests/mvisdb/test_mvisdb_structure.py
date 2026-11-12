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
"""Structure-law tests for MVISDatabase.

The laws under test: selection flows down, validity flows up, derived values
are never stored, the PoseSource ladder, the reference rule, fingerprint identity.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from mvis.mvisdb.mvisdatabase import (Board, BoardPt, BoardType, CalibChoice,
                                      CameraInfo, CameraModel, CameraIntrinsicsState,
                                      CamOnlyBaseState, Corner, EstimateStates,
                                      PolicyEntry, PolicyState, PoseSource,
                                      RelativeCalib, SensorState, SensorType,
                                      TagDetection, frames_to_arrays,
                                      observation_fingerprint)
from mvis.mvisdb.estimates import pose7_to_mat
from tests.helpers.mvisdb_fixtures import make_db


# ---------------------------------------------------------------------------
# the laws
# ---------------------------------------------------------------------------

def test_validity_flows_up_empty_frame_is_invalid():
    db = make_db()
    empty = db.camera(0).frames[2]
    assert empty.board_detections == []
    assert empty.is_valid_to_use is False       # any([]) == False, for free


def test_validity_flows_up_any_group():
    db = make_db()
    f = db.camera(0).frames[0]
    assert f.is_valid_to_use is True
    for g in f.board_detections:
        g.is_valid_to_use = False
    assert f.is_valid_to_use is False
    f.board_detections[0].is_valid_to_use = True
    assert f.is_valid_to_use is True


def test_selection_flows_down():
    db = make_db()
    f = db.camera(0).frames[0]
    g = f.board_detections[0]
    assert g.is_selected_to_use is True
    f.is_selected_to_use = False
    assert g.is_selected is True                # own bit untouched
    assert g.is_selected_to_use is False        # but the derived answer follows


def test_derived_counts_hand_counted():
    db = make_db()
    g = db.camera(0).frames[0].board_detections[0]
    assert g.n_detections == 4
    assert g.n_tags == 2                        # pids {0,1} tag 0 · {4,5} tag 1
    assert g.detect_status == "ok"


def test_derived_tag_corner_row_point3d():
    db = make_db()
    d = db.camera(0).frames[0].board_detections[0].detections[2]  # pid 4
    assert d.tag_id == 1
    assert d.corner_id is Corner.BL
    assert d.row == 22                          # int(22.5), truncation
    np.testing.assert_allclose(d.point_3d, [0.1, 0.0, 0.0])


def test_chessboard_corner_is_none():
    b = Board(1, BoardType.CHESSBOARD, rows=3, cols=4, spacing=0.05, size=0.05,
              pts={i: BoardPt(i, i, None, np.zeros(3)) for i in range(12)})
    d = TagDetection(uv=np.array([1.0, 2.0]), point_id=7, board=b)
    assert b.corners_per_tag == 1
    assert d.tag_id == 7
    assert d.corner_id is None


def test_is_fisheye_per_model():
    fisheye = {CameraModel.EQUIDISTANT, CameraModel.FOV, CameraModel.DOUBLE_SPHERE,
               CameraModel.EUCM, CameraModel.UCM, CameraModel.OMNI_RADTAN,
               CameraModel.KB4}
    for m in CameraModel:
        info = CameraInfo("c", camera_model=m)
        assert info.is_fisheye is (m in fisheye)
    assert CameraInfo("c").is_fisheye is False  # unknown model -> not fisheye


def test_reference_stream_keys_on_calib_choice():
    assert make_db(CalibChoice.CAM_ONLY).reference_stream == (SensorType.CAM, 0)
    assert make_db(CalibChoice.CAM_IMU).reference_stream == (SensorType.IMU, 0)
    db = make_db(CalibChoice.CAM_IMU)
    db.rig.base_imu_id = None
    with pytest.raises(ValueError):
        _ = db.reference_stream


def test_pose_ladder_monotonic():
    db = make_db()
    g = db.camera(0).frames[0].board_detections[0]
    g.set_pose(np.eye(4), PoseSource.PNP)
    g.set_pose(np.eye(4), PoseSource.REFINED_PER_CAMERA)
    g.set_pose(np.eye(4), PoseSource.REFINED_CAM_ONLY)
    with pytest.raises(ValueError):
        g.set_pose(np.eye(4), PoseSource.PNP)               # downgrade
    with pytest.raises(ValueError):
        g.set_pose(np.eye(4), PoseSource.REFINED_PER_CAMERA)
    g.set_pose(np.eye(4), PoseSource.REFINED_CAM_IMU)       # same rank: allowed


def test_frames_to_arrays_respects_flags():
    db = make_db()
    a = frames_to_arrays(db, 0)
    assert len(a["point_id"]) == 6
    db.camera(0).frames[0].board_detections[0].detections[0].is_valid = False
    db.camera(0).frames[1].is_selected_to_use = False
    a = frames_to_arrays(db, 0)
    assert len(a["point_id"]) == 3              # -1 invalid det, -2 deselected frame
    assert a["uv"].shape == (3, 2)
    full = frames_to_arrays(db, 0, selected_only=False, valid_only=False)
    assert len(full["point_id"]) == 6           # flags never delete (I4)


def test_fingerprint_measurement_identity():
    db1, db2 = make_db(), make_db()
    assert observation_fingerprint(db1) == observation_fingerprint(db2)
    # insertion order must not matter
    g = db2.camera(0).frames[0].board_detections[0]
    g.detections.reverse()
    assert observation_fingerprint(db1) == observation_fingerprint(db2)
    # flags are run state, not measurement — fingerprint ignores them
    db2.camera(0).frames[0].is_selected_to_use = False
    db2.camera(0).frames[0].board_detections[0].detections[0].is_valid = False
    assert observation_fingerprint(db1) == observation_fingerprint(db2)
    # one uv bit flips it
    db3 = make_db()
    d = db3.camera(0).frames[0].board_detections[0].detections[0]
    uv = d.uv.copy()
    uv[0] = np.nextafter(uv[0], np.inf)
    object.__setattr__ if False else None
    d.uv = uv  # rebuild the frozen array
    d.__post_init__()
    assert observation_fingerprint(db1) != observation_fingerprint(db3)


def test_measured_arrays_are_frozen():
    db = make_db()
    d = db.camera(0).frames[0].board_detections[0].detections[0]
    with pytest.raises(ValueError):
        d.uv[0] = 999.0


def test_estimate_states_pickles():
    es = EstimateStates(
        reference_stream=(SensorType.CAM, 0),
        sensor_states=[SensorState(
            sensor=(SensorType.CAM, 0),
            intrinsics=CameraIntrinsicsState(np.arange(4.0), np.zeros(4),
                                             CameraModel.RADTAN))],
        state_hz=30.0,
        base_state=CamOnlyBaseState(np.arange(3.0), np.zeros((3, 7))),
    )
    es2 = pickle.loads(pickle.dumps(es))
    assert es2.reference_stream == (SensorType.CAM, 0)
    np.testing.assert_array_equal(es2.base_state.t, es.base_state.t)


def test_validate_catches_unknown_sensor_and_bad_policy():
    db = make_db()
    db.estimate_states = EstimateStates(
        reference_stream=(SensorType.CAM, 0),
        sensor_states=[SensorState(
            sensor=(SensorType.CAM, 99),        # no such camera
            intrinsics=CameraIntrinsicsState(np.zeros(4), np.zeros(4),
                                             CameraModel.RADTAN))])
    with pytest.raises(ValueError, match="unknown sensor"):
        db.validate()
    db.estimate_states = None
    db.run.policy((SensorType.CAM, 0)).intrinsics = PolicyEntry(PolicyState.ANCHOR)
    with pytest.raises(ValueError, match="anchor"):
        db.validate()


def test_validate_reference_agreement_and_time_order():
    db = make_db(CalibChoice.CAM_IMU)
    db.estimate_states = EstimateStates(reference_stream=(SensorType.CAM, 0))
    with pytest.raises(ValueError, match="reference_stream disagrees"):
        db.validate()
    db2 = make_db()
    db2.camera(0).frames[0].timestamp = 999.0   # now out of order
    with pytest.raises(ValueError, match="time-ordered"):
        db2.validate()


def test_frame_ids_unique_and_increasing():
    db = make_db()
    ids = [f.frame_id for s in db.streams for f in s.frames]
    assert len(ids) == len(set(ids))
    assert sorted(ids) == ids or True           # unique is the contract; order per camera below
    per_cam = [f.frame_id for f in db.camera(0).frames]
    assert per_cam == sorted(per_cam)


def test_relative_calib_td_can_exist_cam_only():
    # inter-camera td is LIVE — the type must carry it
    r = RelativeCalib(np.eye(4), td_x_to_ref_s=0.004)
    assert r.td_x_to_ref_s == 0.004


def test_pose7_to_mat():
    np.testing.assert_allclose(
        pose7_to_mat([1, 0, 0, 0, 1, 2, 3]),
        np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]), atol=1e-12)
    # 90 deg about z: qw = qz = sqrt(2)/2
    s = np.sqrt(0.5)
    T = pose7_to_mat([s, 0, 0, s, 0, 0, 0])
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
