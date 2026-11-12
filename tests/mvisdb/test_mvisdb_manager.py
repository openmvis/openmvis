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
"""Tier 1 (I1) manager tests: save = core-once + attach; load = fold; the
direct state() door; write discipline (seal, flock, building, GT-once)."""
from __future__ import annotations

import fcntl
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from mvis.mvisdb.manager import (
    DatabaseManager, NoRunError, RunNotSolvedError, SealedError,
)
from mvis.mvisdb.mvisdatabase import (
    CalibChoice, CalibProvenance, CameraIntrinsicsState, CameraModel,
    CamImuBaseState, EstimateStates, PoseSource, RelativeCalib, Residual,
    RunOrigin, SensorState, SensorType, WorldFrame, observation_fingerprint,
)
from tests.helpers.mvisdb_fixtures import make_db


def mgr() -> DatabaseManager:
    return DatabaseManager()


def rich_db():
    """Fixture db with flags, poses, residuals and an estimate to round-trip."""
    db = make_db(CalibChoice.CAM_IMU)
    f0 = db.camera(0).frames[0]
    f0.is_motion_still = True
    g = f0.board_detections[0]
    g.set_pose(np.diag([1.0, 1.0, 1.0, 1.0]), PoseSource.PNP)
    g.set_pose(np.eye(4) + 0.001, PoseSource.REFINED_PER_CAMERA)
    d = g.detections[1]
    d.is_valid = False
    d.exclude_reason = "outlier_resid_s5"
    d.residual = Residual(3.5, PoseSource.REFINED_PER_CAMERA)
    db.camera(0).frames[1].is_selected_to_use = False
    db.camera(0).frames[1].exclude_reason = "hz_thinned"
    db.run.description = "round-trip fixture"
    db.run.factor_hz[(SensorType.CAM, 0)] = 20.0
    db.estimate_states = EstimateStates(
        reference_stream=(SensorType.IMU, 0),
        state_hz=30.0,
        sensor_states=[
            SensorState((SensorType.CAM, 0),
                        CameraIntrinsicsState(np.array([500.0, 501, 320, 240]),
                                              np.array([-0.1, 0.01, 0.0, 0.0]),
                                              CameraModel.RADTAN, None,
                                              CalibProvenance.ESTIMATED),
                        factor_hz=20.0,
                        relative=RelativeCalib(np.eye(4), td_x_to_ref_s=0.004,
                                               provenance=CalibProvenance.ESTIMATED))],
        base_state=CamImuBaseState(np.arange(4.0), np.random.RandomState(0).rand(4, 7),
                                   np.zeros((4, 3)), np.ones((4, 3)) * 1e-3,
                                   np.ones((4, 3)) * 2e-3),
        boards={0: np.array([1.0, 0, 0, 0, 0.1, 0.2, 0.3])},
        world_frame=WorldFrame(yaw_gauge=0.5),
    )
    return db


def test_save_load_round_trip(tmp_path):
    db = rich_db()
    path = tmp_path / "bag.mvisdb"
    run_id = mgr().save(db, path)
    assert run_id == 1 and db.run.run_id == 1
    back = mgr().load(path, run_id)
    # measurements: identical by fingerprint
    assert observation_fingerprint(back) == observation_fingerprint(db)
    # flags folded back
    f0 = back.camera(0).frames[0]
    assert f0.is_motion_still is True
    g = f0.board_detections[0]
    assert g.pose_cam_to_board.source is PoseSource.REFINED_PER_CAMERA
    np.testing.assert_array_equal(g.pose_cam_to_board.T, np.eye(4) + 0.001)
    d = g.detections[1]
    assert d.is_valid is False and d.exclude_reason == "outlier_resid_s5"
    assert d.residual.residual_px == 3.5
    assert back.camera(0).frames[1].is_selected_to_use is False
    # run + estimate
    assert back.run.calib_choice is CalibChoice.CAM_IMU
    assert back.run.description == "round-trip fixture"
    es, es0 = back.estimate_states, db.estimate_states
    assert es.reference_stream == (SensorType.IMU, 0) and es.state_hz == 30.0
    ss, ss0 = es.sensor_states[0], es0.sensor_states[0]
    np.testing.assert_array_equal(ss.intrinsics.intrinsics, ss0.intrinsics.intrinsics)
    assert ss.factor_hz == 20.0 and ss.relative.td_x_to_ref_s == 0.004
    np.testing.assert_array_equal(es.base_state.T_imu_to_world,
                                  es0.base_state.T_imu_to_world)
    np.testing.assert_array_equal(es.boards[0], es0.boards[0])
    assert es.world_frame.yaw_gauge == 0.5
    back.validate()


def test_sparse_rows_only_non_default(tmp_path):
    db = rich_db()
    path = tmp_path / "bag.mvisdb"
    mgr().save(db, path)
    conn = sqlite3.connect(path)
    # frames touched: f0 (motion_still) + f1 (deselected) — NOT the untouched ones
    assert conn.execute("SELECT COUNT(*) FROM frame_status").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM detection_status").fetchone()[0] == 1
    conn.close()


def test_second_run_attaches_and_first_survives(tmp_path):
    path = tmp_path / "bag.mvisdb"
    db1 = rich_db()
    mgr().save(db1, path)
    db2 = rich_db()
    db2.run.description = "variant top_left"
    r2 = mgr().save(db2, path)
    assert r2 == 2
    listing = mgr().runs(path)
    assert [r["run_id"] for r in listing] == [1, 2]
    assert mgr().load(path, 1).run.description == "round-trip fixture"
    assert mgr().load(path, 2).run.description == "variant top_left"


def test_sealed_core_rejects_different_capture(tmp_path):
    path = tmp_path / "bag.mvisdb"
    mgr().save(rich_db(), path)
    other = rich_db()
    det = other.camera(0).frames[0].board_detections[0].detections[0]
    uv = det.uv.copy(); uv[0] += 1.0
    det.uv = uv; det.__post_init__()
    with pytest.raises(SealedError, match="DIFFERENT capture"):
        mgr().save(other, path)
    mgr().save(other, path, replace=True)       # the cache's way of overwriting
    assert observation_fingerprint(mgr().load(path)) == observation_fingerprint(other)


def test_ground_truth_run_write_once(tmp_path):
    path = tmp_path / "bag.mvisdb"
    gt = rich_db(); gt.run.origin = RunOrigin.GROUND_TRUTH
    mgr().save(gt, path)
    gt2 = rich_db(); gt2.run.origin = RunOrigin.GROUND_TRUTH
    with pytest.raises(SealedError, match="write-once"):
        mgr().save(gt2, path)
    solved = rich_db()                       # a solved run still attaches fine
    assert mgr().save(solved, path) == 2


def test_state_direct_door_and_errors(tmp_path):
    path = tmp_path / "bag.mvisdb"
    db = rich_db()
    mgr().save(db, path)
    es = mgr().state(path, 1)
    assert es.reference_stream == (SensorType.IMU, 0)
    with pytest.raises(NoRunError):
        mgr().state(path, 99)
    bare = make_db(CalibChoice.CAM_ONLY)     # no estimate at all
    mgr().save(bare, path.with_name("bare.mvisdb"))
    with pytest.raises(RunNotSolvedError):
        mgr().state(path.with_name("bare.mvisdb"), 1)


def test_schema_checks_reject_bad_enums(tmp_path):
    path = tmp_path / "bag.mvisdb"
    mgr().save(rich_db(), path)
    conn = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO run (calib_choice) VALUES ('bogus')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE camera SET camera_model='pinhole' WHERE camera_id=0")
    conn.close()


def test_flock_held_during_save(tmp_path):
    path = tmp_path / "bag.mvisdb"
    mgr().save(rich_db(), path)
    lock = open(path.with_suffix(path.suffix + ".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)         # simulate another builder holding it
    fd2 = os.open(path.with_suffix(path.suffix + ".lock"), os.O_RDWR)
    with pytest.raises(BlockingIOError):
        fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd2)
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def test_stale_building_file_is_replaced(tmp_path):
    path = tmp_path / "bag.mvisdb"
    building = path.with_suffix(path.suffix + ".building")
    building.write_bytes(b"crashed build junk")
    mgr().save(rich_db(), path)              # must not trip on the leftover
    assert path.exists() and not building.exists()
    assert mgr().runs(path)[0]["run_id"] == 1


def test_frame_arrays_matches_in_memory(tmp_path):
    from mvis.mvisdb.mvisdatabase import frames_to_arrays
    db = make_db()
    path = tmp_path / "bag.mvisdb"
    mgr().save(db, path)
    disk = mgr().frame_arrays(path, 0)
    mem = frames_to_arrays(db, 0, selected_only=False, valid_only=False)
    np.testing.assert_array_equal(disk["point_id"], mem["point_id"])
    assert disk["uv"].tobytes() == mem["uv"].tobytes()
    np.testing.assert_array_equal(disk["frame_index"], mem["frame_index"])
    np.testing.assert_array_equal(disk["row"], mem["row"])
