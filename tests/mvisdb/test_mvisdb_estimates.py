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
"""I3 step 2a: the CalibrationResult -> EstimateStates writer, cam-only base
state persistence (VarType.T_CAM_TO_BOARD, schema v2), and the m0002
migration."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from mvis.core.types import (
    CalibrationResult, CameraParams, ExtrinsicTransform,
)
from mvis.mvisdb.estimates import (
    estimates_from_calibration_result, mat_to_pose7, pose7_to_mat as _pose7_to_mat,
)
from mvis.mvisdb.manager import DatabaseManager
from mvis.mvisdb.migrations import m0002_cam_only_base_state
from mvis.mvisdb.mvisdatabase import (
    CalibChoice, CalibProvenance, CamOnlyBaseState, SensorType,
)
from tests.helpers.mvisdb_fixtures import make_db
from tests.helpers.mvisdb_fixtures import (_rot, _imu_result)


def _result():
    T10 = np.eye(4)
    T10[:3, :3] = _rot('x', 0.1)
    T10[:3, 3] = [0.12, 0.0, 0.01]
    Tb0 = np.eye(4)
    Tb0[:3, :3] = _rot('z', 0.4)
    Tb0[:3, 3] = [0.3, -0.2, 1.5]
    Tb1 = np.eye(4)
    Tb1[:3, :3] = _rot('z', -0.7)
    Tb1[:3, 3] = [0.1, 0.4, 1.2]
    return CalibrationResult(
        cameras=[
            CameraParams(0, [500.0, 501, 320, 240], [-0.1, 0.01, 0, 0], "radtan"),
            CameraParams(1, [498.0, 499, 321, 239], [0.2, 0.1, 0.05, 0.01],
                         "equidistant", rolling_shutter_readout=0.03),
        ],
        extrinsics=[ExtrinsicTransform(1, 0, T10, np.zeros(3), T10[:3, 3])],
        reference_camera=0,
        time_offsets={1: 0.004},
        board_poses=[Tb0, None, Tb1],
        board_pose_times=[10.0, 10.5, 11.0],
    )


def test_mapper_shapes_and_conventions():
    es = estimates_from_calibration_result(_result())
    assert es.reference_stream == (SensorType.CAM, 0)
    by_id = {s.sensor[1]: s for s in es.sensor_states}
    assert by_id[0].relative is None                      # ref carries no relative
    assert by_id[0].intrinsics.readout_s is None          # 0.0 -> None (global)
    assert by_id[1].intrinsics.readout_s == 0.03
    assert by_id[1].relative.td_x_to_ref_s == 0.004
    assert by_id[1].intrinsics.provenance is CalibProvenance.ESTIMATED
    np.testing.assert_allclose(by_id[1].relative.T_x_to_ref,
                               _result().extrinsics[0].T_to_from)
    # base state: None pose skipped; stored pose is the INVERSE of T_board_cam
    bs = es.base_state
    assert isinstance(bs, CamOnlyBaseState)
    np.testing.assert_allclose(bs.t, [10.0, 11.0])
    T_back = _pose7_to_mat(bs.T_cam_to_board[0])
    np.testing.assert_allclose(T_back @ _result().board_poses[0], np.eye(4),
                               atol=1e-12)


def test_pose7_quaternion_is_unit_and_canonical():
    T = np.eye(4)
    T[:3, :3] = _rot('x', 2.9)          # near-pi rotation exercises the branch
    T[:3, 3] = [1, 2, 3]
    p = mat_to_pose7(T)
    assert abs(np.linalg.norm(p[:4]) - 1) < 1e-12 and p[0] >= 0
    np.testing.assert_allclose(_pose7_to_mat(p), T, atol=1e-12)


def test_cam_only_base_state_round_trip(tmp_path):
    db = make_db(CalibChoice.CAM_ONLY)
    db.estimate_states = estimates_from_calibration_result(_result())
    m = DatabaseManager()
    rid = m.save(db, tmp_path / "b.mvisdb")
    got = m.load(tmp_path / "b.mvisdb", run_id=rid).estimate_states
    assert isinstance(got.base_state, CamOnlyBaseState)
    np.testing.assert_array_equal(got.base_state.t, db.estimate_states.base_state.t)
    np.testing.assert_array_equal(got.base_state.T_cam_to_board,
                                  db.estimate_states.base_state.T_cam_to_board)
    st = m.state(tmp_path / "b.mvisdb", rid)      # direct door sees it too
    assert isinstance(st.base_state, CamOnlyBaseState)


def test_m0002_rebuild_preserves_rows_and_widens_check():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE state_var (
        run_id INTEGER NOT NULL, var_type TEXT NOT NULL CHECK (var_type IN
        ('T_imu_to_world','v','b_g','b_a','T_board_to_world','landmark_xyz')),
        sensor_type TEXT NOT NULL CHECK (sensor_type IN ('cam','imu','board','world')),
        sensor_id INTEGER NOT NULL, dtype TEXT NOT NULL, shape TEXT NOT NULL,
        data BLOB NOT NULL,
        PRIMARY KEY (run_id, var_type, sensor_type, sensor_id)) WITHOUT ROWID""")
    conn.execute("INSERT INTO state_var VALUES (1,'v','imu',0,'float64','[3]',x'00')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO state_var VALUES (1,'T_cam_to_board','cam',0,'float64','[7]',x'00')")
    m0002_cam_only_base_state.up(conn)
    assert conn.execute("SELECT COUNT(*) FROM state_var").fetchone()[0] == 1
    conn.execute(
        "INSERT INTO state_var VALUES (1,'T_cam_to_board','cam',0,'float64','[7]',x'00')")
    assert conn.execute("SELECT COUNT(*) FROM state_var").fetchone()[0] == 2


def test_inverse_mapper_round_trips(tmp_path):
    from mvis.mvisdb.estimates import calibration_result_from_estimates
    src = _result()
    db = make_db(CalibChoice.CAM_ONLY)
    db.estimate_states = estimates_from_calibration_result(src)
    m = DatabaseManager()
    rid = m.save(db, tmp_path / "b.mvisdb")
    back = calibration_result_from_estimates(m.state(tmp_path / "b.mvisdb", rid))
    assert back.reference_camera == src.reference_camera
    assert [c.camera_id for c in back.cameras] == [0, 1]
    for a, b in zip(src.cameras, back.cameras):
        np.testing.assert_array_equal(a.intrinsics, b.intrinsics)
        np.testing.assert_array_equal(a.distortion, b.distortion)
        assert a.distortion_model == b.distortion_model
        assert a.rolling_shutter_readout == b.rolling_shutter_readout
    np.testing.assert_allclose(back.extrinsics[0].T_to_from,
                               src.extrinsics[0].T_to_from, atol=1e-15)
    assert back.time_offsets == src.time_offsets
    # poses: None row dropped on the way in; survivors byte-close both ways
    assert back.board_pose_times == [10.0, 11.0]
    np.testing.assert_allclose(back.board_poses[0], src.board_poses[0], atol=1e-12)
    np.testing.assert_allclose(back.board_poses[1], src.board_poses[2], atol=1e-12)


def test_cam_imu_mapper_and_round_trip(tmp_path):
    from mvis.mvisdb.estimates import estimates_from_cam_imu_result
    from mvis.mvisdb.mvisdatabase import ImuIntrinsicsState
    r = _imu_result()
    es = estimates_from_cam_imu_result(r, state_hz=20.0)
    assert es.reference_stream == (SensorType.IMU, 0)
    assert es.world_frame is not None and es.world_frame.kind == "gravity_aligned"
    by = {s.sensor: s for s in es.sensor_states}
    imu0 = by[(SensorType.IMU, 0)]
    assert isinstance(imu0.intrinsics, ImuIntrinsicsState)
    np.testing.assert_array_equal(imu0.intrinsics.xi24, np.arange(24.0))
    cam0, cam1 = by[(SensorType.CAM, 0)], by[(SensorType.CAM, 1)]
    assert cam0.intrinsics.readout_s == 0.03
    assert cam1.intrinsics.readout_s is None            # tr 0.0 -> None
    assert cam0.relative.td_x_to_ref_s == 0.004         # cams carry td vs IMU
    np.testing.assert_allclose(cam1.relative.T_x_to_ref,
                               r.cameras[1]["calib"].T.matrix())
    imu1 = by[(SensorType.IMU, 1)]
    assert imu1.relative.td_x_to_ref_s == 0.001

    # persist + read back through both doors (the fixture db declares one
    # IMU, so persist a result without the extra IMU — FK-checked by schema)
    r2 = _imu_result(); r2.imus = []
    es2 = estimates_from_cam_imu_result(r2, state_hz=20.0)
    db = make_db(CalibChoice.CAM_IMU)
    db.estimate_states = es2
    m = DatabaseManager()
    rid = m.save(db, tmp_path / "b.mvisdb")
    got = m.state(tmp_path / "b.mvisdb", rid)
    assert got.reference_stream == (SensorType.IMU, 0)
    gb = {s.sensor: s for s in got.sensor_states}
    np.testing.assert_array_equal(gb[(SensorType.IMU, 0)].intrinsics.xi24,
                                  np.arange(24.0))
    np.testing.assert_allclose(gb[(SensorType.CAM, 1)].relative.T_x_to_ref,
                               r.cameras[1]["calib"].T.matrix())
    assert gb[(SensorType.CAM, 0)].relative.td_x_to_ref_s == 0.004


def test_cam_imu_base_state_from_real_values(tmp_path):
    """base_state extraction: real gtsam Values with the solver's key scheme
    (X(k) pose, v(k) 9-vector) -> exact CamImuBaseState, persisted+read."""
    import gtsam
    from mvis.mvisdb.estimates import estimates_from_cam_imu_result
    from mvis.mvisdb.mvisdatabase import CamImuBaseState
    X = gtsam.symbol_shorthand.X
    r = _imu_result()
    K = 3
    vals = gtsam.Values()
    for k in range(K):
        T = np.eye(4); T[:3, :3] = _rot('z', 0.1 * k); T[:3, 3] = [k, 0.5 * k, 1.0]
        vals.insert(X(k), gtsam.Pose3(T))
        vals.insert(gtsam.symbol('v', k),
                    np.concatenate([[0.1 * k, 0, 0], [1e-3, 2e-3, 3e-3],
                                    [0.01, 0.02, 0.03 + 1e-4 * k]]))
    r.values = vals; r.t_states = np.array([10.0, 10.1, 10.2])
    es = estimates_from_cam_imu_result(r, state_hz=10.0)
    bs = es.base_state
    assert isinstance(bs, CamImuBaseState) and len(bs.t) == K
    np.testing.assert_allclose(_pose7_to_mat(bs.T_imu_to_world[2])[:3, 3],
                               [2, 1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(bs.b_a[2], [0.01, 0.02, 0.03 + 2e-4], atol=1e-15)
    np.testing.assert_allclose(bs.v[1], [0.1, 0, 0], atol=1e-15)

    # persist through the manager (state_knot + 4 state_vars) and read back
    r.imus = []
    es2 = estimates_from_cam_imu_result(r, state_hz=10.0)
    db = make_db(CalibChoice.CAM_IMU)
    db.estimate_states = es2
    m = DatabaseManager()
    rid = m.save(db, tmp_path / "b.mvisdb")
    got = m.state(tmp_path / "b.mvisdb", rid).base_state
    assert isinstance(got, CamImuBaseState)
    np.testing.assert_array_equal(got.t, es2.base_state.t)
    np.testing.assert_array_equal(got.T_imu_to_world, es2.base_state.T_imu_to_world)
    np.testing.assert_array_equal(got.b_g, es2.base_state.b_g)


def test_reproj_stats_round_trip_v3(tmp_path):
    """I5 schema v3: per-camera reprojection stats + overall_rms survive
    save/state and come back on the inverse-mapped result."""
    from mvis.core.types import ReprojectionErrorStats
    from mvis.mvisdb.estimates import calibration_result_from_estimates
    r = _result()
    r.reprojection_errors = [
        ReprojectionErrorStats(0, rms=1.5, mean=0.6, median=0.4,
                               p90=0.9, p95=1.1, num_observations=1234),
        ReprojectionErrorStats(1, rms=1.7, mean=0.7, median=0.5,
                               p90=1.0, p95=1.2, num_observations=987),
    ]
    r.overall_rms = 1.6
    db = make_db(CalibChoice.CAM_ONLY)
    db.estimate_states = estimates_from_calibration_result(r)
    m = DatabaseManager()
    rid = m.save(db, tmp_path / "b.mvisdb")
    back = calibration_result_from_estimates(m.state(tmp_path / "b.mvisdb", rid))
    assert back.overall_rms == 1.6
    by = {e.camera_id: e for e in back.reprojection_errors}
    assert by[0].rms == 1.5 and by[0].num_observations == 1234
    assert by[1].p95 == 1.2 and by[1].median == 0.5
