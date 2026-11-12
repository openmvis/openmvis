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
"""basalt-regression B2: Basalt calibration.json -> MVIS conventions (mvis.io.basalt)."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mvis.io import basalt as basalt_io

FIXTURE = "tests/fixtures/basalt/tum_vi_raw_imu1_calibration.json"


def _pose(rotvec, p_m):
    q = Rotation.from_rotvec(rotvec).as_quat()
    return dict(px=p_m[0], py=p_m[1], pz=p_m[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3])


def _write_run(tmp_path, calib, status="ok"):
    (tmp_path / "calibration.json").write_text(json.dumps({"value0": calib}))
    summary = {"status": status, "phases": [{"name": "full", "reason": "converged", "iterations": 5,
                                             "mean_reprojection": 0.12}]}
    (tmp_path / "basalt_status.json").write_text(json.dumps({"status": status, "summary": summary}))
    return str(tmp_path)


def _kb4(fx):
    return {"camera_type": "kb4", "intrinsics": {"fx": fx, "fy": fx, "cx": 320.0, "cy": 240.0,
                                                 "k1": 0.01, "k2": -0.002, "k3": 0.0003, "k4": -0.0001}}


def test_rotation_times_lower_recovers_the_kalibr_split():
    R0 = Rotation.from_rotvec([0.004, -0.007, 0.002]).as_matrix()
    L0 = np.array([[0.946, 0, 0], [0.003, 1.094, 0], [-0.002, 0.004, 1.019]])
    R, L = basalt_io.rotation_times_lower(R0 @ L0)
    assert np.allclose(R, R0) and np.allclose(L, L0)
    assert np.allclose(np.triu(L, 1), 0)


def test_camera_imu_maps_extrinsics_time_and_imu_intrinsics(tmp_path):
    R0 = Rotation.from_rotvec([0.004, -0.007, 0.002]).as_matrix()
    L0 = np.array([[0.946, 0, 0], [0.003, 1.094, 0], [-0.002, 0.004, 1.019]])
    S_g = R0 @ L0 - np.eye(3)
    Da = np.array([[1.007, 0, 0], [0.001, 1.0069, 0], [-0.003, 0.002, 0.972]])
    calib = {
        "T_imu_cam": [_pose([0.1, -2.17, 2.25], [0.0465, -0.0690, -0.0453]),
                      _pose([0.1, -2.22, 2.20], [-0.0546, -0.0690, -0.0491])],
        "intrinsics": [_kb4(382.1), _kb4(380.8)],
        "cam_time_offset_ns": -748000,
        "calib_gyro_bias": [1e-3, 2e-3, 3e-3, *S_g.flatten(order="F")],
        "calib_accel_bias": [0.01, 0.02, 0.03, Da[0, 0] - 1, Da[1, 0], Da[2, 0],
                             Da[1, 1] - 1, Da[2, 1], Da[2, 2] - 1],
        "imu_update_rate": 199.36,
    }
    run = basalt_io.camera_imu(_write_run(tmp_path, calib))
    c0, c1 = run["cams"]
    assert c0["td_ms"] == pytest.approx(-0.748) and c1["td_ms"] == pytest.approx(-0.748)
    assert np.allclose(c0["p_mm"], [46.5, -69.0, -45.3])
    assert np.allclose(c0["rotvec"], [0.1, -2.17, 2.25])
    assert c1["baseline_mm"] == pytest.approx(np.linalg.norm([101.1, 0.0, 3.8]))
    assert c0["intr"] == [382.1, 382.1, 320.0, 240.0] and len(c0["dist"]) == 4
    imu = run["imu"]
    assert np.allclose(imu["Dw_kalibr"], L0) and np.allclose(imu["Da"], Da)
    assert np.allclose(imu["Dw_full"], R0 @ L0)
    assert np.allclose(imu["R_gyro_rotvec"], [0.004, -0.007, 0.002])
    keys = basalt_io.mvis_keys(run)
    assert keys["base.Dw_diag[1]"] == pytest.approx(1.094) and keys["base.Da_diag[2]"] == pytest.approx(0.972)
    assert run["mean_reprojection_px"] == 0.12


def test_runs_that_are_not_ok_are_skipped(tmp_path):
    assert basalt_io.camera_imu(_write_run(tmp_path, {}, status="stalled")) is None


def test_camera_only_baseline_is_relative_to_cam0(tmp_path):
    calib = {"T_imu_cam": [_pose([0, 0, 0], [0, 0, 0]), _pose([0.01, 0.0, 0.0], [0.1011, 0.0, 0.002])],
             "intrinsics": [_kb4(382.1), _kb4(380.8)]}
    run = basalt_io.camera_only(_write_run(tmp_path, calib))
    assert run["cams"][1]["baseline_mm"] == pytest.approx(np.hypot(101.1, 2.0))
    assert np.allclose(run["cams"][0]["p_mm"], 0)


def test_tum_vi_imu1_fixture_matches_the_kalibr_cross_check():
    """Own Basalt run (mvis-dev, --init-knots poses --no-mocap): the TUM-VI raw gyro scale and
    cam-IMU time offset that Kalibr and MVIS also find."""
    calib = basalt_io.read_calibration(FIXTURE)
    _, G = basalt_io.gyro_matrix(calib)
    assert np.allclose(np.diag(G), [0.946, 1.094, 1.018], atol=0.002)
    _, L = basalt_io.rotation_times_lower(G)
    assert np.allclose(np.diag(L), [0.9459, 1.0942, 1.0173], atol=0.002)   # MVIS kalibr-variant Dw_diag
    assert calib["cam_time_offset_ns"] * 1e-6 == pytest.approx(-0.75, abs=0.05)
