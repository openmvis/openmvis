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
"""Task 10 Part 2 — bag spot-check for the real `apply_calibration` orchestrator.

Proves the orchestrator writes the CORRECTED IMU values + SHIFTED camera stamps
into the output bag bytes (plumbing only: no calibration run, no images). We
build a `RigCalib` directly with KNOWN non-identity IMU intrinsics + biases and
a known camera td, run `apply_calibration` on the `tiny_bag` fixture, then
re-open the output bag and INDEPENDENTLY recompute the expected corrected IMU
values (`true_intr.correct(...)`) and shifted camera stamps (`t + round(td*1e9)`)
from the SAME raw values/stamps the fixture wrote.

Fixture raw values (see tests/recalib/conftest.py):
  /imu0           @200Hz, k=0..19: w=(0.01*k,0,0), a=(0,0,9.81),
                  t = 1e9 + k*5_000_000 ns
  /cam0/image_raw @30Hz,  k=0..3:  t = 1e9 + k*33_000_000 ns
The base IMU td=0, so /imu0 record times are unchanged; /cam0 shifts by TD.
"""
import numpy as np
import pytest
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from mvis.imu import ImuModel, RPNG
from mvis.recalib.calib_spec import RigCalib, ImuCalib, MVIS
from mvis.recalib.apply import apply_calibration

# raw values the tiny_bag fixture writes (kept in sync with conftest.py)
_IMU_HZ_DT_NS = 5_000_000
_CAM_HZ_DT_NS = 33_000_000
_T0_NS = 1_000_000_000
_N_IMU = 20
_N_CAM = 4
_TD = 0.002


def _raw_imu(k):
    return np.array([0.01 * k, 0.0, 0.0]), np.array([0.0, 0.0, 9.81])


def test_apply_writes_corrected_imu_and_shifted_cam_stamps(tiny_bag, tmp_path):
    out = str(tmp_path / "corrected.bag")

    # 1. RigCalib with KNOWN non-identity IMU intrinsics + biases + camera td
    import gtsam
    true_intr = ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.zeros(9),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(),
        model=RPNG)
    b_g = np.array([0.01, -0.02, 0.005])
    b_a = np.array([0.05, -0.03, 0.02])
    base = ImuCalib("imu0", "rpng", true_intr, b_g, b_a, 0.0, "base")
    cam = MVIS("cam0", _TD, 0.0, "radtan",
                   np.array([400.0, 400.0, 320.0, 240.0]),
                   np.array([0.0, 0.0, 0.0, 0.0]))  # pixels untouched -> any K
    rig = RigCalib(base, [], [cam])

    # provide a real provenance src path (apply also tolerates <unknown>)
    calib_file = tmp_path / "calib.yaml"
    calib_file.write_text("model: test\n")
    rig._src_path = str(calib_file)

    # 2. run the REAL orchestrator
    summary = apply_calibration(
        tiny_bag, out, rig,
        mapping={"/imu0": "imu0", "/cam0/image_raw": "cam0"},
        base_imu_topic="/imu0", remove_bias=True)
    assert summary["n_imu"] == _N_IMU and summary["n_cam"] == _N_CAM

    # 3. re-open and verify against an INDEPENDENT recompute
    ts = get_typestore(Stores.ROS1_NOETIC)
    imu_msgs, cam_recs = [], []
    with Reader(out) as r:
        conns = list(r.connections)
        for conn, t_ns, raw in r.messages():
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == "/imu0":
                imu_msgs.append((t_ns, msg))
            else:
                cam_recs.append((t_ns, msg))

    assert len(imu_msgs) == _N_IMU and len(cam_recs) == _N_CAM

    # IMU: each output equals true_intr.correct(raw_w, raw_a, b_g, b_a).
    # Records were written global-time-sorted; base td=0 so /imu0 stamps are
    # unchanged and strictly increasing -> match by record-time -> index k.
    imu_msgs.sort(key=lambda r: r[0])
    for k, (t_ns, msg) in enumerate(imu_msgs):
        assert t_ns == _T0_NS + k * _IMU_HZ_DT_NS, \
            f"imu[{k}] record time shifted (td=0 expected): {t_ns}"
        raw_w, raw_a = _raw_imu(k)
        w_exp, a_exp = true_intr.correct(raw_w, raw_a, b_g, b_a)
        got_w = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                          msg.angular_velocity.z])
        got_a = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                          msg.linear_acceleration.z])
        np.testing.assert_allclose(got_w, w_exp, atol=1e-9,
                                   err_msg=f"imu[{k}] angular_velocity")
        np.testing.assert_allclose(got_a, a_exp, atol=1e-9,
                                   err_msg=f"imu[{k}] linear_acceleration")

    # Cam: record time AND header.stamp shifted by round(TD*1e9) ns.
    shift_ns = int(round(_TD * 1e9))
    cam_recs.sort(key=lambda r: r[0])
    for k, (t_ns, msg) in enumerate(cam_recs):
        t_in = _T0_NS + k * _CAM_HZ_DT_NS
        assert t_ns == t_in + shift_ns, \
            f"cam[{k}] record time: {t_ns} != {t_in}+{shift_ns}"
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        assert stamp_ns == t_in + shift_ns, \
            f"cam[{k}] header.stamp: {stamp_ns} != {t_in}+{shift_ns}"
