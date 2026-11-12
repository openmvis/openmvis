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
"""apply_calibration must preserve a source connection's latching/callerid.

Latched topics (e.g. /tf_static, latched camera_info) would lose their latch in
the corrected bag if the writer connections were created without forwarding the
ext fields. We build a tiny bag with a latched, passthrough topic and assert the
output connection is still latched (and the callerid is carried through).
"""
import numpy as np
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore
from mvis.imu import ImuModel
from mvis.recalib.calib_spec import RigCalib, ImuCalib, MVIS
from mvis.recalib.apply import apply_calibration


def _identity_rig():
    intr = ImuModel.from_tangent(np.zeros(24), "rpng")
    base = ImuCalib("imu0", "rpng", intr, np.zeros(3), np.zeros(3), 0.0, "base")
    return RigCalib(base, [], [])


def _bag_with_latched_static(path, ts):
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    String = ts.types["std_msgs/msg/String"]
    with Writer(path) as w:
        c_imu = w.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        # latched, with a callerid — this is a passthrough topic (not in mapping)
        c_tf = w.add_connection(
            "/tf_static", String.__msgtype__, typestore=ts,
            latching=1, callerid="/robot_state_publisher")
        for k in range(5):
            t = 1_000_000_000 + k * 5_000_000
            hdr = Header(seq=k, stamp=Time(sec=t // 10**9, nanosec=t % 10**9),
                         frame_id="imu")
            msg = Imu(header=hdr,
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                      orientation_covariance=np.zeros(9),
                      angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                      angular_velocity_covariance=np.zeros(9),
                      linear_acceleration=Vector3(x=0.0, y=0.0, z=9.81),
                      linear_acceleration_covariance=np.zeros(9))
            w.write(c_imu, t, ts.serialize_ros1(msg, Imu.__msgtype__))
        tf = String(data="static")
        w.write(c_tf, 1_000_000_000,
                ts.serialize_ros1(tf, String.__msgtype__))


def test_latching_and_callerid_preserved(tmp_path):
    ts = get_typestore(Stores.ROS1_NOETIC)
    raw = str(tmp_path / "raw.bag")
    _bag_with_latched_static(raw, ts)

    out = str(tmp_path / "corrected.bag")
    calib_file = tmp_path / "calib.yaml"
    calib_file.write_text("model: identity\n")
    rig = _identity_rig()
    rig._src_path = str(calib_file)

    apply_calibration(raw, out, rig,
                      mapping={"/imu0": "imu0"},
                      base_imu_topic="/imu0", remove_bias=True)

    with Reader(out) as r:
        by_topic = {c.topic: c for c in r.connections}
    assert "/tf_static" in by_topic
    tf_conn = by_topic["/tf_static"]
    assert tf_conn.ext.latching == 1, "latched topic must stay latched"
    assert tf_conn.ext.callerid == "/robot_state_publisher"
