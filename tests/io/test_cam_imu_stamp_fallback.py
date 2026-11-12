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
"""The cam-IMU ROS1 loader reads each message's time from the raw header bytes
(`io.ros_stamp.header_stamp_s`, cleanup plan PR F, F5): a record whose header stamp was never
set (0) takes the bag record time instead of t = 0, and set stamps are bitwise the old
deserialised `header.stamp` value."""
import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from mvis.core.types import FrameObservations
from mvis.io import cam_imu_loader as L
from mvis.io.ros_stamp import header_stamp_s


def _old_hdr_s(msg) -> float:
    """The pre-F5 stamp (cam_imu_loader._hdr_s): header.stamp of the deserialised message,
    no zero-stamp fallback. Kept here as the oracle for the raw-bytes reader."""
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


TS = get_typestore(Stores.ROS1_NOETIC)
T = TS.types
REC0 = 1_700_000_000 * 10**9 + 123_456_789


def _time(ns):
    return T["builtin_interfaces/msg/Time"](sec=ns // 10**9, nanosec=ns % 10**9)


def _imu(ns, k):
    V, Q = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
    return T["sensor_msgs/msg/Imu"](
        header=T["std_msgs/msg/Header"](seq=k, stamp=_time(ns), frame_id="imu"),
        orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.zeros(9),
        angular_velocity=V(x=0.1 * k, y=0.0, z=0.0), angular_velocity_covariance=np.zeros(9),
        linear_acceleration=V(x=0.0, y=0.0, z=9.81), linear_acceleration_covariance=np.zeros(9))


def _image(ns, k):
    return T["sensor_msgs/msg/Image"](
        header=T["std_msgs/msg/Header"](seq=k, stamp=_time(ns), frame_id="cam"), height=8,
        width=8, encoding="mono8", is_bigendian=0, step=8, data=np.zeros(64, np.uint8))


def _bag(path, zero_imu=(), zero_img=(), n=20):
    """IMU at 200 Hz and images at 20 Hz; header = record time + 3 ms, except the listed
    indices, whose header stamp is 0."""
    with Writer(path) as w:
        ci = w.add_connection("/imu0", "sensor_msgs/msg/Imu", typestore=TS)
        cc = w.add_connection("/cam0", "sensor_msgs/msg/Image", typestore=TS)
        for k in range(n * 10):
            rec = REC0 + k * 5_000_000
            w.write(ci, rec, TS.serialize_ros1(_imu(0 if k in zero_imu else rec + 3_000_000, k),
                                               ci.msgtype))
            if k % 10 == 0:
                j = k // 10
                w.write(cc, rec, TS.serialize_ros1(
                    _image(0 if j in zero_img else rec + 3_000_000, j), cc.msgtype))


def test_zero_stamp_imu_record_takes_the_record_time(tmp_path):
    p = str(tmp_path / "z.bag")
    _bag(p, zero_imu={0, 7})
    imu, = L._ros1_imu(p, ["/imu0"])
    t = imu["t"]
    assert t.min() > 1e9                         # no sample at t = 0
    assert (REC0 + 0) / 1e9 in t and (REC0 + 7 * 5_000_000) / 1e9 in t
    assert len(t) == 200


def test_zero_stamp_image_record_takes_the_record_time(tmp_path, monkeypatch):
    p = str(tmp_path / "zi.bag")
    _bag(p, zero_img={0})
    seen = []

    class Board:
        def detect(self, img):
            return []
    monkeypatch.setattr(L, "detect_frame",
                        lambda img, board, cid, t: (seen.append(t), FrameObservations(
                            camera_id=cid, timestamp=t, observations=[], image_size=(8, 8)))[1])
    L.load_ros1(p, ["/cam0"], ["/imu0"], Board(), t0=0.0, window=1e9, min_obs=0)
    assert len(seen) == 20
    assert seen[0] == REC0 / 1e9                 # was 0.0, and the window origin with it
    assert min(seen) > 1e9


def test_set_stamps_are_bitwise_the_deserialised_header(tmp_path):
    from rosbags.rosbag1 import Reader
    p = str(tmp_path / "s.bag")
    _bag(p)
    with Reader(p) as r:
        for conn, rec, raw in r.messages():
            m = TS.deserialize_ros1(raw, conn.msgtype)
            assert np.float64(header_stamp_s(raw, rec)).tobytes() == \
                np.float64(_old_hdr_s(m)).tobytes()


def test_raw_stamp_reader_matches_the_old_import_path():
    from mvis.io.ros1_bag_source import header_stamp_s as old
    assert old is header_stamp_s
    raw = TS.serialize_ros1(_imu(0, 1), "sensor_msgs/msg/Imu")
    assert header_stamp_s(raw, 9 * 10**9) == pytest.approx(9.0)
