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
"""Shared fixtures for recalib apply tests."""
import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


@pytest.fixture
def ts():
    return get_typestore(Stores.ROS1_NOETIC)


@pytest.fixture
def tiny_bag(tmp_path, ts):
    """A 1-IMU + 1-cam bag: known raw IMU samples and image stamps."""
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Image = ts.types["sensor_msgs/msg/Image"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    path = str(tmp_path / "raw.bag")
    with Writer(path) as w:
        c_imu = w.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        c_cam = w.add_connection("/cam0/image_raw", Image.__msgtype__, typestore=ts)
        for k in range(20):
            t = 1_000_000_000 + k * 5_000_000          # 200 Hz
            hdr = Header(seq=k, stamp=Time(sec=t // 10**9, nanosec=t % 10**9),
                         frame_id="imu")
            msg = Imu(header=hdr,
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                      orientation_covariance=np.zeros(9),
                      angular_velocity=Vector3(x=0.01 * k, y=0.0, z=0.0),
                      angular_velocity_covariance=np.zeros(9),
                      linear_acceleration=Vector3(x=0.0, y=0.0, z=9.81),
                      linear_acceleration_covariance=np.zeros(9))
            w.write(c_imu, t, ts.serialize_ros1(msg, Imu.__msgtype__))
        for k in range(4):
            t = 1_000_000_000 + k * 33_000_000          # 30 Hz
            hdr = Header(seq=k, stamp=Time(sec=t // 10**9, nanosec=t % 10**9),
                         frame_id="cam")
            img = Image(header=hdr, height=2, width=2, encoding="mono8",
                        is_bigendian=0, step=2, data=np.zeros(4, np.uint8))
            w.write(c_cam, t, ts.serialize_ros1(img, Image.__msgtype__))
    return path
