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
"""Camera-only readers take an image's time from header.stamp, like the cam-IMU loader.

A bag's record time is when the message reached the recorder. A stamp repair (or a driver's
own stamp) only exists in the header, so reading record time silently undoes it: VI-Rig bag 9's
repaired fisheye2 looked one frame off to the stereo solve and gave an 81 mm baseline.
"""
import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from mvis.core.config import BoardConfig, CalibrationConfig, CameraConfig
from mvis.io.data_loader import DataLoader
from mvis.io.ros1_bag_source import Ros1BagSource, header_stamp_s

TOPIC = "/cam/image_raw"
TS = get_typestore(Stores.ROS1_NOETIC)
RECORD_LAG_NS = 34_000_000
N = 12


def _image(stamp_ns):
    Image = TS.types["sensor_msgs/msg/Image"]
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    stamp = Time(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9)
    return Image(header=Header(seq=1, stamp=stamp, frame_id="cam"), height=4, width=4,
                 encoding="mono8", is_bigendian=0, step=4,
                 data=np.zeros(16, dtype=np.uint8))


def _write_bag(path, header_ns):
    with Writer(path) as w:
        conn = w.add_connection(TOPIC, "sensor_msgs/msg/Image", typestore=TS)
        for i, h in enumerate(header_ns):
            rec = 100 * 10**9 + i * 33_000_000 + RECORD_LAG_NS
            w.write(conn, rec, TS.serialize_ros1(_image(h), conn.msgtype))


@pytest.fixture
def bag(tmp_path):
    p = tmp_path / "x.bag"
    header = [100 * 10**9 + i * 33_000_000 for i in range(N)]
    _write_bag(p, header)
    return p, np.array(header) * 1e-9


def _loader(path):
    cam = CameraConfig(camera_id=0, source_type="rosbag_topic", source_path=str(path), topic=TOPIC)
    board = BoardConfig(board_type="chessboard", rows=5, cols=7, square_size=0.03)
    return DataLoader(CalibrationConfig(cameras=[cam], board=board))


def test_header_stamp_parses_the_raw_message():
    raw = TS.serialize_ros1(_image(5 * 10**9 + 7), "sensor_msgs/msg/Image")
    assert header_stamp_s(raw, 9 * 10**9) == pytest.approx(5 + 7e-9)


def test_header_stamp_falls_back_to_record_time_when_unset():
    raw = TS.serialize_ros1(_image(0), "sensor_msgs/msg/Image")
    assert header_stamp_s(raw, 9 * 10**9) == 9.0


def test_ros1_source_yields_header_stamps(bag):
    path, header = bag
    got = [t for _, t, _ in Ros1BagSource(str(path), [TOPIC])]
    np.testing.assert_allclose(got, header, atol=1e-9)


def test_data_loader_stream_yields_header_stamps(bag):
    path, header = bag
    got = [item[-1] for item in _loader(path).stream()]
    np.testing.assert_allclose(got, header, atol=1e-9)


def test_detection_cache_key_changes_with_the_stamp_source(bag):
    # Caches written from record times must not be reused.
    path, _ = bag
    assert _loader(path).cache_key != str(path)
