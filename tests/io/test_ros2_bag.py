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
"""ROS2 bags read through rosbags (no ROS install): the camera-only source, the cam-IMU loader
and `mvis inspect`, on real sqlite3 bags written by the rosbags Writer."""
import logging
import sqlite3

import cv2
import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mvis.core.config import BoardConfig, CalibrationConfig, CameraConfig
from mvis.detection.board_factory import create_board
from mvis.doctor.check_config import config_needs
from mvis.doctor.inspect import inspect
from mvis.io import cam_imu_loader
from mvis.io.data_loader import DataLoader
from mvis.io.frame_detector import detect_frame
from mvis.io.ros2_bag_source import Ros2BagSource
from mvis.io.rosbag_reader import open_bag

CDR = get_typestore(Stores.ROS2_HUMBLE)
Header = CDR.types["std_msgs/msg/Header"]
Time = CDR.types["builtin_interfaces/msg/Time"]
Image = CDR.types["sensor_msgs/msg/Image"]
Compressed = CDR.types["sensor_msgs/msg/CompressedImage"]
Imu = CDR.types["sensor_msgs/msg/Imu"]
Vec = CDR.types["geometry_msgs/msg/Vector3"]
Quat = CDR.types["geometry_msgs/msg/Quaternion"]

CAM, IMU = "/cam0/image_raw", "/imu/data_raw"
T0 = 10_000 * 10**9             # header clock
RECORD_OFFSET = 7 * 10**9       # the record time is NOT the header time
DT = 100_000_000
H, W = 48, 64
BOARD = BoardConfig(board_type="chessboard", rows=5, cols=7, square_size=0.03)


def _header(i, stamped=True):
    t = T0 + i * DT if stamped else 0
    return Header(stamp=Time(sec=t // 10**9, nanosec=t % 10**9), frame_id="cam")


def _frames(n=4):
    return [np.random.default_rng(i).integers(0, 256, (H, W)).astype(np.uint8) for i in range(n)]


def raw_msg(i, img, stamped=True):
    return Image(_header(i, stamped), H, W, "mono8", 0, W, np.frombuffer(img.tobytes(), np.uint8))


def compressed_msg(i, img, fmt="mono8; png compressed", ext=".png"):
    return Compressed(_header(i), fmt, cv2.imencode(ext, img)[1].reshape(-1))


def write_bag(path, topics, n_imu=10):
    """topics: {topic: [message, ...]}; plus an IMU stream with w = (i, 0, 0), a = (0, 0, 9.81)."""
    with Writer(path, version=9) as w:
        for topic, msgs in topics.items():
            conn = w.add_connection(topic, msgs[0].__msgtype__, typestore=CDR)
            for i, m in enumerate(msgs):
                w.write(conn, T0 + RECORD_OFFSET + i * DT, CDR.serialize_cdr(m, conn.msgtype))
        conn = w.add_connection(IMU, Imu.__msgtype__, typestore=CDR)
        cov = np.zeros(9)
        for i in range(n_imu):
            m = Imu(_header(i), Quat(x=0.0, y=0.0, z=0.0, w=1.0), cov, Vec(x=float(i), y=0.0, z=0.0),
                    cov, Vec(x=0.0, y=0.0, z=9.81), cov)
            w.write(conn, T0 + RECORD_OFFSET + i * DT, CDR.serialize_cdr(m, conn.msgtype))
    return path


def read_source(path, _=None):
    return [(np.array(img), t) for img, t, _cid in Ros2BagSource(str(path), [CAM])]


def read_stream(path, _=None):
    cam = CameraConfig(camera_id=0, source_type="ros2bag", source_path=str(path), topic=CAM)
    loader = DataLoader(CalibrationConfig(cameras=[cam], board=BOARD))
    return [(np.array(img), t) for _cid, img, t in loader.stream()]


def load_cam_imu(path, monkeypatch):
    seen = []

    def spy(img, board, cid=0, t=0.0):
        seen.append((np.array(img), t))
        return detect_frame(np.zeros((8, 8), np.uint8), board, cid, t)

    monkeypatch.setattr(cam_imu_loader, "detect_frame", spy)
    _cams, imus = cam_imu_loader.load_ros1(str(path), [CAM], [IMU], create_board(BOARD), min_obs=0)
    return seen, imus


def read_cam_imu(path, monkeypatch):
    return load_cam_imu(path, monkeypatch)[0]


@pytest.mark.parametrize("read", [read_source, read_stream, read_cam_imu])
def test_frames_and_header_stamps(read, tmp_path, monkeypatch):
    frames = _frames()
    raw = write_bag(tmp_path / "raw", {CAM: [raw_msg(i, f) for i, f in enumerate(frames)]})
    png = write_bag(tmp_path / "png", {CAM: [compressed_msg(i, f) for i, f in enumerate(frames)]})
    for bag in (raw, png):
        got = read(bag, monkeypatch)
        assert len(got) == len(frames)
        for i, (img, t) in enumerate(got):
            np.testing.assert_array_equal(img, frames[i])
            assert img.dtype == np.uint8 and t == pytest.approx((T0 + i * DT) * 1e-9, abs=1e-9)


def test_db3_file_path_reads_like_the_folder(tmp_path):
    bag = write_bag(tmp_path / "bag", {CAM: [raw_msg(i, f) for i, f in enumerate(_frames())]})
    (db3,) = bag.glob("*.db3")
    for (a, ta), (b, tb) in zip(read_source(bag), read_source(db3), strict=True):
        np.testing.assert_array_equal(a, b)
        assert ta == tb


def test_unset_stamp_falls_back_to_record_time(tmp_path):
    bag = write_bag(tmp_path / "bag", {CAM: [raw_msg(0, _frames(1)[0], stamped=False)]})
    assert read_source(bag)[0][1] == pytest.approx((T0 + RECORD_OFFSET) * 1e-9)


def test_jpeg_warns_once(tmp_path, caplog):
    msgs = [compressed_msg(i, f, "jpeg", ".jpg") for i, f in enumerate(_frames(2))]
    bag = write_bag(tmp_path / "jpg", {CAM: msgs})
    with caplog.at_level(logging.WARNING, logger="mvis.io.compressed_image"):
        got = read_source(bag)
    assert got[0][0].shape == (H, W)
    assert len(caplog.records) == 1 and CAM in caplog.records[0].getMessage()


def test_cam_imu_loader_reads_the_imu_on_the_header_clock(tmp_path, monkeypatch):
    bag = write_bag(tmp_path / "bag", {CAM: [raw_msg(i, f) for i, f in enumerate(_frames())]})
    _seen, imus = load_cam_imu(bag, monkeypatch)
    (imu,) = imus
    np.testing.assert_allclose(imu["t"], (T0 + np.arange(10) * DT) * 1e-9)
    np.testing.assert_array_equal(imu["w"][:, 0], np.arange(10.0))
    np.testing.assert_allclose(imu["a"], np.tile([0.0, 0.0, 9.81], (10, 1)))
    (only,) = cam_imu_loader.read_imu_only(str(bag), [IMU], fmt="ros2")
    np.testing.assert_array_equal(only["t"], imu["t"])


def test_big_endian_cdr_is_refused(tmp_path):
    _reader, deserialize = open_bag(write_bag(tmp_path / "bag", {CAM: [raw_msg(0, _frames(1)[0])]}))
    with pytest.raises(ValueError, match="little-endian"):
        deserialize(b"\x00\x00" + bytes(30), Image.__msgtype__)


def test_inspect_lists_streams_and_allows_cam_imu(tmp_path):
    bag = write_bag(tmp_path / "bag", {CAM: [raw_msg(i, f) for i, f in enumerate(_frames())]})
    rep = inspect(str(bag))
    assert rep.format == "ros2" and rep.supported_modes == ["cam", "cam-imu"]
    kinds = {s.name: s.kind for s in rep.streams}
    assert kinds == {CAM: "camera", IMU: "imu"}
    info = next(s.info for s in rep.streams if s.name == CAM)
    assert (info["width"], info["height"], info["encoding"]) == (W, H, "mono8")
    assert not [f for f in rep.findings if f.id == "format.cam_imu"]


def test_config_needs_rosbags_for_ros2(tmp_path):
    cam_imu = tmp_path / "a.yaml"
    cam_imu.write_text("input: {format: ros2, path: x}\n")
    cam = tmp_path / "b.yaml"
    cam.write_text("input: {cameras: [{source_type: ros2bag, source_path: x, topic: /c}]}\n")
    assert "ros1" in config_needs(str(cam_imu)) and "ros1" in config_needs(str(cam))


def test_bag_with_empty_type_hashes_is_read(tmp_path):
    """Bags written by converters (Hilti-Trimble 2026) leave type_description_hash empty;
    rosbags < 0.11.6 refused them."""
    bag = write_bag(tmp_path / "bag", {CAM: [raw_msg(i, f) for i, f in enumerate(_frames())]})
    (db3,) = bag.glob("*.db3")
    db = sqlite3.connect(db3)
    db.execute("update message_definitions set type_description_hash = ''")
    db.execute("update topics set type_description_hash = ''")
    db.commit()
    db.close()
    assert len(read_source(bag)) == 4
