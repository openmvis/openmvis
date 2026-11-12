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
"""CompressedImage topics read like the raw topics they encode, through every bag reader.

A real bag crashed the cam-IMU loader on msg.height. Since G6 every reader decodes through
io.image_decode; every test still runs on all of them, so a reader that bypasses it shows.
"""
import logging
import sys
import types

import cv2
import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from mvis.core.config import BoardConfig, CalibrationConfig, CameraConfig
from mvis.detection.board_factory import create_board
from mvis.doctor.inspect import inspect
from mvis.io import cam_imu_loader
from mvis.io.compressed_image import as_raw_image
from mvis.io.data_loader import DataLoader
from mvis.io.frame_detector import detect_frame
from mvis.io.image_decode import ros_image_to_gray
from mvis.io.ros1_bag_source import Ros1BagSource
from mvis.io.ros2_bag_source import Ros2BagSource

TS = get_typestore(Stores.ROS1_NOETIC)
Header = TS.types["std_msgs/msg/Header"]
Time = TS.types["builtin_interfaces/msg/Time"]
Image = TS.types["sensor_msgs/msg/Image"]
Compressed = TS.types["sensor_msgs/msg/CompressedImage"]
Imu = TS.types["sensor_msgs/msg/Imu"]
Vec = TS.types["geometry_msgs/msg/Vector3"]
Quat = TS.types["geometry_msgs/msg/Quaternion"]

CAM, CAM2, IMU = "/cam0/image_raw", "/cam1/image_raw", "/imu0"
T0 = 100 * 10**9
DT = 100_000_000
H, W = 48, 64
JPEG_LOGGER = "mvis.io.compressed_image"
BOARD = BoardConfig(board_type="chessboard", rows=5, cols=7, square_size=0.03)


def _header(i):
    t = T0 + i * DT
    return Header(seq=i, stamp=Time(sec=t // 10**9, nanosec=t % 10**9), frame_id="cam")


def _frames(n=4, depth=np.uint8, channels=1):
    """Noise with a distinct seed per frame: any lossy step or reorder changes pixels."""
    hi = 256 if depth == np.uint8 else 4096     # 12-bit in 16: the min-max scaled mono16 case
    shape = (H, W) if channels == 1 else (H, W, channels)
    return [np.random.default_rng(i).integers(0, hi, shape).astype(depth) for i in range(n)]


def raw_msg(i, img, encoding):
    channels = 1 if img.ndim == 2 else img.shape[2]
    return Image(header=_header(i), height=H, width=W, encoding=encoding, is_bigendian=0,
                 step=W * channels * img.dtype.itemsize,
                 data=np.frombuffer(img.tobytes(), np.uint8))


def compressed_msg(i, img, fmt="mono8; png compressed", ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return Compressed(header=_header(i), format=fmt, data=buf.reshape(-1))


def write_bag(path, topics):
    """topics: {topic: [message, ...]}; plus a short IMU stream, which load_ros1 reads anyway."""
    with Writer(path) as w:
        for topic, msgs in topics.items():
            conn = w.add_connection(topic, msgs[0].__msgtype__, typestore=TS)
            for i, m in enumerate(msgs):
                w.write(conn, T0 + i * DT, TS.serialize_ros1(m, conn.msgtype))
        conn = w.add_connection(IMU, Imu.__msgtype__, typestore=TS)
        zero = Vec(x=0.0, y=0.0, z=0.0)
        for i in range(10):
            m = Imu(header=_header(i), orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.zeros(9), angular_velocity=zero,
                    angular_velocity_covariance=np.zeros(9), linear_acceleration=zero,
                    linear_acceleration_covariance=np.zeros(9))
            w.write(conn, T0 + i * DT, TS.serialize_ros1(m, conn.msgtype))
    return path


def read_cam_imu(path, monkeypatch):
    seen = []

    def spy(img, board, cid=0, t=0.0):
        seen.append((np.array(img), t))
        return detect_frame(np.zeros((8, 8), np.uint8), board, cid, t)

    monkeypatch.setattr(cam_imu_loader, "detect_frame", spy)
    cam_imu_loader.load_ros1(str(path), [CAM], [IMU], create_board(BOARD), min_obs=0)
    return seen


def read_source(path, _):
    return [(np.array(img), t) for img, t, _cid in Ros1BagSource(str(path), [CAM])]


def _loader(path):
    cam = CameraConfig(camera_id=0, source_type="rosbag_topic", source_path=str(path), topic=CAM)
    return DataLoader(CalibrationConfig(cameras=[cam], board=BOARD))


def read_stream(path, _):
    return [(np.array(img), t) for _cid, img, t in _loader(path).stream()]


READERS = [read_cam_imu, read_source, read_stream]


def assert_same_frames(a, b):
    assert len(a) == len(b) > 0
    for (img_a, t_a), (img_b, t_b) in zip(a, b):
        assert img_a.dtype == img_b.dtype == np.uint8
        np.testing.assert_array_equal(img_a, img_b)
        assert t_a == t_b


@pytest.mark.parametrize("read", READERS)
def test_mono8_png_gives_the_raw_pixels_and_stamps(read, tmp_path, monkeypatch, caplog):
    frames = _frames()
    raw = write_bag(tmp_path / "raw.bag", {CAM: [raw_msg(i, f, "mono8") for i, f in enumerate(frames)]})
    png = write_bag(tmp_path / "png.bag", {CAM: [compressed_msg(i, f) for i, f in enumerate(frames)]})
    with caplog.at_level(logging.WARNING, logger=JPEG_LOGGER):
        got = read(png, monkeypatch)
    assert_same_frames(read(raw, monkeypatch), got)
    np.testing.assert_array_equal(got[0][0], frames[0])
    assert not caplog.records


@pytest.mark.parametrize("read", READERS)
def test_mono16_png_gets_the_shared_mono16_scaling(read, tmp_path, monkeypatch):
    frames = _frames(depth=np.uint16)
    raw = write_bag(tmp_path / "raw.bag", {CAM: [raw_msg(i, f, "mono16") for i, f in enumerate(frames)]})
    png = write_bag(tmp_path / "png.bag", {CAM: [compressed_msg(i, f, "mono16; png compressed")
                                                 for i, f in enumerate(frames)]})
    assert_same_frames(read(raw, monkeypatch), read(png, monkeypatch))


@pytest.mark.parametrize("read", READERS)
def test_colour_png_comes_out_as_uint8_grey(read, tmp_path, monkeypatch):
    frames = _frames(channels=3)
    png = write_bag(tmp_path / "png.bag", {CAM: [compressed_msg(i, f, "bgr8; png compressed")
                                                 for i, f in enumerate(frames)]})
    got = read(png, monkeypatch)
    assert got[0][0].shape == (H, W) and got[0][0].dtype == np.uint8
    np.testing.assert_array_equal(got[0][0], cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))


@pytest.mark.parametrize("read", READERS)
def test_colour_png_matches_raw_colour_in_every_reader(read, tmp_path, monkeypatch):
    frames = _frames(channels=3)
    raw = write_bag(tmp_path / "raw.bag", {CAM: [raw_msg(i, f, "bgr8") for i, f in enumerate(frames)]})
    png = write_bag(tmp_path / "png.bag", {CAM: [compressed_msg(i, f, "bgr8; png compressed")
                                                 for i, f in enumerate(frames)]})
    assert_same_frames(read(raw, monkeypatch), read(png, monkeypatch))


@pytest.mark.parametrize("read", READERS)
def test_jpeg_decodes_and_warns_once_per_topic(read, tmp_path, monkeypatch, caplog):
    flat = np.full((H, W), 120, np.uint8)       # survives JPEG, unlike noise
    msgs = [compressed_msg(i, flat, "jpeg", ".jpg") for i in range(3)]
    jpg = write_bag(tmp_path / "jpg.bag", {CAM: msgs, CAM2: msgs})
    with caplog.at_level(logging.WARNING, logger=JPEG_LOGGER):
        got = read(jpg, monkeypatch)
    assert len(got) == 3 and got[0][0].shape == (H, W) and got[0][0].dtype == np.uint8
    assert np.abs(got[0][0].astype(int) - 120).max() <= 2
    warned = [r.getMessage() for r in caplog.records if r.name == JPEG_LOGGER]
    assert len(warned) == 1 and CAM in warned[0] and "lossy" in warned[0]


def test_each_jpeg_topic_gets_its_own_warning(caplog):
    msg = compressed_msg(0, np.zeros((H, W), np.uint8), "jpeg", ".jpg")
    warned = set()
    with caplog.at_level(logging.WARNING, logger=JPEG_LOGGER):
        for topic in (CAM, CAM2, CAM, CAM2):
            as_raw_image(msg, topic, warned)
    assert sorted(r.getMessage().split(":")[0] for r in caplog.records) == [CAM, CAM2]


def test_colour_jpeg_with_the_image_transport_format_string():
    img = np.dstack([np.full((H, W), v, np.uint8) for v in (40, 90, 200)])
    raw = as_raw_image(compressed_msg(0, img, "rgb8; jpeg compressed bgr8", ".jpg"), CAM, set())
    assert (raw.height, raw.width, raw.encoding) == (H, W, "mono8")
    assert abs(int(raw.data[0, 0]) - 117) <= 2    # luma of BGR (40, 90, 200)


def test_format_string_does_not_hide_a_jpeg(caplog):
    # A driver that labels JPEG bytes "png" still gets the warning.
    msg = compressed_msg(0, np.zeros((H, W), np.uint8), "png", ".jpg")
    with caplog.at_level(logging.WARNING, logger=JPEG_LOGGER):
        as_raw_image(msg, CAM, set())
    assert len(caplog.records) == 1


def test_raw_image_passes_through_untouched():
    msg = raw_msg(0, _frames(1)[0], "mono8")
    assert as_raw_image(msg, CAM, set()) is msg


def _png_bytes():
    return compressed_msg(0, _frames(1)[0]).data


@pytest.mark.parametrize("fmt, data, expect", [
    ("mono8; png compressed", np.frombuffer(b"garbage bytes", np.uint8), "unreadable"),
    ("mono8; png compressed", np.zeros(0, np.uint8), "unreadable"),
    ("mono8; png compressed", _png_bytes()[:200], "unreadable"),
    ("jpeg", np.frombuffer(b"\xff\xd8\xff" + b"x" * 40, np.uint8), "unreadable"),
    ("h264", np.frombuffer(b"\x00\x00\x00\x01\x67", np.uint8), "unsupported"),
    ("16UC1; compressedDepth png", np.r_[np.zeros(12, np.uint8), _png_bytes()], "unsupported"),
])
def test_bad_frame_or_format_names_the_topic_and_stamp(fmt, data, expect):
    msg = Compressed(header=_header(2), format=fmt, data=data)
    with pytest.raises(ValueError) as e:
        as_raw_image(msg, CAM, set())
    assert expect in str(e.value) and CAM in str(e.value) and "100.200000000" in str(e.value)


def test_bad_frame_stops_the_cam_imu_loader(tmp_path, monkeypatch):
    msgs = [compressed_msg(0, _frames(1)[0]),
            Compressed(header=_header(1), format="png", data=_png_bytes()[:200])]
    bag = write_bag(tmp_path / "bad.bag", {CAM: msgs})
    with pytest.raises(ValueError, match="unreadable"):
        read_cam_imu(bag, monkeypatch)


def test_16bit_colour_png_reads_like_raw_bgr16():
    img = np.random.default_rng(7).integers(0, 65536, (H, W, 3)).astype(np.uint16)
    png = ros_image_to_gray(compressed_msg(0, img, "bgr16; png compressed"), CAM, set())
    np.testing.assert_array_equal(png, ros_image_to_gray(raw_msg(0, img, "bgr16")))
    assert png.dtype == np.uint8 and png.shape == (H, W)


@pytest.fixture
def fake_ros2(monkeypatch):
    """rosbag2_py and rclpy come from a ROS 2 install, so fake only those: the messages are real
    CDR, and the SequentialReader replays them with the type table a bag would report. cv_bridge
    is blocked: the reader decodes the message bytes itself (G6)."""
    messages = []
    types_by_topic = {}
    cdr = get_typestore(Stores.ROS2_HUMBLE)

    class Reader:
        def open(self, *_):
            self.queue = list(messages)

        def set_filter(self, _):
            pass

        def get_all_topics_and_types(self):
            return [types.SimpleNamespace(name=k, type=v) for k, v in types_by_topic.items()]

        def has_next(self):
            return bool(self.queue)

        def read_next(self):
            return self.queue.pop(0)

    rb = types.SimpleNamespace(SequentialReader=Reader, StorageFilter=lambda topics: topics,
                               StorageOptions=lambda **k: k, ConverterOptions=lambda **k: k)

    def deserialize(data, cls):
        return cdr.deserialize_cdr(data, cls.__msgtype__)

    for name, mod in {
        "rosbag2_py": rb,
        "rclpy": types.SimpleNamespace(),
        "rclpy.serialization": types.SimpleNamespace(deserialize_message=deserialize),
        "sensor_msgs": types.SimpleNamespace(),
        "sensor_msgs.msg": types.SimpleNamespace(Image=cdr.types["sensor_msgs/msg/Image"],
                                                 CompressedImage=cdr.types["sensor_msgs/msg/CompressedImage"]),
        "cv_bridge": None,                                  # import cv_bridge -> ImportError
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    def add(topic, msgtype, frame_msgs):
        types_by_topic[topic] = msgtype
        for i, m in enumerate(frame_msgs):
            messages.append((topic, bytes(cdr.serialize_cdr(m, msgtype)), T0 + i * DT))
    add.cdr = cdr
    return add


def _ros2_header(cdr, i):
    t = T0 + i * DT
    stamp = cdr.types["builtin_interfaces/msg/Time"](sec=t // 10**9, nanosec=t % 10**9)
    return cdr.types["std_msgs/msg/Header"](stamp=stamp, frame_id="cam")


def test_ros2_png_matches_raw_and_jpeg_warns(fake_ros2, caplog):
    cdr = fake_ros2.cdr
    Comp2, Img2 = cdr.types["sensor_msgs/msg/CompressedImage"], cdr.types["sensor_msgs/msg/Image"]
    frames = _frames(2)
    png = [Comp2(_ros2_header(cdr, i), "mono8; png compressed",
                 cv2.imencode(".png", f)[1].reshape(-1)) for i, f in enumerate(frames)]
    raw = [Img2(_ros2_header(cdr, i), H, W, "mono8", 0, W, np.frombuffer(f.tobytes(), np.uint8))
           for i, f in enumerate(frames)]
    jpg = [Comp2(_ros2_header(cdr, i), "jpeg", cv2.imencode(".jpg", f)[1].reshape(-1))
           for i, f in enumerate(frames)]
    fake_ros2("/png", Comp2.__msgtype__, png)
    fake_ros2("/raw", Img2.__msgtype__, raw)
    fake_ros2("/jpg", Comp2.__msgtype__, jpg)
    with caplog.at_level(logging.WARNING, logger=JPEG_LOGGER):
        got = list(Ros2BagSource("x", ["/png", "/raw", "/jpg"]))
    by_cam = {c: [(img, t) for img, t, cid in got if cid == c] for c in range(3)}
    for (p, t_p), (r, t_r) in zip(by_cam[0], by_cam[1]):
        assert p.ndim == 2 and p.dtype == np.uint8 and t_p == t_r
        np.testing.assert_array_equal(p, r)
        assert r.ndim == 2 and r.dtype == np.uint8
    assert by_cam[2][0][0].shape == (H, W)
    assert len(caplog.records) == 1 and "/jpg" in caplog.records[0].getMessage()


def test_inspect_reports_png_size_and_passes(tmp_path):
    bag = write_bag(tmp_path / "png.bag", {CAM: [compressed_msg(i, f) for i, f in enumerate(_frames())]})
    rep = inspect(str(bag))
    info = next(s.info for s in rep.streams if s.name == CAM)
    assert (info["width"], info["height"], info["encoding"]) == (W, H, "mono8")
    assert not [f for f in rep.findings if f.id in ("camera.lossy", "camera.first_image")]


def test_inspect_flags_jpeg_and_unsupported(tmp_path):
    jpg = write_bag(tmp_path / "jpg.bag", {CAM: [compressed_msg(0, _frames(1)[0], "jpeg", ".jpg")]})
    assert [f.status for f in inspect(str(jpg)).findings if f.id == "camera.lossy"] == ["warn"]
    h264 = Compressed(header=_header(0), format="h264", data=np.zeros(8, np.uint8))
    bad = write_bag(tmp_path / "bad.bag", {CAM: [h264]})
    assert [f.status for f in inspect(str(bad)).findings if f.id == "camera.first_image"] == ["fail"]
