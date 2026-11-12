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
"""One capture key and board fingerprint for every detection cache (cleanup plan PR F, F4):
the key holds the camera topics and the window, the cam-IMU cache file name keeps its hash,
and the board fingerprint is the old DetectionPipeline board config."""
import types

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from mvis.core.config import BoardConfig, CalibrationConfig, CameraConfig
from mvis.detection.apriltag_board import AprilTagBoard
from mvis.io import cam_imu_loader
from mvis.io.capture_key import board_fingerprint, board_signature, capture_key
from mvis.io.data_loader import DataLoader
from mvis.io.detection_pipeline import DetectionPipeline


def _board(**kw):
    return AprilTagBoard(rows=6, cols=6, tag_size=0.088, tag_spacing=0.3, **kw)


def test_capture_key_holds_topics_window_and_knobs():
    k = capture_key("/d/a.bag", ["/cam0", "/cam1"], window=(0.0, 80.0), min_obs=1)
    assert k == "/d/a.bag|topics=['/cam0', '/cam1']|window=(0.0, 80.0)|min_obs=1"
    assert capture_key("/d/a.bag", ["/cam0"]) != capture_key("/d/a.bag", ["/cam1"])
    assert capture_key("/d/a.bag", ["/cam0", "/cam1"]) != capture_key("/d/a.bag", ["/cam1", "/cam0"])
    assert capture_key("/d/a.bag", ["/c"], window=(0, 1)) != capture_key("/d/a.bag", ["/c"], window=(0, 2))
    assert capture_key("/d/a.bag", ["/c"], hz=None) == capture_key("/d/a.bag", ["/c"])


def test_board_fingerprint_is_the_old_pipeline_board_config():
    b = _board()
    old = {k: v for k, v in sorted(vars(b).items()) if isinstance(v, (int, float, str, bool))}
    old["board_type"] = "AprilTagBoard"
    assert board_fingerprint(b) == old == DetectionPipeline(b)._board_config()
    assert board_fingerprint(_board(min_tag_px=8)) != old


def test_board_signature_keeps_its_string_and_old_import_path():
    from mvis.io.frame_detector import board_signature as old_path
    b = _board()
    items = sorted((k, v) for k, v in vars(b).items() if isinstance(v, (int, float, str, bool)))
    assert old_path is board_signature
    assert board_signature(b) == f"AprilTagBoard({items})"


def test_cam_imu_cache_file_names_keep_the_signature_hash(tmp_path):
    """Two signatures on one bag (ELP vs ELP+T265 topics) got ONE file before F4:
    with_suffix cut '.bag_<hash>' off the name."""
    b = _board()
    s1 = cam_imu_loader._sig("/d/4imus_4cams_01.bag", ["/elp/left", "/elp/right"], b, 0.0, 80.0, 1)
    s2 = cam_imu_loader._sig("/d/4imus_4cams_01.bag", ["/t265/fisheye1"], b, 0.0, 80.0, 1)
    p1 = cam_imu_loader._cache(str(tmp_path), "4imus_4cams_01.bag", s1)._path
    p2 = cam_imu_loader._cache(str(tmp_path), "4imus_4cams_01.bag", s2)._path
    assert p1 != p2
    assert p1.name.startswith("cam_imu_4imus_4cams_01_bag_") and p1.suffix == ".mvisdb"


def _cfg(topics, path="/d/x.bag"):
    cams = [CameraConfig(camera_id=i, source_type="rosbag_topic", source_path=path, topic=t)
            for i, t in enumerate(topics)]
    return CalibrationConfig(cameras=cams, board=BoardConfig(board_type="chessboard", rows=5,
                                                             cols=7, square_size=0.03))


def test_camera_only_cache_key_holds_the_topics():
    a = DataLoader(_cfg(["/cam0", "/cam1"])).cache_key
    assert a == "/d/x.bag|topics=['/cam0', '/cam1']|stamp=header"
    assert DataLoader(_cfg(["/cam2", "/cam3"])).cache_key != a


TS = get_typestore(Stores.ROS1_NOETIC)


def _write_bag(path, topics, n=3):
    Image = TS.types["sensor_msgs/msg/Image"]
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    with Writer(path) as w:
        conns = [w.add_connection(t, "sensor_msgs/msg/Image", typestore=TS) for t in topics]
        for i in range(n):
            ns = 100 * 10**9 + i * 10**8
            for c in conns:
                m = Image(header=Header(seq=i, stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9),
                                        frame_id="c"), height=8, width=8, encoding="mono8",
                          is_bigendian=0, step=8, data=np.zeros(64, np.uint8))
                w.write(c, ns, TS.serialize_ros1(m, c.msgtype))


def test_load_ros1_topic_sets_of_one_bag_get_their_own_cache(tmp_path, monkeypatch):
    bag = tmp_path / "rig.bag"
    _write_bag(bag, ["/a", "/b"])
    monkeypatch.setattr(cam_imu_loader, "_ros1_imu", lambda *a, **k: [])
    board = types.SimpleNamespace(detect=lambda img: [], rows=1)
    cache = tmp_path / "cache"
    ca, _ = cam_imu_loader.load_ros1(str(bag), ["/a"], [], board, cache_dir=str(cache), min_obs=0)
    cb, _ = cam_imu_loader.load_ros1(str(bag), ["/b"], [], board, cache_dir=str(cache), min_obs=0)
    assert ca[0]["cache_path"] != cb[0]["cache_path"]
    assert len(list(cache.glob("*.mvisdb"))) == 2
