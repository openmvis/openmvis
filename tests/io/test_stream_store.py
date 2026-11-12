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
"""Shared per-stream detection store (io.stream_store): a camera-only run and a
cam-IMU run on one ROS1 bag detect each frame once, and get the same frames."""
import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from mvis.core.types import FrameObservations, Observation
from mvis.io import cam_imu_loader as L
from mvis.io import detection_pipeline as DP
from mvis.io.stream_store import StreamDetectionStore

TS = get_typestore(Stores.ROS1_NOETIC)
T = TS.types
REC0 = 1_700_000_000 * 10**9
N_IMG = 6                      # per camera, 10 Hz
TOPICS = ["/cam0/image_raw", "/cam1/image_raw"]


# ---------- synthetic bag: two chessboard cameras + an IMU ----------

def _chessboard(rows=5, cols=7, sq=30, shift=0):
    h, w = (rows + 3) * sq, (cols + 3) * sq + 20
    img = np.full((h, w), 200, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = sq + r * sq, sq + c * sq + shift
                img[y0:y0 + sq, x0:x0 + sq] = 30
    return img


def _time(ns):
    return T["builtin_interfaces/msg/Time"](sec=ns // 10**9, nanosec=ns % 10**9)


def _image(ns, k, img):
    h, w = img.shape
    return T["sensor_msgs/msg/Image"](
        header=T["std_msgs/msg/Header"](seq=k, stamp=_time(ns), frame_id="cam"),
        height=h, width=w, encoding="mono8", is_bigendian=0, step=w,
        data=img.reshape(-1).copy())


def _imu(ns, k):
    V, Q = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
    return T["sensor_msgs/msg/Imu"](
        header=T["std_msgs/msg/Header"](seq=k, stamp=_time(ns), frame_id="imu"),
        orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.zeros(9),
        angular_velocity=V(x=0.0, y=0.0, z=0.0), angular_velocity_covariance=np.zeros(9),
        linear_acceleration=V(x=0.0, y=0.0, z=9.81), linear_acceleration_covariance=np.zeros(9))


@pytest.fixture(scope="module")
def bag(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("bag") / "rig.bag")
    with Writer(p) as w:
        cons = [w.add_connection(t, "sensor_msgs/msg/Image", typestore=TS) for t in TOPICS]
        ci = w.add_connection("/imu0", "sensor_msgs/msg/Imu", typestore=TS)
        for k in range(N_IMG * 20):                       # IMU 200 Hz
            rec = REC0 + k * 5_000_000
            w.write(ci, rec, TS.serialize_ros1(_imu(rec, k), ci.msgtype))
            if k % 20 == 0:                               # images 10 Hz
                j = k // 20
                for c, con in enumerate(cons):
                    img = _chessboard(shift=2 * j + 5 * c)
                    w.write(con, rec, TS.serialize_ros1(_image(rec, j, img), con.msgtype))
    return p


def _board():
    from mvis.core.config import BoardConfig
    from mvis.detection.board_factory import create_board
    return create_board(BoardConfig(board_type="chessboard", rows=5, cols=7,
                                    square_size=0.03))


def _cam_config(bag):
    from mvis.core.config import CalibrationConfig, CameraConfig, BoardConfig
    cams = [CameraConfig(camera_id=i, source_type="rosbag_topic", source_path=bag,
                         topic=t) for i, t in enumerate(TOPICS)]
    return CalibrationConfig(cameras=cams, board=BoardConfig(
        board_type="chessboard", rows=5, cols=7, square_size=0.03))


def _no_detection(monkeypatch):
    """Any detection now fails the test: the store must serve every frame."""
    def boom(*a, **k):
        raise AssertionError("detected although the stream store should have served")
    monkeypatch.setattr(DP, "detect_frame", boom)
    monkeypatch.setattr(L, "detect_frame", boom)


def _frames_equal(a, b):
    assert [f.timestamp for f in a] == [f.timestamp for f in b]
    for fa, fb in zip(a, b):
        assert [o.point_id for o in fa.observations] == [o.point_id for o in fb.observations]
        np.testing.assert_array_equal(np.array([o.pixel for o in fa.observations]),
                                      np.array([o.pixel for o in fb.observations]))


def _cam_only(bag, store, cache_path=None):
    from mvis.io.data_loader import DataLoader
    loader = DataLoader(_cam_config(bag))
    det = DP.DetectionPipeline(_board(), cache_path=cache_path, stream_store=store)
    return det.run_streaming(loader, cache_key=loader.cache_key, target_hz=100.0), det


# ---------- unit: the store itself ----------

def _fo(t, n):
    return FrameObservations(camera_id=3, timestamp=t, image_size=(64, 48), observations=[
        Observation(point_id=i, pixel=[float(i), float(i)], row=i) for i in range(n)])


def test_store_roundtrip_window_min_obs_and_camera_id(tmp_path):
    store = StreamDetectionStore(tmp_path)
    board = _board()
    frames = [_fo(1.0, 4), _fo(2.0, 1), _fo(3.0, 0), _fo(4.0, 6)]
    store.put("x.bag", "/cam0", board, frames, frames_read=4, lo=0.5, hi=10.0)
    got = store.get("x.bag", "/cam0", board, camera_id=1, lo=0.5, hi=10.0)
    assert [f.timestamp for f in got] == [1.0, 2.0, 4.0]          # 0-corner frame not stored
    assert {f.camera_id for f in got} == {1}
    got = store.get("x.bag", "/cam0", board, lo=1.5, hi=3.5, min_obs=1)
    assert [f.timestamp for f in got] == [2.0]                    # narrower window, filtered
    assert [f.timestamp for f in store.get("x.bag", "/cam0", board, lo=0.5, hi=10.0,
                                           min_obs=4)] == [1.0, 4.0]
    assert store.get("x.bag", "/cam0", board) is None             # entry doesn't cover ±inf
    assert store.get("x.bag", "/cam0", board, lo=0.0, hi=10.0) is None
    assert store.get("x.bag", "/cam1", board, lo=0.5, hi=10.0) is None


def test_store_misses_on_other_board(tmp_path):
    from mvis.core.config import BoardConfig
    from mvis.detection.board_factory import create_board
    store = StreamDetectionStore(tmp_path)
    store.put("x.bag", "/cam0", _board(), [_fo(1.0, 4)], frames_read=1)
    other = create_board(BoardConfig(board_type="chessboard", rows=5, cols=7,
                                     square_size=0.04))
    assert store.get("x.bag", "/cam0", _board()) is not None
    assert store.get("x.bag", "/cam0", other) is None


def test_put_keeps_a_wider_entry(tmp_path):
    store = StreamDetectionStore(tmp_path)
    board = _board()
    store.put("x.bag", "/cam0", board, [_fo(1.0, 4), _fo(4.0, 4)], frames_read=2)
    store.put("x.bag", "/cam0", board, [_fo(4.0, 4)], frames_read=1, lo=3.0, hi=5.0)
    got = store.get("x.bag", "/cam0", board)                      # still the ±inf entry
    assert [f.timestamp for f in got] == [1.0, 4.0]
    store.put("x.bag", "/cam1", board, [_fo(4.0, 4)], frames_read=1, lo=3.0, hi=5.0)
    store.put("x.bag", "/cam1", board, [_fo(1.0, 4), _fo(4.0, 4)], frames_read=2)
    assert [f.timestamp for f in store.get("x.bag", "/cam1", board)] == [1.0, 4.0]


def test_store_without_meta_is_a_miss(tmp_path):
    store = StreamDetectionStore(tmp_path)
    store.put("x.bag", "/cam0", _board(), [_fo(1.0, 4)], frames_read=1)
    for m in (tmp_path / "streams").glob("*.json"):
        m.unlink()
    assert store.get("x.bag", "/cam0", _board()) is None


# ---------- integration: camera-only and cam-IMU share one detection ----------

def test_camera_only_fills_store_and_reuses_it(bag, tmp_path, monkeypatch):
    store = StreamDetectionStore(tmp_path / "cache")
    first, _ = _cam_only(bag, store)
    assert all(len(first.all_frames[c]) > 0 for c in (0, 1))
    assert len(list((tmp_path / "cache" / "streams").glob("*.mvisdb"))) == 2
    _no_detection(monkeypatch)
    second, det2 = _cam_only(bag, store)
    for c in (0, 1):
        _frames_equal(first.all_frames[c], second.all_frames[c])
    assert det2._last_db is not None                             # carrier set like a fresh run


def test_cam_imu_reuses_camera_only_detections(bag, tmp_path, monkeypatch):
    fresh_cams, _ = L.load_ros1(bag, TOPICS, ["/imu0"], _board(), min_obs=4)
    store_dir = tmp_path / "cache"
    _cam_only(bag, StreamDetectionStore(store_dir))
    _no_detection(monkeypatch)
    cams, imus = L.load_ros1(bag, TOPICS, ["/imu0"], _board(), min_obs=4,
                             cache_dir=str(store_dir))
    for fc, c in zip(fresh_cams, cams):
        np.testing.assert_array_equal(fc["t_cam"], c["t_cam"])
        for a, b in zip(fc["uv_list"], c["uv_list"]):
            np.testing.assert_array_equal(a, b)
    assert len(imus[0]["t"]) == N_IMG * 20


def test_cam_imu_window_entry_does_not_serve_camera_only(bag, tmp_path):
    store_dir = tmp_path / "cache"
    L.load_ros1(bag, TOPICS, ["/imu0"], _board(), t0=0.15, window=1e9, min_obs=4,
                cache_dir=str(store_dir))                        # skips the first frames
    store = StreamDetectionStore(store_dir)
    assert store.get(bag, TOPICS[0], _board()) is None           # camera-only wants it all
    lo = REC0 / 1e9 + 0.15
    assert store.get(bag, TOPICS[0], _board(), lo=lo, hi=lo + 1e9) is not None


def test_cam_imu_partial_miss_keeps_camera_only_entry(bag, tmp_path):
    """A cam-IMU run that misses on one topic re-detects its window; it must not
    cut a full-recording entry another topic already had down to that window."""
    store_dir = tmp_path / "cache"
    store = StreamDetectionStore(store_dir)
    _cam_only(bag, store)                                  # fills both topics, ±inf
    full = store.get(bag, TOPICS[0], _board())
    for m in (store_dir / "streams").glob("*cam1*.json"):
        m.unlink()                                         # cam1 now misses
    L.load_ros1(bag, TOPICS, ["/imu0"], _board(), t0=0.15, window=1e9, min_obs=4,
                cache_dir=str(store_dir))
    got = store.get(bag, TOPICS[0], _board())
    assert got is not None                                 # ±inf entry kept
    _frames_equal(got, full)


def test_single_camera_sweep_path_fills_the_store(bag, tmp_path, monkeypatch):
    """One camera takes the detection.sweep path: it must fill the store too, with
    the same frames a two-camera run stores for that topic."""
    from mvis.core.config import CalibrationConfig, CameraConfig, BoardConfig
    from mvis.io.data_loader import DataLoader
    cfg = CalibrationConfig(cameras=[CameraConfig(
        camera_id=0, source_type="rosbag_topic", source_path=bag, topic=TOPICS[1])],
        board=BoardConfig(board_type="chessboard", rows=5, cols=7, square_size=0.03))
    store = StreamDetectionStore(tmp_path / "solo")
    det = DP.DetectionPipeline(_board(), stream_store=store)
    solo = det.run_streaming(DataLoader(cfg), cache_key="solo", target_hz=100.0)
    both = StreamDetectionStore(tmp_path / "both")
    _cam_only(bag, both)
    _frames_equal(store.get(bag, TOPICS[1], _board()), both.get(bag, TOPICS[1], _board()))
    _no_detection(monkeypatch)
    again = DP.DetectionPipeline(_board(), stream_store=store).run_streaming(
        DataLoader(cfg), cache_key="solo", target_hz=100.0)
    _frames_equal(solo.all_frames[0], again.all_frames[0])


def test_camera_only_config_reads_input_cache_dir(tmp_path):
    from mvis.core.config import CamRunConfig
    p = tmp_path / "cam.yaml"
    p.write_text("mode: cam\ninput:\n  cache_dir: /tmp/x\n  cameras:\n"
                 "  - {camera_id: 0, source_type: rosbag_topic, source_path: a.bag, topic: /c}\n"
                 "board: {board_type: chessboard, rows: 5, cols: 7, square_size: 0.03}\n")
    assert CamRunConfig.from_yaml(str(p)).input.cache_dir == "/tmp/x"
