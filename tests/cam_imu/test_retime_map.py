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
"""Sensor-clock retiming plumbing: stamp-map application, the sidecar round trip,
and the bag restamp tool on a synthetic ROS1 bag."""
import json

import numpy as np
import pytest
import yaml

from mvis.calibration.pipeline_cam_imu import _apply_retime_map
from mvis.cli.main import load_retime_map, write_sensor_clock_sidecar


def _cam(n=6, t0=100.0, dt=0.04):
    t = t0 + dt * np.arange(n)
    return dict(t_cam=t, ids_list=[np.array([i]) for i in range(n)],
                uv_list=[np.array([[i, i]], float) for i in range(n)],
                frames=object(), retime_sensor_clock=True, retime_map=np.zeros((0, 2)))


def test_apply_retime_map_drops_sorts_and_realigns():
    cam = _cam()
    # map frames 0,1,2,4 (3 and 5 missing -> dropped); new times reverse 1 and 2
    stamps = cam["t_cam"][[0, 1, 2, 4]]
    caps = np.array([100.00, 100.09, 100.05, 100.17])
    cam2 = dict(cam)
    n = _apply_retime_map(cam2, cam, stamps, caps)
    assert n == 4
    assert np.allclose(cam2["t_cam"], [100.00, 100.05, 100.09, 100.17])
    assert [int(i[0]) for i in cam2["ids_list"]] == [0, 2, 1, 4]       # realigned
    assert [int(u[0, 0]) for u in cam2["uv_list"]] == [0, 2, 1, 4]
    assert cam2["frames"] is None and cam2["retime_sensor_clock"] is False
    assert "retime_map" not in cam2 and "factor_hz" not in cam2        # rate not forced
    assert len(cam["t_cam"]) == 6                                        # input untouched


def test_sidecar_round_trip(tmp_path):
    rep = {0: {"period_s": 1 / 29.625, "phase_s": 0.1, "rate_hz": 29.625, "coherence": 0.98,
               "scatter_ms": 0.7, "retimed": True, "n_retimed": 3, "n_frames": 4,
               "retime_map": [[1.0, 1.01], [2.0, 2.02], [3.0, 3.03]], "dropped_stamps": [4.0]},
           1: {"retimed": False, "reason": "too few frames with an offset"}}
    p = tmp_path / "cam_imu_sensor_clock.json"
    compact = write_sensor_clock_sidecar(rep, str(p), ["/a/image_raw", "/b/image_raw"])
    assert "retime_map" not in compact["0"] and compact["0"]["sidecar"] == str(p)
    assert compact["0"]["n_retimed"] == 3 and compact["1"]["retimed"] is False
    json.dumps(compact)
    sc = json.load(open(p))
    assert sc["format"].startswith("mvis-sensor-clock") and sc["cameras"]["0"]["topic"] == "/a/image_raw"
    m = load_retime_map(str(p), 0)
    assert m.shape == (3, 2) and np.allclose(m[:, 1], [1.01, 2.02, 3.03])
    with pytest.raises(ValueError):
        load_retime_map(str(p), 1)
    # no path -> compact only, no file
    compact2 = write_sensor_clock_sidecar(rep, None)
    assert "sidecar" not in compact2["0"] and "retime_map" not in compact2["0"]


def test_restamp_bag_moves_only_mapped_topics(tmp_path):
    rosbags = pytest.importorskip("rosbags")
    from rosbags.rosbag1 import Reader, Writer
    from rosbags.typesys import Stores, get_typestore
    from mvis.recalib.restamp_sensor_clock import load_stamp_maps, restamp_bag, companion_topics

    ts = get_typestore(Stores.ROS1_NOETIC)
    Image = ts.types["sensor_msgs/msg/Image"]
    CamInfo = ts.types["sensor_msgs/msg/CameraInfo"]
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    RoI = ts.types["sensor_msgs/msg/RegionOfInterest"]
    Q = ts.types["geometry_msgs/msg/Quaternion"]; V3 = ts.types["geometry_msgs/msg/Vector3"]

    def hdr(t, seq):
        return Header(seq=seq, stamp=Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))), frame_id="c")

    src = tmp_path / "in.bag"; out = tmp_path / "out.bag"
    stamps = [10.00, 10.04, 10.08, 10.12, 10.16]
    with Writer(src) as w:
        ci = w.add_connection("/cam/image_raw", Image.__msgtype__, typestore=ts)
        cc = w.add_connection("/cam/camera_info", CamInfo.__msgtype__, typestore=ts)
        im = w.add_connection("/imu", Imu.__msgtype__, typestore=ts)
        for k, t in enumerate(stamps):
            rec = int(t * 1e9) + 2_000_000
            img = Image(header=hdr(t, k), height=1, width=2, encoding="mono8", is_bigendian=0, step=2,
                        data=np.array([k, k], np.uint8))
            w.write(ci, rec, ts.serialize_ros1(img, Image.__msgtype__))
            info = CamInfo(header=hdr(t, k), height=1, width=2, distortion_model="plumb_bob",
                           D=np.zeros(0), K=np.zeros(9), R=np.zeros(9), P=np.zeros(12),
                           binning_x=0, binning_y=0, roi=RoI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
            w.write(cc, rec + 1, ts.serialize_ros1(info, CamInfo.__msgtype__))
            imu = Imu(header=hdr(t, k), orientation=Q(x=0, y=0, z=0, w=1), orientation_covariance=np.zeros(9),
                      angular_velocity=V3(x=0, y=0, z=0), angular_velocity_covariance=np.zeros(9),
                      linear_acceleration=V3(x=0, y=0, z=9.8), linear_acceleration_covariance=np.zeros(9))
            w.write(im, rec + 2, ts.serialize_ros1(imu, Imu.__msgtype__))
    # map: frames 0,1,3 re-stamped, frame 2 dropped, frame 4 unmatched (outside window)
    sidecar = tmp_path / "x_sensor_clock.json"
    sidecar.write_text(json.dumps({"format": "mvis-sensor-clock/1", "cameras": {"0": {
        "topic": "/cam/image_raw", "period_s": 0.03376, "rate_hz": 29.625, "coherence": 0.98,
        "scatter_ms": 0.7, "phase_s": 0.0,
        # 10.04 + 0.3 µs: float64 seconds never match the message stamp exactly
        "map": [[10.00, 9.995], [10.0400003, 10.0288], [10.12, 10.0963]],
        "dropped_stamps": [10.08]}}}))
    maps = load_stamp_maps(str(sidecar))

    def read(path):
        with Reader(path) as r:
            got = {}
            for conn, rec, raw in r.messages():
                msg = ts.deserialize_ros1(raw, conn.msgtype)
                got.setdefault(conn.topic, []).append((rec, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))
        return got

    # default: frames without a capture time (frame 4) are dropped
    prov = restamp_bag(str(src), str(out), maps, {0: companion_topics("/cam/image_raw")})
    st = prov["topics"]
    assert st["/cam/image_raw"] == {"restamped": 3, "dropped": 1, "unmatched": 1}
    assert st["/cam/camera_info"] == {"restamped": 3, "dropped": 1, "unmatched": 1}
    assert prov["passthrough_messages"] == 5 and prov["unmatched_frames"] == "dropped"
    got = read(out)
    assert [round(s, 4) for _, s in got["/cam/image_raw"]] == [9.995, 10.0288, 10.0963]
    assert [round(s, 4) for _, s in got["/cam/camera_info"]] == [9.995, 10.0288, 10.0963]
    assert [round(s, 2) for _, s in got["/imu"]] == stamps                 # untouched
    recs = [rec for rec, _ in got["/cam/image_raw"]]
    assert recs == sorted(recs) and recs[0] == int(10.00 * 1e9) + 2_000_000  # record times kept
    # the provenance sidecar reads back and says what happened to the bag
    pv = yaml.safe_load((tmp_path / "out.bag.sensor_clock_provenance.yaml").read_text())
    assert prov["provenance"] == str(tmp_path / "out.bag.sensor_clock_provenance.yaml")
    assert pv["source_bag"] == str(src) and pv["output_bag"] == str(out)
    assert pv["source_size"] == src.stat().st_size and pv["output_size"] == out.stat().st_size
    assert pv["topics"] == st and pv["passthrough_messages"] == 5 and pv["unmatched_frames"] == "dropped"
    assert pv["tool"] == "mvis.recalib.restamp_sensor_clock"
    cam0 = pv["cameras"][0]
    assert cam0["topics"] == companion_topics("/cam/image_raw")
    assert cam0["n_map"] == 3 and cam0["n_dropped_in_map"] == 1
    assert cam0["clock"]["rate_hz"] == 29.625 and cam0["clock"]["coherence"] == 0.98
    # --keep-unmatched: frame 4 stays with its recorded stamp
    out2 = tmp_path / "out2.bag"
    restamp_bag(str(src), str(out2), maps, {0: companion_topics("/cam/image_raw")}, keep_unmatched=True)
    assert [round(s, 4) for _, s in read(out2)["/cam/image_raw"]] == [9.995, 10.0288, 10.0963, 10.16]


def test_merge_shared_clock_joins_lattices_and_flags_conflicts():
    from mvis.recalib.restamp_sensor_clock import _StampIndex, merge_shared_clock
    T = 0.03376
    def rec(phase, stamps, idx, dropped=()):
        caps = [int(round((phase + n * T) * 1e9)) for n in idx]
        return {"topic": None, "map": _StampIndex(np.asarray(stamps, np.int64), caps),
                "dropped": _StampIndex(np.asarray(dropped, np.int64)),
                "clock": {"period_s": T, "phase_s": phase, "rate_hz": 1 / T, "coherence": 0.9, "scatter_ms": 0.7}}
    s = [int(1e9 * (10 + 0.04 * k)) for k in range(5)]
    # left: frames 0-3 (phase +0.0003 s); right: frames 1-4 with frame 3 one index off,
    # and its phase wrapped by one period (lattice phases are only defined modulo T)
    left = rec(0.0003, s[:4], [0, 1, 2, 4], dropped=[s[4]])
    right = rec(-0.0003 + T, s[1:], [0, 1, 4, 4])
    m = merge_shared_clock({0: left, 1: right}, [0, 1], require_all=False)
    assert m["merge"]["n_frames"] == 4 and m["merge"]["n_conflict"] == 1
    assert abs(m["merge"]["phase_spread_ms"] - 0.6) < 1e-9 and abs(m["clock"]["phase_s"]) < 1e-12
    assert m["map"].find(s[0]) == int(round(0 * T * 1e9))            # left-only frame
    assert m["map"].find(s[4]) == int(round(5 * T * 1e9))            # right-only frame (left dropped it)
    assert m["map"].find(s[3]) is None and m["dropped"].find(s[3])   # index conflict -> dropped
    assert m["map"].find(s[1]) == int(round(1 * T * 1e9))            # shared frame, common phase


def test_merge_shared_clock_drops_two_frames_on_one_sensor_index():
    from mvis.recalib.restamp_sensor_clock import _StampIndex, merge_shared_clock
    T = 0.03376
    def rec(stamps, idx):
        caps = [int(round(n * T * 1e9)) for n in idx]
        return {"topic": None, "map": _StampIndex(np.asarray(stamps, np.int64), caps),
                "dropped": _StampIndex(np.zeros(0, np.int64)),
                "clock": {"period_s": T, "phase_s": 0.0, "rate_hz": 1 / T, "coherence": 0.9, "scatter_ms": 0.7}}
    s = [int(1e9 * (10 + 0.04 * k)) for k in range(4)]
    # left maps frame 2 to index 3, right maps frame 3 to index 3 as well (bag 7 case)
    left = rec([s[0], s[1], s[2]], [0, 1, 3])
    right = rec([s[0], s[1], s[3]], [0, 1, 3])
    m = merge_shared_clock({0: left, 1: right}, [0, 1], require_all=False)
    assert m["merge"]["n_frames"] == 2 and m["merge"]["n_index_collision"] == 2
    assert m["map"].find(s[2]) is None and m["map"].find(s[3]) is None
    assert m["dropped"].find(s[2]) and m["dropped"].find(s[3])
    caps = list(m["map"].v)
    assert len(set(caps)) == len(caps)


def test_merge_shared_clock_require_all_keeps_only_both_sided_frames():
    from mvis.recalib.restamp_sensor_clock import _StampIndex, merge_shared_clock
    T = 0.03376
    def rec(stamps, idx):
        return {"topic": None, "map": _StampIndex(np.asarray(stamps, np.int64), [int(round(n * T * 1e9)) for n in idx]),
                "dropped": _StampIndex(np.zeros(0, np.int64)),
                "clock": {"period_s": T, "phase_s": 0.0, "rate_hz": 1 / T, "coherence": 0.9, "scatter_ms": 0.7}}
    s = [int(1e9 * (10 + 0.04 * k)) for k in range(4)]
    maps = {0: rec(s[:3], [0, 1, 2]), 1: rec(s[1:], [1, 2, 3])}
    union = merge_shared_clock(maps, [0, 1], require_all=False)
    both = merge_shared_clock(maps, [0, 1])
    assert union["merge"]["n_frames"] == 4 and union["merge"]["n_one_sided"] == 2
    assert both["merge"]["n_frames"] == 2 and both["map"].find(s[0]) is None and both["dropped"].find(s[0])
    assert both["map"].find(s[1]) is not None and both["map"].find(s[3]) is None
