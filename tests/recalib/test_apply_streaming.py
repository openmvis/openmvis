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
"""apply_calibration streams the bag: the write order must equal a stable sort of every
message by its output time (what the buffered implementation did), and a failed run must
not leave a partial bag behind."""
import hashlib

import numpy as np
import pytest
from rosbags.rosbag1 import Reader, Writer

import mvis.recalib.apply as apply_mod
from mvis.recalib.apply import apply_calibration

from .test_apply_options import _OFF, _rig, T0


def _mixed_bag(path, ts, cam_headers, imu_headers, other_times):
    """IMU (record = header + 3 ms), camera (record = header + 9 ms) and an unmapped custom
    topic, with header stamps that need not follow record order."""
    T = ts.types
    Imu, Image, Header = T["sensor_msgs/msg/Imu"], T["sensor_msgs/msg/Image"], T["std_msgs/msg/Header"]
    Time, Q, V = T["builtin_interfaces/msg/Time"], T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Vector3"]

    def hdr(ns):
        return Header(0, Time(ns // 10**9, ns % 10**9), "f")

    d = "float64 value\n"
    with Writer(path) as w:
        ci = w.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        cc = w.add_connection("/cam0/image_raw", Image.__msgtype__, typestore=ts)
        cx = w.add_connection("/other", "test_msgs/msg/Scalar", msgdef=d,
                              md5sum=hashlib.md5(d.strip().encode()).hexdigest())
        msgs = []
        for k, ns in enumerate(imu_headers):
            m = Imu(hdr(ns), Q(0.0, 0.0, 0.0, 1.0), np.zeros(9), V(0.0, 0.0, 0.0),
                    np.zeros(9), V(float(k), 0.0, 9.81), np.zeros(9))
            msgs.append((ns + 3_000_000, ci, ts.serialize_ros1(m, Imu.__msgtype__)))
        for k, ns in enumerate(cam_headers):
            m = Image(hdr(ns), 2, 2, "mono8", 0, 2, np.full(4, k, np.uint8))
            msgs.append((ns + 9_000_000, cc, ts.serialize_ros1(m, Image.__msgtype__)))
        for k, t in enumerate(other_times):
            msgs.append((t, cx, np.float64(k).tobytes()))
        for t, c, raw in sorted(msgs, key=lambda r: r[0]):
            w.write(c, t, raw)


def _expected_order(path, rig, td):
    """Reference: buffer everything, then stable-sort by output time (the old code path)."""
    shift = {"/imu0": round(rig.base_imu.td * 1e9), "/cam0/image_raw": round(rig.cams[0].td * 1e9)}
    rows = []
    with Reader(path) as r:
        for c, t, raw in r.messages():
            if td and c.topic in shift:
                sec, nsec = np.frombuffer(bytes(raw[4:12]), "<u4")
                t = int(sec) * 10**9 + int(nsec) + shift[c.topic]
            rows.append((t, c.topic))
    return sorted(rows, key=lambda x: x[0])


class _RecordingWriter(Writer):
    """Keeps the (time, topic) of every write call: the order in the file. Reading the bag
    back does not show it, since rosbags merges equal times by connection."""
    calls = []

    def write(self, connection, timestamp, data):
        _RecordingWriter.calls.append((timestamp, connection.topic))
        super().write(connection, timestamp, data)


@pytest.mark.parametrize("cam_td", [-0.0250, 0.0, 0.0123, 0.040])
def test_stream_order_equals_stable_full_sort(tmp_path, ts, cam_td, monkeypatch):
    ms = 1_000_000
    imu = [T0 + k * 5 * ms for k in range(30)]
    imu[10], imu[11] = imu[11], imu[10]                 # header stamps out of record order
    cam = [T0 + k * 33 * ms for k in range(5)]
    cam[2] = cam[1] + 3 * ms                            # irregular camera spacing
    # unmapped records exactly at some shifted camera/IMU times: ties keep read order
    other = [T0 + 9 * ms, T0 + 33 * ms + 9 * ms, T0 + 100 * ms + 3 * ms, T0 + 140 * ms]
    raw = str(tmp_path / "raw.bag")
    _mixed_bag(raw, ts, cam, imu, other)
    rig = _rig(t_ga=1e-3, cam_td=cam_td, tmp_path=tmp_path)
    mapping = {"/imu0": "imu0", "/cam0/image_raw": "cam0"}
    for i, kw in enumerate(({}, {**_OFF, "td": True}, _OFF)):
        out = str(tmp_path / f"out{i}.bag")
        _RecordingWriter.calls = []
        monkeypatch.setattr(apply_mod, "Writer", _RecordingWriter)
        apply_calibration(raw, out, rig, mapping, "/imu0", **kw)
        assert _RecordingWriter.calls == _expected_order(raw, rig, kw.get("td", True))


def test_failed_run_removes_partial_bag(tmp_path, ts):
    """A camera td that moves a stamp below zero fails in the write pass, after the output
    was opened; the partial bag is removed and no sidecar is written."""
    raw, out = str(tmp_path / "raw.bag"), tmp_path / "out.bag"
    _mixed_bag(raw, ts, [T0, T0 + 10**9], [T0 + k * 5_000_000 for k in range(5)], [T0])
    rig = _rig(cam_td=-(T0 + 10**9) / 1e9 + 0.5, tmp_path=tmp_path)   # first frame < 0
    with pytest.raises(ValueError, match="outside ROS1 range"):
        apply_calibration(raw, str(out), rig, {"/imu0": "imu0", "/cam0/image_raw": "cam0"},
                          "/imu0")
    assert not out.exists() and not (tmp_path / "out.bag.recalib.yaml").exists()


def test_existing_output_is_left_alone(tmp_path, ts):
    raw, out = str(tmp_path / "raw.bag"), tmp_path / "out.bag"
    _mixed_bag(raw, ts, [T0], [T0], [])
    out.write_bytes(b"keep")
    with pytest.raises(Exception, match="exists"):
        apply_calibration(raw, str(out), _rig(tmp_path=tmp_path),
                          {"/imu0": "imu0"}, "/imu0")
    assert out.read_bytes() == b"keep"
