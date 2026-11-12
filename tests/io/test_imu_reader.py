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
"""The shared IMU readers (`io.imu_reader`, `cam_imu_loader._ros1_imu` / `_euroc_imu`) return
the arrays the four loader copies returned, byte for byte: windowed and whole-bag, ROS1 and
EuRoC, with stamps out of order and comment lines."""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from mvis.io import cam_imu_loader as L


def _old_hdr_s(msg) -> float:
    """The pre-F5 stamp (cam_imu_loader._hdr_s): header.stamp of the deserialised message,
    no zero-stamp fallback. Kept here as the oracle for the raw-bytes reader."""
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def _bag(tmp_path, topics=("/imu0", "/imu1/data"), n=300, seed=0):
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    Imu, Header, Time, Q, V = (T["sensor_msgs/msg/Imu"], T["std_msgs/msg/Header"],
                               T["builtin_interfaces/msg/Time"], T["geometry_msgs/msg/Quaternion"],
                               T["geometry_msgs/msg/Vector3"])
    rng = np.random.default_rng(seed)
    path = str(tmp_path / "imu.bag")
    with Writer(path) as w:
        conns = {tp: w.add_connection(tp, Imu.__msgtype__, typestore=ts) for tp in topics}
        for k in range(n):
            bag_ns = 1_000_000_000 + k * 5_000_000
            for tp in topics:
                t_ns = bag_ns + int(rng.integers(-20_000_000, 20_000_000))  # header jitter, unsorted
                v = rng.normal(0, 3, 6)
                msg = Imu(header=Header(seq=k, stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9),
                                        frame_id="imu"),
                          orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.zeros(9),
                          angular_velocity=V(x=v[0], y=v[1], z=v[2]),
                          angular_velocity_covariance=np.zeros(9),
                          linear_acceleration=V(x=v[3], y=v[4], z=v[5]),
                          linear_acceleration_covariance=np.zeros(9))
                w.write(conns[tp], bag_ns, ts.serialize_ros1(msg, Imu.__msgtype__))
    return path


def _old_ros1(bag_path, imu_topics, lo=None, hi=None):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    imus = {tp: dict(topic=tp, name=tp.strip("/").split("/")[0], t=[], w=[], a=[]) for tp in imu_topics}
    with Reader(bag_path) as r:
        cons = [c for c in r.connections if c.topic in set(imu_topics)]
        for conn, _bt, raw in r.messages(connections=cons):
            m = ts.deserialize_ros1(raw, conn.msgtype)
            t = _old_hdr_s(m)
            if lo is not None and (t < lo - 0.5 or t > hi + 0.5):
                continue
            d = imus[conn.topic]
            d["t"].append(t)
            d["w"].append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            d["a"].append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
    return [L._finalize_imu(imus[tp]) for tp in imu_topics]


def _mav0(tmp_path, subdirs=("imu0", "imu1"), n=400, seed=1):
    rng = np.random.default_rng(seed)
    for s in subdirs:
        os.makedirs(tmp_path / s)
        with open(tmp_path / s / "data.csv", "w", newline="") as f:
            wr = csv.writer(f)
            f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_y,w_z,a_x [m s^-2],a_y,a_z\n")
            for k in rng.permutation(n):
                t_ns = 1_403_636_579_758_555_392 + int(k) * 5_000_000 + int(rng.integers(0, 999))
                wr.writerow([t_ns] + [repr(float(x)) for x in rng.normal(0, 2, 6)])
                if k % 97 == 0:
                    f.write("# a comment line\n")
                if k % 131 == 0:
                    f.write("\n")
    return str(tmp_path)


def _old_euroc(mav0_dir, subdirs, lo=None, hi=None, strip=False):
    imus = []
    for ims in subdirs:
        sub = ims.strip("/").split("/")[0] if strip else ims
        d = dict(topic=sub, name=sub, t=[], w=[], a=[])
        for row in csv.reader(open(os.path.join(mav0_dir, sub, "data.csv"))):
            if not row or row[0].startswith("#"):
                continue
            t = int(row[0]) * 1e-9
            if lo is not None and not (lo - 0.5 <= t <= hi + 0.5):
                continue
            d["t"].append(t)
            d["w"].append([float(x) for x in row[1:4]])
            d["a"].append([float(x) for x in row[4:7]])
        imus.append(L._finalize_imu(d))
    return imus


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.keys() == y.keys() and x["topic"] == y["topic"] and x["name"] == y["name"]
        for k in ("t", "w", "a"):
            assert x[k].dtype == y[k].dtype and x[k].shape == y[k].shape
            assert x[k].tobytes() == y[k].tobytes()


def test_ros1_whole_and_windowed(tmp_path):
    bag = _bag(tmp_path)
    topics = ["/imu0", "/imu1/data"]
    _same(L.read_imu_only(bag, topics), _old_ros1(bag, topics))
    _same(L._ros1_imu(bag, topics[::-1]), _old_ros1(bag, topics[::-1]))
    for lo, hi in ((1.2, 1.9), (0.0, 0.3), (1.0 + 1e-9, 2.0 - 1e-9), (5.0, 6.0)):
        _same(L._ros1_imu(bag, topics, window=(lo, hi)), _old_ros1(bag, topics, lo, hi))


def test_euroc_whole_and_windowed(tmp_path):
    root = _mav0(tmp_path)
    _same(L.read_imu_only(root, ["imu0", "/imu1"], fmt="euroc"),
          _old_euroc(root, ["imu0", "/imu1"], strip=True))
    t0 = 1_403_636_579_758_555_392 * 1e-9
    for lo, hi in ((t0 + 0.3, t0 + 1.1), (t0 - 5, t0 - 1), (t0, t0 + 10)):
        _same(L._euroc_imu(root, ["imu0", "imu1"], window=(lo, hi)),
              _old_euroc(root, ["imu0", "imu1"], lo, hi))


@pytest.mark.parametrize("seed", [0, 1])
def test_inspect_imu_sample_values(tmp_path, seed):
    from types import SimpleNamespace as NS
    from mvis.io.imu_reader import euroc_imu_values, ros1_imu_sample
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, 6)
    m = NS(angular_velocity=NS(x=v[0], y=v[1], z=v[2]), linear_acceleration=NS(x=v[3], y=v[4], z=v[5]))
    assert ros1_imu_sample(m) == ([v[0], v[1], v[2]], [v[3], v[4], v[5]])
    row = ["123"] + [repr(float(x)) for x in v]
    assert euroc_imu_values(row) == ([float(x) for x in row[1:4]], [float(x) for x in row[4:7]])
