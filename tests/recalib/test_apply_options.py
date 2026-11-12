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
"""apply_calibration behaviours folded in from tools/ros1_recalib (cleanup D9).

Each test pins one difference the two paths used to have: padded image rows, integer-ns
t_ga interpolation, unsorted / duplicate IMU stamps, the td switch, the strict input
policy, per-connection output metadata and the provenance flags."""
import hashlib

import numpy as np
import pytest
import yaml
from rosbags.rosbag1 import Reader, Writer

from mvis.imu import ImuModel
from mvis.recalib.apply import apply_calibration
from mvis.recalib.calib_spec import RigCalib, ImuCalib, MVIS

_MAP = {"/imu0": "imu0", "/cam0/image_raw": "cam0"}
_OFF = dict(remove_bias=False, imu_intrinsics=False, td=False, t_ga=False, undistort=False)
T0 = 1_700_000_000_123_456_789          # a real-epoch stamp: float64 spacing here is 256 ns


def _rig(t_ga=0.0, cam_td=0.0, model="radtan", tmp_path=None):
    intr = ImuModel.from_tangent(np.zeros(24), "rpng")
    imu = ImuCalib("imu0", "rpng", intr, np.array([1.0, 2.0, 3.0]),
                   np.array([0.5, 0.5, 0.5]), 0.0, "base", t_ga=t_ga)
    cam = MVIS("cam0", cam_td, 0.0, model, np.array([20.0, 21.0, 15.5, 16.0]),
               np.array([-0.2, 0.05, 0.001, 0.0]))
    rig = RigCalib(imu, [], [cam])
    if tmp_path is not None:
        calib = tmp_path / "calib.yaml"
        calib.write_text("model: test\n")
        rig._src_path = str(calib)
    return rig


def _bag(path, ts, imu_stamps, accel_x=None, images=(), custom=False):
    """imu_stamps: header ns per IMU message (record time = header + 7 ms);
    images: (header_ns, encoding, h, w, step, data)."""
    T = ts.types
    Imu, Image, Header = T["sensor_msgs/msg/Imu"], T["sensor_msgs/msg/Image"], T["std_msgs/msg/Header"]
    Time, Q, V = T["builtin_interfaces/msg/Time"], T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Vector3"]

    def hdr(ns):
        return Header(0, Time(ns // 10**9, ns % 10**9), "f")

    accel_x = list(range(len(imu_stamps))) if accel_x is None else accel_x
    with Writer(path) as w:
        ci = w.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        if images:
            cc = w.add_connection("/cam0/image_raw", Image.__msgtype__, typestore=ts)
        for ns, ax in zip(imu_stamps, accel_x):
            m = Imu(hdr(ns), Q(0.0, 0.0, 0.0, 1.0), np.zeros(9), V(3.0, 4.0, 5.0),
                    np.zeros(9), V(float(ax), 0.0, 9.81), np.zeros(9))
            w.write(ci, ns + 7_000_000, ts.serialize_ros1(m, Imu.__msgtype__))
        for ns, enc, h, wd, step, data in images:
            m = Image(hdr(ns), h, wd, enc, 0, step, np.asarray(data, np.uint8).reshape(-1))
            w.write(cc, ns + 9_000_000, ts.serialize_ros1(m, Image.__msgtype__))
        if custom:
            d = "float64 value\n"
            for caller in ("/a", "/b"):
                cx = w.add_connection("/custom", "test_msgs/msg/Scalar", msgdef=d,
                                      md5sum=hashlib.md5(d.strip().encode()).hexdigest(),
                                      callerid=caller, latching=1)
                w.write(cx, T0, b"\x00" * 8)


def _read(path, ts, topic):
    with Reader(path) as r:
        return [(t, ts.deserialize_ros1(raw, c.msgtype))
                for c, t, raw in r.messages() if c.topic == topic]


def _stamp(m):
    return m.header.stamp.sec * 10**9 + m.header.stamp.nanosec


def _imu_stamps(n=5, dt=5_000_000):
    return [T0 + k * dt for k in range(n)]


def test_t_ga_interpolates_on_integer_ns_offsets(tmp_path, ts):
    """accel x = sample index at 5 ms steps; a 1.234567 ms t_ga shift is exactly
    k + 1.234567 / 5 when interpolated on stamps relative to the first one. On absolute
    float64 ns (1.7e18, spacing 256 ns) the same query is off by up to ~5e-5."""
    raw, out = str(tmp_path / "raw.bag"), str(tmp_path / "out.bag")
    _bag(raw, ts, _imu_stamps())
    apply_calibration(raw, out, _rig(t_ga=1.234567e-3, tmp_path=tmp_path), {"/imu0": "imu0"},
                      "/imu0", **{**_OFF, "t_ga": True})
    got = [m.linear_acceleration.x for _, m in _read(out, ts, "/imu0")]
    np.testing.assert_allclose(got[:-1], np.arange(4) + 1.234567 / 5, rtol=0, atol=1e-12)
    assert got[-1] == 4.0                                   # clamped at the end


def test_td_restamp_is_integer_ns_exact_and_switchable(tmp_path, ts):
    raw = str(tmp_path / "raw.bag")
    _bag(raw, ts, _imu_stamps(2), images=[(T0 + 1, "mono8", 2, 2, 2, np.zeros(4))])
    td = 0.0123456789
    on, off = str(tmp_path / "on.bag"), str(tmp_path / "off.bag")
    apply_calibration(raw, on, _rig(cam_td=td, tmp_path=tmp_path), _MAP, "/imu0",
                      **{**_OFF, "td": True})
    [(t, m)] = _read(on, ts, "/cam0/image_raw")
    assert t == _stamp(m) == T0 + 1 + 12_345_679          # header + round(td * 1e9), in ns
    apply_calibration(raw, off, _rig(cam_td=td, tmp_path=tmp_path), _MAP, "/imu0", **_OFF)
    [(t, m)] = _read(off, ts, "/cam0/image_raw")
    assert (t, _stamp(m)) == (T0 + 1 + 9_000_000, T0 + 1)  # record latency + header kept


def test_padded_rows_undistort(tmp_path, ts):
    import cv2
    yy, xx = np.mgrid[0:16, 0:16]
    pixels = ((xx * 7 + yy * 3) % 256).astype(np.uint8)
    padded = np.full((16, 20), 255, np.uint8)
    padded[:, :16] = pixels
    raw, out = str(tmp_path / "raw.bag"), str(tmp_path / "out.bag")
    _bag(raw, ts, _imu_stamps(2), images=[(T0, "mono8", 16, 16, 20, padded)])
    rig = _rig(tmp_path=tmp_path)
    apply_calibration(raw, out, rig, _MAP, "/imu0", **{**_OFF, "undistort": True})
    [(_, m)] = _read(out, ts, "/cam0/image_raw")
    K = np.array([[20.0, 0, 15.5], [0, 21.0, 16.0], [0, 0, 1]])
    assert m.step == m.width == 16
    np.testing.assert_array_equal(m.data.reshape(16, 16),
                                  cv2.undistort(pixels, K, rig.cams[0].dist))


def test_non_mono8_undistort_passes_through_or_fails_strict(tmp_path, ts):
    raw = str(tmp_path / "raw.bag")
    data = np.arange(48) % 256
    _bag(raw, ts, _imu_stamps(2), images=[(T0, "bgr8", 4, 4, 12, data)])
    out = str(tmp_path / "out.bag")
    apply_calibration(raw, out, _rig(tmp_path=tmp_path), _MAP, "/imu0",
                      **{**_OFF, "undistort": True})
    [(_, m)] = _read(out, ts, "/cam0/image_raw")
    np.testing.assert_array_equal(m.data, data)
    with pytest.raises(ValueError, match="encoding"):
        apply_calibration(raw, str(tmp_path / "strict.bag"), _rig(tmp_path=tmp_path), _MAP,
                          "/imu0", **{**_OFF, "undistort": True}, strict=True)
    assert not (tmp_path / "strict.bag").exists()


def test_unsorted_imu_stamps_are_sorted_before_t_ga(tmp_path, ts):
    """Header stamps 0, 10, 5 ms (record order) with accel x = header ms: np.interp on the
    unsorted series was meaningless; sorted, a +2.5 ms shift gives x + 2.5 (end clamped)."""
    raw, out = str(tmp_path / "raw.bag"), str(tmp_path / "out.bag")
    stamps = [T0, T0 + 10_000_000, T0 + 5_000_000]
    _bag(raw, ts, stamps, accel_x=[0.0, 10.0, 5.0])
    apply_calibration(raw, out, _rig(t_ga=2.5e-3, tmp_path=tmp_path), {"/imu0": "imu0"},
                      "/imu0", **{**_OFF, "t_ga": True})
    got = {_stamp(m): m.linear_acceleration.x for _, m in _read(out, ts, "/imu0")}
    assert got == pytest.approx({T0: 2.5, T0 + 5_000_000: 7.5, T0 + 10_000_000: 10.0})


def test_duplicate_imu_stamps_kept_by_default_rejected_when_strict(tmp_path, ts):
    raw = str(tmp_path / "raw.bag")
    _bag(raw, ts, [T0, T0, T0 + 5_000_000])
    rig = _rig(t_ga=1e-3, tmp_path=tmp_path)
    s = apply_calibration(raw, str(tmp_path / "out.bag"), rig, {"/imu0": "imu0"}, "/imu0",
                          **{**_OFF, "t_ga": True})
    assert s["n_imu"] == 3
    with pytest.raises(ValueError, match="duplicate"):
        apply_calibration(raw, str(tmp_path / "strict.bag"), rig, {"/imu0": "imu0"}, "/imu0",
                          **{**_OFF, "t_ga": True}, strict=True)


def test_bias_only_without_intrinsics(tmp_path, ts):
    raw, out = str(tmp_path / "raw.bag"), str(tmp_path / "out.bag")
    _bag(raw, ts, _imu_stamps(1), accel_x=[10.0])
    rig = _rig(tmp_path=tmp_path)
    rig.base_imu.intr = ImuModel.from_tangent(np.r_[np.ones(6), np.zeros(18)], "rpng")
    apply_calibration(raw, out, rig, {"/imu0": "imu0"}, "/imu0",
                      **{**_OFF, "remove_bias": True})
    [(_, m)] = _read(out, ts, "/imu0")
    w, a = m.angular_velocity, m.linear_acceleration
    assert [w.x, w.y, w.z, a.x, a.y, a.z] == pytest.approx([2.0, 2.0, 2.0, 9.5, -0.5, 9.31])


def test_all_off_copies_bytes_and_keeps_each_connection(tmp_path, ts):
    raw, out = str(tmp_path / "raw.bag"), str(tmp_path / "out.bag")
    _bag(raw, ts, _imu_stamps(3), images=[(T0, "mono8", 2, 2, 2, np.zeros(4))], custom=True)
    apply_calibration(raw, out, _rig(tmp_path=tmp_path), _MAP, "/imu0", **_OFF)

    def dump(p):
        with Reader(p) as r:
            conns = sorted((c.topic, c.msgtype, c.msgdef.data, c.digest, c.ext.callerid,
                            c.ext.latching) for c in r.connections)
            return conns, [(c.topic, c.ext.callerid, t, bytes(raw)) for c, t, raw in r.messages()]
    assert dump(out) == dump(raw)
    assert sum(1 for c in dump(out)[0] if c[0] == "/custom") == 2   # two publishers


def test_missing_mapped_topic_lenient_by_default_strict_rejects(tmp_path, ts):
    raw = str(tmp_path / "raw.bag")
    _bag(raw, ts, _imu_stamps(2))                      # no camera topic in the bag
    out = str(tmp_path / "out.bag")
    s = apply_calibration(raw, out, _rig(tmp_path=tmp_path), _MAP, "/imu0")
    assert s["n_cam"] == 0 and s["corrected_topics"] == ["/imu0"]
    prov = yaml.safe_load(open(s["provenance"]))
    assert prov["corrected_topics"] == ["/imu0"]
    assert prov["corrections"] == {"remove_bias": True, "undistort": False,
                                   "imu_intrinsics": True, "td": True, "t_ga": True}
    assert prov["timeline"] == "base_imu_clock"
    with pytest.raises(ValueError, match="absent"):
        apply_calibration(raw, str(tmp_path / "strict.bag"), _rig(tmp_path=tmp_path), _MAP,
                          "/imu0", strict=True)
    assert not (tmp_path / "strict.bag").exists()


def test_provenance_records_disabled_operations(tmp_path, ts):
    raw, out = str(tmp_path / "raw.bag"), str(tmp_path / "out.bag")
    _bag(raw, ts, _imu_stamps(2))
    s = apply_calibration(raw, out, _rig(tmp_path=tmp_path), {"/imu0": "imu0"}, "/imu0",
                          **{**_OFF, "remove_bias": True}, provenance_extra={"tool": "x"})
    prov = yaml.safe_load(open(s["provenance"]))
    assert prov["corrections"] == {"remove_bias": True, "undistort": False,
                                   "imu_intrinsics": False, "td": False, "t_ga": False}
    assert prov["timeline"] == "original timestamps" and prov["tool"] == "x"
    assert prov["operation_counts"]["remove_bias"] == 2
