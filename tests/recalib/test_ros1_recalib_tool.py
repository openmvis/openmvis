# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
# Copyright (C) 2026 Yuantai-Z
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
# SPDX-FileCopyrightText: 2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-only
"""Independent bag-level checks for the additive ROS1 correction tool."""
from dataclasses import replace
from pathlib import Path
import hashlib
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import yaml
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

from tools.ros1_recalib.export import CorrectionOptions, export_ros1


_REPO = Path(__file__).resolve().parents[2]
_CONTROLS = ("imu_intrinsics", "remove_bias", "td", "t_ga", "undistort")
_OFF = CorrectionOptions(**dict.fromkeys(_CONTROLS, False))


@pytest.fixture
def capture(tmp_path):
    """Raw data with independently calculable scale, bias and timing effects."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Image = ts.types["sensor_msgs/msg/Image"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    Vector = ts.types["geometry_msgs/msg/Vector3"]

    def header(stamp, frame):
        return Header(0, Time(stamp // 10**9, stamp % 10**9), frame)

    bag = tmp_path / "raw.bag"
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    padded = np.full((8, 12), 255, dtype=np.uint8)
    padded[:, :8] = pixels
    custom_def = "float64 value\n"
    with Writer(bag) as writer:
        ci = writer.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        cc = writer.add_connection("/cam0/image_raw", Image.__msgtype__, typestore=ts)
        cx = writer.add_connection(
            "/custom", "recalib_test/msg/Scalar", msgdef=custom_def,
            md5sum=hashlib.md5(custom_def.strip().encode()).hexdigest(),
            latching=1, callerid="/fixture_node")
        for k in range(3):
            stamp = 10**9 + k * 10_000_000
            msg = Imu(
                header(stamp, "imu"), Quaternion(0.0, 0.0, 0.0, 1.0),
                np.zeros(9), Vector(3.0, 4.0, 5.0), np.zeros(9),
                Vector(10.0 + 2 * k, 20.0 + 4 * k, 30.0 + 6 * k), np.zeros(9))
            writer.write(ci, stamp + 7_000_000, ts.serialize_ros1(msg, Imu.__msgtype__))
        msg = Image(header(1_005_000_000, "camera"), 8, 8, "mono8", 0, 12,
                    padded.reshape(-1))
        writer.write(cc, 1_016_000_000, ts.serialize_ros1(msg, Image.__msgtype__))
        writer.write(cx, 1_001_000_000, struct.pack("<d", 7.0))

    # RPNG's upper-triangular diagonal positions are 0, 2 and 5 in each
    # six-vector. Tangent values add to identity; all cross terms are zero.
    xi = np.zeros(24)
    xi[[0, 2, 5]] = [1.0, 2.0, 3.0]       # Dw = diag(2, 3, 4)
    xi[[6, 8, 11]] = [4.0, 5.0, 6.0]     # Da = diag(5, 6, 7)
    calibration = {
        "imus": [{"name": "imu0", "role": "base", "model": "rpng",
                  "imu_intr_xi": xi.tolist(), "b_g": [1.0, 0.5, -0.25],
                  "b_a": [1.0, 2.0, 3.0], "topic": "/imu0", "t_ga": 0.005}],
        "cameras": [{"name": "cam0", "rostopic": "/cam0/image_raw", "td": 0.02,
                     "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 3.5, "cy": 3.5},
                     "distortion": {"model": "radtan", "coeffs": [0.0] * 4}}],
    }
    calib = tmp_path / "calib.yaml"
    calib.write_text(yaml.safe_dump(calibration))
    return {"bag": bag, "calib": calib, "ts": ts, "pixels": pixels}


def _records(path):
    with Reader(path) as reader:
        return [(c.topic, c.msgtype, t, bytes(raw)) for c, t, raw in reader.messages()]


def _connections(path):
    with Reader(path) as reader:
        return sorted((c.topic, c.msgtype, c.msgdef.data, c.digest,
                       c.ext.latching, c.ext.callerid) for c in reader.connections)


def _messages(path, topic, ts):
    return [(t, ts.deserialize_ros1(raw, kind))
            for name, kind, t, raw in _records(path) if name == topic]


def _stamp(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def _vector(value):
    return [value.x, value.y, value.z]


def test_all_operations_off_preserves_payloads_clocks_and_connection_metadata(capture, tmp_path):
    output = tmp_path / "unchanged.bag"
    summary = export_ros1(capture["calib"], capture["bag"], output, options=_OFF)
    assert output.is_file()
    assert _records(output) == _records(capture["bag"])
    assert _connections(output) == _connections(capture["bag"])
    assert summary


def test_missing_mapped_topic_rejected_without_output(capture, tmp_path):
    output = tmp_path / "missing.bag"
    with pytest.raises(ValueError):
        export_ros1(capture["calib"], capture["bag"], output,
                    mapping={"/absent_imu": "imu0", "/cam0/image_raw": "cam0"},
                    base_imu_topic="/absent_imu")
    assert not output.exists()
    assert not Path(str(output) + ".recalib.yaml").exists()


def test_same_topic_multiple_publishers_keep_distinct_connections(capture, tmp_path):
    source = tmp_path / "multiple-publishers.bag"
    with Reader(capture["bag"]) as reader, Writer(source) as writer:
        connections = {
            c.id: writer.add_connection(
                c.topic, c.msgtype, msgdef=c.msgdef.data, md5sum=c.digest,
                callerid=c.ext.callerid, latching=c.ext.latching)
            for c in reader.connections
        }
        custom = next(c for c in reader.connections if c.topic == "/custom")
        other = writer.add_connection(
            custom.topic, custom.msgtype, msgdef=custom.msgdef.data,
            md5sum=custom.digest, callerid="/other_publisher", latching=0)
        for connection, timestamp, data in reader.messages():
            writer.write(connections[connection.id], timestamp, data)
        writer.write(other, 1_002_000_000, struct.pack("<d", 8.0))

    output = tmp_path / "multiple-publishers-corrected.bag"
    export_ros1(capture["calib"], source, output, options=_OFF)
    assert _connections(output) == _connections(source)

    def custom_records(path):
        with Reader(path) as reader:
            return [(t, bytes(raw), c.ext.callerid, c.ext.latching)
                    for c, t, raw in reader.messages() if c.topic == "/custom"]

    assert custom_records(output) == custom_records(source)


def test_padded_mono8_undistortion_exports_a_readable_bag(capture, tmp_path):
    output = tmp_path / "undistorted.bag"
    export_ros1(capture["calib"], capture["bag"], output,
                options=replace(_OFF, undistort=True))
    [(record_time, image)] = _messages(output, "/cam0/image_raw", capture["ts"])
    assert record_time == 1_016_000_000
    assert _stamp(image) == 1_005_000_000
    assert image.encoding == "mono8"
    assert image.height == image.width == image.step == 8
    np.testing.assert_array_equal(image.data.reshape(8, 8), capture["pixels"])


def test_requested_unsupported_undistortion_publishes_no_outputs(capture, tmp_path):
    calibration = yaml.safe_load(capture["calib"].read_text())
    calibration["cameras"][0]["distortion"] = {
        "model": "double_sphere", "coeffs": [0.5, 0.5]}
    capture["calib"].write_text(yaml.safe_dump(calibration))
    output = tmp_path / "unsupported.bag"
    with pytest.raises(ValueError):
        export_ros1(capture["calib"], capture["bag"], output,
                    options=replace(_OFF, undistort=True))
    assert not output.exists()
    assert not Path(str(output) + ".recalib.yaml").exists()


@pytest.mark.parametrize("intrinsics,bias,expected_w,expected_a", [
    (False, False, [3.0, 4.0, 5.0], [10.0, 20.0, 30.0]),
    (True, False, [6.0, 12.0, 20.0], [50.0, 120.0, 210.0]),
    (False, True, [2.0, 3.5, 5.25], [9.0, 18.0, 27.0]),
    (True, True, [4.0, 10.5, 21.0], [45.0, 108.0, 189.0]),
])
def test_intrinsics_and_bias_are_independent_preserving_record_latency(
        capture, tmp_path, intrinsics, bias, expected_w, expected_a):
    output = tmp_path / "numeric.bag"
    export_ros1(capture["calib"], capture["bag"], output,
                options=replace(_OFF, imu_intrinsics=intrinsics, remove_bias=bias))
    records = _messages(output, "/imu0", capture["ts"])
    time, msg = records[0]
    np.testing.assert_allclose(_vector(msg.angular_velocity), expected_w)
    np.testing.assert_allclose(_vector(msg.linear_acceleration), expected_a)
    assert time == 1_007_000_000
    assert _stamp(msg) == 10**9
    assert all(record_time - _stamp(message) == 7_000_000
               for record_time, message in records)


@pytest.mark.parametrize("enabled,expected_a", [
    (False, [[10.0, 20.0, 30.0], [12.0, 24.0, 36.0], [14.0, 28.0, 42.0]]),
    (True, [[11.0, 22.0, 33.0], [13.0, 26.0, 39.0], [14.0, 28.0, 42.0]]),
])
def test_t_ga_interpolates_acceleration_only_when_selected(capture, tmp_path, enabled, expected_a):
    output = tmp_path / "accelerometer.bag"
    export_ros1(capture["calib"], capture["bag"], output,
                options=replace(_OFF, t_ga=enabled))
    messages = _messages(output, "/imu0", capture["ts"])
    np.testing.assert_allclose([_vector(m.linear_acceleration) for _, m in messages], expected_a)
    np.testing.assert_allclose([_vector(m.angular_velocity) for _, m in messages],
                               [[3.0, 4.0, 5.0]] * 3)
    assert [t for t, _ in messages] == [1_007_000_000, 1_017_000_000, 1_027_000_000]


def test_td_uses_header_clock_and_globally_orders_output(capture, tmp_path):
    output = tmp_path / "shifted.bag"
    export_ros1(capture["calib"], capture["bag"], output, options=replace(_OFF, td=True))
    [(time, image)] = _messages(output, "/cam0/image_raw", capture["ts"])
    assert time == _stamp(image) == 1_025_000_000
    records = _records(output)
    assert [t for _, _, t, _ in records] == sorted(t for _, _, t, _ in records)
    custom = [r for r in records if r[0] == "/custom"]
    assert custom == [r for r in _records(capture["bag"]) if r[0] == "/custom"]


@pytest.mark.parametrize("target", ["existing", "input", "input_symlink", "calibration"])
def test_protected_destinations_are_unchanged(capture, tmp_path, target):
    if target == "existing":
        output = tmp_path / "occupied.bag"
        output.write_bytes(b"preexisting output must survive")
    elif target == "input":
        output = capture["bag"]
    elif target == "input_symlink":
        output = tmp_path / "input-link.bag"
        output.symlink_to(capture["bag"])
    else:
        output = capture["calib"]
    previous_output = output.read_bytes()
    previous_input = capture["bag"].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        export_ros1(capture["calib"], capture["bag"], output, options=_OFF)
    assert output.read_bytes() == previous_output
    assert capture["bag"].read_bytes() == previous_input


@pytest.mark.parametrize("config_value,cli_value", [(True, False), (False, True)])
def test_cli_overrides_all_config_controls_and_resolves_relative_paths(
        capture, tmp_path, config_value, cli_value):
    config = tmp_path / "run.yaml"
    output = tmp_path / "cli.bag"
    config.write_text(yaml.safe_dump({
        "calib": capture["calib"].name, "input": capture["bag"].name,
        "output": output.name,
        "corrections": dict.fromkeys(_CONTROLS, config_value),
    }))
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    flags = ["--" + ("" if cli_value else "no-") + name.replace("_", "-")
             for name in _CONTROLS]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    completed = subprocess.run(
        [sys.executable, str(_REPO / "tools/ros1_recalib/run.py"),
         "--config", str(config), *flags], cwd=elsewhere, env=env,
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file()
    provenance = yaml.safe_load(Path(str(output) + ".recalib.yaml").read_text())
    assert provenance["requested_options"] == dict.fromkeys(_CONTROLS, cli_value)
    assert provenance["calib_sha256"] == hashlib.sha256(capture["calib"].read_bytes()).hexdigest()
    if not cli_value:
        assert _records(output) == _records(capture["bag"])
        assert _connections(output) == _connections(capture["bag"])
    else:
        [(time, image)] = _messages(output, "/cam0/image_raw", capture["ts"])
        assert time == _stamp(image) == 1_025_000_000
        assert image.step == 8
        np.testing.assert_array_equal(image.data.reshape(8, 8), capture["pixels"])
        imu_time, imu = _messages(output, "/imu0", capture["ts"])[0]
        assert imu_time == _stamp(imu) == 10**9
        np.testing.assert_allclose(_vector(imu.angular_velocity), [4.0, 10.5, 21.0])
        np.testing.assert_allclose(_vector(imu.linear_acceleration), [50.0, 120.0, 210.0])


def test_cli_explicit_paths_export_without_config(capture, tmp_path):
    output = tmp_path / "explicit.bag"
    completed = subprocess.run(
        [sys.executable, str(_REPO / "tools/ros1_recalib/run.py"),
         "--calib", str(capture["calib"]), "--input", str(capture["bag"]),
         "--output", str(output),
         *["--no-" + name.replace("_", "-") for name in _CONTROLS]],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _records(output) == _records(capture["bag"])


def test_config_controls_apply_without_flags_and_cli_output_overrides_config(capture, tmp_path):
    config = tmp_path / "config-only.yaml"
    config.write_text(yaml.safe_dump({
        "calib": capture["calib"].name, "input": capture["bag"].name,
        "output": "unused.bag", "corrections": dict.fromkeys(_CONTROLS, False),
    }))
    elsewhere = tmp_path / "cli-working-directory"
    elsewhere.mkdir()
    completed = subprocess.run(
        [sys.executable, str(_REPO / "tools/ros1_recalib/run.py"),
         "--config", str(config), "--output", "selected.bag"],
        cwd=elsewhere, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _records(elsewhere / "selected.bag") == _records(capture["bag"])
    assert not (tmp_path / "unused.bag").exists()


@pytest.mark.parametrize("corrections", [
    {"td": "false"}, {"unknown_operation": False}, {"td": None},
])
def test_cli_rejects_ambiguous_or_unknown_controls_without_output(capture, tmp_path, corrections):
    output = tmp_path / "invalid.bag"
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump({
        "calib": str(capture["calib"]), "input": str(capture["bag"]),
        "output": str(output), "corrections": corrections,
    }))
    completed = subprocess.run(
        [sys.executable, str(_REPO / "tools/ros1_recalib/run.py"), "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert not output.exists()


# --- review follow-ups: undistortion, t_ga edges, publish, parity ---

def _types(ts):
    T = ts.types
    return (T["sensor_msgs/msg/Imu"], T["sensor_msgs/msg/Image"],
            T["sensor_msgs/msg/CompressedImage"], T["std_msgs/msg/Header"],
            T["builtin_interfaces/msg/Time"], T["geometry_msgs/msg/Quaternion"],
            T["geometry_msgs/msg/Vector3"])


def _write_custom_bag(path, ts, imu_rows=(), images=(), compressed=()):
    """imu_rows: (record_ns, header_ns, accel); images: (record_ns, header_ns,
    encoding, h, w, step, data); compressed: (record_ns, header_ns, bytes)."""
    Imu, Image, Compressed, Header, Time, Quaternion, Vector = _types(ts)

    def header(stamp, frame):
        return Header(0, Time(stamp // 10**9, stamp % 10**9), frame)

    with Writer(path) as writer:
        ci = writer.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        if images:
            cc = writer.add_connection("/cam0/image_raw", Image.__msgtype__, typestore=ts)
        if compressed:
            cz = writer.add_connection("/cam0/image_raw", Compressed.__msgtype__,
                                       typestore=ts)
        for record, stamp, accel in imu_rows:
            msg = Imu(header(stamp, "imu"), Quaternion(0.0, 0.0, 0.0, 1.0),
                      np.zeros(9), Vector(3.0, 4.0, 5.0), np.zeros(9),
                      Vector(*map(float, accel)), np.zeros(9))
            writer.write(ci, record, ts.serialize_ros1(msg, Imu.__msgtype__))
        for record, stamp, enc, h, w, step, data in images:
            msg = Image(header(stamp, "camera"), h, w, enc, 0, step,
                        np.asarray(data, dtype=np.uint8).reshape(-1))
            writer.write(cc, record, ts.serialize_ros1(msg, Image.__msgtype__))
        for record, stamp, payload in compressed:
            msg = Compressed(header(stamp, "camera"), "jpeg",
                             np.frombuffer(payload, dtype=np.uint8))
            writer.write(cz, record, ts.serialize_ros1(msg, Compressed.__msgtype__))


def _set_calib(calib_path, **camera):
    doc = yaml.safe_load(calib_path.read_text())
    doc["cameras"][0].update(camera)
    calib_path.write_text(yaml.safe_dump(doc))


_IMU_ROWS = [(10**9 + k * 10_000_000, 10**9 + k * 10_000_000, [k, 0.0, 0.0])
             for k in range(3)]


@pytest.mark.parametrize("model,coeffs", [
    ("radtan", [-0.2, 0.05, 0.001, 0.0]),
    ("equidistant", [0.1, -0.02, 0.003, 0.0]),
])
def test_undistortion_matches_opencv_reference(capture, tmp_path, model, coeffs):
    import cv2
    ts = capture["ts"]
    yy, xx = np.mgrid[0:32, 0:32]
    pixels = ((xx * 7 + yy * 3) % 256).astype(np.uint8)
    source = tmp_path / "gradient.bag"
    _write_custom_bag(source, ts, imu_rows=_IMU_ROWS, images=[
        (1_010_000_000, 1_005_000_000, "mono8", 32, 32, 32, pixels)])
    _set_calib(capture["calib"],
               intrinsics={"fx": 20.0, "fy": 21.0, "cx": 15.5, "cy": 16.0},
               distortion={"model": model, "coeffs": coeffs})
    output = tmp_path / "undistorted.bag"
    export_ros1(capture["calib"], source, output,
                options=replace(_OFF, undistort=True))
    [(_, image)] = _messages(output, "/cam0/image_raw", ts)
    K = np.array([[20.0, 0.0, 15.5], [0.0, 21.0, 16.0], [0.0, 0.0, 1.0]])
    D = np.array(coeffs)
    if model == "radtan":
        expected = cv2.undistort(pixels, K, D)
    else:
        expected = cv2.fisheye.undistortImage(pixels, K, D, Knew=K)
    assert not np.array_equal(expected, pixels)
    np.testing.assert_array_equal(image.data.reshape(32, 32), expected)


def test_non_mono8_undistort_rejected_without_output(capture, tmp_path):
    ts = capture["ts"]
    source = tmp_path / "color.bag"
    _write_custom_bag(source, ts, imu_rows=_IMU_ROWS, images=[
        (1_010_000_000, 1_005_000_000, "bgr8", 4, 4, 12, np.zeros(48))])
    output = tmp_path / "color-out.bag"
    with pytest.raises(ValueError, match="encoding"):
        export_ros1(capture["calib"], source, output,
                    options=replace(_OFF, undistort=True))
    assert not output.exists()
    assert not Path(str(output) + ".recalib.yaml").exists()


def test_t_ga_negative_offset_sorts_headers_and_counts_edge_clamps(capture, tmp_path):
    """Header stamps out of record order are sorted before interpolation;
    a negative t_ga clamps the first sample at the start of the series."""
    ts = capture["ts"]
    # record order: headers 0, 20, 10 ms; accel x = header ms
    rows = [(1_001_000_000, 10**9, [0.0, 0.0, 0.0]),
            (1_022_000_000, 10**9 + 20_000_000, [20.0, 0.0, 0.0]),
            (1_023_000_000, 10**9 + 10_000_000, [10.0, 0.0, 0.0])]
    source = tmp_path / "unordered.bag"
    _write_custom_bag(source, ts, imu_rows=rows, images=[
        (1_010_000_000, 1_005_000_000, "mono8", 2, 2, 2, np.zeros(4))])
    doc = yaml.safe_load(capture["calib"].read_text())
    doc["imus"][0]["t_ga"] = -0.005
    capture["calib"].write_text(yaml.safe_dump(doc))
    output = tmp_path / "unordered-out.bag"
    export_ros1(capture["calib"], source, output, options=replace(_OFF, t_ga=True))
    got = {_stamp(m): m.linear_acceleration.x
           for _, m in _messages(output, "/imu0", ts)}
    # a'(t) = a(t - 5 ms): 0 ms clamps to 0; 10 ms -> 5; 20 ms -> 15
    assert got == pytest.approx({10**9: 0.0, 10**9 + 10_000_000: 5.0,
                                 10**9 + 20_000_000: 15.0})
    prov = yaml.safe_load(Path(str(output) + ".recalib.yaml").read_text())
    assert prov["t_ga"]["topics"]["/imu0"]["edge_clamped_messages"] == 1


def test_t_ga_duplicate_header_stamps_rejected_without_output(capture, tmp_path):
    ts = capture["ts"]
    rows = [(1_001_000_000, 10**9, [0.0, 0.0, 0.0]),
            (1_002_000_000, 10**9, [0.0, 0.0, 0.0]),
            (1_011_000_000, 10**9 + 10_000_000, [1.0, 0.0, 0.0])]
    source = tmp_path / "duplicate.bag"
    _write_custom_bag(source, ts, imu_rows=rows, images=[
        (1_010_000_000, 1_005_000_000, "mono8", 2, 2, 2, np.zeros(4))])
    output = tmp_path / "duplicate-out.bag"
    with pytest.raises(ValueError, match="duplicate"):
        export_ros1(capture["calib"], source, output, options=replace(_OFF, t_ga=True))
    assert not output.exists()


def test_compressed_image_td_only_restamps_header_and_keeps_payload(capture, tmp_path):
    ts = capture["ts"]
    payload = bytes(range(50))
    source = tmp_path / "compressed.bag"
    _write_custom_bag(source, ts, imu_rows=_IMU_ROWS,
                      compressed=[(1_016_000_000, 1_005_000_000, payload)])
    output = tmp_path / "compressed-out.bag"
    export_ros1(capture["calib"], source, output, options=replace(_OFF, td=True))
    [(time, msg)] = _messages(output, "/cam0/image_raw", ts)
    assert time == _stamp(msg) == 1_025_000_000
    assert msg.format == "jpeg" and bytes(msg.data) == payload
    with pytest.raises(ValueError, match="sensor_msgs/msg/Image"):
        export_ros1(capture["calib"], source, tmp_path / "no.bag",
                    options=replace(_OFF, undistort=True))


@pytest.mark.parametrize("err", ["EPERM", "EOPNOTSUPP"])
def test_publish_falls_back_to_exclusive_copy_without_hard_links(
        capture, tmp_path, monkeypatch, err):
    import errno
    from tools.ros1_recalib import export as export_mod

    def no_link(src, dst):
        raise OSError(getattr(errno, err), "hard links unsupported")

    monkeypatch.setattr(export_mod.os, "link", no_link)
    output = tmp_path / "copied.bag"
    export_ros1(capture["calib"], capture["bag"], output, options=_OFF)
    assert _records(output) == _records(capture["bag"])
    assert Path(str(output) + ".recalib.yaml").is_file()
    # the copy path still refuses to replace a file created after preflight
    staged = tmp_path / "staged"
    staged.write_bytes(b"new")
    occupied = tmp_path / "occupied"
    occupied.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        export_mod._publish_one(staged, occupied)
    assert occupied.read_bytes() == b"keep"


def test_publish_rolls_back_bag_when_sidecar_target_appears(capture, tmp_path, monkeypatch):
    from tools.ros1_recalib import export as export_mod
    output = tmp_path / "race.bag"
    sidecar = Path(str(output) + ".recalib.yaml")
    real_publish_one = export_mod._publish_one

    def racing(staged, target):
        if target == sidecar:
            sidecar.write_text("someone else")
        return real_publish_one(staged, target)

    monkeypatch.setattr(export_mod, "_publish_one", racing)
    with pytest.raises(FileExistsError):
        export_ros1(capture["calib"], capture["bag"], output, options=_OFF)
    assert not output.exists()
    assert sidecar.read_text() == "someone else"


def test_imu_and_camera_corrections_match_library_apply_calibration(capture, tmp_path):
    """The standalone exporter and mvis.recalib.apply share td/t_ga/IMU
    conventions; on the same bag every mapped message must agree."""
    from mvis.recalib.apply import apply_calibration
    from mvis.recalib.calib_spec import load_rig_calib

    ts = capture["ts"]
    tool_out = tmp_path / "tool.bag"
    export_ros1(capture["calib"], capture["bag"], tool_out,
                options=CorrectionOptions())
    lib_out = tmp_path / "lib.bag"
    rig = load_rig_calib(str(capture["calib"]))
    apply_calibration(str(capture["bag"]), str(lib_out), rig,
                      {"/imu0": "imu0", "/cam0/image_raw": "cam0"}, "/imu0")

    for topic in ("/imu0", "/cam0/image_raw"):
        tool_msgs = _messages(tool_out, topic, ts)
        lib_msgs = _messages(lib_out, topic, ts)
        assert [t for t, _ in tool_msgs] == [t for t, _ in lib_msgs]
        assert [_stamp(m) for _, m in tool_msgs] == [_stamp(m) for _, m in lib_msgs]
        if topic == "/imu0":
            np.testing.assert_allclose(
                [_vector(m.angular_velocity) for _, m in tool_msgs],
                [_vector(m.angular_velocity) for _, m in lib_msgs], atol=1e-12)
            np.testing.assert_allclose(
                [_vector(m.linear_acceleration) for _, m in tool_msgs],
                [_vector(m.linear_acceleration) for _, m in lib_msgs], atol=1e-12)
