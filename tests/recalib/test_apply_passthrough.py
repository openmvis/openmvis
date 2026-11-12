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
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from mvis.imu import ImuModel
from mvis.recalib.calib_spec import RigCalib, ImuCalib, MVIS
from mvis.recalib.apply import apply_calibration


def _identity_rig():
    intr = ImuModel.from_tangent(np.zeros(24), "rpng")
    base = ImuCalib("imu0", "rpng", intr, np.zeros(3), np.zeros(3), 0.0, "base")
    cam = MVIS("cam0", 0.0, 0.0, "radtan", np.zeros(4), np.zeros(4))
    return RigCalib(base, [], [cam])


def test_identity_calib_preserves_counts_and_topics(tiny_bag, tmp_path):
    out = str(tmp_path / "corrected.bag")
    # The CLI (Task 8) sets rig._src_path to the real calib file so provenance
    # can hash it; simulate that here with a real file on disk.
    calib_file = tmp_path / "calib.yaml"
    calib_file.write_text("model: identity\n")
    rig = _identity_rig()
    rig._src_path = str(calib_file)
    summary = apply_calibration(
        tiny_bag, out, rig,
        mapping={"/imu0": "imu0", "/cam0/image_raw": "cam0"},
        base_imu_topic="/imu0", remove_bias=True)
    assert summary["n_imu"] == 20 and summary["n_cam"] == 4
    with Reader(out) as r:
        topics = {c.topic for c in r.connections}
        n = sum(1 for _ in r.messages())
    assert topics == {"/imu0", "/cam0/image_raw"}
    assert n == 24


def test_custom_type_passthrough_survives(tiny_bag, tmp_path, ts):
    """A passthrough topic whose message type the typestore does NOT know (e.g.
    realsense2_camera/msg/IMUInfo) must be copied through, not crash the writer
    on a missing msgdef. Regression: apply-calib re-generated every connection's
    def from the typestore, raising TypesysError on unknown custom types."""
    from rosbags.rosbag1 import Reader as R1, Writer as W1
    # append a custom-type topic to the raw bag (explicit msgdef + md5sum, so the
    # source bag itself carries no typestore-registered definition)
    raw2 = str(tmp_path / "raw_custom.bag")
    with R1(tiny_bag) as rin, W1(raw2) as w:
        conns = {}
        for conn, t, data in rin.messages():
            if conn.topic not in conns:
                conns[conn.topic] = w.add_connection(conn.topic, conn.msgtype, typestore=ts)
            w.write(conns[conn.topic], t, data)
        cc = w.add_connection("/t265/accel/imu_info", "realsense2_camera/msg/IMUInfo",
                              msgdef="", md5sum="a02adb3a99530b11ba18a16f40f9512a")
        w.write(cc, 1_000_000_500, b"\x00\x00\x00\x00")

    out = str(tmp_path / "corrected.bag")
    calib_file = tmp_path / "calib.yaml"; calib_file.write_text("model: identity\n")
    rig = _identity_rig(); rig._src_path = str(calib_file)
    summary = apply_calibration(
        raw2, out, rig, mapping={"/imu0": "imu0", "/cam0/image_raw": "cam0"},
        base_imu_topic="/imu0", remove_bias=True)
    assert summary["n_pass"] == 1                     # the custom topic passed through
    with Reader(out) as r:
        topics = {c.topic for c in r.connections}
    assert "/t265/accel/imu_info" in topics
