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
"""mvis.io.sensor_clock: the sidecar's writers and readers share one layout. Round trip through the
CLI reader and the restamp reader: tests/cam_imu/test_retime_map.py."""
import json

import numpy as np
import pytest

from mvis.io.sensor_clock import (
    FORMAT, camera_record, load_retime_map, read_sidecar, write_sensor_clock_sidecar, write_sidecar,
)

REP = {"retimed": True, "period_s": 0.05, "phase_s": 0.001, "rate_hz": 20.0, "coherence": 0.9, "scatter_ms": 0.1234567,
       "retime_map": [[1.0, 1.1], [2.0, 2.1]], "dropped_stamps": [1.5]}


def test_camera_record_layout():
    assert camera_record(REP, "/cam0") == {
        "topic": "/cam0", "period_s": 0.05, "phase_s": 0.001, "rate_hz": 20.0, "coherence": 0.9,
        "scatter_ms": 0.1234567, "map": [[1.0, 1.1], [2.0, 2.1]], "dropped_stamps": [1.5]}
    rep = {k: v for k, v in REP.items() if k != "dropped_stamps"}
    assert camera_record(rep)["topic"] is None and camera_record(rep)["dropped_stamps"] == []


def test_camera_record_accepts_numpy_maps():
    rec = camera_record({"retime_map": np.array([[1.0, 2.0]]), "dropped_stamps": np.array([3.0])})
    assert json.dumps(rec)                                     # plain floats
    assert rec["map"] == [[1.0, 2.0]] and rec["dropped_stamps"] == [3.0]


def test_write_sidecar_is_the_pipeline_dump_bytes(tmp_path):
    """Same bytes as the literal dump pipeline_cam_imu used to write."""
    p = tmp_path / "s.json"
    write_sidecar(p, {"0": camera_record(REP)})
    old = {"format": "mvis-sensor-clock/1", "cameras": {
        "0": {"topic": None, "period_s": REP.get("period_s"), "phase_s": REP.get("phase_s"),
              "rate_hz": REP.get("rate_hz"), "coherence": REP.get("coherence"),
              "scatter_ms": REP.get("scatter_ms"), "map": REP["retime_map"],
              "dropped_stamps": REP.get("dropped_stamps", [])}}}
    assert p.read_text() == json.dumps(old)
    assert FORMAT == "mvis-sensor-clock/1"


def test_read_sidecar_checks_the_format_only_when_asked(tmp_path):
    p = tmp_path / "other.json"
    p.write_text('{"cameras": {}}')
    assert read_sidecar(p) == {"cameras": {}}
    with pytest.raises(ValueError, match="not a sensor-clock sidecar"):
        read_sidecar(p, check_format=True)
    p.write_text('{"format": "mvis-sensor-clock/2", "cameras": {}}')
    assert read_sidecar(p, check_format=True)["format"].endswith("/2")


def test_write_sensor_clock_sidecar_splits_compact_and_maps(tmp_path):
    report = {0: REP, 1: {"retimed": False, "reason": "gs"}}
    p = tmp_path / "sub" / "s.json"
    compact = write_sensor_clock_sidecar(report, str(p), ["/a"])
    assert compact["0"]["sidecar"] == str(p) and "retime_map" not in compact["0"]
    assert "dropped_stamps" not in compact["0"] and compact["0"]["scatter_ms"] == 0.123457
    assert compact["1"] == {"retimed": False, "reason": "gs"}
    doc = json.loads(p.read_text())
    assert doc["format"] == FORMAT and list(doc["cameras"]) == ["0"] and doc["cameras"]["0"]["topic"] == "/a"
    assert write_sensor_clock_sidecar(report, None)["0"].get("sidecar") is None
    assert REP["retime_map"]                                    # the input report is left alone


def test_load_retime_map_returns_the_camera_or_raises(tmp_path):
    p = tmp_path / "s.json"
    write_sidecar(p, {"0": camera_record(REP)})
    assert np.array_equal(load_retime_map(str(p), 0), [[1.0, 1.1], [2.0, 2.1]])
    with pytest.raises(ValueError, match=r"no retime map for camera 3 \(has \['0'\]\)"):
        load_retime_map(str(p), 3)
