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
# tests/recalib/test_provenance.py
"""Standalone TDD test for recalib/provenance.py."""
import hashlib
import yaml
import pytest

from mvis.recalib.provenance import write_provenance


def test_write_provenance_roundtrip(tmp_path):
    # Create a dummy calib file with known bytes
    calib_bytes = b"fake calib content for sha256 test\n"
    calib_file = tmp_path / "calib.yaml"
    calib_file.write_bytes(calib_bytes)

    # Representative args
    out_bag = str(tmp_path / "output")
    source_bag = str(tmp_path / "source.bag")
    mapping = {"/cam0/image_raw": "/cam0/image_corrected",
               "/imu0": "/imu0_corrected"}
    corrected_topics = ["/cam0/image_corrected", "/imu0_corrected"]
    passthrough_topics = ["/gps/fix"]

    returned_path = write_provenance(
        out_bag,
        source_bag=source_bag,
        calib_path=str(calib_file),
        mapping=mapping,
        remove_bias=True,
        undistort=False,
        base_imu_topic="/imu0",
        corrected_topics=corrected_topics,
        passthrough_topics=passthrough_topics,
    )

    # Path checks
    expected_path = out_bag + ".recalib.yaml"
    assert returned_path == expected_path
    assert (tmp_path / "output.recalib.yaml").exists()

    # Load and verify contents
    with open(returned_path) as f:
        doc = yaml.safe_load(f)

    # schema_version present and correct
    assert doc["schema_version"] == 1

    # calib_sha256 matches independently computed hash
    expected_sha = hashlib.sha256(calib_bytes).hexdigest()
    assert doc["calib_sha256"] == expected_sha

    # timeline
    assert doc["timeline"] == "base_imu_clock"

    # corrections
    assert doc["corrections"]["remove_bias"] is True
    assert doc["corrections"]["undistort"] is False

    # td_convention contains expected substring
    assert "t_sensor + td" in doc["td_convention"]

    # topic_mapping, corrected_topics, passthrough_topics round-trip
    assert doc["topic_mapping"] == mapping
    assert doc["corrected_topics"] == corrected_topics
    assert doc["passthrough_topics"] == passthrough_topics


def test_write_provenance_missing_calib(tmp_path):
    """A nonexistent / sentinel calib_path must not crash; sha is None."""
    out_bag = str(tmp_path / "output")
    missing = str(tmp_path / "does_not_exist.yaml")

    returned_path = write_provenance(
        out_bag,
        source_bag=str(tmp_path / "source.bag"),
        calib_path=missing,
        mapping={"/imu0": "/imu0"},
        remove_bias=True,
        undistort=False,
        base_imu_topic="/imu0",
        corrected_topics=["/imu0"],
        passthrough_topics=[],
    )

    assert returned_path == out_bag + ".recalib.yaml"
    assert (tmp_path / "output.recalib.yaml").exists()
    with open(returned_path) as f:
        doc = yaml.safe_load(f)
    assert doc["calib_sha256"] is None

    # The literal "<unknown>" sentinel apply.py uses must also be tolerated.
    out_bag2 = str(tmp_path / "output2")
    returned2 = write_provenance(
        out_bag2,
        source_bag=str(tmp_path / "source.bag"),
        calib_path="<unknown>",
        mapping={"/imu0": "/imu0"},
        remove_bias=True,
        undistort=False,
        base_imu_topic="/imu0",
        corrected_topics=["/imu0"],
        passthrough_topics=[],
    )
    assert (tmp_path / "output2.recalib.yaml").exists()
    with open(returned2) as f:
        doc2 = yaml.safe_load(f)
    assert doc2["calib_sha256"] is None
