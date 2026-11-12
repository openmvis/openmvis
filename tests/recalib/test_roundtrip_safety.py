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
"""`mvis roundtrip` on a non-bag config points the user at the shipped converter,
not at a script outside the public package (F53)."""
from __future__ import annotations

import pytest

from mvis.recalib.roundtrip import run_roundtrip

EUROC_CONFIG = """# a EuRoC-folder cam-IMU config: nothing on disk is read before the format check
input:
  format: euroc
  path: /nonexistent/mav0
  imu_topics: ["/imu0"]
board: {board_type: apriltag, rows: 6, cols: 6, tag_size: 0.03, tag_spacing: 0.3}
camera_input: {path: /nonexistent/camchain.yaml}
output: {path: out/cam_imu.json}
"""


def test_euroc_config_is_sent_to_the_shipped_converter(tmp_path):
    cfg = tmp_path / "cam_imu_euroc.yaml"
    cfg.write_text(EUROC_CONFIG)
    with pytest.raises(ValueError) as e:
        run_roundtrip(str(cfg), out=str(tmp_path / "rt"))
    msg = str(e.value)
    assert "mvis datasets convert --dataset" in msg
    assert "euroc_folder_to_rosbag.convert_euroc_folder_to_rosbag" in msg
    assert "--bag" in msg
    assert "examples/experiments" not in msg
    # the named function exists and takes (src, out.bag)
    from mvis.datasets.converters.euroc_folder_to_rosbag import convert_euroc_folder_to_rosbag
    import inspect
    assert list(inspect.signature(convert_euroc_folder_to_rosbag).parameters)[:2] == \
        ["source_dir", "output_bag"]
    assert not (tmp_path / "rt").exists()      # refused before anything is written
