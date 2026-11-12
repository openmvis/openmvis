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
"""P2 (gyro-only-imu): per-IMU `gyro_only` config option, threaded into aux_imus."""
import pytest
from mvis.core.config import CamImuRunConfig
from mvis.cli.main import _apply_imu_gyro_only


def test_aux_gyro_only_override_sets_flag():
    imu_topics = ["/imu/data", "/gx3_25/data", "/t265/imu"]
    aux = [{"name": "gx3_25"}, {"name": "t265"}]            # = imu_topics[1:]
    overrides = [{}, {}, {"gyro_only": True}]               # t265 (idx 2) gyro-only
    out = _apply_imu_gyro_only(imu_topics, aux, overrides)
    assert out[0].get("gyro_only") is not True              # gx3_25 stays full
    assert out[1].get("gyro_only") is True                  # t265 -> gyro-only


def test_base_gyro_only_raises():
    imu_topics = ["/imu/data", "/t265/imu"]
    aux = [{"name": "t265"}]
    with pytest.raises(ValueError, match="base"):
        _apply_imu_gyro_only(imu_topics, aux, [{"gyro_only": True}, {}])


def test_no_overrides_is_noop():
    aux = [{"name": "t265"}]
    out = _apply_imu_gyro_only(["/imu/data", "/t265/imu"], aux, [])
    assert out[0].get("gyro_only") is None


def test_config_parses_imus_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "input:\n  path: /x.bag\n  imu_topics: [/imu/data, /t265/imu]\n"
        "  imus:\n    - {}\n    - {gyro_only: true}\n"
        "camera_input: {path: /c.yaml}\n")
    cfg = CamImuRunConfig.from_yaml(str(p))
    assert cfg.input.imus[1]["gyro_only"] is True
