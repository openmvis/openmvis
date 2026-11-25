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
"""`mvis check` names the place from the gravity comment next to the source."""
from pathlib import Path

from mvis.doctor.check_config import check_config

ROOT = Path(__file__).resolve().parents[2]


def _gravity_msg(path):
    return next(f.message for f in check_config(str(path)) if f.id == "solve.gravity")


def test_check_names_the_place():
    assert _gravity_msg(ROOT / "examples/configs/tga_mvis/tga_gx3_25_01.yaml") == (
        "|g| 9.8013 m/s^2 (config, Newark, Delaware)")


def test_check_without_a_place(tmp_path):
    src = (ROOT / "examples/configs/tga_mvis/tga_gx3_25_01.yaml").read_text()
    p = tmp_path / "c.yaml"
    p.write_text(src.replace("# m/s², Newark, Delaware", "# m/s²"))
    assert _gravity_msg(p) == "|g| 9.8013 m/s^2 (config)"
