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
"""Task 4 — one SimConfig with gt_source + classmethod templates (sim-cleanup)."""
import pytest

from mvis.simulation.multi_board import MultiBoardSimConfig, SimConfig

TEMPLATES = ["realistic_single_board", "synthetic_multiboard_fisheye",
             "synthetic_multiboard_pinhole", "rolling_shutter", "two_imu"]


def test_simconfig_alias():
    assert SimConfig is MultiBoardSimConfig


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_builds(name):
    getattr(SimConfig, name)().build()      # constructs the scenario, no error


def test_gt_source_inference():
    assert SimConfig.realistic_single_board().gt_source == "captured"   # has traj_file
    assert SimConfig.synthetic_multiboard_fisheye().gt_source == "synthetic"


def test_rolling_shutter_and_two_imu():
    assert SimConfig.rolling_shutter().readout > 0
    assert SimConfig.two_imu().aux_enable is True
