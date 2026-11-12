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
"""Tests for mvis.core.system_info (the report footer's machine line)."""

from mvis.core.system_info import get_system_info, get_system_summary


class TestSystemInfo:

    def test_get_system_info_keys(self):
        info = get_system_info()
        assert "platform" in info
        assert "cpu_cores_logical" in info
        assert "cpu_cores_physical" in info
        assert "ram_gb" in info
        assert "python_version" in info
        assert "gtsam_threads" in info

    def test_system_info_values(self):
        info = get_system_info()
        assert info["cpu_cores_logical"] >= 1
        assert info["cpu_cores_physical"] >= 1
        assert info["ram_gb"] > 0
        assert info["platform"] in ("linux", "darwin", "windows")

    def test_get_system_summary(self):
        summary = get_system_summary()
        assert isinstance(summary, str)
        assert "cores" in summary.lower()
        assert "GB RAM" in summary
