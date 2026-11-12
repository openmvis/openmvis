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
from types import SimpleNamespace
from mvis.recalib.restamp import shifted_time_ns, set_header_stamp_ns


def test_shift_positive_and_negative():
    assert shifted_time_ns(1_000_000_000, 0.002) == 1_002_000_000
    assert shifted_time_ns(1_000_000_000, -0.0015) == 998_500_000


def test_set_header_stamp_sec_nanosec():
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)))
    set_header_stamp_ns(msg, 1_002_000_000)
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 2_000_000
