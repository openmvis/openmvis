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
"""`mvis inspect` reads ROS1 stamps the way the readers do (cleanup plan PR F, F5): the
header stamp from the raw bytes (`io.ros_stamp.header_stamp_s`), and the bag record time
when the stamp was never set (0), so a zero first stamp no longer reports a start of 0 s
and a bogus IMU rate."""
from mvis.doctor.inspect import Report, inspect_ros1
from tests.io.test_cam_imu_stamp_fallback import REC0, _bag


def _streams(path):
    rep = Report(path=path, format="ros1", supported_modes=[])
    inspect_ros1(path, rep)
    return {s.name: s for s in rep.streams}


def test_zero_first_stamps_report_the_record_time(tmp_path):
    p = str(tmp_path / "z.bag")
    _bag(p, zero_imu={0}, zero_img={0})
    s = _streams(p)
    assert s["/cam0"].start_s == REC0 / 1e9      # was 0.0
    assert s["/imu0"].start_s == REC0 / 1e9
    assert s["/imu0"].rate_hz == 200.0


def test_set_stamps_are_the_header_stamps(tmp_path):
    p = str(tmp_path / "s.bag")
    _bag(p)
    s = _streams(p)
    hdr = REC0 + 3_000_000
    assert s["/cam0"].start_s == hdr // 10**9 + 1e-9 * (hdr % 10**9)   # the old expression
    assert s["/imu0"].start_s == s["/cam0"].start_s
