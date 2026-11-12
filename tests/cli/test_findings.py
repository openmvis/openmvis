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
"""mvis.doctor.findings: the worst-status fold and the finding line equal the three folds and
two print lines they replace (inspect.Report.finish, `mvis check`, verdict.Verdict.finish)."""
from __future__ import annotations

import itertools

from mvis.analysis.verdict import Verdict
from mvis.doctor.findings import Finding, format_finding, worst_status
from mvis.doctor.inspect import Finding as InspectFinding, Report

STATUSES = ("pass", "warn", "fail", None, "other")


def _combos():
    for n in range(0, 4):
        yield from itertools.product(STATUSES, repeat=n)


def test_worst_status_equals_the_old_folds():
    for combo in _combos():
        st = set(combo)
        old_inspect = "fail" if "fail" in st else "warn" if "warn" in st else "pass"
        old_check = ("fail" if any(s == "fail" for s in combo) else
                     "warn" if any(s == "warn" for s in combo) else "pass")
        old_verdict = "RED" if "fail" in st else "YELLOW" if "warn" in st else "GREEN"
        assert worst_status(combo) == old_inspect == old_check
        rep = Report("p", "ros1", [])
        for i, s in enumerate(combo):
            rep.find(f"id{i}", "subj", s, "msg")
        assert rep.finish().status == old_inspect
        v = Verdict("cam")
        for i, s in enumerate(combo):
            v.add(f"id{i}", "subj", 1.0, "thr", s)
        assert v.finish().grade == old_verdict


def test_format_finding_line():
    for f in (Finding("imu.accel_units", "/imu0", "warn", "m/s^2?"),
              Finding("a" * 30, "", "pass", ""), Finding("x", "s", "fail", "boom")):
        assert format_finding(f) == f"[{f.status.upper():<4}] {f.id:<22} {f.subject}: {f.message}"


def test_inspect_still_exports_finding():
    assert InspectFinding is Finding
