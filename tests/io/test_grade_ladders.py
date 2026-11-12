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
"""verdict.grade(strict=) plus the display maps equal the three grade ladders they replace
(verdict._grade, cam_imu_report._grade, compare.band_for_pct), and the report_shell badge,
stat_box and html_table give the cam-IMU report's old HTML."""
from __future__ import annotations

import math

import numpy as np

from mvis.analysis import cam_imu_report
from mvis.analysis.report_shell import badge, html_table, stat_box
from mvis.analysis.verdict import _grade as verdict_grade, grade


def _old_verdict(value, good, marginal):
    if value is None or value != value:
        return None
    return "pass" if value <= good else "warn" if value <= marginal else "fail"


def _old_report(value, good, marginal):
    if value is None or value != value:
        return None
    return "good" if value <= good else "marginal" if value <= marginal else "poor"


def _old_band(pct):
    if pct is None:
        return None
    a = abs(pct)
    return "good" if a < 0.5 else ("warn" if a < 2.0 else "poor")


def _values():
    rng = np.random.default_rng(0)
    vals = [None, float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 0.5, 2.0, 1.5, 3.0,
            5.0, 15.0, math.sqrt(2), 2 * math.sqrt(2), 4 * math.sqrt(2),
            np.float64(0.5), np.float64("nan"), 7, 2, 1]
    for t in (0.5, 2.0, 1.5, 3.0, math.sqrt(2), 2 * math.sqrt(2), 5.0, 15.0):
        vals += [np.nextafter(t, -np.inf), np.nextafter(t, np.inf), -t]
    vals += list(rng.normal(0, 5, 3000)) + list(rng.uniform(-20, 20, 3000))
    return vals


THRESHOLDS = [(0.5, 2.0), (1.5, 3.0), (5.0, 15.0), (math.sqrt(2), 2 * math.sqrt(2)),
              (2 * math.sqrt(2), 4 * math.sqrt(2))]


def test_grade_equals_the_old_ladders():
    from mvis.regression.compare import band_for_pct  # maintainers' package (conftest skips)
    for v in _values():
        for good, marginal in THRESHOLDS:
            assert verdict_grade(v, good, marginal) == grade(v, good, marginal) \
                == _old_verdict(v, good, marginal)
            assert cam_imu_report._grade(v, good, marginal) == _old_report(v, good, marginal)
        if v is not None:
            assert band_for_pct(v) == _old_band(v)
    assert band_for_pct(None) is None


def _old_tbl(headers, rows):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


_SYM = {"good": "&#10003;", "marginal": "&#9888;", "poor": "&#10007;"}


def _old_badge(g):
    if not g:
        return ""
    return f'<span class="badge badge-{g}">{_SYM[g]} {g}</span>'


def _old_stat(value, label, g=None):
    extra = f'<div style="margin-top:4px">{_old_badge(g)}</div>' if g else ""
    return (f'<div class="stat-box"><div class="value">{value}</div>'
            f'<div class="label">{label}</div>{extra}</div>')


def test_html_helpers_give_the_old_markup():
    for g in (None, "", "good", "marginal", "poor"):
        assert badge(g) == _old_badge(g)
        for value in (3, "0.123 px", '<span class="badge">x</span>'):
            assert stat_box(value, "Reproj", g) == _old_stat(value, "Reproj", g)
    rows = [["a", 1, None], [2.5, "<b>x</b>", ""], []]
    for headers in ([], ["x"], ["metric", "value", "grade"]):
        assert html_table(headers, rows) == _old_tbl(headers, rows)
        assert html_table(headers, []) == _old_tbl(headers, [])
