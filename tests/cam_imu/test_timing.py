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
"""Tests for the hierarchical Timer."""

import time

import pytest

from mvis.core.timing import Timer


def test_empty_timer_has_no_sections():
    t = Timer()
    out = t.to_dict()
    assert out["sections"] == []
    assert out["total_seconds"] == 0.0
    assert t.to_flat_rows() == []


def test_single_section_records_time_and_metadata():
    t = Timer()
    with t.section("stage1", n_obs=42):
        time.sleep(0.005)
    out = t.to_dict()
    assert len(out["sections"]) == 1
    s = out["sections"][0]
    assert s["name"] == "stage1"
    assert s["seconds"] >= 0.005
    assert s["metadata"] == {"n_obs": 42}
    assert s["self_seconds"] == s["seconds"]   # no children


def test_nested_sections_compute_self_time_correctly():
    t = Timer()
    with t.section("outer"):
        with t.section("inner"):
            time.sleep(0.003)
        time.sleep(0.002)
    out = t.to_dict()
    outer = out["sections"][0]
    inner = outer["children"][0]
    # outer.seconds ≈ 5ms; inner.seconds ≈ 3ms; outer.self ≈ 2ms.
    assert outer["seconds"] >= inner["seconds"]
    assert outer["self_seconds"] == pytest.approx(
        outer["seconds"] - inner["seconds"], abs=1e-6
    )


def test_two_siblings_appear_in_order():
    t = Timer()
    with t.section("a"):
        pass
    with t.section("b"):
        pass
    names = [s["name"] for s in t.to_dict()["sections"]]
    assert names == ["a", "b"]


def test_record_adds_section_without_context():
    t = Timer()
    t.record("from_log", 1.234, source="external")
    s = t.to_dict()["sections"][0]
    assert s["name"] == "from_log"
    assert s["seconds"] == 1.234
    assert s["metadata"] == {"source": "external"}


def test_to_flat_rows_walks_in_tree_order():
    t = Timer()
    with t.section("stage4"):
        with t.section("setup"):
            pass
        with t.section("optimize", n_iter=12):
            pass
    rows = t.to_flat_rows()
    assert [r["path"] for r in rows] == [
        "stage4", "stage4.setup", "stage4.optimize",
    ]
    assert [r["depth"] for r in rows] == [0, 1, 1]
    assert rows[2]["metadata"] == {"n_iter": 12}


def test_empty_section_name_rejected():
    t = Timer()
    with pytest.raises(ValueError, match="non-empty"):
        with t.section(""):
            pass


def test_top_level_total_sums_top_level_sections():
    t = Timer()
    t.record("a", 1.5)
    t.record("b", 2.5)
    out = t.to_dict()
    assert out["total_seconds"] == pytest.approx(4.0)


def test_to_dict_with_unclosed_section_raises():
    t = Timer()
    cm = t.section("open_forever")
    cm.__enter__()
    try:
        with pytest.raises(RuntimeError, match="open sections"):
            t.to_dict()
    finally:
        cm.__exit__(None, None, None)
