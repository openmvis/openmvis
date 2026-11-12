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
"""mvis.recalib.bagio: helpers shared by the bag rewriters."""
from types import SimpleNamespace

import pytest

from mvis.recalib.bagio import copy_connections, header_ns, to_sec_nsec


def _msg(**stamp):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(**stamp)))


def test_header_ns_reads_ros1_and_ros2_field_names():
    assert header_ns(_msg(secs=3, nsecs=7)) == 3_000_000_007
    assert header_ns(_msg(sec=4, nanosec=9)) == 4_000_000_009
    assert header_ns(_msg()) == 0


def test_copy_connections_recreates_each_one_on_the_writer():
    calls = []

    class Writer:
        def add_connection(self, *a, **kw):
            calls.append((a, kw))
            return f"w{len(calls)}"

    conns = [SimpleNamespace(id=5, topic="/a", msgtype="T", msgdef=SimpleNamespace(data="def"), digest="md5",
                             ext=SimpleNamespace(callerid="/x", latching=1)),
             SimpleNamespace(id=9, topic="/b", msgtype="U", msgdef=SimpleNamespace(data="d2"), digest="m2", ext=None)]
    out = copy_connections(SimpleNamespace(connections=conns), Writer())
    assert out == {5: "w1", 9: "w2"}
    assert calls[0] == (("/a", "T"), dict(msgdef="def", md5sum="md5", callerid="/x", latching=1))
    assert calls[1][1]["callerid"] is None and calls[1][1]["latching"] is None


@pytest.mark.parametrize("t,expected", [(1.5, (1, 500_000_000)), (0.0, (0, 0)), (2.9999999999, (3, 0)),
                                        (-0.25, (-1, 750_000_000))])
def test_to_sec_nsec_normalises_nsec(t, expected):
    assert to_sec_nsec(t) == expected


# ---------- integer-ns raw stamp helpers (F16) ----------------------------------------------------
import struct  # noqa: E402

from mvis.recalib.bagio import (NS_PER_S, ns_to_sec_nsec, raw_stamp_ns, seconds_to_ns,  # noqa: E402
                                with_raw_stamp_ns)


def _raw(sec, nsec, tail=b"payload"):
    return struct.pack("<III", 7, sec, nsec) + tail


def test_raw_stamp_round_trip_keeps_every_other_byte():
    raw = _raw(1_700_000_123, 456_789_001)
    assert raw_stamp_ns(raw) == 1_700_000_123 * NS_PER_S + 456_789_001
    out = with_raw_stamp_ns(raw, raw_stamp_ns(raw) + 9_456_000)
    assert out[:4] == raw[:4] and out[12:] == raw[12:]
    assert struct.unpack_from("<II", out, 4) == (1_700_000_123, 466_245_001)
    assert with_raw_stamp_ns(memoryview(raw), raw_stamp_ns(raw)) == raw


def test_integer_shift_is_exact_where_float_seconds_round():
    """Near 1.7e9 s a float-seconds round trip moves the stamp by up to ~0.24 us."""
    sec, nsec = 1_700_000_123, 456_789_001
    got = raw_stamp_ns(with_raw_stamp_ns(_raw(sec, nsec), sec * NS_PER_S + nsec + 1_000_000))
    assert got == sec * NS_PER_S + nsec + 1_000_000
    s2, n2 = to_sec_nsec(sec + nsec * 1e-9 + 1e-3)       # the old float path
    assert s2 * NS_PER_S + n2 != got


@pytest.mark.parametrize("ns,expected", [(0, (0, 0)), (1_500_000_000, (1, 500_000_000)),
                                         (-250_000_000, (-1, 750_000_000))])
def test_ns_to_sec_nsec(ns, expected):
    assert ns_to_sec_nsec(ns) == expected


def test_seconds_to_ns_rounds_to_the_nearest_ns():
    assert seconds_to_ns(-0.0004012054847988395) == -401_205
    assert seconds_to_ns(0.0094564) == 9_456_400
