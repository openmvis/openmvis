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
"""Unit-tests for _resolve_window + optional real-bag smoke test."""
import os
import pytest

from mvis.cli.main import _resolve_window
from mvis.core.paths import datasets_root

BAG_PATH = os.path.join(datasets_root(),
                        "mvis_4imus_3cams/virig_4imus_3cams_april_01.bag")
IMU_TOPICS = ["/imu/data", "/gx3_25/data", "/gx3_35/imu/data", "/t265/imu"]


def test_manual_mode_uses_t0_window():
    class CI:
        window_mode = "manual"; t0 = 3.0; window = 45.0; segment_length = 80.0
        path = "x"; format = "ros1"; imu_topics = ["/imu"]
    assert _resolve_window(CI, None) == (3.0, 45.0, None)   # (t0, win, abs_window)


def test_manual_window_zero_is_whole():
    class CI:
        window_mode = "manual"; t0 = 3.0; window = 0.0; segment_length = 80.0
        path = "x"; format = "ros1"; imu_topics = ["/imu"]
    t0, win, aw = _resolve_window(CI, None)
    assert t0 == 3.0 and win >= 1e9 and aw is None


def test_motion_mode_uses_selector(monkeypatch):
    import mvis.cli.main as M
    monkeypatch.setattr(M, "read_imu_only",
                        lambda *a, **k: [{"t": [0, 1], "w": [[0, 0, 0]] * 2,
                                          "a": [[0, 0, 9.8]] * 2}])
    monkeypatch.setattr(M, "select_motion_window",
                        lambda *a, **k: (12.0, 92.0))

    class CI:
        window_mode = "motion"; t0 = 3.0; window = 45.0; segment_length = 80.0
        path = "x"; format = "ros1"; imu_topics = ["/imu"]
    t0, win, aw = _resolve_window(CI, None)
    assert aw == (12.0, 92.0)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(BAG_PATH), reason="real bag not present")
def test_motion_mode_real_bag_skips_static_start():
    """Motion selector on the 4imus_3cams bag must land past the static start (~0-10 s)."""
    from mvis.io.cam_imu_loader import read_imu_only

    class CI:
        window_mode = "motion"; t0 = 0.0; window = 0.0; segment_length = 80.0
        path = BAG_PATH; format = "ros1"; imu_topics = IMU_TOPICS

    _t0, _win, aw = _resolve_window(CI, None)
    assert aw is not None, "Selector returned None on real bag"
    streams = read_imu_only(BAG_PATH, IMU_TOPICS, fmt="ros1")
    bag_first = min(s["t"][0] for s in streams if len(s["t"]) > 0)
    lo, _hi = aw
    assert lo - bag_first >= 10.0, (
        f"Motion window starts only {lo - bag_first:.1f}s past bag start; "
        "expected ≥10s to skip static section"
    )
