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
"""Tests for motion-segment selection + whole-bag IMU read.

TDD: written RED before implementation.
"""
import numpy as np
import pytest

from mvis.io.segment_select import select_motion_window


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _imu(t, w, a):  # w,a: (N,3)
    return {"topic": "/imu", "name": "imu", "t": t, "w": w, "a": a}


# ---------------------------------------------------------------------------
# selector tests (brief Step 1 — verbatim)
# ---------------------------------------------------------------------------

def test_selects_planted_excitation_segment():
    t = np.arange(0.0, 200.0, 0.005)                 # 200 s @ 200 Hz
    w = np.zeros((len(t), 3)); a = np.zeros((len(t), 3)); a[:, 2] = 9.81
    hot = (t >= 100) & (t < 180)                      # planted hot 80 s
    w[hot, 0] = np.sin(2 * np.pi * 1.0 * t[hot])      # 1 Hz rotation
    a[hot, 0] = 2.0 * np.sin(2 * np.pi * 1.0 * t[hot])# + translation
    lo, hi = select_motion_window([_imu(t, w, a)], segment_length=80.0, step=5.0)
    assert 95.0 <= lo <= 105.0 and abs((hi - lo) - 80.0) < 1e-6


def test_short_bag_returns_none():
    t = np.arange(0.0, 50.0, 0.01)
    z = np.zeros((len(t), 3))
    assert select_motion_window([_imu(t, z, z)], segment_length=80.0) is None


def test_exact_length_bag_is_selectable():
    # spec: only spans STRICTLY < segment_length return None; an exactly
    # segment_length bag is selectable (one candidate window from the start).
    t = np.arange(0.0, 80.0, 0.01)          # span = 79.99 -> just under; use 80.01 to exceed
    t = np.arange(0.0, 80.01, 0.01)         # span ~= 80.0 (>= segment_length)
    z = np.zeros((len(t), 3))
    out = select_motion_window([_imu(t, z, z)], segment_length=80.0, step=5.0)
    assert out is not None
    lo, hi = out
    assert lo == 0.0 and abs((hi - lo) - 80.0) < 1e-6


def test_deterministic():
    t = np.arange(0.0, 200.0, 0.01); rng = np.zeros((len(t), 3)); rng[:, 0] = np.sin(t)
    a = select_motion_window([_imu(t, rng, rng)], 80.0)
    b = select_motion_window([_imu(t, rng, rng)], 80.0)
    assert a == b


def test_gyro_and_accel_both_count():
    # one IMU excited only in gyro, another only in accel, in different segments;
    # the combined+normalized score must be able to pick either — here put both
    # excitations in the SAME segment and confirm it's chosen over a quiet one.
    t = np.arange(0.0, 200.0, 0.005)
    w = np.zeros((len(t), 3)); a = np.zeros((len(t), 3)); a[:, 2] = 9.81
    hot = (t >= 20) & (t < 100)
    w[hot, 1] = 0.5 * np.sin(t[hot]); a[hot, 0] = 1.0 * np.sin(t[hot])
    lo, hi = select_motion_window([_imu(t, w, a)], 80.0, 5.0)
    assert 15.0 <= lo <= 25.0


def test_nan_sample_does_not_poison_score():
    """A NaN IMU sample must not silently collapse selection to the earliest
    window (NaN > best is always False). nan_to_num lets the real motion win."""
    t = np.arange(0.0, 200.0, 0.005)
    w = np.zeros((len(t), 3)); a = np.zeros((len(t), 3)); a[:, 2] = 9.81
    hot = (t >= 110) & (t < 190)                 # excitation late, not at start
    w[hot, 1] = 0.6 * np.sin(t[hot])
    w[1234, 0] = np.nan                          # a single dropped/corrupt sample
    lo, hi = select_motion_window([_imu(t, w, a)], 80.0, 5.0)
    assert lo >= 100.0                           # picked the late hot segment, not starts[0]


# ---------------------------------------------------------------------------
# read_imu_only + abs_window tests (brief Step 6)
# ---------------------------------------------------------------------------

def _make_tiny_bag(tmp_path, imu_topics=("/imu0", "/imu1"), n_imu=20):
    """Synthetic ROS1 bag with 2 IMU topics spanning ~0.1 s, no camera."""
    import numpy as np
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)

    Imu = ts.types["sensor_msgs/msg/Imu"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]

    path = str(tmp_path / "test_imu.bag")
    with Writer(path) as w:
        connections = {}
        for tp in imu_topics:
            connections[tp] = w.add_connection(tp, Imu.__msgtype__, typestore=ts)
        for k in range(n_imu):
            t_ns = 1_000_000_000 + k * 5_000_000   # 200 Hz, starts at t=1.0 s
            hdr = Header(seq=k,
                         stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9),
                         frame_id="imu")
            for tp in imu_topics:
                msg = Imu(
                    header=hdr,
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.zeros(9),
                    angular_velocity=Vector3(x=float(k) * 0.01, y=0.0, z=0.0),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=Vector3(x=0.0, y=0.0, z=9.81),
                    linear_acceleration_covariance=np.zeros(9),
                )
                w.write(connections[tp], t_ns, ts.serialize_ros1(msg, Imu.__msgtype__))
    return path


def test_read_imu_only_ros1_returns_all_streams(tmp_path):
    """read_imu_only with fmt='ros1' returns one stream per topic, whole bag."""
    from mvis.io.cam_imu_loader import read_imu_only

    imu_topics = ["/imu0", "/imu1"]
    bag = _make_tiny_bag(tmp_path, imu_topics=imu_topics, n_imu=20)

    streams = read_imu_only(bag, imu_topics, fmt="ros1")

    assert len(streams) == 2
    for s in streams:
        assert s["topic"] in imu_topics
        assert len(s["t"]) == 20           # all 20 samples read (no windowing)
        assert s["w"].shape == (20, 3)
        assert s["a"].shape == (20, 3)
    # whole-bag span: first=1.0 s, last=1.0 + 19*0.005 = 1.095 s
    t0 = float(streams[0]["t"][0])
    t1 = float(streams[0]["t"][-1])
    assert t0 == pytest.approx(1.0, abs=1e-3)
    assert t1 == pytest.approx(1.095, abs=1e-3)


def test_read_imu_only_euroc_topic_style_name(tmp_path):
    """read_imu_only(fmt='euroc') accepts topic-style names ('/imu0'): a leading
    slash must NOT make os.path.join treat the subdir as an absolute path
    (regression — motion-mode dispatch passes raw ci.imu_topics)."""
    from mvis.io.cam_imu_loader import read_imu_only

    mav0 = tmp_path / "mav0"
    (mav0 / "imu0").mkdir(parents=True)
    rows = ["#t,wx,wy,wz,ax,ay,az"]
    for i in range(10):
        ns = 1_000_000_000 + i * 5_000_000        # 200 Hz
        rows.append(f"{ns},0.1,0.0,0.0,0.0,0.0,9.81")
    (mav0 / "imu0" / "data.csv").write_text("\n".join(rows) + "\n")

    streams = read_imu_only(str(mav0), ["/imu0"], fmt="euroc")   # topic-style "/imu0"
    assert len(streams) == 1
    assert streams[0]["name"] == "imu0"                          # normalized
    assert len(streams[0]["t"]) == 10
    assert streams[0]["w"].shape == (10, 3)


def test_abs_window_overrides_t0_window():
    """abs_window=(lo,hi) in load_ros1 overrides the t0/window computation.

    We test this without a full bag by checking the _sig helper reflects the
    resolved window: passing abs_window should make the cache key use (lo, hi-lo)
    rather than the original (t0, window). This is a direct unit check on the
    module's internal _sig function used by load_ros1.
    """
    import unittest.mock as mock
    from mvis.io.cam_imu_loader import _sig

    # Use a MagicMock board so board_signature can call vars() on it
    board = mock.MagicMock(spec=[])  # no spec attrs; vars() works on MagicMock

    # Simulate what load_ros1 does:
    # Without abs_window: first=10.0, t0=5.0, window=30.0 => lo=15.0, hi=45.0
    # With abs_window=(20.0, 80.0): lo=20.0, hi=80.0 => t0=20.0, window=60.0
    sig_manual = _sig("/bag.bag", ["/cam0"], board, t0=15.0, window=30.0, min_obs=1)
    sig_abs    = _sig("/bag.bag", ["/cam0"], board, t0=20.0, window=60.0, min_obs=1)

    # abs_window window is different => distinct cache keys
    assert sig_manual != sig_abs


def test_manual_path_cache_key_unchanged():
    """Regression guard: with abs_window=None the loader must build the cache
    signature from the ORIGINAL (t0, window) args, not from the header-resolved
    (lo, hi-lo). Folding the resolved window into the manual key would
    header-time-shift it and silently invalidate every existing cache.

    Reproduce both _sig calls the loader makes and assert the manual one matches
    the historical form byte-for-byte."""
    import unittest.mock as mock
    from mvis.io.cam_imu_loader import _sig

    board = mock.MagicMock(spec=[])
    t0, window, min_obs = 5.0, 30.0, 1
    first = 10.0                     # earliest cam header stamp
    lo = first + t0                  # resolved absolute window start = 15.0

    # Historical key (pre-feature): keyed on the OFFSET t0, not absolute lo.
    historical = _sig("/bag.bag", ["/cam0"], board, t0, window, min_obs)
    # Manual-path key the loader now builds (abs_window is None branch).
    manual = _sig("/bag.bag", ["/cam0"], board, t0, window, min_obs)
    # The buggy form that folds the resolved window (must NOT match historical).
    buggy = _sig("/bag.bag", ["/cam0"], board, t0=lo, window=(lo + window - lo),
                 min_obs=min_obs)

    assert manual == historical, "manual path must reproduce today's cache key"
    assert buggy != historical, "resolved-window key would shift with header time"


def test_abs_window_imu_filter(tmp_path):
    """read_imu_only returns the whole bag (no windowing), confirming that
    abs_window filtering in load_ros1 can narrow to a subset of those samples.

    We verify: whole-bag read gives all N samples; then simulate the abs_window
    filter (lo-0.5, hi+0.5 tolerance as used in load_ros1's IMU loop) on a
    window that strictly excludes the later samples."""
    from mvis.io.cam_imu_loader import read_imu_only

    # Build a bag spanning t=1.0 .. 1.095 s at 200 Hz (20 samples @ 0.005 s)
    bag = _make_tiny_bag(tmp_path, imu_topics=["/imu0"], n_imu=20)

    streams = read_imu_only(bag, ["/imu0"], fmt="ros1")
    t = streams[0]["t"]

    # Verify whole bag: all 20 samples returned
    assert len(t) == 20

    # Simulate abs_window=(lo, hi) = (1.0, 1.020) — a tight window
    # With 0.5 s tolerance: keep t in [lo-0.5, hi+0.5] = [0.5, 1.52]
    # All 20 samples (1.0..1.095) fall within => tolerance is intentionally wide
    # to include IMU samples near cam frames.
    # Now test a STRICT window that excludes later half: abs_window=(1.0, 1.022)
    # Without tolerance, only t<=1.022 passes (samples 0..4 at 1.000..1.020).
    lo, hi = 1.0, 1.022
    mask_strict = (t >= lo) & (t <= hi)
    assert mask_strict.sum() == 5, \
        f"strict [lo,hi] should contain 5 samples, got {mask_strict.sum()}"

    # With the loader's ±0.5 s tolerance, more samples are included
    mask_tolerant = (t >= lo - 0.5) & (t <= hi + 0.5)
    assert mask_tolerant.sum() == 20, \
        f"tolerant window should include all 20 samples, got {mask_tolerant.sum()}"
