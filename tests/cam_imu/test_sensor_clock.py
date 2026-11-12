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
"""Sensor-clock recovery (sensor_clock.py): a camera that free-runs on its own
sensor clock but is stamped by a fixed-rate polling loop (the ELP under
usb_cam 0.3.6) has capture times on a lattice phi + n*T_s; fit_sensor_clock
must recover it, prefer the fundamental over its integer fractions, isolate
outliers, and refuse data that is not on a lattice."""
import numpy as np
import pytest

from mvis.calibration.sensor_clock import (SensorClock, check_sensor_clock,
                                           fit_sensor_clock)

T_S = 1.0 / 29.6249          # ELP sensor period measured on mvis_4imus_4cams bag 1
P_PUB = 0.040                # 25 Hz polling loop


def _polled_camera(seed=0, span=60.0, jitter=0.8e-3, latency=0.003, outlier_frac=0.0):
    """(stamps, true capture times, true sensor indices, outlier mask)."""
    rng = np.random.default_rng(seed)
    phase = 0.0123
    sensor = phase + np.arange(0, int(span / T_S)) * T_S
    stamps = np.arange(0.2, span - 0.2, P_PUB) + rng.normal(0, 0.9e-3, int((span - 0.4) / P_PUB) + 1)[
        :len(np.arange(0.2, span - 0.2, P_PUB))]
    idx = np.searchsorted(sensor, stamps - latency) - 1          # latest finished frame
    cap = sensor[idx] + rng.normal(0, jitter, len(idx))
    out = rng.random(len(idx)) < outlier_frac
    cap[out] += rng.uniform(-0.03, 0.03, out.sum())
    return stamps, cap, idx, out


def test_recovers_period_phase_and_indices():
    stamps, cap, idx, _ = _polled_camera()
    clk = fit_sensor_clock(cap)
    assert isinstance(clk, SensorClock)
    assert abs(clk.period - T_S) < 2e-7
    assert np.all(np.diff(clk.index - idx) == 0)               # indices up to a constant
    assert clk.scatter < 1.2e-3 and clk.coherence > 0.9
    assert check_sensor_clock(clk, stamps) == []
    lat = clk.capture_times() - cap
    assert np.abs(lat).max() < 5e-3                             # lattice within the jitter


def test_sawtooth_offsets_are_one_sensor_period_wide():
    """The stamp - capture offsets of a polled camera span ~one sensor period;
    after retiming they collapse onto the lattice."""
    stamps, cap, _, _ = _polled_camera()
    off = stamps - cap
    assert np.ptp(off) > 0.8 * T_S
    clk = fit_sensor_clock(cap)
    assert np.std(cap - clk.capture_times()) < 1.2e-3


def test_prefers_fundamental_over_integer_fractions():
    """T_s/2 and T_s/3 are lattices for the same points; the fundamental must win."""
    _, cap, _, _ = _polled_camera(seed=1)
    clk = fit_sensor_clock(cap, period_range=(T_S / 3.5, 1.5 * T_S))
    assert abs(clk.period - T_S) < 2e-7


def test_period_hint_narrows_search():
    _, cap, _, _ = _polled_camera(seed=2)
    clk = fit_sensor_clock(cap, period_hint=0.034)
    assert abs(clk.period - T_S) < 2e-7


def test_outliers_are_excluded_not_fatal():
    stamps, cap, idx, out = _polled_camera(seed=3, outlier_frac=0.03)
    clk = fit_sensor_clock(cap)
    assert abs(clk.period - T_S) < 5e-7
    assert np.mean(~clk.inlier[out]) > 0.6                     # most outliers flagged
    assert np.mean(clk.inlier[~out]) > 0.98                    # good frames kept
    good = clk.inlier & ~out
    assert np.all(np.diff(clk.index[good] - idx[good]) == 0)
    assert check_sensor_clock(clk, stamps) == []                # outliers are not judged


def test_rejects_times_not_on_a_lattice():
    rng = np.random.default_rng(4)
    stamps = np.arange(0.2, 60.0, P_PUB)
    clk = fit_sensor_clock(stamps + rng.uniform(0, 0.034, len(stamps)))
    reasons = check_sensor_clock(clk, stamps)
    assert any("coherence" in r for r in reasons)


def test_detects_shared_sensor_index():
    """Two published frames claiming the same sensor frame is a failure."""
    stamps, cap, _, _ = _polled_camera(seed=5, jitter=0.2e-3)
    cap = cap.copy(); cap[50] = cap[49]                        # duplicate sensor frame
    clk = fit_sensor_clock(cap)
    clk.inlier[:] = True
    assert any("share" in r for r in check_sensor_clock(clk, stamps))


def test_input_order_is_preserved():
    stamps, cap, idx, _ = _polled_camera(seed=6)
    perm = np.random.default_rng(6).permutation(len(cap))
    clk = fit_sensor_clock(cap[perm])
    assert np.all(np.diff((clk.index - idx[perm])[np.argsort(perm)]) == 0)


def test_needs_ten_frames():
    with pytest.raises(ValueError):
        fit_sensor_clock(np.arange(5) * T_S)
