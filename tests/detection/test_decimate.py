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
"""_decimate must thin a timestamp stream to AT MOST cap_hz without aliasing a
source that is already at ~cap_hz down to cap_hz/2.

Regression: with the old `>= min_dt*(1-1e-6)` threshold, a jittery 30 Hz camera
capped at state_hz=30 dropped ~half its frames (every frame landing a hair under
33.333 ms was rejected), yielding a 15 Hz trajectory instead of 30 Hz."""
import numpy as np

from mvis.calibration.pipeline_cam_imu import _decimate


def _jittery(rate_hz, n, jitter_ms=2.0, seed=0):
    rng = np.random.RandomState(seed)
    dt = 1.0 / rate_hz
    t = np.cumsum(dt + rng.uniform(-jitter_ms, jitter_ms, n) * 1e-3)
    return t - t[0] + 1.0


def test_same_rate_not_aliased():
    """A jittery 30 Hz stream capped at 30 Hz must keep ~all frames (>=95%),
    not alias to ~50%."""
    t = _jittery(30.0, 2400, jitter_ms=2.0)
    keep = _decimate(t, 30.0)
    frac = keep.mean()
    assert frac > 0.95, f"30Hz capped at 30Hz kept only {frac:.0%} (aliasing)"


def test_double_rate_is_halved():
    """A 60 Hz stream capped at 30 Hz must thin to ~30 Hz (keep ~half)."""
    t = _jittery(60.0, 2400, jitter_ms=1.0)
    keep = _decimate(t, 30.0)
    kept = t[keep]
    eff = 1.0 / np.median(np.diff(kept))
    assert 24.0 <= eff <= 33.0, f"60Hz capped at 30Hz gave {eff:.1f} Hz"


def test_cap_is_soft_upper_bound():
    """cap is a SOFT upper bound: the 0.75 jitter tolerance lets the output run up
    to ~1.34x cap when heavily oversampled, but never far above and never aliased
    below."""
    t = _jittery(120.0, 3000, jitter_ms=0.5)
    keep = _decimate(t, 20.0)
    kept = t[keep]
    eff = 1.0 / np.median(np.diff(kept))
    assert 18.0 <= eff <= 20.0 * 1.34, f"cap=20Hz gave {eff:.1f} Hz (out of soft band)"


def test_cap_zero_keeps_all():
    t = _jittery(30.0, 100)
    assert _decimate(t, 0.0).all()
