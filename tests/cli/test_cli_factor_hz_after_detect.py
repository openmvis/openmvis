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
"""G7: `mvis calibrate` selects each camera's factor_hz after detection, on the common-origin
grid of frame_selector (before: ROS1 / timestamped readers subsampled before detection,
each on its own grid, and the other readers ignored factor_hz)."""
from types import SimpleNamespace

import pytest

from mvis.cli.main import _subsample_by_factor_hz


def _frames(cid, t0, n, dt=0.05):
    return [SimpleNamespace(camera_id=cid, timestamp=round(t0 + i * dt, 6)) for i in range(n)]


def _cfg(**rates):
    return SimpleNamespace(cameras=[SimpleNamespace(camera_id=int(k[3:]), factor_hz=v)
                                    for k, v in rates.items()])


def test_synced_cameras_keep_the_same_instants():
    # 20 Hz pair, cam1 starts one frame late: a per-camera grid would pick 0.05, 0.25, ...
    frames = {0: _frames(0, 0.0, 20), 1: _frames(1, 0.05, 19)}
    out = _subsample_by_factor_hz(_cfg(cam0=5.0, cam1=5.0), frames)
    t = {c: [f.timestamp for f in fr] for c, fr in out.items()}
    assert t[0] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert t[1] == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_a_camera_without_a_rate_keeps_every_frame():
    frames = {0: _frames(0, 0.0, 20), 1: _frames(1, 0.0, 20)}
    out = _subsample_by_factor_hz(_cfg(cam0=5.0, cam1=0.0), frames)
    assert (len(out[0]), len(out[1])) == (5, 20)
    assert _subsample_by_factor_hz(_cfg(cam0=0.0, cam1=None), frames) is frames
