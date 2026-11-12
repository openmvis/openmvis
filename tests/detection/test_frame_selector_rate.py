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
"""subsample_frames_by_hz must pick the SAME instants on synchronized cameras.

Regression for the T265 stereo joint-solve quality problem (2026-10-17): the
5 Hz grid was anchored at each camera's OWN first timestamp. On T265 bags
where fisheye2's first frame is one frame (~33 ms) before/after fisheye1's,
the two grids were offset by a whole frame, so every non-ref frame landed
~31-36 ms away from the nearest ref pose node and the joint solve
interpolated it between ref poses 200 ms apart (no_td) -> cam3 P95 10-16 px
on bags 2/3/5/10/12/14 vs ~2 px where the first frames line up.
"""
from types import SimpleNamespace

import numpy as np

from mvis.io.frame_selector import subsample_frames_by_hz


def _frames(ts):
    return [SimpleNamespace(timestamp=float(t)) for t in ts]


def _max_pair_gap(sub):
    t2 = np.array([f.timestamp for f in sub[2]])
    t3 = np.array([f.timestamp for f in sub[3]])
    # frames outside the ref span have no ref counterpart at all (and the
    # joint solve skips them); every other one must coincide with a ref pick
    t3 = t3[(t3 >= t2[0] - 0.005) & (t3 <= t2[-1] + 0.005)]
    return max(np.min(np.abs(t2 - t)) for t in t3)


def test_synced_cameras_with_extra_leading_frame_pick_same_instants():
    period = 1.0 / 30.0
    n = 300
    ref = np.arange(n) * period                  # fisheye1
    # fisheye2: hardware-synced, +2.5 ms stamp offset, but its stream starts
    # ONE FRAME EARLIER (the T265 bag2/3/10 layout)
    other = np.concatenate([[-period + 0.0025], ref + 0.0025])
    sub = subsample_frames_by_hz({2: _frames(ref), 3: _frames(other)}, 5.0)
    assert _max_pair_gap(sub) < 0.005


def test_synced_cameras_with_one_missing_leading_frame_pick_same_instants():
    period = 1.0 / 30.0
    n = 300
    ref = np.arange(n) * period
    other = (ref + 0.0025)[1:]                   # bag5/12/14 layout
    sub = subsample_frames_by_hz({2: _frames(ref), 3: _frames(other)}, 5.0)
    assert _max_pair_gap(sub) < 0.005


def test_single_camera_selection_unchanged():
    period = 1.0 / 30.0
    ts = np.arange(300) * period + 12.345
    sub = subsample_frames_by_hz({0: _frames(ts)}, 5.0)
    got = [f.timestamp for f in sub[0]]
    ideal = np.arange(ts[0], ts[-1], 0.2)
    want = [ts[i] for i in sorted({int(np.argmin(np.abs(ts - t)))
                                   for t in ideal})]
    assert got == want
