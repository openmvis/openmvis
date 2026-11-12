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
"""Tests for mvis.calibration.init_frame_selector."""

from __future__ import annotations

import numpy as np
import pytest

from mvis.core.types import FrameObservations, Observation
from mvis.calibration.init_frame_selector import (
    select_diverse,
    select_init_frames,
    select_uniform_in_time,
)


def _make_frame(centroid_xy: tuple[float, float], size: float, angle: float,
                cam_id: int = 0, ts: float = 0.0) -> FrameObservations:
    """Build a 4-corner FrameObservations at a given centroid/size/angle."""
    cx, cy = centroid_xy
    half = size / 2.0
    base = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    pts = (rot @ base.T).T + np.array([cx, cy])
    obs = [Observation(point_id=i, pixel=pts[i]) for i in range(4)]
    return FrameObservations(camera_id=cam_id, timestamp=ts, observations=obs,
                             image_size=(640, 480))


def test_uniform_returns_all_when_under_cap():
    frames = [_make_frame((100, 100), 50, 0) for _ in range(5)]
    out = select_uniform_in_time(frames, n_keep=10)
    assert len(out) == 5
    assert out is not frames  # returns a copy


def test_uniform_returns_n_keep_when_over_cap():
    frames = [_make_frame((100, 100), 50, 0, ts=float(i)) for i in range(100)]
    out = select_uniform_in_time(frames, n_keep=10)
    assert len(out) == 10
    # Must include first and last for full coverage
    assert out[0].timestamp == 0.0
    assert out[-1].timestamp == 99.0


def test_uniform_zero_keep_returns_input():
    frames = [_make_frame((100, 100), 50, 0)]
    assert len(select_uniform_in_time(frames, n_keep=0)) == 1


def test_diverse_picks_across_distinct_bins():
    """Frames clustered into 4 distinct (position, size, angle) buckets — verify
    the selector picks at least one from each."""
    frames = []
    for cluster_id in range(4):
        cx = 100 + cluster_id * 130  # spread across image width
        cy = 100 + cluster_id * 90
        size = 30 + cluster_id * 50
        angle = cluster_id * (np.pi / 5)
        for _ in range(10):
            frames.append(_make_frame((cx, cy), size, angle))

    out = select_diverse(frames, n_keep=4, image_size=(640, 480))
    assert len(out) == 4
    centroids = [f.pixels.mean(axis=0) for f in out]
    xs = sorted(c[0] for c in centroids)
    # Each cluster should be represented — adjacent picks at least 50px apart
    assert all(xs[i+1] - xs[i] > 50 for i in range(len(xs) - 1))


def test_diverse_falls_back_when_n_keep_exceeds_input():
    frames = [_make_frame((100, 100), 50, 0) for _ in range(3)]
    out = select_diverse(frames, n_keep=10, image_size=(640, 480))
    assert len(out) == 3


def test_diverse_handles_degenerate_metrics():
    """All-identical frames -> all land in same bin -> selector still returns
    n_keep frames by exhausting that one bin."""
    frames = [_make_frame((320, 240), 50, 0, ts=float(i)) for i in range(50)]
    out = select_diverse(frames, n_keep=10, image_size=(640, 480))
    assert len(out) == 10


def test_top_level_dispatch_unknown_strategy_falls_back_to_uniform():
    frames = [_make_frame((100, 100), 50, 0, ts=float(i)) for i in range(20)]
    out = select_init_frames(frames, n_keep=5, image_size=(640, 480),
                             strategy="bogus")
    expected = select_uniform_in_time(frames, 5)
    assert [f.timestamp for f in out] == [f.timestamp for f in expected]


def test_top_level_dispatch_diverse_routes_correctly():
    frames = []
    for i in range(20):
        x = 50 + (i % 4) * 150
        frames.append(_make_frame((x, 240), 60, 0))
    out = select_init_frames(frames, n_keep=4, image_size=(640, 480),
                             strategy="diverse")
    assert len(out) == 4
    xs = sorted(f.pixels.mean(axis=0)[0] for f in out)
    assert xs[-1] - xs[0] > 200  # coverage spans the full x-range
