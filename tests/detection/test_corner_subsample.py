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
"""Tests for corner subsampling."""

import numpy as np
import pytest

from mvis.detection.corner_subsample import (
    filter_observations,
    filter_frames,
)
from mvis.core.types import FrameObservations, Observation


def _make_obs(n_tags: int = 36) -> list[Observation]:
    """Build n_tags*4 observations with point_id = tag*4 + corner."""
    out = []
    for t in range(n_tags):
        for c in range(4):
            pid = t * 4 + c
            out.append(Observation(point_id=pid, pixel=np.array([pid, pid])))
    return out


def test_all_is_passthrough():
    obs = _make_obs(10)
    out = filter_observations(obs, "all")
    assert len(out) == 40
    assert [o.point_id for o in out] == [o.point_id for o in obs]


def test_top_left_keeps_corner_3():
    obs = _make_obs(36)
    out = filter_observations(obs, "top_left")
    assert len(out) == 36
    for o in out:
        assert o.point_id % 4 == 3


def test_tl_alias_matches_top_left():
    obs = _make_obs(36)
    a = filter_observations(obs, "tl")
    b = filter_observations(obs, "top_left")
    assert [o.point_id for o in a] == [o.point_id for o in b]


@pytest.mark.parametrize("name,idx", [
    ("bl", 0), ("br", 1), ("tr", 2), ("tl", 3),
    ("bottom_left", 0), ("bottom_right", 1),
    ("top_right", 2), ("top_left", 3),
])
def test_single_corner_modes(name, idx):
    obs = _make_obs(12)
    out = filter_observations(obs, name)
    assert len(out) == 12
    assert all(o.point_id % 4 == idx for o in out)


def test_two_diag_keeps_bl_and_tr():
    obs = _make_obs(36)
    out = filter_observations(obs, "two_diag")
    assert len(out) == 72
    assert all(o.point_id % 4 in (0, 2) for o in out)


def test_two_anti_keeps_br_and_tl():
    obs = _make_obs(36)
    out = filter_observations(obs, "two_anti")
    assert len(out) == 72
    assert all(o.point_id % 4 in (1, 3) for o in out)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown corner_subsample"):
        filter_observations(_make_obs(1), "banana")


def test_filter_frames_preserves_metadata():
    frames = {
        0: [FrameObservations(
            camera_id=0, timestamp=1.23,
            observations=_make_obs(6), image_size=(640, 480))],
        1: [FrameObservations(
            camera_id=1, timestamp=2.34,
            observations=_make_obs(6), image_size=(640, 480))],
    }
    out = filter_frames(frames, "top_left")
    assert set(out.keys()) == {0, 1}
    for cid, fs in out.items():
        assert len(fs) == 1
        assert fs[0].camera_id == cid
        assert fs[0].image_size == (640, 480)
        assert fs[0].timestamp == (1.23 if cid == 0 else 2.34)
        assert len(fs[0].observations) == 6
        assert all(o.point_id % 4 == 3 for o in fs[0].observations)


def test_filter_frames_all_noop_returns_same_dict():
    frames = {0: [FrameObservations(
        camera_id=0, timestamp=0.0,
        observations=_make_obs(3), image_size=(100, 100))]}
    out = filter_frames(frames, "all")
    assert out is frames
