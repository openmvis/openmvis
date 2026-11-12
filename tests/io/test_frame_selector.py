# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
# Copyright (C) 2026 Yuantai-Z
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
# SPDX-FileCopyrightText: 2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-only
"""Tests for rate-based ``FrameObservations`` selection."""

from mvis.core.types import FrameObservations
from mvis.io import frame_selector


def _make_timed_frames(count: int, frequency_hz: float):
    return [
        FrameObservations(
            camera_id=0,
            timestamp=i / frequency_hz,
            observations=[],
            image_size=(640, 480),
        )
        for i in range(count)
    ]


def test_select_frame_indices_by_hz_preserves_source_mapping():
    """A 10 Hz timeline selected at 5 Hz reports its original indices."""
    frames = _make_timed_frames(count=20, frequency_hz=10.0)
    all_frames = {0: frames}

    selected_indices = frame_selector.select_frame_indices_by_hz(
        all_frames, target_hz=5.0)

    expected = list(range(0, 20, 2))
    assert selected_indices == {0: expected}

    selected_frames = frame_selector.subsample_frames_by_hz(
        all_frames, target_hz=5.0)
    assert len(selected_frames[0]) == len(expected)
    assert all(
        selected is frames[source_index]
        for selected, source_index in zip(selected_frames[0], expected)
    )
