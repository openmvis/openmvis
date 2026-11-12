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
"""Tests for detection cache (JSON format)."""

import json
import numpy as np
import pytest

from mvis.core.types import FrameObservations, Observation
from mvis.io.detection_cache import DetectionCache


class TestDetectionCache:
    def _make_frames(self):
        """Create sample frames for caching."""
        frames = {
            0: [
                FrameObservations(
                    camera_id=0, timestamp=1.0,
                    observations=[
                        Observation(point_id=0, pixel=np.array([100.5, 200.3]), row=200),
                        Observation(point_id=1, pixel=np.array([150.2, 210.7]), row=210),
                    ],
                    image_size=(640, 480),
                ),
                FrameObservations(
                    camera_id=0, timestamp=2.0,
                    observations=[
                        Observation(point_id=0, pixel=np.array([110.0, 205.0]), row=205),
                    ],
                    image_size=(640, 480),
                ),
            ],
        }
        return frames

    def test_save_and_load(self, tmp_path):
        frames = self._make_frames()
        board_config = {"rows": 6, "cols": 6, "tag_size": 0.088}
        source = "/path/to/bag"

        cache = DetectionCache(str(tmp_path / "test_cache"))
        cache.save(frames, source, board_config)

        # File should be JSON
        cache_file = tmp_path / "test_cache.json"
        assert cache_file.exists()

        # Should be readable JSON
        with open(cache_file) as f:
            data = json.load(f)
        assert "cameras" in data
        assert "0" in data["cameras"]
        assert len(data["cameras"]["0"]["frames"]) == 2

        # Load and verify
        loaded, _, _ = cache.load(source, board_config)
        assert 0 in loaded
        assert len(loaded[0]) == 2
        assert loaded[0][0].camera_id == 0
        assert loaded[0][0].timestamp == 1.0
        assert len(loaded[0][0].observations) == 2
        np.testing.assert_allclose(loaded[0][0].observations[0].pixel, [100.5, 200.3])

    def test_cache_invalidation(self, tmp_path):
        frames = self._make_frames()
        board_config = {"rows": 6, "cols": 6, "tag_size": 0.088}
        source = "/path/to/bag"

        cache = DetectionCache(str(tmp_path / "test_cache"))
        cache.save(frames, source, board_config)

        # Same config — valid
        assert cache.is_valid(source, board_config)

        # Different source — invalid
        assert not cache.is_valid("/different/path", board_config)

        # Different board config — invalid
        assert not cache.is_valid(source, {"rows": 9, "cols": 6, "tag_size": 0.025})

    def test_human_readable_json(self, tmp_path):
        """Verify the JSON file is human-readable."""
        frames = self._make_frames()
        cache = DetectionCache(str(tmp_path / "test"))
        cache.save(frames, "/test", {"rows": 6})

        with open(tmp_path / "test.json") as f:
            content = f.read()

        # Should contain readable field names
        assert '"point_id"' in content
        assert '"pixel"' in content
        assert '"timestamp"' in content
        assert '"observations"' in content

    def test_with_detection_history(self, tmp_path):
        frames = self._make_frames()
        histories = {0: [[(1.0, 5, 15.0), (2.0, 10, 30.0)], [(1.0, 8, 20.0)]]}

        cache = DetectionCache(str(tmp_path / "test"))
        cache.save(frames, "/test", {"rows": 6}, detection_histories=histories)

        _, loaded_hist, _ = cache.load("/test", {"rows": 6})
        assert 0 in loaded_hist
        assert len(loaded_hist[0]) == 2
        assert loaded_hist[0][0][0] == (1.0, 5, 15.0)


class TestRerunVisualizer:
    def test_rerun_visualizer_import(self):
        """Test that RerunVisualizer can be imported."""
        from mvis.analysis.rerun_viz import RerunVisualizer
        # Just verify the class exists and has key methods
        assert hasattr(RerunVisualizer, 'log_detection')
        assert hasattr(RerunVisualizer, 'log_board_points')
        assert hasattr(RerunVisualizer, 'log_calibration_result')
