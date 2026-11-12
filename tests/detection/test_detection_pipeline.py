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
"""Tests for DetectionPipeline.run_streaming — thumbnails, report thumbnails, caching."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mvis.io.data_loader import DataLoader
from mvis.io.detection_pipeline import DetectionPipeline, DetectionResult


# ---------- helpers ----------

def _make_chessboard_image(rows=5, cols=7, sq=40):
    """Create a synthetic chessboard image that OpenCV can detect."""
    h = (rows + 1) * sq + 2 * sq  # padding
    w = (cols + 1) * sq + 2 * sq
    img = np.ones((h, w), dtype=np.uint8) * 200  # light gray bg
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = sq + r * sq, sq + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = 30  # dark squares
    return img


def _create_chessboard_images(dir_path, n=8, rows=5, cols=7, sq=40):
    """Create n slightly varied chessboard images."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    base = _make_chessboard_image(rows, cols, sq)
    for i in range(n):
        # Add slight noise for diversity
        noise = np.random.randint(-5, 5, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dir_path / f"img_{i:05d}.png"), img)


def _make_board(rows=5, cols=7, sq_size=0.04):
    from mvis.detection.chessboard import Chessboard
    return Chessboard(rows=rows, cols=cols, square_size=sq_size)


def _loader(cam_dir):
    from mvis.core.config import CalibrationConfig, CameraConfig, BoardConfig
    cam = CameraConfig(camera_id=0, name="cam0", source_type="images",
                       source_path=str(cam_dir))
    board = BoardConfig(board_type="chessboard", rows=5, cols=7,
                        square_size=0.04)
    return DataLoader(CalibrationConfig(cameras=[cam], board=board))


# ---------- Tests ----------

class TestDetectionPipeline:

    def test_detect_from_image_dir(self, tmp_path):
        """Images streamed from a directory are detected."""
        _create_chessboard_images(tmp_path / "cam0", n=5)
        det = DetectionPipeline(_make_board(), thumb_dir=tmp_path / "thumbs")
        result = det.run_streaming(_loader(tmp_path / "cam0"))

        assert isinstance(result, DetectionResult)
        assert 0 in result.all_frames
        assert len(result.all_frames[0]) > 0
        assert 0 in result.image_sizes

    def test_thumbnails_written_to_disk(self, tmp_path):
        """Each detected frame gets a thumbnail JPEG on disk."""
        _create_chessboard_images(tmp_path / "cam0", n=5)
        thumb_dir = tmp_path / "thumbs"
        det = DetectionPipeline(_make_board(), thumb_dir=thumb_dir)
        det.run_streaming(_loader(tmp_path / "cam0"))

        thumbs_on_disk = list((thumb_dir / "cam0").glob("*.jpg"))
        assert len(thumbs_on_disk) == 5  # one per input image

    def test_no_thumb_dir_no_crash(self, tmp_path):
        """Works without thumb_dir (no thumbnails saved)."""
        _create_chessboard_images(tmp_path / "cam0", n=1)
        det = DetectionPipeline(_make_board(), thumb_dir=None)
        result = det.run_streaming(_loader(tmp_path / "cam0"))
        assert result.thumb_dir is None
        assert result.detection_images == {0: []}

    def test_report_thumbnails_limited(self, tmp_path):
        """detection_images has ≤35 loaded entries, rest are None."""
        _create_chessboard_images(tmp_path / "cam0", n=50)
        det = DetectionPipeline(_make_board(), thumb_dir=tmp_path / "thumbs")
        result = det.run_streaming(_loader(tmp_path / "cam0"))

        imgs = result.detection_images[0]
        loaded = [img for img in imgs if img is not None]
        assert len(loaded) <= 35

    def test_detection_cache(self, tmp_path):
        """Second run loads from cache, skips detection."""
        _create_chessboard_images(tmp_path / "cam0", n=5)
        board = _make_board()
        cache_path = str(tmp_path / "cache" / "detections.json")
        det = DetectionPipeline(board, cache_path=cache_path,
                                thumb_dir=tmp_path / "thumbs")

        # First run — detects. The cache is the sqlite store (no legacy .json).
        r1 = det.run_streaming(_loader(tmp_path / "cam0"), cache_key="test")
        assert Path(cache_path).with_suffix(".mvisdb").exists()
        n1 = len(r1.all_frames[0])

        # Second run — loads from cache
        det2 = DetectionPipeline(board, cache_path=cache_path,
                                 thumb_dir=tmp_path / "thumbs")
        r2 = det2.run_streaming(_loader(tmp_path / "cam0"), cache_key="test")
        assert len(r2.all_frames[0]) == n1

    def test_detection_cache_never_writes_legacy_json(self, tmp_path, monkeypatch):
        """The sqlite store is the only cache; MVIS_SQLITE_CACHE=0 no longer selects JSON."""
        monkeypatch.setenv("MVIS_SQLITE_CACHE", "0")
        _create_chessboard_images(tmp_path / "cam0", n=5)
        cache_path = str(tmp_path / "cache" / "detections.json")
        det = DetectionPipeline(_make_board(), cache_path=cache_path,
                                thumb_dir=tmp_path / "thumbs")
        det.run_streaming(_loader(tmp_path / "cam0"), cache_key="test")
        assert not Path(cache_path).exists()
        assert Path(cache_path).with_suffix(".mvisdb").exists()

    def test_full_rate_frames_returned(self, tmp_path):
        """all_frames_full_rate has all detected frames (before filtering)."""
        _create_chessboard_images(tmp_path / "cam0", n=10)
        det = DetectionPipeline(_make_board(), thumb_dir=tmp_path / "thumbs")
        result = det.run_streaming(_loader(tmp_path / "cam0"))

        assert len(result.all_frames_full_rate[0]) >= len(result.all_frames[0])
