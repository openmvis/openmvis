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
"""Tests for streaming detection — DataLoader.stream() + DetectionPipeline.run_streaming()."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mvis.io.data_loader import DataLoader
from mvis.io.detection_pipeline import DetectionPipeline, DetectionResult


# ---------- helpers ----------

def _make_chessboard_image(rows=5, cols=7, sq=40):
    h = (rows + 1) * sq + 2 * sq
    w = (cols + 1) * sq + 2 * sq
    img = np.ones((h, w), dtype=np.uint8) * 200
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = sq + r * sq, sq + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = 30
    return img


def _create_chessboard_images(dir_path, n=8, rows=5, cols=7, sq=40):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    base = _make_chessboard_image(rows, cols, sq)
    for i in range(n):
        noise = np.random.randint(-5, 5, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dir_path / f"img_{i:05d}.png"), img)


def _create_test_images(dir_path, n=5, size=(100, 100)):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.random.randint(0, 255, size, dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"img_{i:05d}.png"), img)


def _create_euroc_dir(cam_dir, n=5, dt_ns=100_000_000):
    cam_dir = Path(cam_dir)
    data_dir = cam_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(cam_dir / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for i in range(n):
            ts = 1_000_000_000 + i * dt_ns
            fname = f"{ts}.png"
            f.write(f"{ts},{fname}\n")
            img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
            cv2.imwrite(str(data_dir / fname), img)


def _make_config(source_type, source_path, topic=None, camera_id=0):
    from mvis.core.config import CalibrationConfig, CameraConfig, BoardConfig
    cam = CameraConfig(camera_id=camera_id, source_type=source_type,
                       source_path=source_path, topic=topic or "")
    board = BoardConfig(board_type="chessboard", rows=5, cols=7,
                        square_size=0.03)
    return CalibrationConfig(cameras=[cam], board=board)


def _make_stereo_config(left_path, right_path):
    from mvis.core.config import CalibrationConfig, CameraConfig, BoardConfig
    cams = [
        CameraConfig(camera_id=0, source_type="images",
                     source_path=left_path),
        CameraConfig(camera_id=1, source_type="images",
                     source_path=right_path),
    ]
    board = BoardConfig(board_type="chessboard", rows=5, cols=7,
                        square_size=0.03)
    return CalibrationConfig(cameras=cams, board=board)


def _make_board(rows=5, cols=7, sq_size=0.04):
    from mvis.detection.chessboard import Chessboard
    return Chessboard(rows=rows, cols=cols, square_size=sq_size)


# ---------- TestDataLoaderStream ----------

class TestDataLoaderStream:

    def test_stream_images_yields_tuples(self, tmp_path):
        """stream() yields (cam_id, image, timestamp) tuples."""
        _create_test_images(tmp_path / "cam0", n=5)
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        frames = list(loader.stream())
        assert len(frames) == 5
        cid, img, ts = frames[0]
        assert isinstance(cid, int)
        assert isinstance(img, np.ndarray)
        assert isinstance(ts, float)

    def test_stream_images_multi_camera(self, tmp_path):
        """stream() yields frames from all cameras."""
        _create_test_images(tmp_path / "left", n=3)
        _create_test_images(tmp_path / "right", n=3)
        config = _make_stereo_config(
            str(tmp_path / "left"), str(tmp_path / "right"))
        loader = DataLoader(config)

        frames = list(loader.stream())
        assert len(frames) == 6
        cam_ids = {cid for cid, _, _ in frames}
        assert cam_ids == {0, 1}

    def test_stream_euroc(self, tmp_path):
        """stream() yields from EuRoC format."""
        _create_euroc_dir(tmp_path / "cam0", n=5)
        config = _make_config("euroc", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        frames = list(loader.stream())
        assert len(frames) == 5
        _, img, ts = frames[0]
        assert isinstance(img, np.ndarray)
        assert ts > 0

    def test_stream_no_subsampling(self, tmp_path):
        """stream() yields ALL frames, no target_hz applied."""
        _create_test_images(tmp_path / "cam0", n=20)
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        frames = list(loader.stream())
        assert len(frames) == 20


# ---------- TestDetectionPipelineStreaming ----------

class TestDetectionPipelineStreaming:

    def test_run_streaming_detects_all(self, tmp_path):
        """run_streaming() detects all frames."""
        _create_chessboard_images(tmp_path / "cam0", n=10)
        board = _make_board()
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        det = DetectionPipeline(board, thumb_dir=tmp_path / "thumbs",
                                cache_path=str(tmp_path / "cache/det.mvisdb"))
        result = det.run_streaming(loader, cache_key="test")

        assert isinstance(result, DetectionResult)
        assert len(result.all_frames[0]) > 0
        assert 0 in result.image_sizes

    def test_run_streaming_caches_all_detections(self, tmp_path):
        """Cache stores ALL detected frames, not just diversity-filtered."""
        _create_chessboard_images(tmp_path / "cam0", n=15)
        board = _make_board()
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        cache_path = str(tmp_path / "cache/det.mvisdb")
        det = DetectionPipeline(board, cache_path=cache_path,
                                thumb_dir=tmp_path / "thumbs")
        result = det.run_streaming(loader, cache_key="test")

        # Verify through the cache API (the sqlite store since I2a)
        from mvis.io.detection_pipeline import _make_cache
        cached, _, _ = _make_cache(cache_path).load("test", det._board_config())
        # Cache has ALL detected frames >= diversity-filtered count
        assert len(cached[0]) >= len(result.all_frames[0])

    def test_run_streaming_cache_hit_skips_loading(self, tmp_path):
        """Second run loads from cache, no images loaded."""
        _create_chessboard_images(tmp_path / "cam0", n=5)
        board = _make_board()
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        cache_path = str(tmp_path / "cache/det.mvisdb")
        det = DetectionPipeline(board, cache_path=cache_path,
                                thumb_dir=tmp_path / "thumbs")

        r1 = det.run_streaming(loader, cache_key="test")
        r2 = det.run_streaming(loader, cache_key="test")

        assert len(r2.all_frames[0]) == len(r1.all_frames[0])

    def test_run_streaming_different_target_hz(self, tmp_path):
        """Different target_hz on cache hit returns different frame count."""
        _create_chessboard_images(tmp_path / "cam0", n=20)
        board = _make_board()
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        cache_path = str(tmp_path / "cache/det.mvisdb")
        det = DetectionPipeline(board, cache_path=cache_path,
                                thumb_dir=tmp_path / "thumbs")

        r1 = det.run_streaming(loader, cache_key="test", target_hz=10.0)
        r2 = det.run_streaming(loader, cache_key="test", target_hz=2.0)
        assert len(r2.all_frames[0]) <= len(r1.all_frames[0])

    def test_run_streaming_thumbnails_on_disk(self, tmp_path):
        """Thumbnails written to disk during streaming."""
        _create_chessboard_images(tmp_path / "cam0", n=5)
        board = _make_board()
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        thumb_dir = tmp_path / "thumbs"
        det = DetectionPipeline(board, thumb_dir=thumb_dir)
        det.run_streaming(loader, cache_key="test")

        thumbs = list((thumb_dir / "cam0").glob("*.jpg"))
        assert len(thumbs) == 5

    def test_run_streaming_result_has_frame_observations(self, tmp_path):
        """Result contains FrameObservations, not raw images."""
        _create_chessboard_images(tmp_path / "cam0", n=10)
        board = _make_board()
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)

        det = DetectionPipeline(board, thumb_dir=tmp_path / "thumbs")
        result = det.run_streaming(loader, cache_key="test")

        for f in result.all_frames[0]:
            assert hasattr(f, 'observations')
            assert hasattr(f, 'timestamp')
            assert len(f.observations) > 0


def test_parallel_detection_output_does_not_depend_on_thread_completion_order(monkeypatch):
    """G0: the per-camera workers finish in any order; the returned image sizes
    (stored in capture.spec_json) must be keyed by camera id, not by which
    thread finished first.  cam0 is made slow so cam1 always finishes first."""
    import time
    import types
    from mvis.io import detection_pipeline as dp

    def fake_detect(img, board, cid, ts):
        if cid == 0:
            time.sleep(0.05)
        obs = [object()] * 10
        return types.SimpleNamespace(observations=obs, image_size=(640 + cid, 480),
                                     timestamp=ts)
    monkeypatch.setattr(dp, "detect_frame", fake_detect)

    class Loader:
        def stream(self):
            for i in range(3):
                for cid in (0, 1):
                    yield cid, np.zeros((4, 4), np.uint8), float(i)

    class Board:
        def min_points_for_pose(self):
            return 4

    pipe = DetectionPipeline.__new__(DetectionPipeline)
    pipe._board = Board(); pipe._thumb_dir = None
    pipe._write_thumbnail = lambda *a, **k: None
    results, sizes, counts = pipe._stream_detect_parallel(Loader(), [0, 1])
    assert list(sizes) == [0, 1]
    assert sizes == {0: (640, 480), 1: (641, 480)}
    assert [len(results[c]) for c in (0, 1)] == [3, 3]


# ---------- detection progress log ----------

def test_detect_progress_is_throttled_and_counts_per_camera(caplog):
    """A progress line every interval, never between: per-camera read and
    detected counts, the overall frame count and rate."""
    import logging
    from mvis.io.detection_pipeline import _DetectProgress

    now = [0.0]
    p = _DetectProgress(interval_s=30.0, clock=lambda: now[0])
    with caplog.at_level(logging.INFO, logger="mvis.io.detection_pipeline"):
        for _ in range(10):
            p.detected(0)
            p.read(0)
        p.read(1)
        assert not caplog.records                     # 0 s < 30 s: silent
        now[0] = 31.0
        p.read(1)
        now[0] = 40.0
        p.read(1)                                     # inside the next interval
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1
    assert "0m31s, 12 frames (0.4/s)" in lines[0]
    assert "cam0 10 read/10 detected" in lines[0]
    assert "cam1 2 read/0 detected" in lines[0]


def test_detect_progress_reads_only_and_disabled(caplog, monkeypatch):
    import logging
    from mvis.io.detection_pipeline import _DetectProgress

    now = [0.0]
    p = _DetectProgress(counts_detections=False, interval_s=1.0, clock=lambda: now[0])
    monkeypatch.setenv("MVIS_DETECT_PROGRESS_S", "0")
    off = _DetectProgress(clock=lambda: now[0])
    with caplog.at_level(logging.INFO, logger="mvis.io.detection_pipeline"):
        now[0] = 5.0
        p.read(2)
        off.read(0)
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1
    assert lines[0].endswith("cam2 1 read")


def test_detect_progress_logged_on_every_detection_path(tmp_path, monkeypatch, caplog):
    """The sweep (single camera), parallel (stereo) and legacy sequential
    (geometry-free fallback) detection paths all report reads and detections
    while they run."""
    import logging
    monkeypatch.setenv("MVIS_DETECT_PROGRESS_S", "1e-9")
    _create_chessboard_images(tmp_path / "L", n=3)
    _create_chessboard_images(tmp_path / "R", n=3)
    board = _make_board()

    def progress_lines():
        return [r.getMessage() for r in caplog.records
                if "detecting:" in r.getMessage()]

    for name, config in (
            ("single", _make_config("images", str(tmp_path / "L"))),
            ("stereo", _make_stereo_config(str(tmp_path / "L"), str(tmp_path / "R")))):
        caplog.clear()
        det = DetectionPipeline(board, cache_path=str(tmp_path / f"{name}.mvisdb"))
        with caplog.at_level(logging.INFO, logger="mvis.io.detection_pipeline"):
            det.run_streaming(DataLoader(config), cache_key=name)
        progress = progress_lines()
        assert progress, name
        assert "cam0" in progress[-1], name
        assert "detected" in progress[-1], name
        assert "/0 detected" not in progress[-1], name   # final line (finish): all counted
        assert "cam0 3 read/3 detected" in progress[-1], name

    caplog.clear()
    det = DetectionPipeline(board)
    with caplog.at_level(logging.INFO, logger="mvis.io.detection_pipeline"):
        det._stream_detect_sequential(
            DataLoader(_make_config("images", str(tmp_path / "L"))))
    progress = progress_lines()
    assert progress
    assert "cam0 3 read/3 detected" in progress[-1]


def test_detect_progress_finish_and_off():
    """finish() logs the final counts once; with progress off (interval 0) nothing is logged."""
    import logging
    from mvis.io.detection_pipeline import _DetectProgress
    t = [0.0]
    p = _DetectProgress(interval_s=10.0, clock=lambda: t[0])
    p.read(0); p.detected(0); p.read(0)
    lg = logging.getLogger("mvis.io.detection_pipeline")
    seen = []
    h = logging.Handler(); h.emit = lambda r: seen.append(r.getMessage())
    lg.addHandler(h); old = lg.level; lg.setLevel(logging.INFO)
    try:
        p.finish()
        _DetectProgress(interval_s=0.0).finish()
    finally:
        lg.removeHandler(h); lg.setLevel(old)
    assert len(seen) == 1 and "cam0 2 read/1 detected" in seen[0]
