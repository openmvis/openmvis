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
"""Tests for the unified DataLoader."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mvis.io.data_loader import DataLoader
from mvis.io.image_source import ImageSource


# ---------- helpers ----------

def _create_test_images(dir_path, n=5, size=(100, 100)):
    """Create n test images in dir_path."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.random.randint(0, 255, size, dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"img_{i:05d}.png"), img)


def _create_euroc_dir(cam_dir, n=5, dt_ns=100_000_000):
    """Create EuRoC-format directory with n images."""
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
    """Create minimal CalibrationConfig for testing."""
    from mvis.core.config import CalibrationConfig, CameraConfig, BoardConfig

    cam = CameraConfig(
        camera_id=camera_id, source_type=source_type,
        source_path=source_path, topic=topic or "",
    )
    board = BoardConfig(board_type="chessboard", rows=5, cols=7,
                        square_size=0.03)
    return CalibrationConfig(cameras=[cam], board=board)


def _make_stereo_config(left_path, right_path):
    """Create stereo config for testing."""
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


# ---------- TestDataLoader ----------

class TestDataLoader:

    def test_stream_images_yields_arrays(self, tmp_path):
        _create_test_images(tmp_path / "cam0", n=5)
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)
        assert loader.is_lazy is True

        frames = list(loader.stream())
        assert len(frames) == 5
        cid, img, ts = frames[0]
        assert cid == 0
        assert isinstance(img, np.ndarray)
        assert isinstance(ts, float)

    def test_stream_images_sorted_by_filename(self, tmp_path):
        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()
        for value, name in [(3, "c.png"), (1, "a.png"), (2, "b.png")]:
            cv2.imwrite(str(cam_dir / name),
                        np.full((10, 10), value, dtype=np.uint8))
        config = _make_config("images", str(cam_dir))
        frames = list(DataLoader(config).stream())
        assert [int(img[0, 0]) for _, img, _ in frames] == [1, 2, 3]
        assert [ts for _, _, ts in frames] == [0.0, 1.0, 2.0]

    def test_stream_images_multi_camera(self, tmp_path):
        _create_test_images(tmp_path / "left", n=3)
        _create_test_images(tmp_path / "right", n=3)
        config = _make_stereo_config(
            str(tmp_path / "left"), str(tmp_path / "right"))
        frames = list(DataLoader(config).stream())
        assert sum(1 for cid, _, _ in frames if cid == 0) == 3
        assert sum(1 for cid, _, _ in frames if cid == 1) == 3

    def test_stream_images_empty_dir(self, tmp_path):
        (tmp_path / "cam0").mkdir()
        config = _make_config("images", str(tmp_path / "cam0"))
        assert list(DataLoader(config).stream()) == []

    def test_stream_images_missing_dir(self, tmp_path):
        # a missing camera folder fails loudly (before make_sources it gave 0 frames)
        config = _make_config("images", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            list(DataLoader(config).stream())

    def test_cache_key_images(self, tmp_path):
        _create_test_images(tmp_path / "cam0", n=1)
        config = _make_config("images", str(tmp_path / "cam0"))
        loader = DataLoader(config)
        # the source dir plus the camera dirs (capture_key, F4)
        assert loader.cache_key.startswith(f"{tmp_path}|topics=")
        assert str(tmp_path / "cam0") in loader.cache_key

    def test_is_lazy_false_for_rosbag(self):
        config = _make_config("rosbag_topic", "/fake/bag.bag",
                              topic="/cam0/image_raw")
        loader = DataLoader(config)
        assert loader.is_lazy is False

    def test_unknown_source_type_raises(self):
        config = _make_config("unknown_type", "/fake")
        with pytest.raises(ValueError, match="not supported"):
            list(DataLoader(config).stream())


# ---------- TestDataLoaderEuroc ----------

class TestDataLoaderEuroc:

    def test_stream_euroc_yields_every_frame(self, tmp_path):
        _create_euroc_dir(tmp_path / "cam0", n=10, dt_ns=50_000_000)
        config = _make_config("euroc", str(tmp_path / "cam0"))
        assert len(list(DataLoader(config).stream())) == 10

    def test_stream_euroc_returns_numpy(self, tmp_path):
        _create_euroc_dir(tmp_path / "cam0", n=3, dt_ns=100_000_000)
        config = _make_config("euroc", str(tmp_path / "cam0"))
        loader = DataLoader(config)
        assert loader.is_lazy is False

        _, img, ts = next(iter(loader.stream()))
        assert isinstance(img, np.ndarray)
        assert isinstance(ts, float)

    def test_stream_euroc_16bit_conversion(self, tmp_path):
        cam_dir = tmp_path / "cam0"
        data_dir = cam_dir / "data"
        data_dir.mkdir(parents=True)
        img16 = np.random.randint(0, 65535, (50, 50), dtype=np.uint16)
        cv2.imwrite(str(data_dir / "1000000000.png"), img16)
        with open(cam_dir / "data.csv", "w") as f:
            f.write("#timestamp [ns],filename\n")
            f.write("1000000000,1000000000.png\n")
        config = _make_config("euroc", str(cam_dir))
        _, img, _ = next(iter(DataLoader(config).stream()))
        assert img.dtype == np.uint8


# ---------- TestDataLoaderStreamImages ----------

class TestDataLoaderStreamImages:

    def test_stream_reads_the_pixels(self, tmp_path):
        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()
        original = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(cam_dir / "img_000.png"), original)
        config = _make_config("images", str(cam_dir))
        _, img, _ = next(iter(DataLoader(config).stream()))
        assert img.shape == (100, 100)
        np.testing.assert_array_equal(img, original)

    def test_stream_images_16bit_conversion(self, tmp_path):
        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()
        img16 = np.random.randint(0, 65535, (20, 20), dtype=np.uint16)
        cv2.imwrite(str(cam_dir / "img_000.png"), img16)
        config = _make_config("images", str(cam_dir))
        _, img, _ = next(iter(DataLoader(config).stream()))
        assert img.dtype == np.uint8

    def test_stream_is_lazy(self, tmp_path):
        _create_test_images(tmp_path / "cam0", n=3)
        config = _make_config("images", str(tmp_path / "cam0"))
        it = DataLoader(config).stream()
        assert iter(it) is it   # a generator: one image at a time


# ---------- TestImageSourceLen ----------

class TestImageSourceLen:

    def test_len_single_camera(self, tmp_path):
        _create_test_images(tmp_path, n=7)
        src = ImageSource(str(tmp_path))
        assert len(src) == 7

    def test_len_multi_camera(self, tmp_path):
        _create_test_images(tmp_path / "a", n=3)
        _create_test_images(tmp_path / "b", n=5)
        src = ImageSource([str(tmp_path / "a"), str(tmp_path / "b")])
        assert len(src) == 8

    def test_len_empty(self, tmp_path):
        src = ImageSource(str(tmp_path))
        assert len(src) == 0
