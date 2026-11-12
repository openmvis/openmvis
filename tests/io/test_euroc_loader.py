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
"""Tests for EurocImageSource."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mvis.io.euroc_image_source import EurocImageSource


def _build_euroc_tree(root: Path, stamps_ns: list[int],
                      write_images: bool = True,
                      header: str = "#timestamp [ns],filename") -> Path:
    """Create a mav0/cam0/{data.csv,data/*.png} tree. Return the cam0 dir."""
    cam_dir = root / "mav0" / "cam0"
    img_dir = cam_dir / "data"
    img_dir.mkdir(parents=True)

    csv = cam_dir / "data.csv"
    lines = [header]
    for ts in stamps_ns:
        fname = f"{ts}.png"
        lines.append(f"{ts},{fname}")
        if write_images:
            img = np.full((4, 4), ts % 255, dtype=np.uint8)
            cv2.imwrite(str(img_dir / fname), img)
    csv.write_text("\n".join(lines) + "\n")
    return cam_dir


def test_iter_yields_sorted_triples(tmp_path):
    stamps = [30, 10, 50, 20, 40]
    cam_dir = _build_euroc_tree(tmp_path, stamps)
    src = EurocImageSource(str(cam_dir), camera_id=2)

    assert src.num_cameras() == 1
    assert len(src) == 5

    out = list(src)
    assert len(out) == 5
    ts_seconds = [t for _, t, _ in out]
    assert ts_seconds == sorted(ts_seconds)
    assert ts_seconds[0] == pytest.approx(10e-9)
    for img, _, cid in out:
        assert cid == 2
        assert img.shape == (4, 4)


def test_missing_images_skipped(tmp_path):
    stamps = [100, 200, 300]
    cam_dir = _build_euroc_tree(tmp_path, stamps)
    # Delete the middle image
    (cam_dir / "data" / "200.png").unlink()

    src = EurocImageSource(str(cam_dir))
    out = list(src)
    assert len(out) == 2
    assert [int(round(t * 1e9)) for _, t, _ in out] == [100, 300]


def test_bad_rows_skipped(tmp_path):
    cam_dir = tmp_path / "mav0" / "cam0"
    (cam_dir / "data").mkdir(parents=True)
    img = np.zeros((4, 4), dtype=np.uint8)
    cv2.imwrite(str(cam_dir / "data" / "10.png"), img)
    cv2.imwrite(str(cam_dir / "data" / "20.png"), img)

    (cam_dir / "data.csv").write_text(
        "#timestamp [ns],filename\n"
        "\n"                      # empty
        "# another comment\n"
        "not_a_number,10.png\n"   # bad ts
        "onlyonecol\n"            # malformed
        "10,10.png\n"
        "20,20.png\n"
    )
    src = EurocImageSource(str(cam_dir))
    out = list(src)
    assert len(out) == 2


def test_missing_csv_raises(tmp_path):
    cam_dir = tmp_path / "mav0" / "cam0"
    (cam_dir / "data").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        EurocImageSource(str(cam_dir))


def test_missing_data_dir_raises(tmp_path):
    cam_dir = tmp_path / "mav0" / "cam0"
    cam_dir.mkdir(parents=True)
    (cam_dir / "data.csv").write_text("#timestamp [ns],filename\n")
    with pytest.raises(FileNotFoundError):
        EurocImageSource(str(cam_dir))


def test_uint16_normalised_to_uint8(tmp_path):
    cam_dir = tmp_path / "mav0" / "cam0"
    (cam_dir / "data").mkdir(parents=True)
    img16 = np.full((4, 4), 256 * 100, dtype=np.uint16)
    cv2.imwrite(str(cam_dir / "data" / "10.png"), img16)
    (cam_dir / "data.csv").write_text("#timestamp [ns],filename\n10,10.png\n")

    src = EurocImageSource(str(cam_dir))
    out = list(src)
    assert len(out) == 1
    img, _, _ = out[0]
    assert img.dtype == np.uint8
    assert int(img[0, 0]) == 100
