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
"""G6 step 2: io.data_source.make_sources, the one camera-only reader factory (DataLoader
and, from G7, `mvis calibrate`): one reader per bag, one per other camera, config ids."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from mvis.core.config import BoardConfig, CalibrationConfig, CameraConfig
from mvis.io.data_loader import DataLoader
from mvis.io.data_source import make_sources, stream_cameras
from mvis.io.ros1_bag_source import Ros1BagSource


def _cam(cid, kind, path, topic=""):
    return CameraConfig(camera_id=cid, source_type=kind, source_path=str(path), topic=topic)


def _images(d, values):
    Path(d).mkdir(parents=True)
    for i, v in enumerate(values):
        cv2.imwrite(str(Path(d) / f"{i:03d}.png"), np.full((8, 8), v, np.uint8))


def test_one_reader_per_bag_with_the_config_ids():
    cams = [_cam(4, "rosbag_topic", "/a.bag", "/left"), _cam(7, "ros1bag", "/a.bag", "/right"),
            _cam(2, "ros1bag", "/b.bag", "/cam")]
    (a, a_ids), (b, b_ids) = make_sources(cams)
    assert isinstance(a, Ros1BagSource) and a._topics == ["/left", "/right"]
    assert (a_ids, b._topics, b_ids) == ([4, 7], ["/cam"], [2])


def test_stream_maps_local_indices_to_config_ids_across_types(tmp_path):
    _images(tmp_path / "left", [10, 11])
    _images(tmp_path / "right", [20])
    stamps = tmp_path / "t.csv"
    stamps.write_text("000.png,5.5\n")
    cams = [_cam(3, "images", tmp_path / "left"),
            _cam(9, "timestamped_images", f"{tmp_path / 'right'},{stamps}")]
    got = [(cid, int(img[0, 0]), t) for cid, img, t in stream_cameras(cams)]
    assert got == [(3, 10, 0.0), (3, 11, 1.0), (9, 20, 5.5)]


@pytest.mark.parametrize("cam, match", [
    (_cam(0, "ros1bag", "/a.bag"), "needs a topic"),
    (_cam(0, "lidar", "/x"), "not supported")])
def test_bad_cameras_raise(cam, match):
    with pytest.raises(ValueError, match=match):
        make_sources([cam])


def test_cache_key_names_every_bag_only_for_several_bags():
    board = BoardConfig(board_type="chessboard", rows=5, cols=7, square_size=0.03)
    key = lambda cams: DataLoader(CalibrationConfig(cameras=cams, board=board)).cache_key  # noqa: E731
    one = key([_cam(0, "ros1bag", "/a.bag", "/l"), _cam(1, "ros1bag", "/a.bag", "/r")])
    assert one == "/a.bag|topics=['/l', '/r']|stamp=header"     # unchanged by G6
    two = key([_cam(0, "ros1bag", "/a.bag", "/l"), _cam(1, "ros1bag", "/b.bag", "/r")])
    assert two != key([_cam(0, "ros1bag", "/a.bag", "/l"), _cam(1, "ros1bag", "/c.bag", "/r")])
