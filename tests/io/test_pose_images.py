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
"""Per-pose camera images for the moving frustum in the cam-only 3D plot.

Real detection thumbnails are matched to pose times through
`thumbnails/cam<N>/index.csv`; poses without one get a rendered view (board
projected through the calibrated model, detected corners on top).
"""

import base64

import cv2
import numpy as np

from mvis.analysis.pose_images import load_thumb_index, pose_images
from mvis.analysis.plot_3d import plot_camera_poses, progress_spec
from mvis.analysis.report_shell import plotly_to_html
from mvis.core.types import CameraParams, FrameObservations, Observation
from mvis.detection.chessboard import Chessboard

BOARD = Chessboard(rows=7, cols=5, square_size=0.03)
CAM = CameraParams(camera_id=0, intrinsics=np.array([300.0, 300.0, 160.0, 120.0]),
                   distortion=np.zeros(4), distortion_model="radtan",
                   image_size=(320, 240))


def _pose(z=-0.5):
    # camera on the board's -z side looking at it (T_board_cam, x/y centred)
    T = np.eye(4)
    T[:3, 3] = [0.06, 0.09, z]
    return T


def _write_thumbs(tmp_path, stamps, value=200):
    d = tmp_path / "thumbnails" / "cam0"
    d.mkdir(parents=True)
    lines = []
    for i, t in enumerate(stamps):
        cv2.imwrite(str(d / f"{i:05d}.jpg"), np.full((240, 320, 3), value, np.uint8))
        lines.append(f"{i},{t:.9f}")
    (d / "index.csv").write_text("\n".join(lines) + "\n")
    return tmp_path / "thumbnails"


def test_load_thumb_index(tmp_path):
    thumbs = _write_thumbs(tmp_path, [1.0, 1.1, 1.2])
    ts, paths = load_thumb_index(thumbs, 0)
    np.testing.assert_allclose(ts, [1.0, 1.1, 1.2])
    assert paths[2].name == "00002.jpg"
    assert load_thumb_index(tmp_path / "nope", 0) is None


def test_real_thumbnail_matched_by_time(tmp_path):
    thumbs = _write_thumbs(tmp_path, [1.0, 1.1, 1.2], value=200)
    imgs, src = pose_images([_pose(), _pose()], [1.1, 5.0], CAM, BOARD,
                            frames=[], thumb_dir=thumbs, width=32)
    assert imgs.shape == (2, 24, 32) and imgs.dtype == np.uint8
    assert src == ["real", "rendered"]          # 5.0 s has no thumbnail
    assert abs(int(imgs[0].mean()) - 200) <= 2


def test_rendered_view_shows_board_and_detections():
    # one detected corner at the image centre, marked white
    obs = [Observation(point_id=0, pixel=np.array([160.0, 120.0]))]
    frames = [FrameObservations(camera_id=0, timestamp=2.0, observations=obs,
                                image_size=(320, 240))]
    imgs, src = pose_images([_pose()], [2.0], CAM, BOARD, frames=frames,
                            thumb_dir=None, width=64)
    assert src == ["rendered"]
    img = imgs[0]
    assert img.max() == 255                      # detected corner
    assert len(np.unique(img)) >= 3              # background / board / corners


def test_plot_carries_images_for_the_frustum():
    poses = [_pose(-0.5 - 0.01 * k) for k in range(3)]
    imgs = np.stack([np.full((24, 32), 40 * k, np.uint8) for k in range(3)])
    fig = plot_camera_poses(poses, BOARD, images=imgs)
    spec = progress_spec(fig)
    v = spec["views"][0]
    assert (v["iw"], v["ih"]) == (32, 24)
    raw = np.frombuffer(base64.b64decode(v["img"]), np.uint8)
    np.testing.assert_array_equal(raw.reshape(3, 24, 32), imgs)
    assert v["img_idx"] == [0, 1, 2]
    surf = fig.data[spec["glyph"]["image"]]
    assert surf.type == "surface"
    np.testing.assert_array_equal(np.asarray(surf.surfacecolor), imgs[-1])
    assert "plane" not in spec["glyph"]          # the image replaces the tint
    html = plotly_to_html(fig)
    assert "surfacecolor" in html


def test_detection_writes_thumbnail_index(tmp_path):
    from mvis.io.detection_pipeline import DetectionPipeline
    (tmp_path / "cam0").mkdir()
    img = np.zeros((240, 320), np.uint8)
    det = DetectionPipeline(BOARD, thumb_dir=tmp_path)
    for i, t in enumerate([10.0, 10.1, 10.2]):
        det._write_thumbnail(0, i, img, [], t)
    ts, paths = load_thumb_index(tmp_path, 0)
    np.testing.assert_allclose(ts, [10.0, 10.1, 10.2])
    assert all(p.exists() for p in paths)
    # a new run (new pipeline) restarts the index instead of appending
    det2 = DetectionPipeline(BOARD, thumb_dir=tmp_path)
    det2._write_thumbnail(0, 0, img, [], 20.0)
    ts, _ = load_thumb_index(tmp_path, 0)
    np.testing.assert_allclose(ts, [20.0])


def test_pose_panel_coverage_and_views(tmp_path):
    from mvis.analysis.pose_images import pose_panel
    thumbs = _write_thumbs(tmp_path, [1.0, 1.1, 1.2])
    # corner in the top-left cell at t=1.0, bottom-right cell at t=1.2
    fr = [FrameObservations(camera_id=0, timestamp=1.0, image_size=(320, 240),
                            observations=[Observation(0, np.array([5.0, 5.0]))]),
          FrameObservations(camera_id=0, timestamp=1.2, image_size=(320, 240),
                            observations=[Observation(1, np.array([315.0, 235.0])),
                                          Observation(2, np.array([314.0, 236.0]))])]
    p = pose_panel([_pose()] * 3, [1.0, 1.1, 1.2], CAM, BOARD, fr,
                   thumb_dir=thumbs, width=40)
    assert (p["w"], p["h"]) == (320, 240)
    assert p["uv"] == [[5, 5], [], [315, 235, 314, 236]]
    assert p["n"] == [1, 1, 3]
    assert p["cov"] == [round(100 / 48, 1), round(100 / 48, 1), round(200 / 48, 1)]
    assert p["jpg_idx"] == [0, 1, 2] and all(p["jpg"])


def test_pose_panel_caps_stored_views():
    from mvis.analysis.pose_images import pose_panel
    p = pose_panel([_pose()] * 50, np.arange(50.0), CAM, BOARD, [],
                   width=20, max_images=10)
    assert len(p["jpg"]) == len(p["jpg_idx"]) == 10
    assert p["jpg_idx"][0] == 0 and p["jpg_idx"][-1] == 49


def test_html_side_panel_rendered_with_panel():
    from mvis.analysis.pose_images import pose_panel
    poses = [_pose(-0.5 - 0.01 * k) for k in range(3)]
    p = pose_panel(poses, [0.0, 0.1, 0.2], CAM, BOARD, [], width=20)
    html = plotly_to_html(plot_camera_poses(poses, BOARD, times=[0.0, 0.1, 0.2],
                                            panel=p))
    assert 'class="mvis-3d"' in html and 'class="mvis-panel"' in html
    assert "Image coverage so far" in html and "data:image/jpeg;base64," in html
    assert 'class="mvis-panel"' not in plotly_to_html(
        plot_camera_poses(poses, BOARD, times=[0.0, 0.1, 0.2]))


def test_camera_views_match_each_camera_frames():
    from mvis.analysis.pose_images import camera_views
    cam1 = CameraParams(camera_id=1, intrinsics=CAM.intrinsics, distortion=np.zeros(4),
                        distortion_model="radtan", image_size=(320, 240))
    fr = lambda cid, t, pid: FrameObservations(
        camera_id=cid, timestamp=t, image_size=(320, 240),
        observations=[Observation(pid, np.array([100.0, 100.0]))])
    frames = {0: [fr(0, 1.0, 3), fr(0, 1.1, 4)], 1: [fr(1, 1.002, 7), fr(1, 1.042, 8)]}
    T_ext = np.eye(4); T_ext[0, 3] = -0.1
    v = camera_views([_pose(), _pose()], [1.0, 1.1], [cam1, CAM], {1: T_ext}, 0,
                     BOARD, frames)
    assert [x["camera_id"] for x in v] == [0, 1]                 # ref first
    assert v[0]["ids"] == [[3], [4]]
    assert v[1]["ids"] == [[7], []]      # 1.1 s: nearest cam1 frame 58 ms > 20 ms tol
    assert v[1]["images"].shape[0] == 2 and len(v[1]["panel"]["uv"]) == 2


def test_frustum_images_capped_with_index():
    from mvis.analysis import plot_3d
    n = plot_3d._MAX_FRUSTUM_IMAGES + 50
    poses = [_pose(-0.5 - 0.001 * k) for k in range(n)]
    imgs = np.zeros((n, 6, 8), np.uint8)
    v = progress_spec(plot_camera_poses(poses, BOARD, images=imgs))["views"][0]
    assert len(v["img_idx"]) == plot_3d._MAX_FRUSTUM_IMAGES
    assert v["img_idx"][0] == 0 and v["img_idx"][-1] == n - 1
    assert len(base64.b64decode(v["img"])) == plot_3d._MAX_FRUSTUM_IMAGES * 48
