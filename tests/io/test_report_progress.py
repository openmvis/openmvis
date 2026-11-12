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
"""Calibration progress bar under the 3D trajectory plots (html-reports).

Both reports' trajectory figures carry a progress spec (covered-path + current
marker traces); `plotly_to_html` renders a play/scrub strip under any figure
that has one, and nothing extra for figures without it.
"""

import numpy as np
import gtsam

from mvis.analysis.plot_3d import (
    add_progress, plot_cam_imu_scene, plot_camera_poses, progress_spec)
from mvis.analysis.report_shell import plotly_to_html
from mvis.detection.chessboard import Chessboard
import plotly.graph_objects as go


def _values(n=8):
    X = gtsam.symbol_shorthand.X
    vals = gtsam.Values()
    for k in range(n):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1 * k, 0.0, 0.0)))
    return vals


def test_scene_progress_spec_uses_state_times():
    t = 100.0 + 0.5 * np.arange(8)
    fig = plot_cam_imu_scene(_values(), t_states=t)
    spec = progress_spec(fig)
    assert spec is not None
    assert len(spec["xyz"]) == 8
    np.testing.assert_allclose(spec["t"], 0.5 * np.arange(8))   # relative to start
    assert fig.data[spec["done"]].name == "covered"
    assert fig.data[spec["cur"]].name == "current"
    # starts at the end of the run: the static view still shows the whole path
    assert len(fig.data[spec["done"]].x) == 8
    assert np.isclose(float(fig.data[spec["cur"]].x[0]), 0.7)


def test_scene_progress_without_times_counts_states():
    spec = progress_spec(plot_cam_imu_scene(_values()))
    assert spec is not None and spec["t"] is None and spec["unit"] == "state"


def test_scene_progress_ignores_mismatched_times():
    spec = progress_spec(plot_cam_imu_scene(_values(), t_states=np.arange(3.0)))
    assert spec["t"] is None


def test_camera_only_poses_get_progress():
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    poses = []
    for k in range(5):
        T = np.eye(4)
        T[:3, 3] = [0.02 * k, 0.0, -0.5]
        poses.append(T)
    spec = progress_spec(plot_camera_poses(poses, board))
    assert spec is not None and len(spec["xyz"]) == 5 and spec["unit"] == "pose"


def test_single_pose_has_no_progress():
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    T = np.eye(4); T[2, 3] = -0.5
    assert progress_spec(plot_camera_poses([T], board)) is None


def test_html_has_progress_strip_only_when_spec_present():
    fig = plot_cam_imu_scene(_values(), t_states=np.arange(8.0))
    html = plotly_to_html(fig)
    assert 'class="mvis-progress"' in html
    assert 'type="range"' in html and 'max="7"' in html
    assert "100 %" in html                  # initial label = whole run
    assert "Plotly.restyle" in html
    # the spec is consumed by the strip, not serialized twice into the figure
    assert "mvis_progress" not in html
    plain = plotly_to_html(go.Figure(go.Scatter3d(x=[0], y=[0], z=[0])))
    assert "mvis-progress" not in plain


def test_add_progress_leaves_input_figure_traces_untouched():
    fig = go.Figure()
    add_progress(fig, np.zeros((4, 3)))
    assert len(fig.data) == 2


def test_camera_only_poses_progress_in_seconds():
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    poses = [np.eye(4) for _ in range(3)]
    for k, T in enumerate(poses):
        T[:3, 3] = [0.02 * k, 0.0, -0.5]
    spec = progress_spec(plot_camera_poses(poses, board, times=[10.0, 10.1, 10.4]))
    np.testing.assert_allclose(spec["t"], [0.0, 0.1, 0.4])


def _names(fig, spec):
    return {k: fig.data[i].name for k, i in spec["glyph"].items()}


def test_camera_only_progress_moves_frustum_and_axes():
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    poses = [np.eye(4) for _ in range(3)]
    for k, T in enumerate(poses):
        T[:3, 3] = [0.02 * k, 0.0, -0.5]
    fig = plot_camera_poses(poses, board)
    spec = progress_spec(fig)
    assert len(spec["R"]) == 3 and len(spec["R"][0]) == 9
    assert _names(fig, spec) == {"frustum": "current frustum",
                                 "plane": "current image plane",
                                 "x": "current x", "y": "current y",
                                 "z": "current z", "ray": "current ray"}
    # identity pose looks along +z (board frame) = display +x: from the camera
    # at display x = -0.5 the ray ends on the board plane x = 0
    ray = fig.data[spec["glyph"]["ray"]]
    assert np.isclose(float(ray.x[1]), 0.0)
    # glyph starts at the last pose: the optical axis (camera z) leaves from it
    z = fig.data[spec["glyph"]["z"]]
    last = np.array([z.x[0], z.y[0], z.z[0]])
    np.testing.assert_allclose(last, spec["xyz"][-1])


def test_scene_progress_moves_imu_axes_only():
    fig = plot_cam_imu_scene(_values())
    spec = progress_spec(fig)
    assert set(spec["glyph"]) == {"x", "y", "z"}
    assert spec["fr"] == 0


def test_html_carries_glyph_update():
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    poses = [np.eye(4) for _ in range(3)]
    for k, T in enumerate(poses):
        T[:3, 3] = [0.02 * k, 0.0, -0.5]
    html = plotly_to_html(plot_camera_poses(poses, board))
    assert "glyph" in html and '"fr": ' in html


def test_aprilgrid_board_drawn_with_tags_and_kept_corners():
    from mvis.detection.board_factory import create_board
    from types import SimpleNamespace as NS
    board = create_board(NS(board_type="apriltag", rows=2, cols=3, tag_size=0.05,
                            tag_spacing=0.3, tag_family="t36h11"))
    T = np.eye(4); T[2, 3] = -0.5
    T2 = T.copy(); T2[0, 3] = 0.05
    fig = plot_camera_poses([T, T2], board, corner_subsample="top_left")
    assert fig.data[0].name == "Board corners (6 used / 24)"
    tags = [d for d in fig.data if d.name == "tags"]
    assert len(tags) == 1 and len(tags[0].i) == 2 * 6      # 2 triangles per tag
    assert any(d.name == "board" and d.type == "mesh3d" for d in fig.data)


def test_cam_imu_scene_rig_camera_rides_with_imu():
    """Base camera frustum sits at X(k) . T_CtoI; the IMU triad at X(k); the
    board is drawn as a plate in the world and the ray hits its plane."""
    from mvis.detection.board_factory import create_board
    from types import SimpleNamespace as NS
    board = create_board(NS(board_type="apriltag", rows=2, cols=2, tag_size=0.05,
                            tag_spacing=0.3, tag_family="t36h11"))
    vals = _values()                                   # IMU at (0.1k, 0, 0)
    # board 1 m ahead along world +z, facing the rig (board z -> world -z)
    T_BinW = gtsam.Pose3(gtsam.Rot3.Rx(np.pi), gtsam.Point3(0.0, 0.0, 1.0))
    T_ci = np.eye(4); T_ci[:3, 3] = [0.0, 0.05, 0.0]   # camera 5 cm along IMU y
    imgs = np.zeros((8, 6, 8), np.uint8)
    fig = plot_cam_imu_scene(vals, board=board, T_BinW=T_BinW,
                             t_states=np.arange(8.0),
                             base_cam=dict(T_CtoI=T_ci, images=imgs, panel=None))
    spec = progress_spec(fig)
    g = spec["glyph"]
    assert {"frustum", "image", "ray", "x", "y", "z"} <= set(g)
    fr = fig.data[g["frustum"]]
    np.testing.assert_allclose([fr.x[0], fr.y[0], fr.z[0]], [0.7, 0.05, 0.0], atol=1e-9)
    xa = fig.data[g["x"]]
    np.testing.assert_allclose([xa.x[0], xa.y[0], xa.z[0]], [0.7, 0.0, 0.0], atol=1e-9)
    ray = fig.data[g["ray"]]                           # camera z = world +z -> board
    assert np.isclose(float(ray.z[1]), 1.0)
    assert spec["views"][0]["cam"][9:] == [0.0, 0.05, 0.0]
    assert any(d.name == "tags" for d in fig.data)


def test_cam_imu_loader_thumb_writer(tmp_path):
    from mvis.io.cam_imu_loader import _ThumbWriter, _thumbs_for
    from mvis.analysis.pose_images import load_thumb_index
    from types import SimpleNamespace as NS
    cache = NS(_path=str(tmp_path / "cam_imu_bag_abc"))
    w = _ThumbWriter(cache, Chessboard(rows=7, cols=5, square_size=0.03))
    img = np.zeros((48, 64), np.uint8)
    for t in (5.0, 5.05):
        w(0, img, NS(observations=[], timestamp=t))
    td = _thumbs_for(str(tmp_path / "cam_imu_bag_abc.thumbs"), 0)
    ts, paths = load_thumb_index(td, 0)
    np.testing.assert_allclose(ts, [5.0, 5.05])
    assert all(p.exists() for p in paths)
    assert _thumbs_for(td, 1) is None


def _two_cam_fig(ids1):
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    poses = [np.eye(4) for _ in range(3)]
    for k, T in enumerate(poses):
        T[:3, 3] = [0.02 * k, 0.0, -0.5]
    T_ext = np.eye(4); T_ext[0, 3] = -0.1           # ext T_to_from (ref <- cam1)
    views = [dict(camera_id=0, ids=[[0, 1], [1], [0, 1, 2]]),
             dict(camera_id=1, ids=ids1)]
    return plot_camera_poses(poses, board, extrinsics=[(1, T_ext)],
                             times=[0.0, 0.1, 0.2], views=views)


def test_multi_camera_views_and_covis():
    fig = _two_cam_fig([[5], [], [1, 2, 3]])
    spec = progress_spec(fig)
    v = spec["views"]
    assert [x["name"] for x in v] == ["cam0", "cam1"]
    assert v[0]["cam"] is None and v[1]["cam"][9:] == [0.1, 0.0, 0.0]
    # last pose: 1, 2 seen by both -> purple, 0 cam0 only, 3 cam1 only
    cv = spec["covis"]
    corners = fig.data[cv["trace"]].marker
    assert corners.color[1] == corners.color[2] == cv["both"]
    assert corners.color[0] == v[0]["color"] and corners.color[3] == v[1]["color"]
    html = plotly_to_html(fig)
    assert 'class="mvis-progress-cam"' in html and '<option value="1">cam1</option>' in html


def test_single_camera_has_no_selector_or_covis():
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    poses = [np.eye(4) for _ in range(3)]
    for k, T in enumerate(poses):
        T[:3, 3] = [0.02 * k, 0.0, -0.5]
    fig = plot_camera_poses(poses, board, views=[dict(camera_id=0, ids=[[0]] * 3)])
    spec = progress_spec(fig)
    assert "covis" not in spec
    assert 'class="mvis-progress-cam"' not in plotly_to_html(fig)


def test_cam_imu_scene_all_cameras_and_thinned_timeline():
    from mvis.detection.board_factory import create_board
    from types import SimpleNamespace as NS
    board = create_board(NS(board_type="apriltag", rows=2, cols=2, tag_size=0.05,
                            tag_spacing=0.3, tag_family="t36h11"))
    T_BinW = gtsam.Pose3(gtsam.Rot3.Rx(np.pi), gtsam.Point3(0.0, 0.0, 1.0))
    keep = [0, 3, 7]
    views = [dict(name="cam0", cam=(np.eye(3), np.zeros(3)), ids=[[0], [1], [0, 1]]),
             dict(name="cam1", cam=(np.eye(3), np.array([0.1, 0, 0])), ids=[[], [1], [1]])]
    fig = plot_cam_imu_scene(_values(), board=board, T_BinW=T_BinW,
                             t_states=np.arange(8.0), views=views, state_idx=keep)
    spec = progress_spec(fig)
    assert len(spec["xyz"]) == 3 and spec["t"] == [0.0, 3.0, 7.0]
    assert spec["unit"] == "step"
    assert [v["name"] for v in spec["views"]] == ["cam0", "cam1"]
    assert "covis" in spec                      # 2 cameras with ids
    html = plotly_to_html(fig)
    assert '<option value="1">cam1</option>' in html


def test_panel_uv_packed_as_int16_roundtrip():
    import base64
    from mvis.analysis.plot_3d import _pack_uv
    p = _pack_uv(dict(uv=[[5, 6, 300, 200], [], [1279, 959]], cov=[1, 1, 2], n=[2, 2, 3]))
    assert "uv" not in p and p["uvo"] == [0, 4, 4, 6]
    raw = np.frombuffer(base64.b64decode(p["uvb"]), "<i2")
    assert raw.tolist() == [5, 6, 300, 200, 1279, 959]
