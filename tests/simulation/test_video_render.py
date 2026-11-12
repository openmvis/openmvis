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
"""informative-video P1: _draw_path draws a growing camera path (up to the current frame) over a
faint full-path ghost, with start/end/current markers."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from mvis.simulation.video import _draw_path


def _ax():
    return plt.figure().add_subplot(111, projection="3d")


def test_draw_path_grows_up_to_index_over_a_ghost():
    path = np.array([[i, 0, 0] for i in range(10)], float)
    ax = _ax()
    _draw_path(ax, path, upto=3)
    bright = ax.lines[-1]                           # traversed path drawn on top
    assert len(bright.get_data_3d()[0]) == 4        # indices 0..3 inclusive
    ghost = ax.lines[0]                             # faint full path for context
    assert len(ghost.get_data_3d()[0]) == 10
    assert ghost.get_alpha() is not None and ghost.get_alpha() < 0.2


def test_draw_path_full_when_upto_none_has_no_ghost():
    path = np.array([[i, 0, 0] for i in range(10)], float)
    ax = _ax()
    _draw_path(ax, path, upto=None)
    assert len(ax.lines) == 1
    assert len(ax.lines[0].get_data_3d()[0]) == 10


def test_draw_path_has_start_end_and_current_markers():
    path = np.array([[i, 0, 0] for i in range(10)], float)
    ax = _ax()
    _draw_path(ax, path, upto=3)
    assert len(ax.collections) == 3                 # start + end + current

    ax2 = _ax()
    _draw_path(ax2, path, upto=None)
    assert len(ax2.collections) == 2                # start + end only (no current marker)


def test_residual_color_thresholds():
    """P3: per-corner reprojection residual colored by magnitude (green<0.5, orange<1, red>=1 px)."""
    from mvis.simulation.video import _residual_color
    assert _residual_color(0.3) == "tab:green"
    assert _residual_color(0.7) == "tab:orange"
    assert _residual_color(1.5) == "tab:red"
    assert _residual_color(0.5) == "tab:orange"   # boundary: 0.5 is no longer "green"
    assert _residual_color(1.0) == "tab:red"


def test_camera_source_label_topic_folder_fallback():
    """rig-view P1: per-camera source id for the panel — ROS topic (ros1), euroc cam folder, else cam<c>."""
    from mvis.simulation.video import camera_source_label
    assert camera_source_label(["/blackfly/image_raw", "/t265/f1"], "", "ros1", 0) == "/blackfly/image_raw"
    assert camera_source_label([], "/d/MIC05_imucalib2/mav0", "euroc", 1) == "mav0/cam1"
    assert camera_source_label([], "", "ros1", 2) == "cam2"
    assert camera_source_label(["/a"], "", "ros1", 3) == "cam3"     # index past the topic list


def test_rig_extent_center_and_half_with_pad():
    import numpy as np
    from mvis.simulation.video import _rig_extent
    origins = [[0, 0, 0], [0.1, 0, 0], [0, 0.05, 0]]
    c, half = _rig_extent(origins, pad=0.0)
    assert np.allclose(c, [0.05, 0.025, 0.0])
    assert abs(half - 0.05) < 1e-9                 # half of the max span (0.10 in x)
    c2, half2 = _rig_extent(origins, pad=0.02)
    assert abs(half2 - 0.07) < 1e-9                # pad adds to the half-extent


def test_draw_sensor_frame_adds_three_axes_and_a_label():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mvis.simulation.video import _draw_sensor_frame
    ax = plt.figure().add_subplot(111, projection="3d")
    _draw_sensor_frame(ax, np.eye(4), "cam0", kind="cam")
    assert len(ax.lines) == 3                       # 3 RGB body axes
    assert len(ax.texts) == 1 and "cam0" in ax.texts[0].get_text()


def test_parse_rig_view():
    """rig-view tweak: '--rig-view=elev,azim' -> (elev, azim); default on empty/bad."""
    from mvis.simulation.video import parse_rig_view
    assert parse_rig_view("30,-45") == (30.0, -45.0)
    assert parse_rig_view("") == (20.0, -60.0)           # default
    assert parse_rig_view(None) == (20.0, -60.0)
    assert parse_rig_view("bogus") == (20.0, -60.0)      # malformed -> default
    assert parse_rig_view("10,20,30") == (20.0, -60.0)   # wrong arity -> default
