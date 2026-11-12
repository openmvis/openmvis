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
"""analysis.display_frame: the display transform and the equal-aspect cube equal the copies they
replace (plot_3d, rerun_viz and simulation.visualization), byte for byte."""
from __future__ import annotations

import numpy as np

from mvis.analysis.display_frame import T_DISP, cube_bounds, to_display_points, to_display_pose


def _old_rerun_pts(pts):
    p = np.asarray(pts)
    out = np.empty_like(p)
    out[:, 0] = p[:, 2]
    out[:, 1] = p[:, 0]
    out[:, 2] = p[:, 1]
    return out


_OLD_T = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def _old_plotly(pts):
    pts = np.asarray(pts, float)
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2.0, max(float((hi - lo).max()) / 2.0, 1e-3) * 1.05


def _old_sim(pts):
    mn, mx = pts.min(0), pts.max(0)
    return (mn + mx) / 2, max((mx - mn).max() / 2, 0.1)


def _same(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def test_display_transform_matches_the_rerun_copy():
    assert T_DISP.tobytes() == _OLD_T.tobytes()
    rng = np.random.default_rng(1)
    for n in (1, 2, 49, 1000):
        pts = rng.normal(0, 1, (n, 3))
        assert to_display_points(pts).tobytes() == _old_rerun_pts(pts).tobytes()
        T = np.eye(4)
        T[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        T[:3, 3] = rng.normal(0, 2, 3)
        assert to_display_pose(T).tobytes() == (_OLD_T @ T).tobytes()


def test_cube_bounds_matches_both_copies():
    rng = np.random.default_rng(2)
    sets = [rng.normal(0, s, (n, 3)) for s in (1e-5, 1e-3, 0.05, 1.0, 30.0) for n in (1, 2, 7, 500)]
    sets += [np.zeros((4, 3)), np.full((3, 3), 2.5)]
    for pts in sets:
        c, r = cube_bounds(pts, 1e-3, pad=1.05)
        c0, r0 = _old_plotly(pts)
        assert _same(c, c0) and _same(r, r0)
        c, r = cube_bounds(pts, 0.1)
        c0, r0 = _old_sim(pts)
        assert _same(c, c0) and _same(r, r0)
        for k in range(3):                            # the limits the axes receive
            assert _same(c[k] - r, c0[k] - r0) and _same(c[k] + r, c0[k] + r0)
