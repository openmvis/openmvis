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
"""Signed per-corner errors for the cam-IMU report's per-camera scatter + corner heatmap.

`corner_errors` gives the (u, v) errors behind `corner_norms` (bitwise: the norms are their row
norms), the corner table keeps them beside the norms from the same projection pass, and
`CornerTable.by_camera` splits them per camera into the report's {frame: vectors} and
{board corner id: [norms]}. The report draws one scatter + heatmap row per camera."""
import matplotlib
matplotlib.use("Agg")
import numpy as np

from mvis.analysis.cam_imu_report import _reproj_camera_sections, _reproj_table_html
from mvis.calibration.pipeline_cam_imu import (_corner_norm_table, _frame_corner_errors,
                                               _frame_corner_norms)
from tests.cam_imu.test_diagnostics_once import _eq, _fixture


def test_corner_errors_norms_are_corner_norms_bitwise():
    for make in (False, True):
        cam_factors, values, _b, _g = _fixture(make)
        for c, j, fac in cam_factors:
            v = _frame_corner_errors(fac, c, j, values)     # make_factor keys or the legacy path
            assert v.shape == (len(fac.uv), 2)
            assert _eq(np.linalg.norm(v, axis=1), _frame_corner_norms(fac, c, j, values))


def test_table_vectors_and_by_camera():
    cam_factors, values, _b, _g = _fixture(True)
    for k, (_c, _j, fac) in enumerate(cam_factors):
        fac.report_pids = np.arange(len(fac.uv)) + k     # any ids, one per corner
    t = _corner_norm_table(cam_factors, values)
    assert t.vec.shape == (60, 2)
    assert _eq(np.isnan(t.vec[:, 0]), np.isnan(t.e))
    per = t.by_camera(cam_factors)
    assert len(per) == 2
    # camera 0: three frames, frame 1 lost its guarded corner; camera 1: frame 2 fully guarded -> absent
    assert sorted(per[0]["vectors_by_frame"]) == [0, 1, 2]
    assert len(per[0]["vectors_by_frame"][1]) == 11
    assert sorted(per[1]["vectors_by_frame"]) == [0, 1]
    assert len(per[1]["vectors_by_frame"][1]) == 6
    n0 = sum(len(v) for v in per[0]["corner_errors"].values())
    assert n0 == 35 and all(np.isfinite(x) for v in per[0]["corner_errors"].values() for x in v)


def test_by_camera_without_ids_gives_vectors_only():
    cam_factors, values, _b, _g = _fixture(True)
    per = _corner_norm_table(cam_factors, values).by_camera(cam_factors)
    assert per[0]["vectors_by_frame"] and per[0]["corner_errors"] == {}
    assert "n_tags" not in per[0]


def test_by_camera_counts_tags_from_ids():
    cam_factors, values, _b, _g = _fixture(True)
    t = _corner_norm_table(cam_factors, values)
    # one corner per tag (corner_subsample: tl): tags == finite corners, not corners / 4
    for _c, _j, fac in cam_factors:
        fac.report_pids = np.arange(len(fac.uv)) * 4
    per = t.by_camera(cam_factors)
    assert [p["n_tags"] for p in per] == [12 + 11 + 12, 8 + 6 + 0]
    # all four corners: a tag with one guard-skipped corner still counts once per frame
    for _c, _j, fac in cam_factors:
        fac.report_pids = np.arange(len(fac.uv))
    assert [p["n_tags"] for p in t.by_camera(cam_factors)] == [3 + 3 + 3, 2 + 2 + 0]
    # one frame split into sub-frame factors (split_rows): a tag in two of them counts once
    split = [(c, 1 if c == 0 else j, fac) for c, j, fac in cam_factors]
    for _c, _j, fac in split:
        fac.report_pids = np.arange(len(fac.uv)) * 4
    assert t.by_camera(split)[0]["n_tags"] == 12


class _Board:
    """6 x 6 grid of points (9 tags of 4), enough for the heatmap and per-tag plots."""
    _num_tags = 9

    def all_points_3d(self):
        g = np.stack(np.meshgrid(np.arange(6) * 0.1, np.arange(6) * 0.1), -1).reshape(-1, 2)
        return np.c_[g, np.zeros(len(g))]

    def num_points(self):
        return 36


def test_report_table_and_camera_sections():
    rng = np.random.default_rng(0)
    cd = dict(vectors_by_frame={f: rng.normal(0, 0.2, (20, 2)) for f in range(10)},
              corner_errors={p: [0.1, 0.2] for p in range(36)}, n_tags=123)
    d = {"reproj_corners_by_cam": [cd, cd]}
    table = _reproj_table_html(d, _Board())
    assert table.count("<td>cam") == 2 and "<th>Tags</th>" in table
    assert "<td>123</td>" in table and "<td>200</td>" in table              # tags from the ids
    # no ids (n_tags absent) or a chessboard: no Tags column, never corners / 4
    no_ids = {"reproj_corners_by_cam": [{**cd, "n_tags": None}]}
    for tbl in (_reproj_table_html(no_ids, _Board()), _reproj_table_html(d, None)):
        assert "Tags" not in tbl and "<td>50</td>" not in tbl and "<td>200</td>" in tbl
    secs = _reproj_camera_sections(d, _Board())
    assert [c for c, _b in secs] == [0, 1]
    body = secs[1][1]
    # tiles, then per-frame, per-tag, u/v histogram, vectors, then the scatter + heatmap row
    assert body.count('class="stat-box"') == 5 and body.count("<img") == 6
    assert body.count('<div class="plot-row">') == 1
    # no board: no per-tag, no heatmap; old results: nothing
    assert _reproj_camera_sections({"reproj_corners_by_cam": [cd]}, None)[0][1].count("<img") == 4
    assert _reproj_camera_sections({}, None) == [] and _reproj_table_html({}, None) == ""
