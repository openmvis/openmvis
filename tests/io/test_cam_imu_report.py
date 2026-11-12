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
"""Cam-IMU HTML report rendering (cam-imu-report R2+).

Renders the report from a minimal duck-typed CamImuCalibResult and asserts it
carries the shared shell (header / TOC / section cards / stat boxes / grade
badges) and the expected sections. The 3D-geometry assertions land in R3.
"""

from types import SimpleNamespace as NS

import json
import re

import numpy as np
import gtsam

from mvis.analysis.cam_imu_report import (
    cam_imu_html_report, cam_imu_text_report)
from tests.helpers.cam_imu_report_fakes import _fake_calib, _fake_result


def view_specs(html):
    """{kind: spec} of the canvas views (analysis.views) embedded in a report."""
    out = {}
    for m in re.finditer(r'<div class="mvv" data-mvv="(scene|rig)"[^>]*>.*?<script type="application/json">(.*?)</script>',
                         html, flags=re.S):
        out[m.group(1)] = json.loads(m.group(2))
    return out


def test_html_uses_shared_shell_and_sections():
    html = cam_imu_html_report(_fake_result())
    for marker in ('<!DOCTYPE html>', 'report-header', 'nav id="toc"',
                   'report-footer', 'stat-box', 'badge badge-good',
                   'id="sec-summary"', 'id="sec-cameras"', 'id="sec-imu"'):
        assert marker in html, f"missing {marker!r}"
    assert len(html) > 5000


def test_html_aux_and_reference_sections():
    ref = {"label": "Bench", "lever_mm": {0: [50.0, -20.0, 10.0]},
           "rotation_deg": {0: 3.5}}
    html = cam_imu_html_report(_fake_result(aux=True), reference=ref)
    assert 'id="sec-aux"' in html
    assert 'id="sec-reference"' in html
    assert "Bench" in html


def test_failed_status_badge():
    r = _fake_result()
    r.ok = False
    assert "badge-poor" in cam_imu_html_report(r)


def test_text_report_unchanged_shape():
    txt = cam_imu_text_report(_fake_result(aux=True))
    assert "Joint Camera-IMU Calibration Report" in txt
    assert "IMU imu0 (base)" in txt


def _result_with_geometry():
    """A result carrying converged Values (trajectory + board pose) so the 3D
    scene plot has data to draw."""
    r = _fake_result(aux=True)
    X = gtsam.symbol_shorthand.X
    B0 = gtsam.symbol('b', 0)
    vals = gtsam.Values()
    for k in range(8):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1 * k, 0.02 * k, 0.0)))
    vals.insert(B0, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 2.0)))
    r.values = vals
    r.T_BinW = vals.atPose3(B0)
    return r


def test_3d_section_rig_and_scene_embedded():
    from mvis.detection.chessboard import Chessboard
    r = _result_with_geometry()
    html = cam_imu_html_report(r, board=Chessboard(7, 7, 0.10))
    assert 'id="sec-3d"' in html
    # both canvas views embedded, their player inlined once
    specs = view_specs(html)
    assert set(specs) == {"scene", "rig"}
    assert html.count("MVIS report views: the calibration process player") == 1 and "Plotly.newPlot" not in html
    assert "Solved rig" in html and "Calibration process" in html
    assert len(specs["scene"]["xyz"]) == 8 and specs["scene"]["board"]["outline"]


def test_3d_section_rig_only_without_values():
    """No converged Values -> rig plot still renders, scene is skipped, no crash."""
    r = _fake_result()           # no .values attribute
    html = cam_imu_html_report(r)
    assert 'id="sec-3d"' in html
    assert set(view_specs(html)) == {"rig"}


def test_write_cam_imu_report_plumbs_board(tmp_path):
    from mvis.analysis.cam_imu_report import write_cam_imu_report
    from mvis.detection.chessboard import Chessboard
    html_p = tmp_path / "rep.html"
    txt_p = tmp_path / "rep.txt"
    txt = write_cam_imu_report(_result_with_geometry(), html_path=str(html_p),
                               text_path=str(txt_p), board=Chessboard(7, 7, 0.10))
    assert txt_p.read_text().startswith("=")
    html = html_p.read_text()
    assert 'id="sec-3d"' in html and set(view_specs(html)) == {"scene", "rig"}
    side = json.loads((tmp_path / "rep_views.json").read_text())      # the public copy
    assert side["scene"]["v"] == 2 and side["rig"]["cams"]


# ---------------------------------------------------------------- H3 additions
def _rich_result():
    """_result_with_geometry + everything the comprehensive sections feed on:
    per-frame reproj, t_states, IMU interval costs / split, timings, t_ga."""
    r = _result_with_geometry()
    n = 40
    r.diagnostics.update(
        reproj_per_frame=np.abs(np.sin(np.linspace(0, 6, n))) * 0.6 + 0.1,
        reproj_frame_cam=np.zeros(n, dtype=int),
        imu_interval_cost=(np.abs(np.cos(np.linspace(0, 4, 7))) * 30 + 5).tolist(),
        imu_cost_split={"theta": 10.0, "p": 5.0, "v": 3.0, "bg": 1.0, "ba": 0.5},
        timings={"detect": 3.2, "seed": 1.1, "solve": 20.5},
        t_ga=-0.00519, t_ga_sigma=0.0001,
    )
    r.t_states = np.linspace(0.0, 3.5, 8)
    return r


def test_tga_rendered_in_html():
    html = cam_imu_html_report(_rich_result())
    assert "t_ga" in html and "-5.190" in html            # ms, in the Base IMU table


def test_config_section_rendered():
    cfg = {"mode": "board", "imu_model": "calyx", "estimate_t_ga": True,
           "sigma_px": 1.0}
    html = cam_imu_html_report(_rich_result(), config=cfg)
    assert 'id="sec-config"' in html
    assert "estimate_t_ga" in html and "calyx" in html


def test_reproj_and_motion_sections_rendered():
    html = cam_imu_html_report(_rich_result())
    assert 'id="sec-reproj"' in html                      # per-frame RMS plot
    assert 'id="sec-motion"' in html                      # excitation plot
    assert html.count("data:image/png;base64") >= 2


def test_imu_residual_and_timing_sections_rendered():
    html = cam_imu_html_report(_rich_result())
    assert 'id="sec-imu-resid"' in html
    assert 'id="sec-timing"' in html
    assert "theta" in html                                # cost-split table


def test_intrinsics_sigma_rendered():
    html = cam_imu_html_report(_rich_result())
    assert "intr &sigma;" in html or "intr σ" in html


def test_minimal_result_still_renders_without_new_sections():
    """Old-style result (no per-frame data / t_states / timings) -> the new
    sections are skipped, nothing crashes, old sections intact."""
    html = cam_imu_html_report(_fake_result(aux=True))
    for absent in ('id="sec-reproj"', 'id="sec-motion"',
                   'id="sec-imu-resid"', 'id="sec-timing"', 'id="sec-config"'):
        assert absent not in html
    for present in ('id="sec-summary"', 'id="sec-cameras"', 'id="sec-imu"',
                    'id="sec-aux"'):
        assert present in html


def test_config_rendered_as_copyable_yaml_block():
    cfg = {"mode": "board", "solve": {"sigma_px": 1.0, "estimate_t_ga": True}}
    html = cam_imu_html_report(_rich_result(), config=cfg)
    assert "<pre" in html and "estimate_t_ga: true" in html


def test_config_raw_string_echoed_verbatim():
    raw = "mode: board\nsolve:\n  sigma_px: 1.0   # comment kept\n"
    html = cam_imu_html_report(_rich_result(), config=raw)
    assert "# comment kept" in html


def test_base_imu_full_intrinsics_matrices():
    r = _rich_result()
    r.imu_intr_sigma = np.full(24, 1e-3)
    xi = np.zeros(24)
    xi[0] = 0.004                                  # gyro scale-x perturbation
    r.imu_intr_xi = xi
    html = cam_imu_html_report(r)
    assert "IMU intrinsics (full)" in html
    assert "Dw (gyro)" in html and "Da (accel)" in html and "Tg" in html


def test_aux_imu_base_style_table_with_spatial_temporal():
    r = _rich_result()
    xi24 = np.zeros(24)
    xi24[0] = 0.003
    r.imus[0]["intr_xi24"] = xi24                  # freed aux intrinsics
    html = cam_imu_html_report(r)
    assert "T_IstoI lever p_IsinI (mm)" in html
    assert "td vs base (ms)" in html
    assert html.count("IMU intrinsics (full)") >= 1


def test_3d_report_is_self_contained_no_cdn():
    """The 3D views are inline canvas code: no external script, no plotly.js
    bundle (CSP/offline viewers block CDNs; the old section was megabytes)."""
    html = cam_imu_html_report(_result_with_geometry())
    assert "<script src=" not in html and "cdn.plot.ly" not in html
    assert "plotly-graph-div" not in html and "window.mvisViews" in html
    assert len(html) < 1_000_000


def test_summary_uses_median_p95_not_rms():
    """The headline reprojection metric is median + p95;
    RMS appears only as the legacy fallback when median/p95 are absent."""
    html = cam_imu_html_report(_rich_result())
    assert "Reproj median" in html and "Reproj p95" in html
    assert "Reproj RMS" not in html
    legacy = _fake_result()
    del legacy.diagnostics["reproj_median_px"], legacy.diagnostics["reproj_p95_px"]
    html2 = cam_imu_html_report(legacy)
    assert "Reproj RMS" in html2


def test_config_and_result_files_folded_with_copy(tmp_path):
    y = tmp_path / "run_calib.yaml"; y.write_text("cam0:\n  fx: 400.0\n")
    j = tmp_path / "cam_imu.json"; j.write_text('{"ok": true}')
    html = cam_imu_html_report(_fake_result(), config="solve:\n  max_iterations: 40\n",
                               result_files=[str(y), str(j), str(tmp_path / "missing.yaml")])
    assert 'id="sec-results"' in html and "Result files" in html
    res = html[html.index('id="sec-results"'):html.index('id="sec-config"')]
    assert res.count('<details class="appendix">') == 2         # missing file skipped
    assert "run_calib.yaml</code> &middot; YAML" in res and "cam_imu.json</code> &middot; JSON" in res
    assert "fx: 400.0" in res and 'class="copy-btn"' in res
    cfg = html[html.index('id="sec-config"'):]
    assert '<details class="appendix"><summary>Run configuration' in cfg
    assert "max_iterations: 40" in cfg


def test_brand_is_openmvis():
    html = cam_imu_html_report(_fake_result())
    assert "<h1>OpenMVIS</h1>" in html
    assert cam_imu_text_report(_fake_result()).splitlines()[1].startswith("OpenMVIS")


def test_result_file_over_cap_shows_head_only(tmp_path):
    from mvis.analysis.report_shell import result_files_html
    big = tmp_path / "x_full.yaml"
    big.write_text("".join(f"line{i}: {i}\n" for i in range(5000)))
    html = result_files_html([str(big)], max_bytes=1024)
    assert "of 5,000 lines shown" in html
    assert "line0: 0" in html and "line4999" not in html


def test_result_files_and_config_close_the_report(tmp_path):
    y = tmp_path / "c_calib.yaml"; y.write_text("a: 1\n")
    html = cam_imu_html_report(_result_with_geometry(), config="a: 1\n",
                               result_files=[str(y)])
    assert html.index('id="sec-3d"') < html.index('id="sec-results"') < html.index('id="sec-config"')


def test_rotations_are_reported_in_rad():
    """Rows carry the log(R) rotation vector in rad; no *_deg rotation keys remain."""
    from mvis.analysis.cam_imu_report import _cam_rows, _imu_block
    r = _fake_result(aux=True)
    rows = _cam_rows(r)
    base, aux = _imu_block(r)
    for row in rows:
        assert {"rot_rad", "rot_angle_rad", "rot_sig_rad"} <= set(row)
        assert row["rot_angle_rad"] == float(np.linalg.norm(row["rot_rad"]))
        assert not [k for k in row if k.startswith("rot") and k.endswith("_deg")]
    for a in aux:
        assert "rot_deg" not in a and "rot_angle_deg" not in a
    txt = cam_imu_text_report(r)
    assert "T_CtoI rot (rad)" in txt and "rot (deg)" not in txt
    html = cam_imu_html_report(r)
    assert "rot |rad|" in html and "rotation (rad)" in html and "rotation (°)" not in html


def test_reference_rotation_deg_and_rad_give_the_same_delta():
    from mvis.analysis.cam_imu_report import _cam_rows, _ref_deltas
    rows = _cam_rows(_fake_result())
    a = _ref_deltas(rows, {"rotation_deg": {0: 3.5}})["cams"][0]["rotation_delta_rad"]
    b = _ref_deltas(rows, {"rotation_rad": {0: float(np.radians(3.5))}})["cams"][0]["rotation_delta_rad"]
    assert a == b
    assert abs(a - abs(rows[0]["rot_angle_rad"] - np.radians(3.5))) < 1e-15
    txt = cam_imu_text_report(_fake_result(), {"rotation_deg": {0: 3.5}})
    assert re.search(r"rotation Δ [0-9.]+ rad", txt)


def test_reference_rotation_entries_are_read_lazily_per_camera():
    """Entries for cameras the run lacks, and None entries, are ignored as before."""
    from mvis.analysis.cam_imu_report import _cam_rows, _ref_deltas
    rows = _cam_rows(_fake_result())                       # one camera, id 0
    want = abs(rows[0]["rot_angle_rad"] - np.radians(3.5))
    d = _ref_deltas(rows, {"rotation_deg": {0: 3.5, 1: None, 2: [1.0, 2.0, 3.0]}})
    assert abs(d["cams"][0]["rotation_delta_rad"] - want) < 1e-15
    assert _ref_deltas(rows, {"rotation_deg": {0: None}})["cams"] == []
    assert _ref_deltas(rows, {"rotation_rad": {0: None}, "rotation_deg": {0: 3.5}})["cams"][0][
        "rotation_delta_rad"] == d["cams"][0]["rotation_delta_rad"]
    txt = cam_imu_text_report(_fake_result(), {"rotation_deg": {0: 3.5, 1: None}})
    assert "rotation Δ" in txt


def test_reference_rotation_rad_wins_over_deg_per_camera():
    from mvis.analysis.cam_imu_report import _cam_rows, _ref_deltas
    rows = _cam_rows(_fake_result())
    both = _ref_deltas(rows, {"rotation_rad": {0: 0.1}, "rotation_deg": {0: 3.5}})
    assert both["cams"][0]["rotation_delta_rad"] == abs(rows[0]["rot_angle_rad"] - 0.1)
    # rotation_rad lists another camera only: camera 0 falls back to its rotation_deg entry
    other = _ref_deltas(rows, {"rotation_rad": {1: 0.1}, "rotation_deg": {0: 3.5}})
    assert other["cams"][0]["rotation_delta_rad"] == abs(rows[0]["rot_angle_rad"] - float(np.radians(3.5)))


def test_calyx_base_imu_reports_R_wtoI_in_rad():
    from mvis.analysis.cam_imu_report import _imu_block
    from mvis.calibration.factors.lie_groups import log_so3
    from mvis.imu.imu_model import CALYX, ImuModel
    r = _fake_result()
    xi = np.zeros(24)
    xi[:9] = [0.004, -0.003, 0.002, 0.01, -0.006, 0.003, 0.008, 0.005, -0.007]   # full calyx Dw
    r.imu_intr_xi, r.imu_intr_sigma = xi, np.full(24, 1e-3)
    R_wtoI, _ = ImuModel.from_tangent(xi, CALYX).decompose()
    want = np.asarray(log_so3(R_wtoI), float)
    assert np.linalg.norm(want) > 1e-4
    base, _ = _imu_block(r)
    assert "R_wtoI_deg" not in base
    assert np.array_equal(np.asarray(base["R_wtoI_rad"]), want)
    html = cam_imu_html_report(r)
    assert "R_wtoI (rad)" in html
    assert " ".join(f"{x:.5f}" for x in want) in html
