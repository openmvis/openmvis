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
"""Report / label formatting for the custom camera models (kb4, omni_radtan).

Regression tests for the integration-review "reports-misc" findings:
  (a) cam-IMU text prints kb4 distortion with its names
  (b) report.print_report labels kb4 / omni coefficients with DIST_NAMES
  (c) html_report mixed-model distortion table: no zero padding, per-cell names
  (d) [removed 2026-10-25 with the corner-coverage figure]
  (e) sim_recovery_2imu1cam._print_absolute prints every intrinsic row
  (f) demo_video/render_final fisheye tag derived from the registry (kb4)
Existing models keep their historical text exactly.
"""

import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import gtsam

from mvis.analysis.cam_imu_report import cam_imu_html_report, cam_imu_text_report
from mvis.analysis.html_report import _distortion_table_rows
from mvis.analysis.report import print_report
from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.core.camera_models import DIST_NAMES
from mvis.core.types import CalibrationResult, CameraParams, ExtrinsicTransform

REPO = Path(__file__).resolve().parents[2]

KB4 = [300.0, 301.0, 321.0, 239.0, 0.02, -0.01, 3e-3, -5e-4]
OMNI = [400.0, 401.0, 320.0, 240.0, 1.5249, -0.28, 0.07, 5e-4, -2.47e-3]
RADTAN = [400.0, 400.0, 320.0, 240.0, -0.294155, 0.087975, 0.000472, -0.000425]
EQUI = [380.0, 381.0, 320.0, 240.0, 0.01, -0.002, 0.0003, -0.00004]


# ---- (a) cam-IMU report -----------------------------------------------------

def _camimu_result(model, intr):
    cal = CameraCalib(gtsam.Pose3(), 0.0, 0.0, intr, model=model)
    return NS(cameras=[dict(calib=cal, image_size=(640, 480), n_frames=10,
                            reproj_rms_px=0.3)],
              ok=True, diagnostics={}, b_g=np.zeros(3), b_a=np.zeros(3),
              b_g_drift=np.zeros(3), b_a_drift=np.zeros(3),
              imu_intr_sigma=None, imu_intr_xi=None, imus=[], imu_model="calyx")


def test_camimu_text_kb4_names():
    txt = cam_imu_text_report(_camimu_result("kb4", KB4))
    line = next(l for l in txt.splitlines() if l.strip().startswith("dist"))
    assert line == "  dist (k1 k2 k3 k4): 0.02000 -0.01000 0.00300 -0.00050"


def test_camimu_text_existing_models_unchanged():
    txt = cam_imu_text_report(_camimu_result("radtan", RADTAN))
    old = " ".join(f"{x:.5f}" for x in RADTAN[4:])        # historical format
    assert f"  dist: {old}" in txt.splitlines()
    html = cam_imu_html_report(_camimu_result("equidistant", EQUI))
    assert "0.01000 -0.00200 0.00030 -0.00004" in html


# ---- (b) cam-only text report ------------------------------------------------

def _cam(cid, model, intr):
    return CameraParams(cid, np.array(intr[:4]), np.array(intr[4:]), model, (640, 480))


def _dist_block(model, intr):
    lines = print_report(CalibrationResult(cameras=[_cam(0, model, intr)])).splitlines()
    i = lines.index(f"  Distortion ({model}):")
    return lines[i + 1:i + 1 + len(intr) - 4]


def test_print_report_kb4_names_and_significant_digits():
    assert _dist_block("kb4", KB4) == [
        "    k1 = 0.02", "    k2 = -0.01", "    k3 = 0.003", "    k4 = -0.0005"]


def test_print_report_omni_labels_xi():
    assert _dist_block("omni_radtan", OMNI) == [
        "    xi = 1.5249", "    k1 = -0.28", "    k2 = 0.07", "    p1 = 0.0005",
        "    p2 = -0.00247"]


def test_print_report_existing_models_unchanged():
    assert _dist_block("radtan", RADTAN) == [
        "    d0 = -0.294155", "    d1 = 0.087975", "    d2 = 0.000472", "    d3 = -0.000425"]
    assert _dist_block("double_sphere", [300, 300, 320, 240, -0.2, 0.6]) == [
        "    d0 = -0.200000", "    d1 = 0.600000"]


# ---- (c) cam-only HTML distortion table --------------------------------------

def test_html_mixed_rig_no_zero_padding_and_named_cells():
    cams = [_cam(0, "radtan", RADTAN), _cam(1, "omni_radtan", OMNI), _cam(2, "kb4", KB4)]
    rows = _distortion_table_rows(cams)
    assert rows[0] == ("<table><tr><th>Camera</th><th>Model</th>"
                       + "".join(f"<th>d{i}</th>" for i in range(5)) + "</tr>")
    radtan, omni, kb4 = rows[1:]
    assert radtan == ("<tr><td>0</td><td>radtan</td>"
                      "<td class='num'>k1=-0.294155</td><td class='num'>k2=0.087975</td>"
                      "<td class='num'>p1=0.000472</td><td class='num'>p2=-0.000425</td>"
                      + "<td class='num'></td>" + "</tr>")
    assert "<td class='num'>xi=1.524900</td>" in omni     # omni keeps '.6f'
    assert omni.count("<td class='num'></td>") == 0
    assert "0.000000" not in omni
    assert "<td class='num'>k4=-0.000500</td>" in kb4
    assert kb4.count("<td class='num'></td>") == 1


def test_html_single_model_rig_unchanged():
    rows = _distortion_table_rows([_cam(0, "radtan", RADTAN), _cam(1, "radtan", RADTAN)])
    assert rows[0] == ("<table><tr><th>Camera</th><th>Model</th><th>k1</th><th>k2</th>"
                       "<th>p1</th><th>p2</th></tr>")
    assert rows[1] == ("<tr><td>0</td><td>radtan</td><td class='num'>-0.294155</td>"
                       "<td class='num'>0.087975</td><td class='num'>0.000472</td>"
                       "<td class='num'>-0.000425</td></tr>")
    rows = _distortion_table_rows([_cam(0, "fov", [300, 300, 320, 240, 0.9])])
    assert rows[1] == "<tr><td>0</td><td>fov</td><td class='num'>0.900000</td></tr>"


def _printed_intr(model, intr):
    from scripts.sim_recovery_2imu1cam import _print_absolute
    names = ["fx", "fy", "cx", "cy"] + (
        ["k1", "k2", "p1", "p2"] if model in ("radtan", "equidistant")
        else list(DIST_NAMES[model]))
    row = lambda p, g: dict(param=p, gt=g, est=g, err=0.0, sigma=1.0, z=0.0, pass_=True)
    rows = [row(f"cam0/{n}", v) for n, v in zip(names, intr)]
    rows += [row(f"cam0/{k}{a}", 0.0) for k in ("rot", "pos") for a in range(3)]
    rows += [row("cam0/td", 0.0)]
    gt = NS(cams=[dict(model=model, T_CtoI=np.eye(4).tolist(), intr8=list(intr))],
            imus=[], imu_model="calyx")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_absolute(gt, dict(rows=rows, reproj_rms_px=0.3, n_fail=0))
    printed = [l.split()[0] for l in buf.getvalue().splitlines()
               if l.startswith("  ") and len(l.split()) == 5]
    return names, [p for p in printed if p in names]


def test_sim_recovery_2imu1cam_prints_all_kb4_and_omni_rows():
    for model, intr in (("kb4", KB4), ("omni_radtan", OMNI), ("radtan", RADTAN)):
        names, printed = _printed_intr(model, intr)
        assert printed == names, model


# ---- (f) demo video fisheye tag -------------------------------------------------

def _fisheye_tag_fn():
    """Load render_final._fisheye_tag without importing pyvista (not in the env)."""
    src = (REPO / "demo_video" / "render_final.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_fisheye_tag")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "render_final", "exec"), ns)
    return ns["_fisheye_tag"]


def test_render_final_fisheye_tag_from_registry():
    tag = _fisheye_tag_fn()
    assert tag(dict(is_fisheye=False, model="kb4")) == "  fisheye"
    for m in ("equidistant", "fov", "double_sphere", "eucm", "ucm", "omni_radtan"):
        assert tag(dict(is_fisheye=(m == "equidistant"), model=m)) == "  fisheye", m
    assert tag(dict(is_fisheye=False, model="radtan")) == ""
    assert tag(dict(is_fisheye=True)) == "  fisheye"      # legacy scene, no model key
    assert tag(dict(is_fisheye=False)) == ""


def test_render_final_does_not_import_mvis():
    """render_final runs in the demo_video env (no mvis installed); its wide-model
    tuple must still cover every non-radtan registry model."""
    from mvis.core.camera_models import MODELS
    tree = ast.parse((REPO / "demo_video" / "render_final.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert not any(a.name.split(".")[0] == "mvis" for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "mvis"
    tag = _fisheye_tag_fn()
    for m in MODELS:
        assert tag(dict(is_fisheye=False, model=m)) == ("" if m == "radtan" else "  fisheye"), m


def test_camera_only_report_uses_the_shared_page_shell(tmp_path, monkeypatch):
    """html_report assembles its page through report_shell.render_page, not its own copy."""
    import numpy as np
    from mvis.analysis import html_report
    from mvis.core.types import CalibrationResult, CameraParams
    from mvis.detection.chessboard import Chessboard

    seen = {}
    real = html_report.render_page

    def spy(title, toc, sections, style):
        seen.update(title=title, toc=list(toc), n=len(sections), style=style)
        return real(title, toc, sections, style)

    monkeypatch.setattr(html_report, "render_page", spy)
    result = CalibrationResult(
        cameras=[CameraParams(camera_id=0, intrinsics=np.array([500.0, 500.0, 320.0, 240.0]),
                              distortion=np.array([-0.1, 0.01, 0.001, -0.001]), distortion_model="radtan",
                              image_size=(640, 480))],
        reprojection_errors=[], overall_rms=0.2, reference_camera=0)
    out = tmp_path / "r.html"
    html_report.generate_html_report(result, {0: []}, Chessboard(rows=7, cols=5, square_size=0.03), {}, {}, None,
                                     title="My title", output_path=str(out))
    assert seen["title"] == "My title" and seen["n"] > 0
    assert seen["toc"][0] == ("sec-summary", "1. Summary")
    html = out.read_text()
    assert "<title>" in html and "My title" in html and "report-footer" in html and 'id="scroll-top"' in html
