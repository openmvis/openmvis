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
"""Report layout from the report-refinements pass, in both reports: footer, Summary (graded tiles, facts
line, no metric table), camera chips and graded cells, camera tabs, Appendix, camera-only title and corner
accounting, the shared plot style, and the page script / print rules. JS behaviour is checked by structure."""
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mvis.analysis.report_shell import REPORT_RC, cam_label, graded, render_page, report_plot_style


@pytest.fixture(scope="module")
def cam_imu_html():
    from mvis.analysis.cam_imu_report import cam_imu_html_report
    from tests.io.test_cam_imu_report import _result_with_geometry
    r = _result_with_geometry()
    for e in r.cameras:
        e.update(reproj_median_px=0.142, reproj_p95_px=0.312, reproj_mean_px=0.155, reproj_std_px=0.089,
                 image_size=(1280, 1024))
    return cam_imu_html_report(r, config="solve:\n  sigma_g: 6.0e-4\n")


def test_footer_has_no_author_name_and_links_openmvis():
    html = render_page("t", [], ["<div class='section'></div>"])
    assert "by Yulin Yang" not in html and "yangyulin.net" not in html
    assert 'href="https://openmvis.com"' in html and "contact@openmvis.com" in html


def test_cam_imu_summary_tiles_each_number_once_and_a_facts_line(cam_imu_html):
    summary = cam_imu_html[cam_imu_html.index('id="sec-summary"'):cam_imu_html.index('id="sec-3d"')]
    assert "<th>Metric</th>" not in summary                          # the old metric table is gone
    assert "Camera <span class='nt'>χ²/dof</span>" in summary and "IMU <span class='nt'>χ²/dof</span>" in summary
    assert "<div class='facts'>" in summary and "Trajectory states" in summary
    assert summary.count("REPROJ") == 0 and summary.count("Reproj median") == 1


def test_camera_chip_and_graded_cell_helpers():
    from mvis.analysis.views.spec import COLORS
    assert cam_label(1) == f"<span class='chip' style='background:{COLORS[1]}'></span>cam1"
    assert graded("0.142", "good") == "<span class='g g-good'>0.142</span>" and graded("0.1", None) == "0.1"


def test_cam_imu_appendix_holds_timing_results_config_with_their_ids(cam_imu_html):
    i = cam_imu_html.index('id="sec-appendix"')
    assert cam_imu_html.index('id="sec-config"') > i                    # id kept, inside the Appendix
    assert re.search(r"<h3>[A-C]\. Configuration</h3>", cam_imu_html)   # lettered in order (no timing here)
    assert "Configuration</h2>" not in cam_imu_html                     # no longer its own numbered section


def test_page_script_builds_camera_tabs_aligns_numbers_and_opens_folds_for_print():
    html = render_page("t", [], ["<div class='section'></div>"])
    assert "Select a camera:" in html and "cam-tabs" in html            # tab bar with its prompt
    assert "classList.add('num')" in html                               # numeric column alignment
    assert "beforeprint" in html and "afterprint" in html               # folds open for printing
    assert ".fold-all, .copy-btn, .cam-tabs .tab-bar { display: none !important; }" in html


def test_cam_imu_per_camera_blocks_are_tab_panels():
    from mvis.analysis import cam_imu_report as R
    d = {"reproj_corners_by_cam": []}
    orig, orig_t = R._reproj_camera_sections, R._reproj_table_html
    try:
        R._reproj_camera_sections = lambda d, board: [(0, "<p>a</p>"), (1, "<p>b</p>")]
        R._reproj_table_html = lambda d, board: "<table></table>"
        from tests.io.test_cam_imu_report import _result_with_geometry
        html = R.cam_imu_html_report(_result_with_geometry())
    finally:
        R._reproj_camera_sections, R._reproj_table_html = orig, orig_t
    assert "<div class='cam-tabs'>" in html
    assert html.count('class="cam-panel"') == 2 and 'id="sec-reproj-cam1" data-label="cam1"' in html


def test_report_plot_style_applies_the_shared_rc_inside_the_report():
    import matplotlib.pyplot as plt

    @report_plot_style
    def draw():
        return plt.rcParams["axes.spines.top"], plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    top, c0 = draw()
    from mvis.analysis.views.spec import COLORS
    assert top is False and c0 == COLORS[0] and REPORT_RC["axes.grid"] is True
    assert plt.rcParams["axes.spines.top"] is True                      # restored outside the report


def test_cam_only_title_names_the_recording_and_cameras():
    from mvis.analysis.html_report import _cam_only_title
    res = NS(cameras=[NS(camera_id=0), NS(camera_id=1)])
    ci = {"cameras_info": {0: {"source_path": "/d/run/rig.bag"}, 1: {"source_path": "/d/run/rig.bag"}}}
    assert _cam_only_title("04_configs / cam0", res, ci) == "Camera calibration · rig.bag · cam0, cam1"
    assert _cam_only_title("ds / coll", res, None) == "Camera calibration · ds / coll"


def test_regenerate_reads_camera_sources_from_the_run_config(tmp_path):
    from mvis.analysis.regenerate_html import _camera_sources
    cfg = tmp_path / "cam.yaml"
    cfg.write_text("input:\n  cameras:\n  - camera_id: 0\n    source_path: /d/rig.bag\n")
    assert _camera_sources(cfg) == {0: {"source_path": "/d/rig.bag"}}
    assert _camera_sources(tmp_path / "missing.yaml") == {}


def test_cam_only_corner_accounting_is_one_line_per_camera_with_the_table_folded():
    from mvis.analysis.html_report import _corner_accounting_rows
    acc = {0: dict(used=15670, past_fold=0, past_fold_pass1=0, other_invalid=0, no_pose_frames=0, outliers=0,
                   max_theta_deg=69.6, fold_deg=None, fold_level="none"),
           1: dict(used=900, past_fold=12, past_fold_pass1=0, other_invalid=0, no_pose_frames=3, outliers=0,
                   max_theta_deg=95.0, fold_deg=93.0, fold_level="alarm")}
    html = "".join(_corner_accounting_rows(NS(corner_accounting=acc)))
    assert "<b>cam0</b>: 15,670 corners used; none left out" in html
    assert "<b>cam1</b>: 900 corners used; left out: 12 past the fold, 3 in no-pose frames" in html
    assert "(fold ALARM)" in html and "Full corner accounting" in html and "</table></div></details>" in html


def test_header_and_favicon_carry_the_openmvis_logo():
    from mvis.analysis.report_shell import LOGO_SVG
    html = render_page("t", [], ["<div class='section'></div>"])
    assert f'<span class="brand-logo">{LOGO_SVG}</span>OpenMVIS</h1>' in html
    assert '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,' in html
