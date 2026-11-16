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
"""Per-camera reprojection mean / median / std (Kalibr's three) in the cam_imu.json summary
and the text report, with the inlier population next to them; old results keep their output."""
import numpy as np

from mvis.analysis.cam_imu_report import _text_reproj_line
from mvis.calibration.reproj_stats import reproj_entry
from mvis.cli.main import _reproj_extra


def _entry():
    return reproj_entry(np.array([0.1, 0.2, 0.3, 0.4, 12.0]), 1.0)


def test_summary_adds_mean_std_and_inliers_after_the_old_fields():
    e = _entry()
    cam = _reproj_extra(e, "")
    assert list(cam) == ["reproj_mean", "reproj_std", "reproj_p90", "reproj_max", "reproj_n_corners", "reproj_inlier"]
    assert cam["reproj_max"] == 12.0 and cam["reproj_p90"] == round(e["reproj_p90_px"], 3)
    assert cam["reproj_mean"] == round(e["reproj_mean_px"], 3) and cam["reproj_n_corners"] == 5
    keep = np.array([0.1, 0.2, 0.3, 0.4])
    assert cam["reproj_inlier"] == dict(mean=0.25, median=0.25, std=round(float(np.std(keep)), 3),
                                        p90=round(float(np.percentile(keep, 90)), 3),
                                        p95=round(float(np.percentile(keep, 95)), 3), max=0.4,
                                        n_corners=4, n_excluded=1, threshold_px=3.0)
    assert list(_reproj_extra(e, "_px"))[:2] == ["reproj_mean_px", "reproj_std_px"]


def test_old_result_without_mean_keeps_its_summary():
    assert _reproj_extra({"reproj_rms_px": 0.5, "reproj_median_px": 0.2, "reproj_p95_px": 0.9}, "") == {}


def test_text_report_line_gives_mean_median_std_p90_p95_max():
    e = _entry()
    r = dict(n_frames=7, reproj_rms_px=e["reproj_rms_px"], reproj_mean_px=e["reproj_mean_px"],
             reproj_median_px=e["reproj_median_px"], reproj_std_px=e["reproj_std_px"],
             reproj_p90_px=e["reproj_p90_px"], reproj_p95_px=e["reproj_p95_px"], reproj_max_px=e["reproj_max_px"])
    line = _text_reproj_line(r)
    assert line.startswith("  reproj mean 2.600 / med 0.300 / std 4.701 / p90 ") and line.endswith("(7 frames)")
    assert " / max 12.00 px" in line and " / p95 " in line
    assert _text_reproj_line(dict(r, reproj_mean_px=None)).startswith("  reproj med 0.300 / p95")
