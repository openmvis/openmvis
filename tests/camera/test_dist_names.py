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
"""camera_models.dist_names / NAMED_DIST_MODELS and the html_report distortion headers give
the names the report-local tables gave (the cam-IMU row, the text report and html_report)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np

from mvis.analysis.html_report import _distortion_table_rows
from mvis.core.camera_models import DIST_NAMES, MODELS, N_DIST, dist_names

_OLD_HEADERS = {
    "radtan": ("k1", "k2", "p1", "p2"),
    "equidistant": ("k1", "k2", "k3", "k4"),
    "fov": ("omega", "", "", ""),
    "double_sphere": ("xi", "alpha", "", ""),
    "eucm": ("alpha", "beta", "", ""),
    "ucm": ("alpha", "", "", ""),
    "omni_radtan": DIST_NAMES["omni_radtan"],
    "kb4": DIST_NAMES["kb4"],
}


def test_dist_names_equal_the_old_inline_rules():
    for m in MODELS + ("unknown", None):
        old_report = DIST_NAMES.get(m) if m in ("omni_radtan", "kb4", "ucm") else None
        old_cam_imu = DIST_NAMES[m] if m in ("omni_radtan", "kb4", "ucm") else None
        assert dist_names(m) == old_report == old_cam_imu


def _old_rows(cameras):
    rows = []
    models_present = set(c.distortion_model for c in cameras)
    n_generic = max([4] + [len(c.distortion) for c in cameras])
    generic = tuple(f"d{i}" for i in range(n_generic))
    mixed = len(models_present) != 1
    headers = _OLD_HEADERS.get(models_present.pop(), generic) if not mixed else generic
    rows.append("<table><tr><th>Camera</th><th>Model</th>"
                + "".join(f"<th>{h}</th>" for h in headers if h) + "</tr>")
    n_cols = sum(1 for h in headers if h)
    for cam in cameras:
        coefs = list(cam.distortion)
        if mixed:
            names = DIST_NAMES.get(cam.distortion_model, ())
            cells = [(f"{names[i]}=" if i < len(names) else "") + "{:.6f}".format(coefs[i])
                     if i < len(coefs) else "" for i in range(n_cols)]
        else:
            coefs += [0.0] * n_cols
            cells = ["{:.6f}".format(coefs[i]) for i in range(n_cols)]
        rows.append(f"<tr><td>{cam.camera_id}</td><td>{cam.distortion_model}</td>"
                    + "".join(f"<td class='num'>{c}</td>" for c in cells) + "</tr>")
    return rows


def test_distortion_table_rows_unchanged():
    rng = np.random.default_rng(0)
    cams = [NS(camera_id=i, distortion_model=m, distortion=list(rng.normal(0, 0.1, N_DIST[m])))
            for i, m in enumerate(MODELS)]
    for one in cams:                                  # single-model tables: named headers
        assert _distortion_table_rows([one, one]) == _old_rows([one, one])
    assert _distortion_table_rows(cams) == _old_rows(cams)          # mixed: generic headers
    odd = NS(camera_id=9, distortion_model="weird", distortion=[0.1, 0.2])
    assert _distortion_table_rows([odd]) == _old_rows([odd])
