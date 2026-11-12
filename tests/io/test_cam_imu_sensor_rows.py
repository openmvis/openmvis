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
"""result_schema.cam_imu_sensor_rows (step A): the cam_imu.json per-sensor fields the CLI writes
from it are byte-identical to the inline block they replace, over varied fake results (models,
aux IMUs, pinned / freed intrinsics, t_ga, baselines)."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import gtsam
import numpy as np
import pytest

from mvis.calibration.factors.lie_groups import log_so3
from mvis.cli.main import _summary_sensor_fields
from mvis.imu import ImuModel
from mvis.io.result_schema import calyx_dw_decomposed, cam_imu_sensor_rows, imu_diag

MODELS = [("radtan", [400., 401., 320., 240., -0.28, 0.07, 5e-4, -4e-4]),
          ("omni_radtan", [380., 381., 322., 238., 1.1, -0.2, 0.05, 1e-4, -2e-4]),
          ("kb4", [190., 190.5, 256., 255., 0.01, -0.003, 0.001, -2e-4]),
          ("ucm", [250., 251., 320., 240., 0.62])]


def _pose(rng, s_rot=1.0, s_t=0.05):
    return gtsam.Pose3(gtsam.Rot3.Expmap(rng.normal(0, s_rot, 3)), gtsam.Point3(*rng.normal(0, s_t, 3)))


def fake_result(v):
    rng = np.random.default_rng(v)
    cams = []
    for c in range(1 + v % 3):
        m, intr = MODELS[(v + c) % len(MODELS)]
        intr = np.array(intr) * (1 + rng.normal(0, 1e-3, len(intr)))
        cal = NS(model=m, intr=intr, td=float(rng.normal(0, 0.01)), tr=0.0, T=_pose(rng))
        e = dict(calib=cal, reproj_rms_px=float(rng.uniform(0.1, 1)),
                 reproj_median_px=(None if v % 2 else float(rng.uniform(0.1, 1))),
                 reproj_p95_px=(None if v % 2 else float(rng.uniform(1, 3))))
        if c > 0 and v % 4 != 3:
            e["T_CtoC0"] = _pose(rng, 0.1, 0.1)
        if v % 5 == 0:
            e["corner_accounting"] = {"used": int(rng.integers(100, 1000))}
        cams.append(e)
    imus = []
    for k in range(v % 3):
        a = dict(name=f"aux{k}", td=float(rng.normal(0, 0.005)))
        if k == 0:
            a["t_ga"] = float(rng.normal(0, 0.003))
            a["t_ga_sigma"] = float(rng.uniform(0, 1e-4)) if v % 2 else None
        if v % 3:
            a["chi2_dof"] = float(rng.uniform(0.5, 2))
        if (v + k) % 4:
            a["T_IstoI"] = _pose(rng, 1.0, 0.1)
        if (v + k) % 2 == 0:
            a["intr_xi24"] = rng.normal(0, 0.01, 24) if v % 7 else np.zeros(24)
        imus.append(a)
    d = {}
    if v % 2 == 0:
        d["t_ga"] = float(rng.normal(0, 0.003))
        if v % 4 == 0:
            d["t_ga_sigma"] = 1e-4
    return NS(imu_model=("calyx" if v % 3 else "rpng"), imu_intr_sigma=(None if v % 4 == 1 else 1),
              imu_intr_xi=(None if v % 6 == 5 else rng.normal(0, 0.01, 24)),
              diagnostics=d, cameras=cams, imus=imus)


def _old_fields(r, summary):
    """The block calibrate_cam_imu wrote inline before (verbatim)."""
    d = r.diagnostics
    for c, e in enumerate(r.cameras):
        cal = e["calib"]
        cd = dict(name=f"cam{c}", td_ms=round(1e3 * cal.td, 3),
                  reproj=round(float(e["reproj_rms_px"]), 3),
                  reproj_median=(round(float(e["reproj_median_px"]), 3)
                                 if e.get("reproj_median_px") is not None else None),
                  reproj_p95=(round(float(e["reproj_p95_px"]), 3)
                              if e.get("reproj_p95_px") is not None else None),
                  intr=[round(float(v), 5) for v in cal.intr],
                  p_CinI_mm=[round(1e3 * v, 2)
                             for v in np.asarray(cal.T.translation(), float)],
                  rotvec=[round(v, 5) for v in log_so3(cal.T.rotation().matrix())])
        if e.get("corner_accounting") is not None:
            cd["corner_accounting"] = e["corner_accounting"]
        if c > 0 and "T_CtoC0" in e:
            cd["p_CtoC0_mm"] = [round(1e3 * v, 2)
                                for v in np.asarray(e["T_CtoC0"].translation(), float)]
        summary["cams"].append(cd)
    model = getattr(r, "imu_model", "rpng")
    if getattr(r, "imu_intr_xi", None) is not None:
        xi = np.asarray(r.imu_intr_xi, float)
        summary["imu_model"] = model
        ii = ImuModel.from_tangent(xi, model)
        summary["Dw_diag"], summary["Da_diag"] = imu_diag(ii, 5)
        if model == "calyx":
            summary.update(calyx_dw_decomposed(ii.Dw(), 5))
    if d.get("t_ga") is not None:
        summary["t_ga_ms"] = round(1e3 * d["t_ga"], 4)
        if d.get("t_ga_sigma") is not None:
            summary["t_ga_sigma_ms"] = round(1e3 * d["t_ga_sigma"], 4)
    summary["imus"] = []
    for m in getattr(r, "imus", []) or []:
        a = dict(name=m.get("name", "aux"),
                 td_ms=round(1e3 * float(m.get("td", 0.0)), 3))
        if m.get("t_ga") is not None:
            a["t_ga_ms"] = round(1e3 * float(m["t_ga"]), 4)
            if m.get("t_ga_sigma") is not None:
                a["t_ga_sigma_ms"] = round(1e3 * float(m["t_ga_sigma"]), 4)
        if "chi2_dof" in m:
            a["chi2_dof"] = round(float(m["chi2_dof"]), 3)
        if "T_IstoI" in m:
            T = m["T_IstoI"]
            a["p_IsinI_mm"] = [round(1e3 * v, 2)
                               for v in np.asarray(T.translation(), float)]
            a["rotvec"] = [round(v, 5) for v in log_so3(T.rotation().matrix())]
        xi24 = m.get("intr_xi24")
        if xi24 is not None:
            xi24 = np.asarray(xi24, float)
            if xi24.size >= 12 and np.max(np.abs(xi24)) > 1e-9:
                ii = ImuModel.from_tangent(xi24, model)
                a["Dw_diag"], a["Da_diag"] = imu_diag(ii, 5)
                if model == "calyx":
                    a.update(calyx_dw_decomposed(ii.Dw(), 5))
        summary["imus"].append(a)


@pytest.mark.parametrize("v", range(24))
def test_summary_sensor_fields_bytes(v):
    r = fake_result(v)
    old, new = {"ok": True, "cams": []}, {"ok": True, "cams": []}
    _old_fields(r, old)
    _summary_sensor_fields(r, new)
    assert json.dumps(new, indent=2) == json.dumps(old, indent=2)
    assert [type(x) for x in new["cams"][0]["rotvec"]] == [type(x) for x in old["cams"][0]["rotvec"]]


def test_require_sigma_gates_only_the_base_imu():
    r = fake_result(1)                       # imu_intr_sigma None, xi set
    assert cam_imu_sensor_rows(r, require_sigma=False)["base"]["intr"] is not None
    assert cam_imu_sensor_rows(r, require_sigma=True)["base"]["intr"] is None
    r = fake_result(5)                       # xi None
    assert cam_imu_sensor_rows(r, require_sigma=False)["base"]["intr"] is None
