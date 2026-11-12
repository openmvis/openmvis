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
"""Task 1 — the shared recovery report/metrics module (sim-cleanup plan)."""
import numpy as np
import gtsam

from mvis.imu import create_imu_model
from mvis.simulation.report import recovery_errors, format_recovery_table


class _StubCalib:
    def __init__(self, T, td, tr, intr, model="radtan"):
        self.T, self.td, self.tr = T, float(td), float(tr)
        self.intr = np.asarray(intr, float)
        self.model = model


class _StubRes:
    def __init__(self, calib, imu_intr_xi, diagnostics, ok):
        self.calib, self.imu_intr_xi = calib, np.asarray(imu_intr_xi, float)
        self.diagnostics, self.ok = diagnostics, ok


def _case():
    gt_T = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0.05, -0.02, 0.01))
    est_T = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0.051, -0.02, 0.01))  # +1 mm x
    gt = dict(T_CtoI=gt_T, cam_intrinsics=np.array([400., 400, 320, 240, 0, 0, 0, 0]),
              imu_intr=create_imu_model("calyx"),          # identity -> Dw/Da diag 1
              board_poses=[gt_T], readout=0.017)
    res = _StubRes(
        _StubCalib(est_T, td=0.001, tr=0.016,
                   intr=np.array([400.1, 400, 320, 240, 0, 0, 0, 0])),
        imu_intr_xi=np.zeros(24),                          # -> identity reconstruction
        diagnostics={'imu_model': 'calyx', 'reproj_rms_px': 0.9, 'board_poses': {}},
        ok=True)
    return gt, res


def test_recovery_errors_contract():
    gt, res = _case()
    e = recovery_errors(gt, res)
    assert e['ok'] is True
    assert abs(e['lever_mm']['x'] - 1.0) < 1e-6
    assert abs(e['lever_mm']['norm'] - 1.0) < 1e-6
    assert abs(e['cam_intr']['fx'] - 0.1) < 1e-6
    assert abs(e['td_ms'] - 1.0) < 1e-6
    assert e['R_CtoI_deg'] < 1e-9
    assert max(e['Da'].values()) < 1e-9 and max(e['Dw'].values()) < 1e-9
    assert abs(e['readout_ms']['err'] - 1.0) < 1e-6


def test_format_recovery_table_runs():
    gt, res = _case()
    s = format_recovery_table(gt, res)
    assert 't_CtoI' in s and 'fx' in s and 'Da' in s


def test_intr_row_names_keep_the_historical_radtan_names():
    from mvis.core.camera_models import DIST_NAMES
    from mvis.simulation.synth import intr_row_names
    assert intr_row_names("radtan") == ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
    assert intr_row_names("equidistant") == intr_row_names("radtan")
    assert intr_row_names("kb4") == ["fx", "fy", "cx", "cy"] + list(DIST_NAMES["kb4"])


def _absolute_inputs():
    from types import SimpleNamespace as NS
    row = lambda p, g, z=0.5, err=0.001: dict(param=p, gt=g, est=g + err, err=err, sigma=1.0, z=z, pass_=True)
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
    rows = [row(f"cam0/{n}", 100.0 + k, z=(4.0 if n == "cx" else 0.5)) for k, n in enumerate(names)]
    rows += [row(f"cam0/{k}{a}", 0.0) for k in ("rot", "pos") for a in range(3)]
    rows += [row("cam0/td", 0.002), row("imu0/Dw0", 1.0), row("aux/Dw1", 1.01), row("aux/td", 0.001, z=-3.0)]
    rows += [row(f"aux/pose{a}", 0.0) for a in range(6)]
    T = np.eye(4); T[:3, 3] = [0.1, 0.0, 0.0]
    gt = NS(cams=[dict(model="radtan", T_CtoI=np.eye(4).tolist(), intr8=[0.0] * 8)],
            imus=[dict(name="imu0", is_base=True, xi24=[0.0] * 24),
                  dict(name="aux", is_base=False, xi24=[0.0] * 24, T_IstoI=T.tolist())],
            imu_model="calyx")
    return gt, dict(rows=rows, reproj_rms_px=0.3, n_fail=2)


def test_format_absolute_lists_every_sensor_and_flags_large_z():
    from mvis.simulation.report import format_absolute
    gt, res = _absolute_inputs()
    txt = format_absolute(gt, res)
    lines = txt.splitlines()
    assert txt.startswith("\n") and "(1 cam + 2 imu, reproj=0.300 px)" in lines[1]
    assert "--- CAMERA 0 ---" in lines and "--- IMU aux ---" in lines
    assert "--- IMU imu0  (reference frame) ---" in lines
    cx = next(l for l in lines if l.strip().startswith("cx"))
    assert cx.endswith(" *")
    assert sum("extrinsic" in l for l in lines) == 2 and sum(l.startswith("  td:") for l in lines) == 2
    assert lines[-1] == "  reproj = 0.300 px (noise floor),  total |z|>=3: 2/%d  (* = |z|>=3)" % len(res["rows"])
    labelled = format_absolute(gt, res, rig="custom", cam_labels=["Blackfly"])
    assert "(custom, reproj" in labelled and "--- CAMERA 0 (Blackfly) ---" in labelled
