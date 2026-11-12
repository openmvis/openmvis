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
"""Phase 1 (sim-from-result): ground_truth_from_result maps a CamImuCalibResult to a
GroundTruth (trajectory + board + every camera + base/aux IMUs)."""
import json
from dataclasses import asdict

import numpy as np
import gtsam

from mvis.calibration.pipeline_cam_imu import X
from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.simulation.from_result import ground_truth_from_result, GroundTruth


class _StubResult:
    pass


def _stub_result():
    vals = gtsam.Values()
    Ts = [gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0.1 * i, 0.0, 0.0)) for i in range(4)]
    for k, T in enumerate(Ts):
        vals.insert(X(k), T)
    intr = np.array([400., 400, 320, 240, 0, 0, 0, 0])
    cal0 = CameraCalib(gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0.05, 0, 0)),
                       0.001, 0.0, intr, model="radtan")          # base, global
    cal1 = CameraCalib(gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0.10, 0, 0)),
                       0.002, 0.017, intr, model="equidistant")   # fisheye, rolling
    r = _StubResult()
    r.t_states = np.array([0.0, 0.1, 0.2, 0.3]); r.values = vals
    r.cameras = [{'calib': cal0}, {'calib': cal1}]
    r.imu_intr_xi = np.zeros(24); r.imu_model = "calyx"
    r.imus = [dict(name='aux1', td=0.001, intr_xi24=np.zeros(24),
                   T_IstoI=gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0.1, -0.05, 0)))]
    r.b_g = np.zeros(3); r.b_a = np.zeros(3)
    r.T_BinW = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 2))
    return r, Ts


def test_ground_truth_from_result_fields():
    r, Ts = _stub_result()
    gt = ground_truth_from_result(r, image_sizes=[(640, 480), (848, 800)])
    assert isinstance(gt, GroundTruth)
    assert len(gt.cams) == 2 and len(gt.imus) == 2                 # 2 cams, base + 1 aux
    assert gt.cams[0]['image_size'] == [640, 480]
    assert gt.cams[1]['is_fisheye'] is True
    assert abs(gt.cams[1]['readout'] - 0.017) < 1e-9 and gt.cams[0]['readout'] == 0.0
    assert gt.imus[0]['is_base'] and gt.imus[0]['T_IstoI'] is None
    assert gt.imus[1]['name'] == 'aux1' and gt.imus[1]['T_IstoI'] is not None
    assert len(gt.T_states) == 4 and np.allclose(gt.T_states[1], Ts[1].matrix())
    assert gt.imu_model == "calyx" and gt.g_inW[2] < 0


def test_ground_truth_from_result_json_roundtrips():
    r, _ = _stub_result()
    gt = ground_truth_from_result(r, image_sizes=[(640, 480), (848, 800)])
    GroundTruth(**json.loads(json.dumps(asdict(gt))))             # serialises like gt_from_json
