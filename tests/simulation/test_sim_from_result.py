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
import numpy as np
import gtsam
import pytest
from mvis.simulation.from_result import cal_from_vec, fit_bspline_from_states


def _stub_result():
    import gtsam, numpy as np
    from mvis.calibration.factors.camera_calib import CameraCalib   # the .calib type
    X = gtsam.symbol_shorthand.X

    class R: pass
    r = R()
    T_CtoI = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.01, -0.02, 0.03))
    cc = CameraCalib(T_CtoI, 0.004, 0.0,
                     np.array([400., 410., 320., 240., -0.1, 0.01, 0., 0.]),
                     model="radtan")
    r.cameras = [{"calib": cc, "image_size": (640, 480)}]
    r.imus = []                                  # base-only stub
    r.imu_intr_xi = np.zeros(24)
    r.b_g = np.array([0.001, 0.0, -0.002]); r.b_a = np.array([0.02, -0.01, 0.0])
    r.T_BinW = gtsam.Pose3(gtsam.Rot3.Rx(0.1), gtsam.Point3(0.0, 0.0, 1.5))
    dt = 1.0 / 30.0
    ts = 1.0 + np.arange(10) * dt
    vals = gtsam.Values()
    for k, t in enumerate(ts):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3.Rz(0.2 * t), gtsam.Point3(t, 0., 0.)))
    r.values = vals; r.t_states = ts; r.imu_model = "calyx"
    return r


def test_capture_and_roundtrip(tmp_path):
    from mvis.simulation.from_result import capture_gt, gt_to_json, gt_from_json
    gt = capture_gt(_stub_result())
    assert len(gt.cams) == 1
    assert gt.cams[0]["intr8"][0] == 400.0
    assert gt.imus[0]["is_base"] is True
    p = tmp_path / "gt.json"
    gt_to_json(gt, str(p))
    gt2 = gt_from_json(str(p))
    assert gt2.imu_model == "calyx"
    assert np.allclose(gt2.b_a, gt.b_a)
    assert len(gt2.T_states) == 10


def test_cal_from_vec_radtan():
    cal = cal_from_vec([400.0, 410.0, 320.0, 240.0, -0.1, 0.01, 0.0, 0.0], is_fisheye=False)
    assert isinstance(cal, gtsam.Cal3DS2)
    assert cal.fx() == pytest.approx(400.0)
    assert cal.px() == pytest.approx(320.0)


def test_cal_from_vec_fisheye():
    cal = cal_from_vec([300.0, 300.0, 320.0, 240.0, 0.01, 0.001, 0.0, 0.0], is_fisheye=True)
    assert isinstance(cal, gtsam.Cal3Fisheye)
    assert cal.fy() == pytest.approx(300.0)


def test_fit_bspline_passes_through_states():
    # a smooth helical T_ItoG path sampled at 30 Hz
    dt = 1.0 / 30.0
    ts = 1.0 + np.arange(120) * dt
    T_states = []
    for k, t in enumerate(ts):
        R = gtsam.Rot3.Rz(0.3 * t).matrix()
        p = np.array([np.cos(t), np.sin(t), 0.1 * t])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p
        T_states.append(M)
    spline, max_err = fit_bspline_from_states(ts, T_states, dt)
    assert max_err < 5e-3                      # spline tracks the states (< 5 mm)
    # queryable across the interior and pose is finite
    R_GtoI, pos = spline.get_pose(ts[60])
    assert np.all(np.isfinite(pos))
    assert abs(np.linalg.det(R_GtoI) - 1.0) < 1e-6
