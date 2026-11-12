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
# tests/simulation/test_sim_recovery_driver.py
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_build_synth_shapes_and_recovers_intrinsics():
    """End-to-end on a tiny synthetic GT: build synth -> recover -> intrinsics
    row exists and base reproj is at the noise floor. Small/fast (1 cam, base
    IMU, ~6 s) — the REAL 3cam study is run via the script, not in CI."""
    import gtsam
    from mvis.simulation.from_result import GroundTruth, cal_from_vec
    from mvis.simulation.synth import build_synth, recover, zscore_table
    # minimal GT: identity-ish helix trajectory at 30 Hz, one radtan cam, base IMU
    dt = 1.0 / 30.0
    ts = 1.0 + np.arange(180) * dt
    T_states = []
    for t in ts:
        M = np.eye(4)
        M[:3, :3] = gtsam.Rot3.Rz(0.4 * np.sin(t)).matrix()
        M[:3, 3] = [0.2 * np.sin(t), 0.2 * np.cos(t), 0.0]
        T_states.append(M.tolist())
    Tb = np.eye(4); Tb[:3, 3] = [0.0, 0.0, 1.2]
    gt = GroundTruth(
        cams=[dict(intr8=[400., 400., 320., 240., -0.05, 0.0, 0.0, 0.0],
                   T_CtoI=np.eye(4).tolist(), td=0.0, is_fisheye=False,
                   image_size=[640, 480])],
        imus=[dict(name="base", is_base=True, xi24=[0.0] * 24,
                   T_IstoI=None, td=0.0)],
        b_g=[0.0, 0.0, 0.0], b_a=[0.0, 0.0, 0.0], T_BinW=Tb.tolist(),
        t_states=ts.tolist(), T_states=T_states, imu_model="calyx",
        g_inW=[0.0, 0.0, -9.81])
    cam_times = [ts[5:-5:3]]                      # ~10 Hz frames inside coverage
    imu_times = [ts]                              # base IMU at state rate (sim only)

    # Attach board points local (required by build_synth)
    pts = []
    for i in range(6):
        for j in range(6):
            pts.append([0.05 * j, 0.05 * i, 0.0])
    gt.board_points_local = np.array(pts)

    cams, t_imu, w_m, a_m, aux = build_synth(
        gt, cam_times, imu_times, sigma_px=1.0,
        imu_sigmas=dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002),
        seed=0)
    assert cams[0]["pin_intrinsics"] is False
    assert cams[0]["uv_list"] and len(cams[0]["t_cam"]) > 10
    # board points in WORLD from GT board pose
    res = recover(cams, _board_points_world(gt), t_imu, w_m, a_m, aux, None)
    rows = zscore_table(gt, res)
    names = {r["param"] for r in rows}
    assert any(n.startswith("cam0/fx") for n in names)     # intrinsics checked
    assert res.diagnostics["reproj_rms_px"] < 3.0          # noise-floor-ish


def _board_points_world(gt):
    # mirror the script helper for the test
    import numpy as np
    from mvis.detection.board_factory import create_board
    # a small synthetic board: reuse a 6x6 apriltag grid's local points
    # (the script uses the real board; the test just needs a planar grid)
    pts = []
    for i in range(6):
        for j in range(6):
            pts.append([0.05 * j, 0.05 * i, 0.0])
    P = np.array(pts)
    Tb = np.array(gt.T_BinW)
    return (Tb[:3, :3] @ P.T).T + Tb[:3, 3]
