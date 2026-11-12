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
"""Permanent guard: a small seeded closed-loop cam-IMU recovery. Marked slow
(real GTSAM solve).

Scope (established by the 3cam study, docs/plans/mvis/sim-recovery-3cam):
this fast synthetic scenario (8 s, 1 cam + base IMU) robustly recovers the
CAMERA-side parameters — intrinsics (fx,fy,cx,cy + distortion), camera-to-IMU
rotation, and the in-plane (XY) lever — within 3 sigma, so the test asserts
those strictly. They exercise the entire closed loop (synth corners + synth IMU
-> joint solve -> compare), and a broken IMU path would corrupt the trajectory
and blow up the camera reproj, so the camera assertions transitively guard the
IMU path too.

OUT OF SCOPE here (validated only by the real-motion STUDY, not this unit test):
IMU gyro/accel SCALE (Dw/Da), inter-sensor td, and the camera depth-lever
(pos2). A short, modestly-excited synthetic sequence cannot observe IMU scale
(it needs sustained, varied angular rate over a long trajectory — the study saw
~2% there vs >10% in a scenario this small), so asserting on them here would be
testing excitation, not the estimator.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from mvis.simulation.synth import build_synth, recover, zscore_table
from tests.helpers.sim_scenarios import (_STRICT, _small_gt)


def test_seeded_camera_recovery_within_3sigma():
    gt = _small_gt()
    gt.board_points_local = np.array(
        [[0.05 * j, 0.05 * i, 0.0] for i in range(6) for j in range(6)])
    ts = np.asarray(gt.t_states)
    # IMU at a realistic 400 Hz (not the 30 Hz state rate); spline is queryable
    # anywhere inside the state span.
    imu_t = np.arange(ts[2], ts[-2], 1.0 / 400.0)
    cams, t_imu, w_m, a_m, aux = build_synth(
        gt, [ts[5:-5:3]], [imu_t], sigma_px=1.0,
        imu_sigmas=dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002),
        seed=7)
    res = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, None)
    assert res.ok
    assert res.diagnostics["reproj_rms_px"] < 2.0       # ~ sigma_px noise floor

    rows = zscore_table(gt, res)
    strict = [r for r in rows if _STRICT.match(r["param"])]
    assert strict, "no camera-side rows produced — zscore_table/recover wiring broke"
    bad = [r for r in strict if r["sigma"] == r["sigma"] and abs(r["z"]) >= 3.0]
    assert not bad, \
        f"camera params outside 3 sigma: {[(r['param'], round(r['z'], 2)) for r in bad]}"
