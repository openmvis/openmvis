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
"""Minimal duck-typed cam-IMU results for the report tests."""
from types import SimpleNamespace as NS

import numpy as np
import gtsam


def _fake_calib(rot=(0.02, -0.03, 0.05), trans=(0.05, -0.02, 0.01)):
    return NS(model="radtan", is_fisheye=False,
              intr=np.array([400., 400., 320., 240., -0.28, 0.07, 5e-4, -4e-4]),
              td=0.01, tr=0.0, DIM=16,
              T=gtsam.Pose3(gtsam.Rot3.Expmap(np.array(rot)), gtsam.Point3(*trans)))


def _fake_result(aux=False):
    cam = dict(calib=_fake_calib(), calib_sigma=np.full(16, 0.01),
               reproj_rms_px=0.42, reproj_median_px=0.35, reproj_p95_px=0.88,
               n_frames=300)
    imus = []
    if aux:
        T = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                        gtsam.Point3(0.08, -0.04, 0.03))
        imus = [dict(name="aux", td=0.004, chi2_dof=1.05, T_IstoI=T,
                     b_g_s=np.zeros(3), b_a_s=np.zeros(3))]
    return NS(ok=True, imu_model="calyx",
              b_g=np.array([0.01, -0.02, 0.005]), b_a=np.array([0.05, -0.03, 0.02]),
              b_g_drift=np.zeros(3), b_a_drift=np.zeros(3),
              imu_intr_sigma=None, imu_intr_xi=np.zeros(24),
              diagnostics=dict(reproj_rms_px=0.42, reproj_median_px=0.35,
                               reproj_p95_px=0.88, cam_chi2_dof=1.1,
                               imu_chi2_dof=0.9, n_states=600, iterations=12),
              cameras=[cam], imus=imus)
