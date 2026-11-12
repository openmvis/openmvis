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
"""apply-calib side of the gyro<->accel time offset (aci-gyro-accel-td).

The corrected accel at stamp t must be the measured accel at t + t_ga
(undoing "measured at t = physical at t - t_ga"), interpolated on the
header clock; calib YAMLs round-trip the t_ga field.
"""
import numpy as np

from mvis.recalib.apply import shifted_accel
from mvis.recalib.calib_spec import _imu_from_block


def test_shifted_accel_undoes_delay():
    """A delayed sinusoid, shifted by +t_ga, matches the un-delayed signal."""
    hz, t_ga = 200.0, 2.3e-3
    t = np.arange(0.0, 2.0, 1.0 / hz)
    def a_phys(tt):
        return np.stack([np.sin(2 * np.pi * 1.3 * tt),
                         np.cos(2 * np.pi * 0.7 * tt),
                         np.sin(2 * np.pi * 2.1 * tt + 0.5)], axis=-1)
    A_meas = a_phys(t - t_ga)                       # measured = physical at t - t_ga
    T_ns = t * 1e9
    q = t[10:-10]                                   # stay off the clamped edges
    A_corr = np.stack([shifted_accel(T_ns, A_meas, tq * 1e9 + t_ga * 1e9) for tq in q])
    assert np.max(np.abs(A_corr - a_phys(q))) < 2e-3    # linear-interp error only


def test_shifted_accel_zero_offset_identity():
    T_ns = np.arange(10) * 5e6
    A = np.arange(30, dtype=float).reshape(10, 3)
    for k in range(10):
        assert np.allclose(shifted_accel(T_ns, A, T_ns[k]), A[k])


def test_calib_spec_parses_t_ga():
    blk = dict(name="imu0", role="base", model="calyx",
               imu_intr_xi=np.zeros(24).tolist(),
               b_g=[0, 0, 0], b_a=[0, 0, 0], t_ga=0.00123)
    imu = _imu_from_block(blk)
    assert abs(imu.t_ga - 0.00123) < 1e-12
    assert _imu_from_block({k: v for k, v in blk.items() if k != "t_ga"}).t_ga == 0.0
