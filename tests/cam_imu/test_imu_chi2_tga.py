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
"""The reported IMU chi2 is scored at the estimated t_ga, the value the solver used.

`_imu_cost_split` called `whitened_error` without t_ga, which scored every factor at t_ga = 0. A
good solve with a large offset then read high: Looper `robot_arm` 1, t_ga -5.54 ms, reported 29.93
where the estimated t_ga gives 0.141.
"""
import gtsam
import numpy as np

from mvis.calibration import pipeline_cam_imu as P
from mvis.calibration.pipeline_cam_imu import I0, O0, X, _Vk, _imu_cost_split


class _Factor:
    """Residual is zero at `t_ga_true` and grows with distance from it."""

    def __init__(self, t_ga_true):
        self.t_ga_true = t_ga_true

    def whitened_error(self, pose0, pose1, vb0, vb1, xi_in, t_ga=0.0):
        return np.full(15, (t_ga - self.t_ga_true) * 1e3)


def _values(n, t_ga=None):
    v = gtsam.Values()
    for k in range(n + 1):
        v.insert(X(k), gtsam.Pose3())
        v.insert(_Vk(k), np.zeros(9))
    v.insert(I0, np.zeros(24))
    if t_ga is not None:
        v.insert(O0, np.array([t_ga]))
    return v


def test_cost_is_scored_at_the_estimated_t_ga():
    sums, per_interval = _imu_cost_split([_Factor(-0.005)] * 2, _values(2, -0.005), O0)
    assert sum(sums.values()) == 0.0
    assert np.array_equal(per_interval, np.zeros(2))


def test_cost_moves_with_the_t_ga_in_the_values():
    sums, _ = _imu_cost_split([_Factor(-0.005)], _values(1, -0.004), O0)
    assert sum(sums.values()) == 15 * 1.0 ** 2


def test_without_a_t_ga_key_the_cost_is_scored_at_zero():
    # estimate_t_ga off: no key in the values, so this stays the plain factor cost
    sums, _ = _imu_cost_split([_Factor(0.0)], _values(1))
    assert sum(sums.values()) == 0.0


def test_a_t_ga_key_missing_from_the_values_is_an_error_not_a_silent_zero():
    import pytest
    with pytest.raises(RuntimeError):
        _imu_cost_split([_Factor(0.0)], _values(1), O0)


def test_the_run_passes_the_t_ga_key_when_it_estimates_t_ga():
    import inspect
    src = inspect.getsource(P._run_cam_imu_calibration_multi_once)
    assert "_imu_cost_split(imu_factors, res, O0 if est_t_ga_run(cfg) else None)" in src

