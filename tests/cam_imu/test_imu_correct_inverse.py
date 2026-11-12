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
"""Phase 1 (roundtrip-identity): correct() must be the EXACT inverse of measure() for
non-identity intrinsics, including the gyro Tg*a coupling. If this holds, a corrected
dataset MUST recalibrate to identity when the estimator uses the same model — so any
non-collapse is a model/estimator issue, not weak observability."""
import numpy as np
from mvis.imu import ImuModel, parse_imu_model


def test_correct_is_exact_inverse_of_measure():
    rng = np.random.default_rng(0)
    model = parse_imu_model("calyx")
    worst = 0.0
    for _ in range(200):
        M = ImuModel.from_tangent(rng.normal(0, 0.12, 24), model)   # non-identity Dw/Da/Tg
        b_g, b_a = rng.normal(0, 0.01, 3), rng.normal(0, 0.05, 3)
        w, a = rng.normal(0, 1.5, 3), rng.normal(0, 5, 3)
        wm, am = M.measure(w, a, b_g, b_a)
        wc, ac = M.correct(wm, am, b_g, b_a)
        worst = max(worst, np.max(np.abs(wc - w)), np.max(np.abs(ac - a)))
    assert worst < 1e-9, f"correct() not the inverse of measure(): {worst:.2e}"


def test_same_xi_under_wrong_model_scrambles_intrinsics():
    """The footgun: a calyx xi read as rpng gives different Dw/Da (mechanism for a
    wrong correction that still passes the accel-on-gravity check)."""
    xi = np.r_[[0.15, 0.02, -0.01, 0.0, 0.12, 0.03, 0.0, 0.0, 0.10], [0.02, 0, 0.01, 0.03, 0, 0.02], np.zeros(9)]
    Mc = ImuModel.from_tangent(xi, parse_imu_model("calyx"))
    Mr = ImuModel.from_tangent(xi, parse_imu_model("rpng"))
    assert not np.allclose(np.diag(Mc.Dw()), np.diag(Mr.Dw()), atol=1e-3)
