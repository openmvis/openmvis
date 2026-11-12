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
"""ImuFactorACI_In_SE3 (5-key): residual + analytic Jacobian validation.

The intrinsics variable is the 24-dim ImuModel tangent ξ_in around intr_init.
Jacobians (5 blocks) are validated by central finite differences at a residual≈0
state with biases/intrinsics OFF init (so the bias/intrinsic terms are exercised
while the first-order approximation is exact).
"""

import numpy as np
import gtsam
import pytest

from mvis.imu import KALIBR, RPNG, CALYX
from mvis.calibration.factors.numdiff import numerical_jacobian
from tests.helpers.imu_factor_cases import (_build_factor, _consistent_state)


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_residual_zero_at_consistent_state(model):
    f, pre = _build_factor(model)
    s = _consistent_state(f)
    e = f.whitened_error(*s)
    assert np.max(np.abs(e)) < 1e-6
    assert e.shape == (15,)


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_jacobians_fd(model):
    f, _ = _build_factor(model)
    pose0, pose1, vb0, vb1, xi_in = _consistent_state(f)
    _, H = f.error_and_jacobians(pose0, pose1, vb0, vb1, xi_in)
    _, H_fd = numerical_jacobian(
        f.whitened_error, [pose0, pose1, vb0, vb1, xi_in])
    for i, (Ha, Hn) in enumerate(zip(H, H_fd)):
        assert np.allclose(Hn, Ha, rtol=1e-4, atol=1e-5), f"block {i}"


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_custom_factor_in_graph(model):
    f, _ = _build_factor(model)
    pose0, pose1, vb0, vb1, xi_in = _consistent_state(f)
    X = gtsam.symbol_shorthand.X
    V = gtsam.symbol_shorthand.V
    keys = [X(0), X(1), V(0), V(1), gtsam.symbol('Y', 0)]
    factor = f.make_factor(*keys)

    values = gtsam.Values()
    values.insert(keys[0], pose0)
    values.insert(keys[1], pose1)
    values.insert(keys[2], vb0)
    values.insert(keys[3], vb1)
    values.insert(keys[4], xi_in)

    err = factor.unwhitenedError(values)
    assert err.shape == (15,)
    assert np.allclose(err, f.whitened_error(pose0, pose1, vb0, vb1, xi_in))
    assert factor.linearize(values) is not None
