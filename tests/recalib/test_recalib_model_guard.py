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
"""Phase 0: the recalib reader must fail loud on a model-less imu_intr_xi (footgun guard)."""
import numpy as np
import pytest
from mvis.recalib.calib_spec import _imu_from_block


def test_xi_without_model_raises():
    blk = {"name": "imu0", "role": "base", "imu_intr_xi": [0.0] * 24}   # no 'model'
    with pytest.raises(ValueError, match="model"):
        _imu_from_block(blk)


def test_xi_with_model_reads_calyx_layout():
    xi = list(np.r_[np.full(9, 0.1), np.zeros(15)])                     # calyx Dw full-9
    blk = {"name": "imu0", "role": "base", "model": "calyx", "imu_intr_xi": xi}
    imu = _imu_from_block(blk)
    assert imu.model == "calyx"
    assert imu.intr.Dw().shape == (3, 3)


def test_aux_without_xi_is_identity_no_raise():
    imu = _imu_from_block({"name": "aux1", "role": "aux"})              # no xi, no model -> ok
    assert np.allclose(imu.intr.Dw(), np.eye(3))
