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
"""`tests/_blas_threads.py`: the in-process OpenBLAS thread pin the golden
test relies on (cam-IMU speed-up S0 fix round 1)."""
import numpy as np
import pytest

from tests import _blas_threads as B

pytestmark = pytest.mark.skipif(B.get_num_threads() is None, reason="no OpenBLAS in this process")


def test_pin_and_restore_every_loaded_openblas():
    before = B.num_threads_per_lib()
    assert before and all(n >= 1 for n in before.values())
    assert B.get_num_threads() == before[B.loaded_libs()[0]]
    with B.blas_threads(1) as n:
        assert n == 1
        assert set(B.num_threads_per_lib().values()) == {1}
        np.linalg.solve(np.eye(64) * 2.0, np.ones(64))     # a BLAS call under the pin
    assert B.num_threads_per_lib() == before
    with B.blas_threads(None) as n:                        # None: leave as is
        assert n == B.get_num_threads()


def test_numpy_lib_is_reported_first():
    libs = B.loaded_libs()
    assert any("numpy" in p for p in libs), libs
    assert "numpy" in libs[0]
    assert "numpy" in B.describe()
