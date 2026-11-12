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
"""Frozen copy of the `mvis.imu.imu_model` leaves the oracles use (S0, never
edited): the Dw / Da / Tg builders, their parameter Jacobians, the
`dw_jacobian` / `da_jacobian` model dispatch and the `R_wtoI` / `R_atoI`
properties.

The live `ImuModel` instance is still the VALUE carrier (x_dw, x_da, x_tg, R
and the per-model layout constants MODEL / N_DW / HAS_R) — `wrap_intr` turns
it into a frozen view whose `Dw()`, `Da()`, `Tg()`, `R_wtoI`, `R_atoI` go
through the copied builders below instead of the live methods, so the verbatim
copies of `imu_segment.py` / `aci.py` keep their `intr.Dw()` spelling.
"""

from __future__ import annotations

import numpy as np

KALIBR = 0
RPNG = 1
CALYX = 2

_I3 = np.eye(3)
_E = np.eye(3)  # columns e1, e2, e3


# --- shared correction-matrix builders (the per-model math, in one place) ---
def lower_tri(x: np.ndarray) -> np.ndarray:
    """3x3 lower-triangular matrix from a 6-vector (Kalibr Dw/Da convention)."""
    return np.array([
        [x[0], 0.0,  0.0],
        [x[1], x[3], 0.0],
        [x[2], x[4], x[5]],
    ])


def upper_tri(x: np.ndarray) -> np.ndarray:
    """3x3 upper-triangular matrix from a 6-vector (RPNG / Calyx-Da convention)."""
    return np.array([
        [x[0], x[1], x[3]],
        [0.0,  x[2], x[4]],
        [0.0,  0.0,  x[5]],
    ])


def full3x3(x: np.ndarray) -> np.ndarray:
    """Full 3x3 matrix from a 9-vector, row-major (Calyx Dw convention)."""
    return np.array([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]],
        [x[6], x[7], x[8]],
    ])


def tg_matrix(x_tg: np.ndarray) -> np.ndarray:
    """Build the 3x3 Tg (gyro g-sensitivity) matrix, column-major from a 9-vector.
    Model-independent (Tg has the same parameterization for all models)."""
    return np.array([
        [x_tg[0], x_tg[3], x_tg[6]],
        [x_tg[1], x_tg[4], x_tg[7]],
        [x_tg[2], x_tg[5], x_tg[8]],
    ])


# --- shared parameter Jacobians of the correction matrices (used by the ACI
#     integrator); `lower`/`upper`/`full` mirror the triangular conventions ---
def jac_lower(v: np.ndarray) -> np.ndarray:
    """d(M·v)/d(x) for a lower-triangular 3x3 M(x6): 3x6 (Kalibr Dw/Da)."""
    v1, v2, v3 = v
    H = np.zeros((3, 6))
    H[:, 0:3] = v1 * _I3
    H[:, 3] = v2 * _E[:, 1]
    H[:, 4] = v2 * _E[:, 2]
    H[:, 5] = v3 * _E[:, 2]
    return H


def jac_upper(v: np.ndarray) -> np.ndarray:
    """d(M·v)/d(x) for an upper-triangular 3x3 M(x6): 3x6 (RPNG / Calyx-Da)."""
    v1, v2, v3 = v
    H = np.zeros((3, 6))
    H[:, 0] = v1 * _E[:, 0]
    H[:, 1] = v2 * _E[:, 0]
    H[:, 2] = v2 * _E[:, 1]
    H[:, 3:6] = v3 * _I3
    return H


def jac_full(v: np.ndarray) -> np.ndarray:
    """d(M·v)/d(x) for a full row-major 3x3 M(x9): 3x9 (Calyx Dw). Row i depends
    only on x[3i:3i+3] with d(row_i)/d(x[3i+j]) = v[j] -> block-diagonal of vᵀ."""
    H = np.zeros((3, 9))
    for i in range(3):
        H[i, 3 * i:3 * i + 3] = v
    return H


def tg_jacobian(a_inI: np.ndarray) -> np.ndarray:
    """d(Tg·a_inI)/d(x_Tg): 3x9 (matches ACI_In::get_H_Tg). Model-independent."""
    a1, a2, a3 = a_inI
    H = np.zeros((3, 9))
    H[:, 0:3] = a1 * _I3
    H[:, 3:6] = a2 * _I3
    H[:, 6:9] = a3 * _I3
    return H


# --- per-model static math: (build_dw, build_da, dw_param_jacobian,
#     da_param_jacobian, N_DW, HAS_R), as the three subclasses define them ---
_MODEL_MATH = {
    KALIBR: (lower_tri, lower_tri, jac_lower, jac_lower, 6, True),
    RPNG:   (upper_tri, upper_tri, jac_upper, jac_upper, 6, True),
    CALYX:  (full3x3,   upper_tri, jac_full,  jac_upper, 9, False),
}


def dw_jacobian(w_inw: np.ndarray, model: int) -> np.ndarray:
    """d(Dw·w_inw)/d(x_Dw): 3x6 (Kalibr/RPNG) or 3x9 (Calyx)."""
    return _MODEL_MATH[int(model)][2](w_inw)


def da_jacobian(a_ina: np.ndarray, model: int) -> np.ndarray:
    """d(Da·a_ina)/d(x_Da): 3x6 (matches ACI_In::get_H_Da)."""
    return _MODEL_MATH[int(model)][3](a_ina)


class _FrozenIntr:
    """Frozen view of an ImuModel value: the same x_dw/x_da/x_tg/R and layout,
    with the correction matrices and rotations built by the copies above
    (mirrors `ImuModel.Dw/Da/Tg/R_wtoI/R_atoI/n_dw/has_R/da_off/tg_off/r_off`)."""
    __slots__ = ("model", "x_dw", "x_da", "x_tg", "R", "_math")

    def __init__(self, model: int, x_dw, x_da, x_tg, R):
        self.model = int(model)
        self._math = _MODEL_MATH[self.model]
        self.x_dw = np.asarray(x_dw, float)
        self.x_da = np.asarray(x_da, float)
        self.x_tg = np.asarray(x_tg, float)
        self.R = np.asarray(R, float)

    # --- correction matrices (consumed by the ACI factor) ---
    def Dw(self) -> np.ndarray:
        return self._math[0](self.x_dw)

    def Da(self) -> np.ndarray:
        return self._math[1](self.x_da)

    def Tg(self) -> np.ndarray:
        return tg_matrix(self.x_tg)

    @property
    def R_wtoI(self) -> np.ndarray:
        return self.R if self.model == KALIBR else np.eye(3)

    @property
    def R_atoI(self) -> np.ndarray:
        return self.R if self.model == RPNG else np.eye(3)

    # --- model-dependent tangent layout ---
    @property
    def n_dw(self) -> int:
        return self._math[4]

    @property
    def has_R(self) -> bool:
        return self._math[5]

    @property
    def da_off(self) -> int:
        return self._math[4]

    @property
    def tg_off(self) -> int:
        return self._math[4] + 6

    @property
    def r_off(self) -> int:
        return self._math[4] + 15


def wrap_intr(intr) -> _FrozenIntr:
    """ImuModel (live value object) -> frozen view. Idempotent."""
    if isinstance(intr, _FrozenIntr):
        return intr
    return _FrozenIntr(intr.model, intr.x_dw, intr.x_da, intr.x_tg, intr.R)
