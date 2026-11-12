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
"""Frozen copy of `mvis.calibration.factors.lie_groups` (S0, never edited).

Only the functions the oracles use are copied: skew, exp_so3, log_so3, the
SO(3) right / left-inverse / right-inverse Jacobians, _Q_matrix and
se3_right_jacobian. Verbatim; no imports from the live module.
"""

import numpy as np
import gtsam

# Small-angle threshold for Taylor expansion
_EPS = 1e-10


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]× from a 3-vector.

    [v]× · w = v × w (cross product).
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential: rotation matrix from a 3-vector (gtsam Rot3.Expmap)."""
    return gtsam.Rot3.Expmap(np.asarray(phi, float)).matrix()


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: 3-vector from a rotation matrix (gtsam Rot3.Logmap)."""
    return gtsam.Rot3.Logmap(gtsam.Rot3(np.asarray(R, float)))


# =====================================================================
# SO(3) Jacobians
# =====================================================================

def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3): J_r(φ).

    J_r(φ) = I - (1-cos‖φ‖)/‖φ‖² · [φ]× + (‖φ‖-sin‖φ‖)/‖φ‖³ · [φ]×²

    For small ‖φ‖: J_r ≈ I - 0.5·[φ]×

    Args:
        phi: 3-vector (rotation axis × angle).

    Returns:
        3×3 matrix.
    """
    theta = np.linalg.norm(phi)
    if theta < _EPS:
        return np.eye(3) - 0.5 * skew(phi)

    phi_x = skew(phi)
    phi_x2 = phi_x @ phi_x
    return (np.eye(3)
            - (1.0 - np.cos(theta)) / (theta ** 2) * phi_x
            + (theta - np.sin(theta)) / (theta ** 3) * phi_x2)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): J_l(φ).

    J_l(φ) = I + (1-cos‖φ‖)/‖φ‖² · [φ]× + (‖φ‖-sin‖φ‖)/‖φ‖³ · [φ]×²

    Note: J_l(φ) = J_r(-φ).
    """
    return so3_right_jacobian(-phi)


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of the left Jacobian of SO(3): J_l⁻¹(φ).

    J_l⁻¹(φ) = I - 0.5·[φ]× + (1/‖φ‖² - (1+cos‖φ‖)/(2‖φ‖sin‖φ‖)) · [φ]×²

    For small ‖φ‖: J_l⁻¹ ≈ I + 0.5·[φ]×
    """
    theta = np.linalg.norm(phi)
    if theta < _EPS:
        return np.eye(3) + 0.5 * skew(phi)

    phi_x = skew(phi)
    phi_x2 = phi_x @ phi_x
    c = 1.0 / (theta ** 2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * phi_x + c * phi_x2


def so3_right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of the right Jacobian of SO(3): J_r⁻¹(φ).

    J_r⁻¹(φ) = J_l⁻¹(-φ).
    """
    return so3_left_jacobian_inv(-phi)


# =====================================================================
# SE(3) Jacobians
# =====================================================================

def _Q_matrix(xi: np.ndarray) -> np.ndarray:
    """Q matrix for the SE(3) Jacobian (Barfoot eq. 9.84).

    Used in the upper-right block of the SE(3) right/left Jacobian.

    Args:
        xi: 6-vector [rho; phi] where rho is translation, phi is rotation.

    Returns:
        3×3 matrix.
    """
    rho = xi[:3]
    phi = xi[3:]
    theta = np.linalg.norm(phi)

    rho_x = skew(rho)
    phi_x = skew(phi)

    if theta < _EPS:
        return 0.5 * rho_x

    theta2 = theta ** 2
    theta3 = theta ** 3
    theta4 = theta ** 4
    theta5 = theta ** 5
    st = np.sin(theta)
    ct = np.cos(theta)

    # Barfoot eq. 9.84
    Q = (0.5 * rho_x
         + (theta - st) / theta3 * (phi_x @ rho_x + rho_x @ phi_x + phi_x @ rho_x @ phi_x)
         - (1.0 - theta2 / 2.0 - ct) / theta4 * (phi_x @ phi_x @ rho_x + rho_x @ phi_x @ phi_x
                                                    - 3.0 * phi_x @ rho_x @ phi_x)
         - 0.5 * ((1.0 - theta2 / 2.0 - ct) / theta4
                   - 3.0 * (theta - st - theta3 / 6.0) / theta5) * (
             phi_x @ rho_x @ phi_x @ phi_x + phi_x @ phi_x @ rho_x @ phi_x))

    return Q


def se3_right_jacobian(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SE(3): J_r(ξ).

    GTSAM convention: ξ = [ω₁,ω₂,ω₃, v₁,v₂,v₃] (rotation first).

    In GTSAM convention, J_r has the structure:
        J_r = [J_r(φ),     0₃ₓ₃  ]
              [Q_gtsam,    J_r(φ) ]

    where Q_gtsam is computed from the Barfoot Q with convention mapping.

    Args:
        xi: 6-vector in GTSAM convention [rot; trans].

    Returns:
        6×6 matrix in GTSAM convention.
    """
    phi = xi[:3]   # rotation part (GTSAM: first 3)
    rho = xi[3:]   # translation part (GTSAM: last 3)

    Jr_so3 = so3_right_jacobian(phi)

    # Q matrix from Barfoot uses [trans; rot] ordering internally
    xi_barfoot = np.array([rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]])
    Qr = _Q_matrix(xi_barfoot)

    # Build directly in GTSAM convention [rot; trans]
    J = np.zeros((6, 6))
    J[:3, :3] = Jr_so3      # rot-rot block
    J[3:, 3:] = Jr_so3      # trans-trans block
    J[3:, :3] = Qr.T        # trans-rot coupling (transpose for convention)
    return J
