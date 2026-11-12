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
"""Frozen oracles for the cam-IMU speed-up work (S0, 2026-10-31).

Every module here is a verbatim copy of the live code as it stood before the
first kernel step landed (the S0 commit). Only
the imports change — the copies import each other, never the live modules
they shadow — so the oracles stay self-contained while the live code is
rewritten (`make_projector`, `_Projector.project`, `_gtsam_project`,
`lie_groups`, `imu_model.dw/da_jacobian`, `R_wtoI`/`R_atoI`,
`_rs_row_offset` are copied too).

    ref_lie_groups          skew / exp / log / SO(3), SE(3) Jacobians
    ref_imu_model           Dw/Da/Tg builders + parameter Jacobians, R_wtoI/R_atoI
    ref_projection          _gtsam_project / _kernel_project / _Projector
    ref_camera_factor       _rs_row_offset
    ref_aci                 integrate_RAJ, ACIPreintegrator
    ref_imu_segment         _Corrected / _State / _step / Propagated / ImuSegment
    ref_camera_imu_factor   CameraCalibImuFactor
    ref_camera_interp_factor CameraCalibInterpFactor (+ _interp_pose)
    ref_diagnostics         _slice_imu, _propagate_state, _densify_velocities,
                            _frame_corner_norms, _reproj_rms, _reproj_corner_norms
    (K2, 2026-11-01, frozen from the pre-K2 tree of the pre-release history:)
    ref_camera_calib_factor CameraCalibFactor (the cam-only-style single-state factor)
    ref_imu_factor          ImuFactorACI_In (base IMU factor)
    ref_imu_factor_gyro     ImuFactorACI_Multi_In_SO3 (gyro-only aux factor)
    (2026-11-07, moved out of src when it lost its last caller:)
    ref_intrinsic_jacobians intrinsic_jacobian_radtan / intrinsic_jacobian_fisheye

THESE COPIES ARE NEVER EDITED. `tests/cam_imu/test_speedup_equivalence.py` compares
the live code against them on a seeded fixture; a step that is meant to be
bitwise uses `exact=True`, a rounding step uses `assert_close`.
`ref_camera_imu_factor` and `ref_camera_interp_factor` carry the board-heights
support (`board_pids` / `n_board_points` / `uses_heights_key` / `_pB` / `H_h`)
the live factors have.

The only non-verbatim pieces are the two small adapters that replace
attribute access on live objects by the copied leaf functions:
`ref_imu_model.wrap_intr` (an ImuModel value -> a frozen view whose
`Dw()/Da()/Tg()/R_wtoI/R_atoI` are the copied builders) and
`ref_projection._RefCameraModel` (a cal_vec -> the copied per-model
`project`). Where a copy calls them the line is marked `# S0 adapter`.
"""
