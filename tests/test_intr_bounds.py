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
"""solve.bound_intrinsics: the valid-domain hinge on double_sphere / omni_radtan intrinsics."""
import gtsam
import numpy as np

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.pipeline_cam_imu import intr_bound_factor
from mvis.core.config import CamImuCalibConfig


def _err(base, xi):
    fac = intr_bound_factor(gtsam.symbol('c', 0), base)
    v = gtsam.Values()
    v.insert(gtsam.symbol('c', 0), xi)
    return fac.unwhitenedError(v), fac.linearize(v).jacobian()[0]


def test_ds_zero_inside_domain_and_hinged_outside():
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, [287.0, 287.0, 414.0, 405.0, 0.0, 0.65],
                       model="double_sphere")
    e, _ = _err(base, np.zeros(base.DIM))
    assert np.all(e == 0)
    xi = np.zeros(base.DIM)
    xi[8] = -300.0                 # fx -> -13
    xi[13] = -0.85                 # alpha -> -0.2
    e, A = _err(base, xi)
    assert np.isclose(e.max(), 13.0) and np.isclose(sorted(e)[-2], 0.2)
    assert np.count_nonzero(np.abs(A).sum(axis=0)) == 2   # only the violated fx, alpha columns


def test_omni_xi_lower_bound_only_and_unbounded_models():
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, [560.0, 560.0, 414.0, 405.0, 0.96, 0, 0, 0, 0],
                       model="omni_radtan")
    xi = np.zeros(base.DIM)
    xi[12] = 6.0                   # xi = 6.96: valid (large-xi near-equivalence), no penalty
    assert np.all(_err(base, xi)[0] == 0)
    xi[12] = -2.0                  # xi = -1.04
    assert np.isclose(_err(base, xi)[0].max(), 1.04)
    eucm = CameraCalib(gtsam.Pose3(), 0.0, 0.0, [285.0, 285.0, 414.0, 405.0, 0.72, 0.86], model="eucm")
    assert intr_bound_factor(gtsam.symbol('c', 0), eucm) is None
    assert CamImuCalibConfig().bound_intrinsics is False


def test_kb4_focal_lower_bound_only():
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, [285.0, 286.0, 424.0, 400.0, -0.01, 0.05, -0.05, 0.01],
                       model="kb4")
    xi = np.zeros(base.DIM)
    xi[12:16] = [5.0, -5.0, 5.0, -5.0]   # distortion terms carry no bound
    assert np.all(_err(base, xi)[0] == 0)
    xi[9] = -290.0                        # fy -> -4
    e, A = _err(base, xi)
    assert np.isclose(e.max(), 4.0) and np.count_nonzero(e) == 1
    assert np.count_nonzero(np.abs(A).sum(axis=0)) == 1 and A[:, 9].any()


def test_generic_fisheye_names_get_the_kb4_bound():
    from mvis.core.camera_models import resolve_model_alias
    for name in ("fisheye", "kannala_brandt", "kannala-brandt"):
        model = resolve_model_alias(name)
        base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, [285.0, 286.0, 424.0, 400.0, 0, 0, 0, 0],
                           model=model)
        assert intr_bound_factor(gtsam.symbol('c', 0), base) is not None, name
    equi = CameraCalib(gtsam.Pose3(), 0.0, 0.0, [285.0, 286.0, 424.0, 400.0, 0, 0, 0, 0],
                       model="equidistant")
    assert intr_bound_factor(gtsam.symbol('c', 0), equi) is None
