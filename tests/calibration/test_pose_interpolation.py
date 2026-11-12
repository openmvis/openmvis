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
"""Tests for SE(3) interpolation helpers used by the projection factor."""

import numpy as np
import gtsam


class TestSE3Interpolation:
    """Test the SLERP interpolation function."""

    def test_identity_at_endpoints(self):
        from mvis.calibration.factors.pose_interpolation import se3_interpolate as _se3_interpolate

        p0 = gtsam.Pose3(gtsam.Rot3.Yaw(0.1), gtsam.Point3(1, 0, 0))
        p1 = gtsam.Pose3(gtsam.Rot3.Yaw(0.5), gtsam.Point3(2, 0, 0))

        r0 = _se3_interpolate(p0, p1, 0.0)
        r1 = _se3_interpolate(p0, p1, 1.0)

        np.testing.assert_allclose(r0.matrix(), p0.matrix(), atol=1e-10)
        np.testing.assert_allclose(r1.matrix(), p1.matrix(), atol=1e-10)

    def test_midpoint(self):
        from mvis.calibration.factors.pose_interpolation import se3_interpolate as _se3_interpolate

        p0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))
        p1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2, 0, 0))

        mid = _se3_interpolate(p0, p1, 0.5)
        t = mid.translation()
        np.testing.assert_allclose([t[0], t[1], t[2]], [1, 0, 0], atol=1e-10)

    def test_extrapolates_lambda(self):
        """Lambda outside [0,1] extrapolates (no clamp) for correct Jacobians."""
        from mvis.calibration.factors.pose_interpolation import se3_interpolate as _se3_interpolate

        p0 = gtsam.Pose3()
        p1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0))

        # lambda < 0 extrapolates beyond p0
        r = _se3_interpolate(p0, p1, -0.5)
        np.testing.assert_allclose(
            [r.translation()[0], r.translation()[1], r.translation()[2]],
            [-0.5, 0, 0], atol=1e-10)

        # lambda > 1 extrapolates beyond p1
        r = _se3_interpolate(p0, p1, 1.5)
        np.testing.assert_allclose(
            [r.translation()[0], r.translation()[1], r.translation()[2]],
            [1.5, 0, 0], atol=1e-10)
