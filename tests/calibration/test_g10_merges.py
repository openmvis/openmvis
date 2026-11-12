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
"""G10 merges: se3_interpolate on interp_pose, the shared ACI step input Jacobians and the
one motion dt rule."""
import gtsam
import numpy as np

from mvis.analysis.motion_analysis import compute_motion_metrics, motion_series, valid_board_poses
from mvis.calibration.factors.aci import _IntrPack, step_input_jacobians
from mvis.calibration.factors.pose_interpolation import (
    interp_pose, se3_interpolate, se3_interpolate_jacobians)
from mvis.imu import ImuModel, RPNG

P0 = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.1, -0.2, 0.3])), gtsam.Point3(1.0, 2.0, 3.0))
P1 = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.4, 0.1, -0.2])), gtsam.Point3(1.5, 1.0, 2.0))


def test_se3_interpolate_is_interp_pose_with_exact_ends():
    for lam in (-0.3, 0.25, 0.7, 1.4):
        assert se3_interpolate(P0, P1, lam).equals(interp_pose(P0, P1, lam)[0], 0.0)
        T, J0, J1, v = se3_interpolate_jacobians(P0, P1, lam)
        eps = 1e-6
        for J, (Ba, Bb) in ((J0, (P0, P1)), (J1, (P1, P0))):
            for k in range(6):
                d = np.zeros(6); d[k] = eps
                Tp = (se3_interpolate(Ba.retract(d), Bb, lam) if J is J0
                      else se3_interpolate(P0, Ba.retract(d), lam))
                np.testing.assert_allclose(T.localCoordinates(Tp) / eps, J[:, k], atol=1e-5)
    assert se3_interpolate(P0, P1, 0.0) is P0 and se3_interpolate(P0, P1, 1.0) is P1


def test_forward_step_jacobians_match_the_old_feed_imu_single_blocks():
    intr = ImuModel(x_dw=np.array([1.01, 0.002, 0.99, -0.001, 0.003, 1.02]),
                    x_da=np.array([0.98, 0.001, 1.01, 0.002, -0.002, 1.0]),
                    x_tg=np.full(9, 1e-3), R=gtsam.Rot3.Expmap([0.01, -0.02, 0.03]).matrix(),
                    model=RPNG)
    p = _IntrPack(intr)
    Rw, Ra, Dw, Da, Tg = intr.R_wtoI, intr.R_atoI, intr.Dw(), intr.Da(), intr.Tg()
    H_b, H_in = step_input_jacobians(p, 1, np.array([0.1, 9.8, 0.2]), np.array([0.3, -0.1, 0.2]),
                                     np.array([0.3, -0.1, 0.2]), np.array([0.1, 9.8, 0.2]))
    old = np.zeros((6, 6))
    old[0:3, 0:3] = -Rw @ Dw
    old[0:3, 3:6] = Rw @ Dw @ Tg @ Ra @ Da
    old[3:6, 3:6] = -Ra @ Da
    np.testing.assert_allclose(H_b, old, rtol=0, atol=1e-15)
    assert H_in.shape == (6, 24) and step_input_jacobians(p, 1, *[np.ones(3)] * 4, False)[1] is None


def _pose(rv, t):
    T = np.eye(4); T[:3, :3] = gtsam.Rot3.Expmap(np.asarray(rv, float)).matrix()
    T[:3, 3] = t
    return T


def test_one_dt_rule_skips_duplicate_stamps_everywhere():
    poses = [_pose([0, 0, 0.1], [0, 0, 1]), _pose([0, 0.2, 0.1], [0, 0, 1.1]),
             _pose([0.3, 0.2, 0.1], [0, 0, 1.2]), _pose([0.3, 0.2, 0.4], [0, 0, 1.3]),
             _pose([0.1, 0.5, 0.4], [0.1, 0, 1.3])]
    times = [0.0, 0.1, 0.1, 0.2, 0.3]           # pair 1->2 shares a stamp
    s = motion_series(valid_board_poses(poses, times))
    assert [t for t, *_ in s] == [0.1, 0.2, 0.3]
    m = compute_motion_metrics(poses, times)
    assert len(m._axis_vectors) == 3
    np.testing.assert_allclose(m.rot_angle_max,
                               max(np.degrees(np.linalg.norm(r)) for _t, _dt, r, _w in s))
