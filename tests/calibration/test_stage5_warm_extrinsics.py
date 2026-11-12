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
"""Stage 5 must warm-start the extrinsics in the convention the optimizer
consumes (T_cam_ref; Optimizer.solve inserts ext.inverse() as E = T_ref_cam).

Regression (2026-10-17): the rerun loaded the pass-1 result's T_to_from
(= E = T_ref_cam) as-is, so the optimizer started pass 2 from the INVERSE
extrinsic. Monado MGC (20 deg, 108 mm): cam1 initial median 405 px, pass 2
never recovered in 5 iterations and was reverted on every round; T265
(~0 deg, 64 mm): initial cost 958k vs pass-1 final 4.6k.
"""
from types import SimpleNamespace

import gtsam
import numpy as np

from mvis.calibration.pipeline_cam import _warm_start_extrinsics


def test_warm_start_extrinsic_round_trips_through_optimizer_convention():
    E = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.33, -0.004, 0.10),
                    gtsam.Point3(0.005, -0.107, -0.018))   # T_ref_cam
    res = SimpleNamespace(extrinsics=[SimpleNamespace(
        from_camera=1, to_camera=0, T_to_from=E.matrix())])
    warm = _warm_start_extrinsics(res)
    # the optimizer inserts warm[1].inverse() as its E variable
    assert np.allclose(warm[1].inverse().matrix(), E.matrix(), atol=1e-12)
