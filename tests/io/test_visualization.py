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
"""Smoke test for the scenario visualization (headless Agg)."""

import os

import numpy as np
import gtsam

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.scenario import make_scenario
from mvis.simulation.visualization import plot_scenario

_TRAJ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tum_room_segment.txt")


def _scenario():
    tr = load_trajectory(_TRAJ, dt=0.05)
    cal = _cal_from_vector(
        np.array([400.0, 400.0, 320.0, 240.0, -0.28, 0.07, 0.0005, -0.0004]), False)
    T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                         gtsam.Point3(0.05, -0.02, 0.01))
    intr = ImuModel(x_dw=np.ones(6), x_da=np.ones(6), x_tg=np.zeros(9),
                         R=np.eye(3), model=RPNG)
    return make_scenario(tr, t0=tr.start_time + 3.0, depth=3.0,
                         board=Chessboard(7, 7, 0.10), cal=cal, T_CtoI=T_CtoI,
                         imu_intr=intr, b_g=np.zeros(3), b_a=np.zeros(3),
                         image_size=(640, 480))


def test_plot_scenario_renders(tmp_path):
    out = tmp_path / "scenario.png"
    fig = plot_scenario(_scenario(), save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert len(fig.axes) == 2          # 3-D view + camera image
    import matplotlib.pyplot as plt
    plt.close(fig)
