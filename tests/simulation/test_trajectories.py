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
"""sine_3d loader: load the in-repo trajectory, recenter + shrink to a tight
workspace, preserving the yaw sweep (rotation untouched)."""

import numpy as np

from mvis.simulation.trajectories import load_sine3d


def test_shrunk_to_one_metre_span():
    traj = load_sine3d(target_span=1.0, window=(0.0, 70.0))
    t_lo = traj.start_time
    ts = np.linspace(t_lo, t_lo + 70.0, 400)
    pos = np.array([traj.get_pose(t)[1] for t in ts])
    span = (pos.max(0) - pos.min(0)).max()
    assert abs(span - 1.0) < 0.10, f"span {span:.2f} m not ~1 m"
    assert np.linalg.norm(pos, axis=1).max() < 0.8, "workspace not centred/tight"


def test_two_turns_preserved():
    # scaling translation must NOT touch rotation: camera still faces all 4 boards twice
    traj = load_sine3d(target_span=1.0, window=(0.0, 70.0))
    ts = np.linspace(traj.start_time, traj.start_time + 70.0, 500)
    yaw = np.unwrap([np.arctan2(traj.get_pose(t)[0].T[1, 0],
                                traj.get_pose(t)[0].T[0, 0]) for t in ts])
    assert yaw.ptp() > 4 * np.pi - 1.0, f"need ~2 turns, got {yaw.ptp():.1f} rad"


def test_sine3d_is_packaged():
    """sine_3d must load from inside the package so a pip-installed mvis has it."""
    import os
    from mvis.simulation.trajectories import PACKAGED_DIR, trajectory_path
    assert trajectory_path("sine_3d") == os.path.join(PACKAGED_DIR, "sine_3d.txt")
    assert os.path.exists(trajectory_path("sine_3d"))
