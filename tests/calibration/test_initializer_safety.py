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
"""The spawn-based PnP worker (``pnp_subprocess: true``) exchanges its pickles
through a private temp directory of its own, removed afterwards, not through a
predictable sibling of a NamedTemporaryFile in the shared temp dir (F124)."""
from __future__ import annotations

import multiprocessing
import os
import tempfile

import numpy as np

from mvis.detection.chessboard import Chessboard
from mvis.calibration import initializer as init_mod
from mvis.calibration.initializer import Initializer
from tests.helpers.cam_frames import _generate_synthetic_observations


class _InlineProcess:
    """Runs the worker in this process (no spawn, no GTSAM re-import), exit code 0."""
    def __init__(self, target, args):
        self._target, self._args = target, args
        self.exitcode = None

    def start(self):
        self._target(*self._args)
        self.exitcode = 0

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class _CrashedProcess(_InlineProcess):
    def start(self):
        self.exitcode = -11            # a segfaulted child, nothing written


def _run(monkeypatch, process_cls, n_frames=4):
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    frames, _ = _generate_synthetic_observations(board, K, dist, num_frames=n_frames,
                                                 noise_sigma=0.1)
    made = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(*a, **kw):
        d = real_mkdtemp(*a, **kw)
        made.append(d)
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", recording_mkdtemp)
    monkeypatch.setattr(multiprocessing, "get_context",
                        lambda method: type("Ctx", (), {"Process": staticmethod(
                            lambda target, args: process_cls(target, args))}))
    poses = Initializer(board).init_board_poses(
        frames, np.array([500.0, 500.0, 320.0, 240.0]), dist, "radtan",
        use_subprocess=True)
    return poses, made, frames


def test_worker_files_live_in_a_private_dir_that_is_removed(monkeypatch):
    poses, made, frames = _run(monkeypatch, _InlineProcess)
    assert len(made) == 1 and os.path.basename(made[0]).startswith("mvis_pnp_")
    assert not os.path.exists(made[0])
    assert len(poses) == len(frames)
    assert sum(p is not None for p in poses) == len(frames)


def test_crashed_worker_still_removes_the_dir(monkeypatch):
    poses, made, frames = _run(monkeypatch, _CrashedProcess)
    assert not os.path.exists(made[0])
    assert poses == [None] * len(frames)


def test_worker_only_touches_its_own_dir(tmp_path):
    """The child writes exactly the output path it is given (inside the private dir)."""
    import pickle
    board = Chessboard(rows=7, cols=5, square_size=0.03)
    inp, out = str(tmp_path / "pnp_in.pkl"), str(tmp_path / "pnp_out.pkl")
    with open(inp, "wb") as f:
        pickle.dump({"board_pts": {}, "frame_data": [], "intrinsics": [500, 500, 320, 240],
                     "dist_coeffs": [0, 0, 0, 0], "distortion_model": "radtan",
                     "pnp_method": "iterative",
                     "min_points_for_pose": board.min_points_for_pose(),
                     "max_reprojection_px": 0.0}, f)
    init_mod._pnp_worker(inp, out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pnp_in.pkl", "pnp_out.pkl"]
    with open(out, "rb") as f:
        assert pickle.load(f) == []
