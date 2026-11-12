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
"""Legacy `mvis calibrate` path (cli.main.calibrate) with a kb4 camera in a rig: no
pre-refinement, so the init poses and distortion reach the joint solve. The single-camera
path is in test_kb4.py. The heavy pieces (detection, init, optimizer) are stubbed; this
pins the wiring."""

from types import SimpleNamespace

import numpy as np

import mvis.cli.main as cli
from mvis.core.types import CalibrationResult, CameraParams

_IMG = (640, 480)
_N = 4                                           # frames per camera


def _config(models):
    return SimpleNamespace(
        board=None, reference_camera=0, corner_subsample="all",
        output_file="unused.yaml", output_format="yaml",
        cameras=[SimpleNamespace(camera_id=i, distortion_model=m, shutter_type="global")
                 for i, m in enumerate(models)],
        optimizer=SimpleNamespace(
            max_init_frames=0, init_frame_selection="uniform", pnp_subprocess=False,
            optimize_rolling_shutter=False, max_iterations=10, pixel_sigma=1.0,
            huber_threshold=1.0, absolute_error_tol=1e-9, relative_error_tol=1e-9))


class _Init:
    """Initializer stub: frame 1 of every camera has no init pose."""
    def __init__(self, board):
        pass

    def init_intrinsics_opencv(self, frames, img_size, dm):
        return np.array([300.0, 300.0, 320.0, 240.0]), np.zeros(4)

    def init_board_poses(self, frames, intr, dist, dm, **kw):
        return [("init", i) if i != 1 else None for i in range(len(frames))]

    def init_extrinsics(self, frames, poses, ref):
        return {}


def _result(cid, model, dist, poses):
    return CalibrationResult(
        cameras=[CameraParams(cid, np.array([301.0, 302.0, 321.0, 239.0]),
                              np.asarray(dist, float), model, _IMG)],
        board_poses_gtsam=list(poses))         # valid_frame_indices left [] (as the optimizer does)


def _patch(monkeypatch, calls):
    monkeypatch.setattr(cli, "create_board", lambda b: None)
    monkeypatch.setattr(cli, "Initializer", _Init)
    monkeypatch.setattr(cli, "_detect_all_frames",
                        lambda cfg, board: ({c.camera_id: list(range(_N)) for c in cfg.cameras},
                                            {c.camera_id: _IMG for c in cfg.cameras}))
    monkeypatch.setattr(cli, "print_report", lambda r: "")
    monkeypatch.setattr(cli.OutputWriter, "write", staticmethod(lambda *a, **k: None))

    class _Opt:
        """Optimizer stub: records each call."""
        def __init__(self, board, opts):
            self.opts = opts

        def solve_single_camera(self, frames, intr, dist, poses, dm, img, camera_id=0):
            calls.append(("single", camera_id))
            return _result(camera_id, dm, dist, [])

        def solve_multi_camera(self, frames, intr, dist, poses, extr, dms, ref, sizes):
            calls.append(("multi", {c: list(p) for c, p in poses.items()},
                          {c: np.asarray(d).copy() for c, d in dist.items()}))
            return CalibrationResult(cameras=[])
    monkeypatch.setattr(cli, "Optimizer", _Opt)


def test_cli_multi_kb4_rig_goes_to_the_joint_solve_directly(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)
    cli.calibrate(_config(["kb4", "radtan"]))
    (_, poses, dist), = calls                     # no single-camera solve before it
    for c in (0, 1):
        assert poses[c] == [("init", 0), None, ("init", 2), ("init", 3)]
        np.testing.assert_array_equal(dist[c], np.zeros(4))
