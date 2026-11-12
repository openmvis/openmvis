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
"""#26: the omni_radtan xi-grid score and the `mvis cam` omni schedule.

* The grid maps board points with the pose solvePnP returns (T_cam_board, so
  transformFrom). Before #26 it used transformTo, scored every xi on inverted
  poses (hundreds of px at every xi) and so picked a grid end, usually 2.5.
* `mvis calibrate` (cli.main.calibrate; `mvis cam` before G8) gives each omni camera the
  pipeline's Stage 2 schedule (omni_schedule: P1 xi under a 1.0 prior with the
  radtan part held, P2 xi held; the P1/P2 call order and options are tested
  in tests/calibration/test_solve_phased.py). Before, the cli solved omni in one go from the
  grid seed; with the corrected score the grid picks xi = 1 on a 180-deg lens (fits at
  2.15-2.28) and that one-shot solve stalls.

Single-camera xi stays at the P1 value unless pipeline Stage 2's diverged-pose
reopt runs (that solve has xi free). This is deliberate: with corners below
about 80 deg single-camera xi is weakly pinned. A final free-xi stage was tried
and dropped (owner, 2026-11-05): on T265 cam1/07 a free xi slid from 1.919 to
1.635 against Kalibr's 1.900, with fx 835 -> 754 and k1 0.170 -> 0.003 moving
with it, at a 0.0025 px lower RMS.
"""
import logging
import re
from types import SimpleNamespace

import numpy as np

from tests.camera.test_omni_p2 import _IMG, _omni_stereo_frames

from mvis.calibration import omni_schedule
from mvis.detection.chessboard import Chessboard


def _cal(xi, fx):
    return np.array([fx, fx + 2, 322.0, 238.0, xi, -0.05, 0.01, 1e-3, 2e-3])


def test_grid_score_uses_board_to_camera_pose(caplog):
    """With the correct map the grid's best mean error is a few px (0.05 px
    data, radtan part at zero) and it picks the true xi when that is on the
    grid; on inverted poses it was > 100 px at every xi."""
    from mvis.calibration.initializer import Initializer
    board = Chessboard(7, 9, 0.04)
    frames = _omni_stereo_frames(board, _cal(1.0, 300.0))[0]
    with caplog.at_level(logging.INFO, logger="mvis.calibration.initializer"):
        intr, dist = Initializer(board).init_intrinsics_opencv(frames, _IMG, "omni_radtan")
    errs = [float(m) for m in re.findall(r"Omni grid: xi=\S+ fx=\S+ err=([\d.]+)px", caplog.text)]
    assert len(errs) == 8 and min(errs) < 10.0, errs
    assert float(dist[0]) == 1.0, dist


def test_cli_single_camera_omni_uses_the_schedule(monkeypatch):
    """`mvis cam` on one omni camera runs P1 + P2 (not one solve from the grid
    seed), keeps the camera's own id, and returns a converged fit."""
    import mvis.cli.main as cli
    board = Chessboard(7, 9, 0.04)
    frames = _omni_stereo_frames(board, _cal(2.2, 600.0), seed=0)[0]
    for f in frames:
        f.camera_id = 4
    out = {}
    monkeypatch.setattr(cli, "create_board", lambda b: board)
    monkeypatch.setattr(cli, "_detect_all_frames", lambda c, b: ({4: frames}, {4: _IMG}))
    monkeypatch.setattr(cli, "print_report", lambda r: "")
    monkeypatch.setattr(cli.OutputWriter, "write",
                        staticmethod(lambda r, *a, **k: out.setdefault("r", r)))
    calls = []
    real = omni_schedule.solve_phased
    monkeypatch.setattr(omni_schedule, "solve_phased",
                        lambda *a, **k: calls.append((a[5], a[-1])) or real(*a, **k))
    cfg = SimpleNamespace(
        board=None, reference_camera=4, corner_subsample="all",
        output_file="unused.yaml", output_format="yaml",
        cameras=[SimpleNamespace(camera_id=4, distortion_model="omni_radtan",
                                 shutter_type="global")],
        optimizer=SimpleNamespace(
            max_init_frames=0, init_frame_selection="uniform", pnp_subprocess=False,
            optimize_rolling_shutter=False, max_iterations=30, pixel_sigma=1.0,
            huber_threshold=1.345, absolute_error_tol=1e-8, relative_error_tol=1e-5))
    cli.calibrate(cfg)
    assert calls == [("omni_radtan", 4)]
    r = out["r"]
    assert r.cameras[0].camera_id == 4
    assert r.overall_rms < 0.08, r.overall_rms
    assert abs(float(r.cameras[0].intrinsics[2]) - 322.0) < 0.3
