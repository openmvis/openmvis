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
"""G1: one two-phase schedule (omni_schedule.solve_phased) for double_sphere
and omni_radtan, used by pipeline Stage 2 and `mvis cam`.

Before G1 pipeline Stage 2 built the omni P2 options by hand from a subset of
the config and dropped board_heights / board_height_sigma (a board-shape solve
lost the board shape in P2), and `mvis cam` solved double_sphere in one go.
"""
import dataclasses
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

from mvis.calibration import omni_schedule
from mvis.calibration.factors.custom_projections import project_double_sphere
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.detection.chessboard import Chessboard

_IMG = (640, 480)


def _non_default_options():
    """OptimizerOptions with every field moved off its default, so a field the
    P2 options lose shows up as a difference."""
    return OptimizerOptions(
        max_iterations=7, pixel_sigma=0.7, huber_threshold=2.1,
        absolute_error_tol=3e-9, relative_error_tol=4e-6,
        lm_lambda_lower=1e-9, lm_lambda_upper=5e3, optimize_intrinsics=False,
        optimize_time_offset=True, dt_prior_sigma=0.25, rs_cameras={3: 0.02},
        fixed_rs_cameras={5: 0.03}, frozen_intrinsics_cameras={6},
        frozen_xi_cameras={8}, readout_prior_sigma=0.02, readout_sigma=False,
        board_heights=True, board_height_sigma=0.002,
        snap_sync_tolerance_s=0.004, verbose=True)


def test_non_default_options_cover_every_field():
    d, o = OptimizerOptions(), _non_default_options()
    same = [f.name for f in dataclasses.fields(OptimizerOptions)
            if getattr(d, f.name) == getattr(o, f.name)]
    assert same == [], same


def test_omni_phase2_options_differ_only_in_frozen_xi():
    p1 = _non_default_options()
    p2 = omni_schedule.phase2_options(p1, "omni_radtan", 3)
    assert p2.frozen_xi_cameras == {8, 3}
    assert p1.frozen_xi_cameras == {8}                    # P1 options untouched
    assert dataclasses.replace(p2, frozen_xi_cameras=p1.frozen_xi_cameras) == p1
    assert p2.board_heights is True and p2.board_height_sigma == 0.002


def test_ds_phase2_options_are_the_phase1_options():
    p1 = _non_default_options()
    assert omni_schedule.phase2_options(p1, "double_sphere", 3) is p1


def test_phase2_options_reject_other_models():
    with pytest.raises(ValueError):
        omni_schedule.phase2_options(OptimizerOptions(), "radtan", 0)


class _FakeOpt:
    """solve_single_camera stub recording (options, frozen_distortion,
    camera_id) per call; P1 moves xi to 1.5, P2 keeps what it is given."""
    log = []

    def __init__(self, board, options):
        self.board, self.options = board, options

    def solve_single_camera(self, frames, intr, dist, poses, dm, img, camera_id=0,
                            frozen_distortion=False):
        from mvis.core.types import CalibrationResult, CameraParams
        self.log.append((self.options, frozen_distortion, camera_id))
        d = np.array(dist, float)
        if frozen_distortion:
            d[0] = 1.5
        return CalibrationResult(cameras=[CameraParams(
            camera_id, np.array([300.0, 300, 320, 240]), d, dm, _IMG)])


@pytest.mark.parametrize("dm,dist", [("omni_radtan", [1.0, 0, 0, 0, 0]),
                                     ("double_sphere", [-0.2, 0.6])])
def test_solve_phased_runs_p2_on_the_p1_options(dm, dist):
    _FakeOpt.log = []
    p1 = _non_default_options()
    r = omni_schedule.solve_phased(_FakeOpt("board", p1), [0, 1], np.zeros(4),
                                   np.array(dist), [None, None], dm, _IMG, 3)
    (o1, fz1, c1), (o2, fz2, c2) = _FakeOpt.log
    assert (fz1, c1, fz2, c2) == (True, 3, False, 3)
    assert o1 is p1
    assert o2 == omni_schedule.phase2_options(p1, dm, 3)
    assert o2.board_heights is True
    assert float(r.cameras[0].distortion[0]) == 1.5       # warm start from P1


# ── real solves ──

def _ds_frames(board, cal, n=30, seed=0, sigma=0.05):
    from mvis.core.types import FrameObservations, Observation
    pts = board.all_points_3d().astype(float)
    rng = np.random.RandomState(seed)
    frames, t, att = [], 0.0, 0
    while len(frames) < n and att < n * 50:
        att += 1
        T = gtsam.Pose3(gtsam.Rot3.Rodrigues(*rng.uniform(-0.6, 0.6, 3)),
                        gtsam.Point3(rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2),
                                     rng.uniform(0.3, 0.55)))
        obs = []
        for k, p in enumerate(pts):
            uv = project_double_sphere(np.asarray(T.transformFrom(gtsam.Point3(*p))), cal)[0]
            if uv is not None and 5 <= uv[0] < _IMG[0] - 5 and 5 <= uv[1] < _IMG[1] - 5:
                obs.append(Observation(point_id=k, timestamp=t,
                                       pixel=uv + rng.standard_normal(2) * sigma))
        if len(obs) < 0.8 * len(pts):
            continue
        frames.append(FrameObservations(camera_id=0, timestamp=t, observations=obs,
                                        image_size=_IMG))
        t += 0.1
    return frames


_DS_CAL = np.array([190.0, 191.0, 322.0, 238.0, -0.18, 0.59])


def _spy_solves(monkeypatch):
    calls = []
    real = Optimizer.solve_single_camera

    def spy(self, *a, **k):
        r = real(self, *a, **k)
        calls.append(SimpleNamespace(options=self.options,
                                     frozen=k.get("frozen_distortion", False),
                                     camera_id=k.get("camera_id"), result=r))
        return r
    monkeypatch.setattr(Optimizer, "solve_single_camera", spy)
    return calls


def test_pipeline_stage2_omni_keeps_board_heights_in_phase2(monkeypatch):
    """A board-shape omni solve estimates the board heights in Stage 2 P2
    (before G1 the P2 options had board_heights False)."""
    from tests.camera.test_omni_p2 import _omni_frames
    from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
    board = Chessboard(7, 9, 0.04)
    calls = _spy_solves(monkeypatch)
    pc = PipelineConfig(
        max_iterations=10, max_iterations_single=10, corner_subsample="all",
        reference_camera=None, rerun_after_outlier_rejection=False,
        distortion_models={0: "omni_radtan"}, board_heights=True,
        board_height_sigma=0.002, verbose=False)
    CalibrationPipeline(board, pc).run({0: _omni_frames(board)}, {0: _IMG})
    p1, p2 = calls[0], calls[1]                     # Stage 2 P1, P2
    assert (p1.frozen, p2.frozen) == (True, False)
    assert p2.options.frozen_xi_cameras == {0}
    assert dataclasses.replace(p2.options, frozen_xi_cameras=None) == p1.options
    assert p2.options.board_heights and p2.options.board_height_sigma == 0.002
    assert p2.result.board_heights is not None


def test_cli_single_camera_ds_uses_the_schedule(monkeypatch):
    """`mvis cam` on one double_sphere camera runs P1 (distortion frozen) then
    P2 (free) on the same options, keeps the camera id, and fits."""
    import mvis.cli.main as cli
    board = Chessboard(7, 9, 0.04)
    frames = _ds_frames(board, _DS_CAL)
    for f in frames:
        f.camera_id = 4
    out = {}
    monkeypatch.setattr(cli, "create_board", lambda b: board)
    monkeypatch.setattr(cli, "_detect_all_frames", lambda c, b: ({4: frames}, {4: _IMG}))
    monkeypatch.setattr(cli, "print_report", lambda r: "")
    monkeypatch.setattr(cli.OutputWriter, "write",
                        staticmethod(lambda r, *a, **k: out.setdefault("r", r)))
    calls = _spy_solves(monkeypatch)
    cfg = SimpleNamespace(
        board=None, reference_camera=4, corner_subsample="all",
        output_file="unused.yaml", output_format="yaml",
        cameras=[SimpleNamespace(camera_id=4, distortion_model="double_sphere",
                                 shutter_type="global")],
        optimizer=SimpleNamespace(
            max_init_frames=0, init_frame_selection="uniform", pnp_subprocess=False,
            optimize_rolling_shutter=False, max_iterations=30, pixel_sigma=1.0,
            huber_threshold=1.345, absolute_error_tol=1e-8, relative_error_tol=1e-5,
            optimize_board_points=True))
    cli.calibrate(cfg)
    assert [(c.frozen, c.camera_id) for c in calls] == [(True, 4), (False, 4)]
    assert calls[0].options is calls[1].options and calls[1].options.board_heights
    r = out["r"]
    assert r is calls[1].result and r.cameras[0].camera_id == 4
    assert r.overall_rms < 0.08, r.overall_rms
    assert abs(float(r.cameras[0].intrinsics[0]) - _DS_CAL[0]) < 1.0
