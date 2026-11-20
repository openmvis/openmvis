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
"""A camera-only solve that runs out of iterations far from its minimum is rescued.

On the Monado Reverb G2 cameras the vanishing-point focal length starts up to 2x too large, and
three cam0 solves were still descending at the 15-iteration limit (MGC05 cam0 460.6 px against
271.1 px). The synthetic fisheye here converges from a 2x seed within 15 iterations, so the tests
force the stop with a small limit."""
import logging

import cv2
import numpy as np
import pytest

from mvis.calibration import optimizer as O
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.core.types import FrameObservations, Observation
from mvis.detection.chessboard import Chessboard

IMG = (640, 480)
F = 270.0
K = np.array([[F, 0, 322.0], [0, F, 229.0], [0, 0, 1]])
D = np.array([0.08, -0.04, 0.23, -0.13])     # equidistant (KB4), as fitted on a Reverb G2 camera


def _scene(n=30, noise=0.3):
    board = Chessboard(rows=7, cols=6, square_size=0.04)
    obj = board.all_points_3d().astype(np.float64)
    rng = np.random.RandomState(1)
    frames = []
    while len(frames) < n:
        rvec = rng.uniform(-0.7, 0.7, 3)
        tvec = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2), rng.uniform(0.35, 0.8)]) - obj.mean(0)
        uv = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)[0].reshape(-1, 2)
        depth = (cv2.Rodrigues(rvec)[0] @ obj.T).T[:, 2] + tvec[2]
        if np.all(depth > 0.05) and np.all((uv > 5) & (uv < np.array(IMG) - 5)):
            uv = uv + rng.randn(*uv.shape) * noise
            frames.append(FrameObservations(
                camera_id=0, timestamp=float(len(frames)), image_size=IMG,
                observations=[Observation(point_id=i, pixel=uv[i], row=int(uv[i, 1])) for i in range(len(uv))]))
    return board, frames


def _result(seed_scale, max_iterations):
    board, frames = _scene()
    intr = np.array([F * seed_scale, F * seed_scale, (IMG[0] - 1) / 2, (IMG[1] - 1) / 2])
    dist = np.zeros(4)
    poses = Initializer(board).init_board_poses(frames, intr, dist, "equidistant")
    return Optimizer(board, OptimizerOptions(max_iterations=max_iterations)).solve_single_camera(
        frames, intr, dist, poses, "equidistant", IMG)


def _solve(seed_scale, max_iterations):
    r = _result(seed_scale, max_iterations)
    return r.cameras[0], r.reprojection_errors[0].rms


def _no_rescue(monkeypatch):
    monkeypatch.setattr(O, "rescue_if_stuck", lambda graph, start, values, opt, *a, **k: (values, opt, None))


def test_solve_cut_off_far_from_the_minimum_is_rescued(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="mvis.calibration.lm_rescue"):
        cam, rms = _solve(2.0, max_iterations=5)
    assert "ran out of iterations" in caplog.text
    _no_rescue(monkeypatch)
    cut_cam, cut_rms = _solve(2.0, max_iterations=5)
    assert abs(cam.intrinsics[0] - F) < 0.5 * abs(cut_cam.intrinsics[0] - F)
    assert rms < cut_rms


def test_converged_solve_is_bitwise_unchanged(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="mvis.calibration.lm_rescue"):
        cam, rms = _solve(1.1, max_iterations=15)
    assert "rescuing" not in caplog.text
    assert cam.intrinsics[0] == pytest.approx(F, rel=0.01)
    _no_rescue(monkeypatch)
    ref_cam, ref_rms = _solve(1.1, max_iterations=15)
    np.testing.assert_array_equal(cam.intrinsics, ref_cam.intrinsics)
    np.testing.assert_array_equal(cam.distortion, ref_cam.distortion)
    assert rms == ref_rms


def test_rescue_report_rides_on_the_result():
    """The report reaches the result (and so the result YAML); a converged solve has none."""
    rescued = _result(2.0, max_iterations=5).lm_rescue
    assert rescued is not None and rescued["kept"] and rescued["err_rescued"] < rescued["err_before"]
    # five iterations per stage do not reach the minimum either, so the result is flagged
    assert rescued["failed"] == (rescued["ratio_after"] >= 0.5)
    assert _result(1.1, max_iterations=15).lm_rescue is None


def test_verdict_grades_a_still_stuck_result_red(tmp_path):
    from mvis.analysis.verdict import assess_cam
    import yaml
    cam = {"id": 0, "intrinsics": {"fx": 270.0, "fy": 270.0, "cx": 320.0, "cy": 240.0},
           "distortion": {"coeffs": [0.0, 0.0, 0.0, 0.0]}, "image_size": [640, 480]}
    errs = {"per_camera": [{"camera": 0, "median": 0.2, "p95": 0.5, "rms": 0.3, "num_observations": 5000}]}
    for failed, grade in ((True, "RED"), (False, None)):
        f = tmp_path / f"r{failed}.yaml"
        f.write_text(yaml.safe_dump({"cameras": [cam], "reprojection_errors": errs,
                                     "lm_rescue": {"failed": failed, "kept": True, "ratio_after": 0.66 if failed else 0.0}}))
        v = assess_cam(str(f))
        assert ("solve.lm_rescue" in [c.id for c in v.checks]) == failed
        if grade:
            assert v.grade == grade
