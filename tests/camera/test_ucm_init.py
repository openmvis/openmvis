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
"""UCM init + schedule (U3): an unseeded camera-only solve on a wide synthetic
UCM rig converges with the default single solve (no two-phase schedule).

Rig: fx 380, alpha 0.64 (~124 deg fold, ~195 deg diagonal FOV at 1024^2),
0.3 px corner noise; boards spread out to ~80 deg off-axis. The seed is the
initializer's VP f0 and the registry default alpha (0.6).
"""
import gtsam
import numpy as np
import pytest

from mvis.calibration.factors.custom_projections import project_ucm
from mvis.core import camera_models as cm

CAL = np.array([380.0, 381.0, 515.0, 508.0, 0.64])
IMG = (1024, 1024)


def _rot_z_to(d):
    """Rotation taking +z onto unit vector d."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    s, c = np.linalg.norm(v), float(z @ d)
    if s < 1e-12:
        return np.eye(3)
    return gtsam.Rot3.Rodrigues(*(v / s * np.arctan2(s, c))).matrix()


def _frames(board, cal=CAL, n=40, seed=0, sigma=0.3, th_max=80.0):
    from mvis.core.types import FrameObservations, Observation
    rng = np.random.default_rng(seed)
    pts = board.all_points_3d().astype(float)
    ctr = pts.mean(0)
    frames, t, att = [], 0.0, 0
    while len(frames) < n and att < n * 50:
        att += 1
        th = np.radians(th_max) * np.sqrt(rng.uniform())      # area-uniform in angle
        ph = rng.uniform(-np.pi, np.pi)
        d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        R = _rot_z_to(d) @ gtsam.Rot3.Rodrigues(*rng.uniform(-0.4, 0.4, 3)).matrix()
        p0 = d * rng.uniform(0.35, 0.7) - R @ ctr
        obs = []
        for i, p in enumerate(pts):
            uv = project_ucm(R @ p + p0, cal)[0]
            if uv is None or not (5 <= uv[0] < IMG[0] - 5 and 5 <= uv[1] < IMG[1] - 5):
                continue
            obs.append(Observation(point_id=i, timestamp=t,
                                   pixel=uv + rng.standard_normal(2) * sigma))
        if len(obs) < 0.6 * len(pts):
            continue
        frames.append(FrameObservations(camera_id=0, timestamp=t, observations=obs,
                                        image_size=IMG))
        t += 0.1
    return frames


def _solve(board, frames, model="ucm", seed_cal=None):
    from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
    from mvis.core.config import CameraConfig
    cc = ([] if seed_cal is None else
          [CameraConfig(camera_id=0, initial_intrinsics=list(seed_cal[:4]),
                        initial_distortion=list(seed_cal[4:]))])
    pc = PipelineConfig(
        camera_configs=cc, max_iterations=50, max_iterations_single=50, pixel_sigma=1.0,
        huber_threshold=1.345, corner_subsample="all", optimize_intrinsics=True,
        optimize_time_offset=False, max_init_frames=100, init_frame_selection="uniform",
        reference_camera=None, rerun_after_outlier_rejection=False,
        distortion_models={0: model}, verbose=False)
    return CalibrationPipeline(board, pc).run({0: frames}, {0: IMG})


def test_default_alpha_in_range():
    assert 0.5 < cm.default_dist("ucm")[0] < 0.75


def test_frames_cover_wide_field():
    """The rig really exercises the periphery (rays past 90 deg)."""
    from mvis.detection.chessboard import Chessboard
    from mvis.calibration.factors.custom_projections import unproject_ucm
    frames = _frames(Chessboard(8, 11, 0.05))
    th = [np.degrees(np.arccos(unproject_ucm(o.pixel, CAL)[2]))
          for f in frames for o in f.observations]
    assert max(th) > 90.0 and len(frames) == 40


@pytest.mark.parametrize("cal,seed", [(CAL, 0), (CAL, 1),
                                      (np.r_[330.0, 331.0, CAL[2:4], 0.5], 0),
                                      (np.r_[450.0, 451.0, CAL[2:4], 0.8], 1)])
def test_unseeded_single_solve_recovers_ucm(cal, seed):
    """Single solve from VP f0 + alpha 0.6, no two-phase schedule (U3 rule (i)):
    it reaches the same minimum as a solve seeded at the truth (so any
    remaining error is the noise draw, e.g. cx -0.85 px at alpha 0.5 seed 0).
    alpha 0.5 / 0.64 / 0.8 x 4 seeds all converged; four cases are kept."""
    from mvis.detection.chessboard import Chessboard
    board = Chessboard(8, 11, 0.05)
    frames = _frames(board, cal=cal, seed=seed)
    r, r_gt = _solve(board, frames), _solve(board, frames, seed_cal=cal)
    cam = r.cameras[0]
    assert cam.distortion_model == "ucm"
    assert not np.any(cam.distortion[1:])           # result pads to 4 (as ds/eucm/fov)
    est = np.r_[cam.intrinsics, cam.distortion[0]]
    est_gt = np.r_[r_gt.cameras[0].intrinsics, r_gt.cameras[0].distortion[0]]
    np.testing.assert_allclose(est, est_gt, atol=1e-3, rtol=0)   # converged
    assert np.all(np.abs(est[:4] - cal[:4]) < 1.0), est - cal    # accurate
    assert abs(est[4] - cal[4]) < 0.005, est[4]
    assert r.overall_rms < 0.45                      # noise floor 0.3 * sqrt(2) = 0.42
    assert cm.dist_sanity_issue("ucm", [est[4]]) is None
