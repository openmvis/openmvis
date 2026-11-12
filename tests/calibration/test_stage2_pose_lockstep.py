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
"""Stage 2 keeps board poses in lockstep with the frame list (2c-9).

When the stage-2 subsample (`stage2_target_hz`) thins the frames, the
per-camera solve must start from each frame's own pose (by timestamp) and
write poses back for every frame of all_frames, not only the subsample.
Found on mvis_4imus_3cams: the Blackfly reference camera (radtan, 15 Hz,
diversity skipped) crashed with "628 poses vs 1726 frames".
"""
import logging

import numpy as np
import pytest

from mvis.calibration.pipeline_cam import (
    CalibrationPipeline, PipelineConfig, _poses_by_identity,
)
from mvis.detection.chessboard import Chessboard

from tests.helpers.cam_frames import _IMG, _K, _gen_frames


class _F:
    def __init__(self, t):
        self.timestamp = t


def test_poses_by_identity_follows_timestamps_not_positions():
    full = [_F(t) for t in (0.0, 0.1, 0.2, 0.3)]
    poses = ["p0", "p1", "p2", "p3"]
    sub = [full[3], full[1]]
    assert _poses_by_identity(sub, full, poses, None) == ["p3", "p1"]
    assert _poses_by_identity(full, full, poses, None) == poses
    # a frame missing from full_frames falls back to pose_by_ts, else None
    extra = [_F(0.5), _F(0.9)]
    assert _poses_by_identity(extra, full, poses, {0.5: "q"}) == ["q", None]


def test_stage2_subsample_keeps_lockstep():
    board = Chessboard(7, 7, 0.05)
    frames = _gen_frames(board, n=40, seed=3)          # 10 Hz
    pc = PipelineConfig(
        max_iterations=10, max_iterations_single=10, pixel_sigma=1.0,
        huber_threshold=1.345, corner_subsample="top_left",
        optimize_intrinsics=True, optimize_time_offset=True, max_init_frames=100,
        init_frame_selection="uniform", reference_camera=None,
        rerun_after_outlier_rejection=False, distortion_models={0: "radtan"},
        stage2_target_hz=2.0, verbose=False)            # thin 10 Hz to ~2 Hz
    pipe = CalibrationPipeline(board, pc)
    result = pipe.run({0: frames}, {0: _IMG})           # asserted lockstep inside
    # single camera: no later joint stage, so the answer is Stage 2 on the
    # ~8 subsampled views; sane, not exact
    cam = result.cameras[0]
    assert abs(cam.fx / _K[0, 0] - 1) < 0.08 and abs(cam.fy / _K[1, 1] - 1) < 0.08
    assert abs(cam.cx - _K[0, 2]) < 10 and abs(cam.cy - _K[1, 2]) < 10


def _subsampled_pipeline():
    board = Chessboard(7, 7, 0.05)
    frames = _gen_frames(board, n=40, seed=3)          # 10 Hz
    pc = PipelineConfig(
        max_iterations=10, max_iterations_single=10, pixel_sigma=1.0,
        huber_threshold=1.345, corner_subsample="top_left",
        optimize_intrinsics=True, optimize_time_offset=True, max_init_frames=100,
        init_frame_selection="uniform", reference_camera=None,
        rerun_after_outlier_rejection=False, distortion_models={0: "radtan"},
        stage2_target_hz=2.0, verbose=False)
    return CalibrationPipeline(board, pc), frames


def test_diverged_gate_uses_full_track_length(monkeypatch, caplog):
    """The 10 % removal gate counts the full pose track, not the subsample.

    3 diverged poses on a ~40-frame track (~8 Stage-2 frames at 2 Hz) are
    under 10 % of the track, so they are removed and the camera is
    re-optimized; against the subsample (3 >= 0.8) they would be kept.
    """
    from mvis.calibration import pipeline_cam

    pipe, frames = _subsampled_pipeline()
    flagged = {5, 6, 7}
    calls = []

    def fake_detect(poses, min_z=0.05, max_z=20.0):
        calls.append(len(poses))
        return set(flagged) if len(calls) == 1 else set()

    monkeypatch.setattr(pipeline_cam, "_detect_diverged_poses_from_state",
                        fake_detect)
    seen = {}
    real_diff = type(pipe.ledger).diff_one

    def spy_diff(self, cid, before, after, stage, reason):
        if stage == "stage2_diverged":
            seen.setdefault(reason, set()).update(
                {f.timestamp for f in before} - {f.timestamp for f in after})
        return real_diff(self, cid, before, after, stage, reason)

    monkeypatch.setattr(type(pipe.ledger), "diff_one", spy_diff)
    with caplog.at_level(logging.WARNING, logger="mvis.calibration.pipeline_cam"):
        result = pipe.run({0: frames}, {0: _IMG})
    # The detector sees the full track (40 minus quality-gated frames), not
    # the ~8-frame Stage-2 subsample.
    assert calls and calls[0] > 30
    assert any("removing and reoptimizing" in r.getMessage()
               for r in caplog.records)
    assert len(seen.get("pose_diverged", ())) == len(flagged)
    assert result is not None


def test_stage2_merge_rejects_result_without_valid_frame_indices(monkeypatch):
    """Poses with no frame identity abort Stage 2 instead of being guessed.

    A positional fallback (non-None init poses, in order) misplaces every
    pose after a skipped low-point frame, so the merge-back fails loudly;
    the raise is outside the per-camera try/except on purpose.
    """
    from mvis.calibration.optimizer import Optimizer

    real_solve = Optimizer.solve_single_camera

    def solve_without_identity(self, *args, **kwargs):
        result = real_solve(self, *args, **kwargs)
        result.valid_frame_indices = []
        return result

    monkeypatch.setattr(Optimizer, "solve_single_camera",
                        solve_without_identity)
    pipe, frames = _subsampled_pipeline()
    with pytest.raises(ValueError, match="valid_frame_indices"):
        pipe.run({0: frames}, {0: _IMG})
