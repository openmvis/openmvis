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
"""Tests for the unified quality gate module."""

import numpy as np
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.detection.quality_gate import (
    QualityGateConfig,
    QualityGateResult,
    PerFrameStat,
    apply_quality_gates,
)
from mvis.core.types import FrameObservations, Observation


def _frame(cam_id, ts, point_ids):
    obs = [Observation(point_id=p, pixel=np.array([float(p) * 10, float(p) * 10]))
           for p in point_ids]
    return FrameObservations(camera_id=cam_id, timestamp=ts,
                             observations=obs, image_size=(640, 480))


@pytest.fixture
def board():
    return Chessboard(rows=7, cols=5, square_size=0.03)


@pytest.fixture
def basic_intrinsics():
    return np.array([500.0, 500.0, 320.0, 240.0])


@pytest.fixture
def basic_dist():
    return np.zeros(4)


class TestQualityGateNoOp:
    """All gates disabled — should be a near no-op (only PnP reinit)."""

    def test_no_gates_preserves_frames(self, board, basic_intrinsics, basic_dist):
        frames = {0: [_frame(0, float(i), list(range(35))) for i in range(10)]}
        cfg = QualityGateConfig(reinit_board_poses=False)
        out_frames, out_poses, result = apply_quality_gates(
            frames, board, {0: basic_intrinsics}, {0: basic_dist},
            {0: "radtan"}, cfg)
        assert len(out_frames[0]) == 10
        assert result.frames_before[0] == 10
        assert result.frames_after[0] == 10


class TestMinTagsGate:
    """Per-frame minimum tag count filter."""

    def test_drops_sparse_frames(self, board):
        # Chessboard: corners_per_tag=1, so tags = points
        frames = {0: [
            _frame(0, 0.0, [0, 1]),        # 2 tags
            _frame(0, 1.0, list(range(10))), # 10 tags
            _frame(0, 2.0, list(range(20))), # 20 tags
        ]}
        cfg = QualityGateConfig(
            min_tags_per_frame=6, reinit_board_poses=False)
        out, _, result = apply_quality_gates(
            frames, board, {}, {}, {}, cfg)
        assert len(out[0]) == 2  # only 10-tag and 20-tag frames kept
        assert result.min_tag_dropped[0] == 1


class TestPnPReinitGate:
    """PnP reinitialisation with reprojection error gate."""

    def test_reinit_produces_poses(self, board, basic_intrinsics, basic_dist):
        """Good synthetic data should produce valid poses."""
        # Generate frames with chessboard corners at plausible pixel locations
        from tests.helpers.cam_frames import _generate_synthetic_observations
        frames_list, _ = _generate_synthetic_observations(
            board, np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            np.array([-0.1, 0.01, 0.001, -0.001, 0.0]),
            num_frames=5, noise_sigma=0.3)
        frames = {0: frames_list}

        cfg = QualityGateConfig(
            reinit_board_poses=True,
            max_pnp_reprojection_px=10.0)
        _, poses, result = apply_quality_gates(
            frames, board, {0: basic_intrinsics}, {0: basic_dist},
            {0: "radtan"}, cfg)
        n_valid = sum(1 for p in poses[0] if p is not None)
        assert n_valid == 5  # all should succeed with good data

    def test_tight_threshold_rejects_poses(self, board, basic_intrinsics, basic_dist):
        """Very tight reprojection threshold should reject some poses."""
        from tests.helpers.cam_frames import _generate_synthetic_observations
        frames_list, _ = _generate_synthetic_observations(
            board, np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            np.array([-0.1, 0.01, 0.001, -0.001, 0.0]),
            num_frames=5, noise_sigma=0.5)
        frames = {0: frames_list}

        cfg = QualityGateConfig(
            reinit_board_poses=True,
            max_pnp_reprojection_px=0.01)  # impossibly tight
        _, poses, result = apply_quality_gates(
            frames, board, {0: basic_intrinsics}, {0: basic_dist},
            {0: "radtan"}, cfg)
        n_valid = sum(1 for p in poses[0] if p is not None)
        assert n_valid < 5  # some should be rejected


class TestPerFrameStats:
    """Per-frame stats for HTML report plotting."""

    def test_per_frame_stats_populated(self, board):
        frames = {0: [_frame(0, float(i), list(range(20))) for i in range(5)]}
        cfg = QualityGateConfig(reinit_board_poses=False)
        _, _, result = apply_quality_gates(
            frames, board, {}, {}, {}, cfg)
        assert 0 in result.per_frame_stats
        assert len(result.per_frame_stats[0]) == 5
        for ps in result.per_frame_stats[0]:
            assert ps.tag_count == 20  # chessboard: 1 corner = 1 tag
            assert ps.obs_count == 20


class TestQualityGateResult:
    """Result logging."""

    def test_log_summary(self, caplog, board):
        import logging
        frames = {0: [_frame(0, float(i), list(range(20))) for i in range(5)]}
        cfg = QualityGateConfig(min_tags_per_frame=10, reinit_board_poses=False)
        with caplog.at_level(logging.INFO):
            _, _, result = apply_quality_gates(
                frames, board, {}, {}, {}, cfg)
        assert "Quality gate cam 0" in caplog.text


class TestStage5Config:
    """Stage 5 uses looser thresholds than Stage 1→2."""

    def test_stage5_looser_min_tags(self, board):
        """Stage 5 config allows 4 tags, Stage 1→2 requires 6."""
        frames = {0: [_frame(0, float(i), list(range(5))) for i in range(10)]}
        # Stage 1→2: 6 tags required → all dropped (only 5 tags per frame)
        cfg_s12 = QualityGateConfig(min_tags_per_frame=6, reinit_board_poses=False)
        out_s12, _, _ = apply_quality_gates(frames, board, {}, {}, {}, cfg_s12)
        assert len(out_s12[0]) == 0

        # Stage 5: 4 tags → all kept
        frames2 = {0: [_frame(0, float(i), list(range(5))) for i in range(10)]}
        cfg_s5 = QualityGateConfig(min_tags_per_frame=4, reinit_board_poses=False)
        out_s5, _, _ = apply_quality_gates(frames2, board, {}, {}, {}, cfg_s5)
        assert len(out_s5[0]) == 10
