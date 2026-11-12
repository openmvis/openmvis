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
"""Tests for detection-quality gating filters."""

from collections import Counter

import numpy as np
import pytest

from mvis.detection.frame_quality import (
    filter_ref_frames_by_pooled_tags,
    filter_observations_by_min_views,
    filter_observations_by_min_views_per_camera,
    filter_frames_by_min_tags,
    filter_frames_by_tag_spread,
    log_tag_coverage,
    _tag_ids,
)
from mvis.core.types import FrameObservations, Observation


def _frame(cam_id: int, ts: float, point_ids: list[int]) -> FrameObservations:
    obs = [Observation(point_id=p, pixel=np.array([float(p), float(p)]))
           for p in point_ids]
    return FrameObservations(camera_id=cam_id, timestamp=ts,
                             observations=obs, image_size=(640, 480))


# ---------------------------------------------------------------------------
# _tag_ids
# ---------------------------------------------------------------------------

def test_tag_ids_apriltag_groups_by_4():
    obs = [Observation(p, np.zeros(2)) for p in [0, 1, 2, 3, 4, 5, 6, 7, 12]]
    assert _tag_ids(obs, corners_per_tag=4) == {0, 1, 3}


def test_tag_ids_chessboard_no_grouping():
    obs = [Observation(p, np.zeros(2)) for p in range(7)]
    assert _tag_ids(obs, corners_per_tag=1) == set(range(7))


# ---------------------------------------------------------------------------
# Pooled-pose filter
# ---------------------------------------------------------------------------

def test_pose_filter_pools_across_cameras():
    """Sparse ref frame survives if synced non-ref camera has many tags."""
    # 36-tag board, threshold = max(4, ceil(0.20 × 36)) = 8 tags pooled.
    # Ref frame at t=1.0 has only 5 tags (corners 0..19).
    # Non-ref frame at t=1.001 has 10 different tags (corners 80..119).
    # Pool: |{0..4} ∪ {20..29}| = 15 tags ≥ 8 → KEEP.
    sparse_ref = _frame(0, 1.0, list(range(20)))           # 5 tags
    fat_other = _frame(1, 1.001, list(range(80, 120)))     # 10 tags (80..119)
    out, stats = filter_ref_frames_by_pooled_tags(
        {0: [sparse_ref], 1: [fat_other]},
        reference_camera=0, num_expected_tags=36, corners_per_tag=4,
    )
    assert stats.n_ref_kept == 1
    assert stats.n_ref_dropped == 0
    assert stats.threshold_tags == 8


def test_pose_filter_drops_isolated_sparse_ref():
    """If the ref frame is sparse AND no synced non-ref helps, drop it."""
    sparse = _frame(0, 1.0, list(range(12)))                # 3 tags
    far_other = _frame(1, 99.0, list(range(120)))           # 30 tags but far away
    out, stats = filter_ref_frames_by_pooled_tags(
        {0: [sparse], 1: [far_other]},
        reference_camera=0, num_expected_tags=36, corners_per_tag=4,
        sync_tolerance_s=0.01,
    )
    assert stats.n_ref_dropped == 1
    assert stats.dropped_pooled_counts == Counter({3: 1})
    assert out[0] == []
    assert out[1] == [far_other]   # non-ref unchanged


def test_pose_filter_keeps_dense_ref():
    """Dense ref frame is always kept regardless of non-ref."""
    dense = _frame(0, 0.5, list(range(80)))     # 20 tags
    out, stats = filter_ref_frames_by_pooled_tags(
        {0: [dense]},
        reference_camera=0, num_expected_tags=36, corners_per_tag=4,
    )
    assert stats.n_ref_kept == 1
    assert stats.n_ref_dropped == 0


def test_pose_filter_auto_sync_tolerance():
    """sync_tolerance_s=None → half the median ref-frame interval."""
    ref_frames = [_frame(0, t, list(range(40))) for t in [0.0, 0.1, 0.2, 0.3]]
    out, stats = filter_ref_frames_by_pooled_tags(
        {0: ref_frames}, reference_camera=0,
        num_expected_tags=36, corners_per_tag=4,
    )
    # median diff = 0.1 s, half = 0.05 s
    assert stats.sync_tolerance_s == pytest.approx(0.05)


def test_pose_filter_floor_overrides_low_fraction():
    """Small board: ceil(0.20 × 10) = 2; floor 4 wins → threshold = 4."""
    # 5 tags but on a small (10-tag) board — should always pass floor.
    f = _frame(0, 1.0, list(range(20)))   # 5 tags
    out, stats = filter_ref_frames_by_pooled_tags(
        {0: [f]}, reference_camera=0, num_expected_tags=10,
        corners_per_tag=4,
    )
    assert stats.threshold_tags == 4
    assert stats.n_ref_kept == 1


def test_pose_filter_no_ref_camera_passes_through():
    f1 = _frame(1, 1.0, list(range(40)))
    out, stats = filter_ref_frames_by_pooled_tags(
        {1: [f1]}, reference_camera=0, num_expected_tags=36,
    )
    # ref camera (id=0) not present → no-op
    assert out == {1: [f1]}
    assert stats.n_ref_in == 0


# ---------------------------------------------------------------------------
# Per-corner observability
# ---------------------------------------------------------------------------

def test_corner_filter_drops_corner_with_too_few_views():
    frames = {
        0: [_frame(0, t, [0, 99]) for t in range(2)],
        1: [_frame(1, t, [0]) for t in range(5)],
    }
    out, stats = filter_observations_by_min_views(
        frames, min_views=5, min_points_for_pose=1,
    )
    assert stats.n_corners_dropped == 1
    assert stats.drop_view_counts == Counter({2: 1})


def test_corner_filter_pools_views_across_cameras():
    """5 cameras each seeing corner 7 once → 5 views total."""
    frames = {cid: [_frame(cid, 0.0, [7])] for cid in range(5)}
    out, stats = filter_observations_by_min_views(
        frames, min_views=5, min_points_for_pose=1,
    )
    assert stats.n_corners_dropped == 0
    for cid in range(5):
        assert len(out[cid]) == 1


def test_corner_filter_drops_frames_below_min_pose_points():
    f1 = _frame(0, 1.0, [10, 11, 12, 13, 7])
    f2 = _frame(0, 2.0, [7, 8])
    f3 = _frame(0, 3.0, [7, 8])
    out, stats = filter_observations_by_min_views(
        {0: [f1, f2, f3]}, min_views=2, min_points_for_pose=4,
    )
    assert stats.n_corners_dropped == 4
    assert len(out[0]) == 0
    assert stats.n_frames_out == 0


# ---------------------------------------------------------------------------
# Per-camera corner observability filter
# ---------------------------------------------------------------------------

def test_per_camera_corner_filter_independent():
    """Corner C observed 5× by cam0 but 1× by cam1 — cam1 loses it."""
    # cam0 sees corner 100 in 5 frames
    cam0_frames = [_frame(0, float(i), [100, 101, 102, 103])
                   for i in range(5)]
    # cam1 sees corner 100 in only 1 frame, corner 200 in 5
    cam1_frames = [_frame(1, float(i), [200, 201, 202, 203])
                   for i in range(5)]
    cam1_frames[0] = _frame(1, 0.0, [100, 200, 201, 202, 203])

    out = filter_observations_by_min_views_per_camera(
        {0: cam0_frames, 1: cam1_frames},
        min_views=3, min_points_for_pose=1)
    # cam0: all corners have 5 views → all kept
    assert len(out[0]) == 5
    for f in out[0]:
        assert 100 in {o.point_id for o in f.observations}

    # cam1: corner 100 has only 1 view → dropped from cam1
    for f in out[1]:
        assert 100 not in {o.point_id for o in f.observations}


def test_per_camera_corner_filter_drops_frames():
    """If all corners in a frame are low-view, the frame is dropped."""
    # cam0: corner 50 appears only once, all other frames use corners 60-63
    frames = [_frame(0, 0.0, [50])]  # only corner in frame
    frames += [_frame(0, float(i), [60, 61, 62, 63]) for i in range(1, 5)]
    out = filter_observations_by_min_views_per_camera(
        {0: frames}, min_views=3, min_points_for_pose=1)
    # frame 0 had only corner 50 (1 view) → dropped
    assert len(out[0]) == 4


# ---------------------------------------------------------------------------
# Per-frame minimum tag count filter
# ---------------------------------------------------------------------------

def test_min_tags_filter_apriltag():
    """Frames with <6 unique tags (corners_per_tag=4) are dropped."""
    # 1 tag = 4 corners (point_ids 0-3)
    f1 = _frame(0, 1.0, [0, 1, 2, 3])  # 1 tag
    # 3 tags = 12 corners
    f2 = _frame(0, 2.0, list(range(12)))  # 3 tags
    # 5 tags = 20 corners
    f3 = _frame(0, 3.0, list(range(20)))  # 5 tags
    # 6 tags = 24 corners
    f4 = _frame(0, 4.0, list(range(24)))  # 6 tags
    # 8 tags = 32 corners
    f5 = _frame(0, 5.0, list(range(32)))  # 8 tags

    kept = filter_frames_by_min_tags(
        [f1, f2, f3, f4, f5], min_tags=6, corners_per_tag=4)
    assert len(kept) == 2
    assert kept[0].timestamp == 4.0
    assert kept[1].timestamp == 5.0


def test_min_tags_filter_top_left():
    """With top_left (corners_per_tag=1), each obs is a unique tag."""
    f1 = _frame(0, 1.0, [3, 7])      # 2 tags
    f2 = _frame(0, 2.0, [3, 7, 11, 15, 19])  # 5 tags
    f3 = _frame(0, 3.0, [3, 7, 11, 15, 19, 23])  # 6 tags
    kept = filter_frames_by_min_tags(
        [f1, f2, f3], min_tags=6, corners_per_tag=1)
    assert len(kept) == 1
    assert kept[0].timestamp == 3.0


def test_min_tags_filter_keeps_all_when_sufficient():
    frames = [_frame(0, float(i), list(range(24))) for i in range(5)]
    kept = filter_frames_by_min_tags(
        frames, min_tags=4, corners_per_tag=4)
    assert len(kept) == 5


# ---------------------------------------------------------------------------
# Tag coverage logging
# ---------------------------------------------------------------------------

def test_tag_coverage_logs(caplog):
    """log_tag_coverage logs the expected format."""
    import logging
    frames = [_frame(0, float(i), list(range(24))) for i in range(10)]
    with caplog.at_level(logging.INFO):
        log_tag_coverage(frames, camera_id=0, num_expected_tags=36,
                         corners_per_tag=4)
    assert "6/36 tags" in caplog.text
    assert "views/tag" in caplog.text


def test_tag_coverage_empty_frames(caplog):
    """log_tag_coverage handles empty frame lists."""
    import logging
    with caplog.at_level(logging.INFO):
        log_tag_coverage([], camera_id=0, num_expected_tags=36,
                         corners_per_tag=4)
    assert "no tags observed" in caplog.text


def test_tag_coverage_flags_sparse_tags(caplog):
    """Tags with <3 views are flagged in the log."""
    import logging
    # 6 tags (0-5), tag 5 observed only once, tag 4 observed twice
    frames = [_frame(0, float(i), list(range(20))) for i in range(5)]  # tags 0-4, 5 times
    frames.append(_frame(0, 5.0, [20, 21, 22, 23]))  # tag 5, once
    frames.append(_frame(0, 6.0, [16, 17, 18, 19, 20, 21, 22, 23]))  # tag 4+5
    with caplog.at_level(logging.INFO):
        log_tag_coverage(frames, camera_id=0, num_expected_tags=36,
                         corners_per_tag=4)
    assert "tags with <3 views" in caplog.text


# ---------------------------------------------------------------------------
# Per-camera corner filter — additional tests
# ---------------------------------------------------------------------------

def test_per_camera_corner_filter_no_drop_when_all_sufficient():
    """All corners have enough views per camera — no drops."""
    frames = {0: [_frame(0, float(i), [10, 11, 12, 13]) for i in range(5)]}
    out = filter_observations_by_min_views_per_camera(
        frames, min_views=3, min_points_for_pose=1)
    assert len(out[0]) == 5
    total_obs = sum(len(f.observations) for f in out[0])
    assert total_obs == 20  # 4 corners × 5 frames, unchanged


def test_per_camera_corner_filter_respects_min_points_for_pose():
    """Frames falling below min_points_for_pose after filter are dropped."""
    # cam0: corners 10,11,12 in 5 frames, corner 99 only in 2 frames
    frames = [_frame(0, float(i), [10, 11, 12, 99]) for i in range(2)]
    frames += [_frame(0, float(i), [10, 11, 12]) for i in range(2, 5)]
    # After filter (min_views=3): corner 99 has 2 views → dropped
    # Frames 0-1 had [10,11,12,99] → [10,11,12] (3 points, >= min_points=3)
    out = filter_observations_by_min_views_per_camera(
        {0: frames}, min_views=3, min_points_for_pose=3)
    assert len(out[0]) == 5  # all frames still have ≥3 points
    for f in out[0]:
        assert 99 not in {o.point_id for o in f.observations}

    # With higher min_points_for_pose=4: all frames drop to ≤3 obs → all dropped
    out2 = filter_observations_by_min_views_per_camera(
        {0: frames}, min_views=3, min_points_for_pose=4)
    assert len(out2[0]) == 0  # all frames have ≤3 obs after filter < 4


def test_per_camera_corner_filter_multi_camera():
    """Multiple cameras filtered independently."""
    # cam0: corner 5 in 10 frames
    cam0 = [_frame(0, float(i), [5, 6, 7]) for i in range(10)]
    # cam1: corner 5 in only 1 frame
    cam1 = [_frame(1, 0.0, [5, 8, 9])]
    cam1 += [_frame(1, float(i), [8, 9]) for i in range(1, 10)]
    out = filter_observations_by_min_views_per_camera(
        {0: cam0, 1: cam1}, min_views=3, min_points_for_pose=1)
    # cam0: corner 5 has 10 views → kept
    assert all(5 in {o.point_id for o in f.observations} for f in out[0])
    # cam1: corner 5 has 1 view → dropped
    assert all(5 not in {o.point_id for o in f.observations} for f in out[1])


# ---------------------------------------------------------------------------
# Per-frame minimum tag filter — additional tests
# ---------------------------------------------------------------------------

def test_min_tags_filter_empty_input():
    """Empty frame list returns empty."""
    kept = filter_frames_by_min_tags([], min_tags=6, corners_per_tag=4)
    assert kept == []


def test_min_tags_filter_boundary():
    """Frame with exactly min_tags is kept."""
    f = _frame(0, 1.0, list(range(24)))  # 6 tags
    kept = filter_frames_by_min_tags([f], min_tags=6, corners_per_tag=4)
    assert len(kept) == 1


def test_min_tags_filter_mixed_corners_per_tag():
    """Works with corners_per_tag=2 (two_diag mode)."""
    # point_ids 0,1 → tag 0; 2,3 → tag 1; etc.
    f1 = _frame(0, 1.0, [0, 1, 2, 3, 4, 5])  # 3 tags
    f2 = _frame(0, 2.0, list(range(12)))       # 6 tags
    kept = filter_frames_by_min_tags(
        [f1, f2], min_tags=4, corners_per_tag=2)
    assert len(kept) == 1
    assert kept[0].timestamp == 2.0


# ---------------------------------------------------------------------------
# Tag spread filter (convex hull)
# ---------------------------------------------------------------------------

def _frame_with_pixels(cam_id, ts, pixels, image_size=(848, 800)):
    """Create a frame with explicit pixel positions."""
    obs = [Observation(point_id=i, pixel=np.array(px)) for i, px in enumerate(pixels)]
    return FrameObservations(camera_id=cam_id, timestamp=ts,
                             observations=obs, image_size=image_size)


def test_tag_spread_drops_clustered():
    """Tags clustered in a tiny region → dropped."""
    # 15 tags all within 100×50px of an 848×800 image
    # Hull area ≈ 5000px² / (848*800) = 0.74% < 1%
    pixels = [(400 + i * 6, 400 + (i % 3) * 15) for i in range(15)]
    f = _frame_with_pixels(0, 1.0, pixels)
    kept = filter_frames_by_tag_spread([f], min_hull_fraction=0.01,
                                        corners_per_tag=1)
    assert len(kept) == 0


def test_tag_spread_keeps_spread():
    """Tags spread across the image → kept."""
    # 15 tags spread across 600×600px of 848×800
    pixels = [(100 + (i % 5) * 150, 100 + (i // 5) * 200) for i in range(15)]
    f = _frame_with_pixels(0, 1.0, pixels)
    kept = filter_frames_by_tag_spread([f], min_hull_fraction=0.01,
                                        corners_per_tag=1)
    assert len(kept) == 1


def test_tag_spread_few_tags_pass_through():
    """Frames with < 3 tags can't compute hull → pass through."""
    pixels = [(100, 100), (200, 200)]
    f = _frame_with_pixels(0, 1.0, pixels)
    kept = filter_frames_by_tag_spread([f], min_hull_fraction=0.01,
                                        corners_per_tag=1)
    assert len(kept) == 1  # not dropped, let min_tags handle it


def test_tag_spread_empty_frame():
    """Empty frame passes through."""
    f = FrameObservations(camera_id=0, timestamp=1.0, observations=[],
                          image_size=(640, 480))
    kept = filter_frames_by_tag_spread([f], min_hull_fraction=0.01,
                                        corners_per_tag=1)
    assert len(kept) == 1


def test_tag_spread_mixed():
    """Mix of spread and clustered frames — only clustered dropped."""
    # Frame 1: clustered (hull < 1%)
    pixels1 = [(400 + i * 5, 400 + i * 3) for i in range(10)]
    f1 = _frame_with_pixels(0, 1.0, pixels1, image_size=(848, 800))
    # Frame 2: spread (hull > 5%)
    pixels2 = [(100 + (i % 4) * 200, 100 + (i // 4) * 200) for i in range(12)]
    f2 = _frame_with_pixels(0, 2.0, pixels2, image_size=(848, 800))
    kept = filter_frames_by_tag_spread([f1, f2], min_hull_fraction=0.01,
                                        corners_per_tag=1)
    assert len(kept) == 1
    assert kept[0].timestamp == 2.0


def test_tag_spread_with_corners_per_tag():
    """With corners_per_tag=4, tag centers are averaged from 4 corners."""
    # 6 tags in a 2D grid, each with 4 corners.
    obs = []
    for tid in range(6):
        cx = 100 + (tid % 3) * 250  # tag centers at x=100, 350, 600
        cy = 150 + (tid // 3) * 300  # y=150, 450
        for corner in range(4):
            dx, dy = (corner % 2) * 10, (corner // 2) * 10
            obs.append(Observation(point_id=tid * 4 + corner,
                                   pixel=np.array([cx + dx, cy + dy])))
    f = FrameObservations(camera_id=0, timestamp=1.0, observations=obs,
                          image_size=(848, 800))
    kept = filter_frames_by_tag_spread([f], min_hull_fraction=0.01,
                                        corners_per_tag=4)
    # 6 tag centers in 2D grid → hull covers ~22% of image
    assert len(kept) == 1
