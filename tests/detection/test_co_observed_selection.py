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
"""Frame selection must be co-observation aware for synchronized rigs.

Regression for the T265 / Monado MGC joint-solve tail (2026-10-17): the
diversity filter ran independently per camera, so ~5-18 % of the non-ref
frames it kept had NO ref frame at the same instant. The joint solve has pose
nodes only on the ref camera and interpolated those frames between ref poses
>= 400 ms apart (no time offset estimated) -> 20-35 px residual tails.

With no time offset estimated, a kept non-ref frame must have a kept ref
frame within half a source frame period. With td estimated, the historical
behaviour (ref keeps every frame, non-ref diversity-filtered) is unchanged.
"""
import numpy as np

from mvis.core.types import FrameObservations, Observation
from mvis.io.detection_pipeline import DetectionPipeline
from mvis.io.frame_selector import subsample_by_detection_diversity

PERIOD = 1.0 / 30.0
SIZE = (848, 800)


def _frame(cid, t, cx, cy, s):
    obs = [Observation(point_id=i, pixel=np.array([cx + dx * s, cy + dy * s]))
           for i, (dx, dy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)])]
    return FrameObservations(camera_id=cid, timestamp=t, observations=obs,
                             image_size=SIZE)


def _rig(n=1800, offset=0.0025, drop_ref=()):
    """Two synced cameras that see the same board motion slightly
    differently (different lens position), so independent diversity picks
    diverge."""
    ts = np.arange(n) * PERIOD
    ref, other = [], []
    for k, t in enumerate(ts):
        if k not in drop_ref:
            ref.append(_frame(2, t, 400 + 250 * np.sin(0.4 * t),
                              400 + 150 * np.cos(0.3 * t), 60 + 20 * np.sin(t)))
        other.append(_frame(3, t + offset,
                            380 + 250 * np.sin(0.4 * t) + 30 * np.sin(2.1 * t),
                            410 + 150 * np.cos(0.3 * t),
                            55 + 25 * np.sin(1.3 * t)))
    return {2: ref, 3: other}


def _select(frames, enable_td=False):
    dp = DetectionPipeline(board=None, thumb_dir=None, cache_path=None)
    res = dp._subsample_and_return(
        frames, {2: SIZE, 3: SIZE}, {}, {}, ref_cam=2,
        enable_td=enable_td, target_hz=5.0)
    return res.all_frames


def _unpaired(sel, tol=PERIOD / 2):
    t2 = np.array([f.timestamp for f in sel[2]])
    return [f.timestamp for f in sel[3]
            if np.min(np.abs(t2 - f.timestamp)) > tol]


def test_every_kept_nonref_frame_has_a_ref_frame_at_the_same_instant():
    sel = _select(_rig())
    assert len(sel[3]) > 100
    assert _unpaired(sel) == []


def test_ref_gaps_do_not_strand_nonref_frames():
    # ref misses detections on some frames (board partly out of view)
    sel = _select(_rig(drop_ref=set(range(300, 1800, 7))))
    assert _unpaired(sel) == []


def test_td_estimated_path_unchanged():
    frames = _rig()
    sel = _select(frames, enable_td=True)
    from mvis.io.frame_selector import subsample_frames_by_hz
    rate = subsample_frames_by_hz(frames, 5.0)
    assert [f.timestamp for f in sel[2]] == [f.timestamp for f in rate[2]]
    idx, _ = subsample_by_detection_diversity(
        [(None, f.timestamp) for f in rate[3]],
        [f.observations for f in rate[3]], min_time_gap=0.05,
        image_size=SIZE)
    assert [f.timestamp for f in sel[3]] == [rate[3][i].timestamp for i in idx]


def test_unsynchronized_cameras_fall_back_to_independent_selection():
    # stamps half a frame apart: no co-instants exist; must not empty cam3
    sel = _select(_rig(offset=PERIOD / 2))
    assert len(sel[3]) > 100


# ---- pipeline stage gates: a ref frame dropped by a per-camera gate must
# take its same-instant non-ref frames with it (no td) ----

def _pipe(optimize_time_offset=False):
    from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
    cfg = PipelineConfig(optimize_time_offset=optimize_time_offset,
                         reference_camera=2)
    return CalibrationPipeline(board=None, config=cfg)


def _gate_case():
    ts = np.arange(20) * 0.2
    ref = [_frame(2, t, 400, 400, 60) for k, t in enumerate(ts) if k != 7]
    other = [_frame(3, t + 0.0025, 400, 400, 60) for t in ts]
    poses = {2: [object() for _ in ref], 3: [object() for _ in other]}
    return {2: ref, 3: other}, poses


def test_drop_unpaired_nonref_without_td():
    frames, poses = _gate_case()
    f2, p2 = _pipe()._drop_unpaired_nonref(frames, poses, ref_cam=2,
                                            stage="stage4_entry")
    assert len(f2[2]) == 19 and len(f2[3]) == 19
    assert all(abs(f.timestamp - 1.4025) > 1e-6 for f in f2[3])
    assert len(p2[3]) == len(f2[3]) and len(p2[2]) == len(f2[2])


def test_drop_unpaired_nonref_noop_with_td():
    frames, poses = _gate_case()
    f2, p2 = _pipe(True)._drop_unpaired_nonref(frames, poses, ref_cam=2,
                                                stage="stage4_entry")
    assert len(f2[3]) == 20 and p2 is poses


def test_ref_frame_without_pose_is_not_a_pose_node():
    frames, poses = _gate_case()
    poses[2][3] = None            # PnP-rejected ref frame (kept for indexing)
    f2, _ = _pipe()._drop_unpaired_nonref(frames, poses, ref_cam=2,
                                           stage="stage4_entry")
    assert len(f2[3]) == 18
