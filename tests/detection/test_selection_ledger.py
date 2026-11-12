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
"""I3 step 2c-1: the SelectionLedger — drops recorded at surgery time, folded
into MVISDatabase flags at attach."""
from __future__ import annotations

import numpy as np

from mvis.calibration.selection_ledger import SelectionLedger
from mvis.core.types import FrameObservations
from mvis.mvisdb.mvisdatabase import CalibChoice
from tests.helpers.mvisdb_fixtures import make_db


def _fo(cid, ts):
    return FrameObservations(camera_id=cid, timestamp=ts, observations=[])


def test_diff_records_removed_frames_only():
    led = SelectionLedger()
    before = {0: [_fo(0, 1.0), _fo(0, 2.0), _fo(0, 3.0)], 1: [_fo(1, 1.0)]}
    after = {0: [_fo(0, 1.0), _fo(0, 3.0)], 1: [_fo(1, 1.0)]}
    n = led.diff(before, after, "stage1_gate", "quality_gate")
    assert n == 1
    assert led.drops == {(0, 2.0): "stage1_gate:quality_gate"}


def test_first_drop_wins():
    led = SelectionLedger()
    led.record(0, 2.0, "stage1_gate", "quality_gate")
    led.record(0, 2.0, "stage5_outlier", "residual_outlier")
    assert led.drops[(0, 2.0)] == "stage1_gate:quality_gate"


def test_camera_vanishing_entirely_is_recorded():
    led = SelectionLedger()
    n = led.diff({0: [_fo(0, 1.0)]}, {}, "stage2_diverged", "pose_diverged")
    assert n == 1 and (0, 1.0) in led.drops


def test_apply_to_db_flags_and_detection_thinning():
    db = make_db(CalibChoice.CAM_ONLY)
    s = db.camera(0)
    ts = [f.timestamp for f in s.frames]
    assert len(ts) >= 3
    led = SelectionLedger()
    led.record(0, ts[0], "stage1_gate", "quality_gate")
    # solve started from frames [0] and [1] only -> frame [2] was thinned
    solve = {0: [_fo(0, ts[0]), _fo(0, ts[1])]}
    n = led.apply_to_db(db, solve_frames=solve)
    f0, f1, f2 = s.frames[:3]
    assert not f0.is_selected_to_use and f0.exclude_reason == "stage1_gate:quality_gate"
    assert f1.is_selected_to_use and f1.exclude_reason is None
    assert not f2.is_selected_to_use and f2.exclude_reason == "detection:hz_subsample"
    other = sum(len(st.frames) for st in db.streams if st.camera_id != 0)
    assert n == 2 + sum(1 for f in s.frames[3:]) + other  # absent cam = thinned


def test_apply_without_solve_frames_flags_ledger_only():
    db = make_db(CalibChoice.CAM_ONLY)
    s = db.camera(0)
    led = SelectionLedger()
    led.record(0, s.frames[0].timestamp, "driver", "hz_subsample")
    assert led.apply_to_db(db) == 1
    assert not s.frames[0].is_selected_to_use
    assert all(f.is_selected_to_use for f in s.frames[1:])


def test_apply_keeps_upstream_exclusion_reasons():
    """2c-3: a carried db already holds detection:diversity — apply_to_db must
    not overwrite it with the generic hz label."""
    db = make_db(CalibChoice.CAM_ONLY)
    s = db.camera(0)
    ts = [f.timestamp for f in s.frames]
    s.frames[0].is_selected_to_use = False
    s.frames[0].exclude_reason = "detection:diversity"
    led = SelectionLedger()
    n = led.apply_to_db(db, solve_frames={0: [_fo(0, t) for t in ts[1:]]})
    assert s.frames[0].exclude_reason == "detection:diversity"
    assert all(f.is_selected_to_use for f in s.frames[1:])


def test_mark_frames_direct_flag_write():
    """2c-4: mark_frames writes flags at the moment of selection; upstream
    exclusions and unknown cameras are untouched."""
    from mvis.calibration.selection_ledger import mark_frames
    db = make_db(CalibChoice.CAM_ONLY)
    s = db.camera(0)
    ts = [f.timestamp for f in s.frames]
    s.frames[1].is_selected_to_use = False
    s.frames[1].exclude_reason = "detection:diversity"
    n = mark_frames(db, {0: {ts[0]}}, "driver", "hz_subsample")
    assert n == len(ts) - 2                       # ts[0] kept, ts[1] upstream
    assert s.frames[0].is_selected_to_use
    assert s.frames[1].exclude_reason == "detection:diversity"
    for f in s.frames[2:]:
        assert not f.is_selected_to_use
        assert f.exclude_reason == "driver:hz_subsample"
    for st in db.streams:                         # absent cams untouched
        if st.camera_id != 0:
            assert all(f.is_selected_to_use for f in st.frames)


def test_write_through_flags_immediately():
    """2c-5: a ledger built with the db flags the frame at record() time;
    upstream reasons still win; apply_to_db afterwards adds nothing new."""
    db = make_db(CalibChoice.CAM_ONLY)
    s = db.camera(0)
    ts = [f.timestamp for f in s.frames]
    s.frames[1].is_selected_to_use = False
    s.frames[1].exclude_reason = "detection:diversity"
    led = SelectionLedger(db=db)
    led.record(0, ts[0], "stage1_gate", "quality_gate")
    led.record(0, ts[1], "stage1_gate", "quality_gate")   # upstream wins
    assert not s.frames[0].is_selected_to_use
    assert s.frames[0].exclude_reason == "stage1_gate:quality_gate"
    assert s.frames[1].exclude_reason == "detection:diversity"
    n = led.apply_to_db(db)                               # idempotent closer
    assert n == 0


def test_record_obs_write_through_d71():
    """D71/2c-6: obs-level drop sets is_valid=False with the condemning
    residual; upstream-invalid obs untouched; first drop wins."""
    from mvis.mvisdb.mvisdatabase import PoseSource
    db = make_db(CalibChoice.CAM_ONLY)
    s = db.camera(0)
    f0 = s.frames[0]
    dets = [d for g in f0.board_detections for d in g.detections]
    assert len(dets) >= 2
    led = SelectionLedger(db=db)
    led.record_obs(0, f0.timestamp, dets[0].point_id,
                   "stage5_outlier", "residual_outlier", residual_px=17.3)
    assert not dets[0].is_valid
    assert dets[0].exclude_reason == "stage5_outlier:residual_outlier"
    assert dets[0].residual.residual_px == 17.3
    assert dets[0].residual.residual_source is PoseSource.REFINED_CAM_ONLY
    assert dets[1].is_valid
    # first drop wins on re-record
    led.record_obs(0, f0.timestamp, dets[0].point_id, "other", "x",
                   residual_px=99.0)
    assert dets[0].residual.residual_px == 17.3


def test_unmark_reverts_a_stage_completely():
    """I4 primitive: unmark(stage) restores frames AND observations that
    stage excluded (first-drop-wins guarantees no earlier reason existed),
    leaving other stages' marks untouched."""
    db = make_db(CalibChoice.CAM_ONLY)
    s = db.camera(0)
    ts = [f.timestamp for f in s.frames]
    led = SelectionLedger(db=db)
    led.record(0, ts[0], "stage2_diverged", "pose_diverged")
    led.record(0, ts[1], "stage5_outlier", "residual_outlier")
    d0 = s.frames[0].board_detections[0].detections[0]
    led.record_obs(0, ts[0], d0.point_id, "stage5_outlier",
                   "residual_outlier", residual_px=9.9)
    assert not s.frames[1].is_selected_to_use and not d0.is_valid
    n = led.unmark("stage5_")
    assert n == 2
    assert s.frames[1].is_selected_to_use            # stage-5 frame restored
    assert s.frames[1].exclude_reason is None
    assert d0.is_valid and d0.residual is None       # stage-5 obs restored
    assert not s.frames[0].is_selected_to_use        # stage-2 mark untouched
    assert (0, ts[1]) not in led.drops
    assert (0, ts[0]) in led.drops
