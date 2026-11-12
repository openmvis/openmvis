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
"""I2b step 1: the detection component (FeatureDetector hierarchy + sweep).

Contracts under test: verbatim emission (order + float64 bits) through the
wrappers; the sweep fills ONLY measurements, in time order, one frame at a
time; cross-path identity with the bridge (same inputs -> same fingerprint);
tag-mode detections are loud, never silently dropped.
"""
from __future__ import annotations

import numpy as np
import pytest

from mvis.core.types import Observation
from mvis.detection import BoardFeatureDetector, TagFeatureDetector, sweep
from mvis.mvisdb.bridge import board_from_geometry, db_from_frame_observations
from mvis.mvisdb.mvisdatabase import (
    BoardType, CalibChoice, CameraInfo, CameraStream, MVISDatabase, Rig, Run,
    observation_fingerprint,
)
from mvis.core.types import FrameObservations


class StubBoard:
    """Legacy-shaped target: geometry + detect(), emitting a SCRAMBLED order
    keyed off the image so the replay column is exercised for real."""
    _rows = 2; _cols = 2; _spacing = 0.03; _tag_size = 0.1; _start_id = 0

    def num_points(self):
        return 16

    def point_3d(self, pid):
        return np.array([0.01 * pid, 0.02 * pid, 0.0])

    def detect(self, image):
        base = float(image[0, 0])
        order = [5, 0, 9] if int(base) % 2 == 0 else [2, 7]
        return [Observation(point_id=p, pixel=np.array([base + p + 0.25, 2.0 * p + 0.5]),
                            row=int(2.0 * p + 0.5)) for p in order]


def make_source(n0=3, n1=2):
    """(image, t, cam) stream — interleaved cams, time-ordered per cam."""
    frames = []
    for i in range(n0):
        frames.append((np.full((8, 8), 2 * i, np.uint8), 100.0 + 0.1 * i, 0))
    for i in range(n1):
        frames.append((np.full((8, 8), 2 * i + 1, np.uint8), 100.05 + 0.1 * i, 1))
    return frames


def make_db():
    db = MVISDatabase(name="sweep-test", rig=Rig(), run=Run(CalibChoice.CAM_ONLY))
    db.boards[0] = board_from_geometry(StubBoard(), board_type=BoardType.APRILGRID)
    db.streams = [CameraStream(0, CameraInfo("cam0")), CameraStream(1, CameraInfo("cam1"))]
    return db


def test_board_detector_verbatim_emission():
    det = BoardFeatureDetector(StubBoard(), board_id=0)
    img = np.full((8, 8), 4, np.uint8)
    out = det.detect(img)
    assert [d.point_id for d in out] == [5, 0, 9]              # scrambled order kept
    assert [d.detector_emission_order for d in out] == [0, 1, 2]
    ref = StubBoard().detect(img)
    for a, b in zip(out, ref):
        assert a.uv.tobytes() == np.asarray(b.pixel, float).tobytes()
        assert a.row == b.row


def test_sweep_fills_measurements_only():
    db = make_db()
    n = sweep(make_source(), [BoardFeatureDetector(StubBoard(), 0)], db)
    assert n == 5
    assert len(db.camera(0).frames) == 3 and len(db.camera(1).frames) == 2
    f = db.camera(0).frames[0]
    g = f.board_detections[0]
    assert [d.point_id for d in g.detections] == [5, 0, 9]
    assert f.is_selected_to_use is True                         # no flags touched
    assert db.estimate_states is None                           # no estimates touched
    db.validate()                                               # time order holds


def test_sweep_matches_bridge_fingerprint():
    """The cross-path identity: sweeping a source == bridging the equivalent
    FrameObservations — one fingerprint, two roads."""
    db_a = make_db()
    sweep(make_source(), [BoardFeatureDetector(StubBoard(), 0)], db_a)

    frames = {0: [], 1: []}
    for img, t, cam in make_source():
        obs = StubBoard().detect(img)
        frames[cam].append(FrameObservations(camera_id=cam, timestamp=t,
                                             observations=obs, image_size=(8, 8)))
    db_b = db_from_frame_observations(frames, StubBoard(),
                                      board_type=BoardType.APRILGRID)
    assert observation_fingerprint(db_a) == observation_fingerprint(db_b)


def test_sweep_min_obs_skips_frames():
    db = make_db()
    # odd-pixel frames emit 2 observations, even emit 3 — threshold at 3
    n = sweep(make_source(), [BoardFeatureDetector(StubBoard(), 0)], db, min_obs=3)
    assert n == 3                                               # the two 2-obs frames skipped
    assert len(db.camera(1).frames) == 0


def test_tag_mode_is_loud_not_silent():
    class StubTagImpl:
        def detect(self, image):
            return [Observation(point_id=1, pixel=np.array([1.0, 2.0]), row=2)]
    tag = TagFeatureDetector(StubTagImpl())
    assert [d.point_id for d in tag.detect(np.zeros((4, 4), np.uint8))] == [1]
    db = make_db()
    with pytest.raises(NotImplementedError, match="no schema home"):
        sweep(make_source(1, 0), [tag], db)


def test_stream_via_sweep_equals_legacy_loop():
    """I2b step 2 gate (unit level): the sweep-backed sequential path returns
    EXACTLY what the legacy loop returned — frames, sizes, counts, bytes."""
    from mvis.io.detection_pipeline import DetectionPipeline

    class ApriltagBoard:                       # name chosen so the sweep path engages
        _rows = 2; _cols = 2; _spacing = 0.03; _tag_size = 0.1; _start_id = 0
        def num_points(self): return 16
        def point_3d(self, pid): return np.array([0.01 * pid, 0.02 * pid, 0.0])
        def min_points_for_pose(self): return 3
        def detect(self, image):
            base = float(image[0, 0])
            if int(base) == 99:                # a frame below the keep threshold
                return [Observation(point_id=1, pixel=np.array([1.0, 2.0]), row=2)]
            order = [5, 0, 9] if int(base) % 2 == 0 else [2, 7, 11, 3]
            return [Observation(point_id=p,
                                pixel=np.array([base + p + 0.25, 2.0 * p + 0.5]),
                                row=int(2.0 * p + 0.5))
                    for p in order]

    class StubLoader:
        def stream(self):
            for i in range(4):
                yield 0, np.full((6, 9), 2 * i, np.uint8), 50.0 + 0.1 * i
            yield 0, np.full((6, 9), 99, np.uint8), 50.45      # dropped by min filter

    legacy = DetectionPipeline(ApriltagBoard())._stream_detect_sequential(StubLoader())
    swept = DetectionPipeline(ApriltagBoard())._stream_detect_via_sweep(StubLoader())
    for a, b, name in zip(legacy, swept, ("frames", "sizes", "counts")):
        assert type(a) is type(b), name
    la, sa = legacy[0], swept[0]
    assert set(la) == set(sa) == {0}
    assert legacy[1] == swept[1] == {0: (9, 6)}
    assert legacy[2] == swept[2] == {0: 5}                     # counts incl. dropped
    assert len(la[0]) == len(sa[0]) == 4                       # 4 kept, 1 dropped
    for x, y in zip(la[0], sa[0]):
        assert x.timestamp == y.timestamp
        assert tuple(x.image_size) == tuple(y.image_size)
        assert [o.point_id for o in x.observations] == [o.point_id for o in y.observations]
        for ox, oy in zip(x.observations, y.observations):
            assert ox.pixel.tobytes() == oy.pixel.tobytes()
            assert ox.row == oy.row


def test_parallel_via_sweep_equals_legacy(tmp_path):
    """I2b step 3: the parallel path's component round-trip is field-identical
    to the legacy return."""
    from mvis.io.detection_pipeline import DetectionPipeline

    class ApriltagBoard:
        _rows = 2; _cols = 2; _spacing = 0.03; _tag_size = 0.1; _start_id = 0
        def num_points(self): return 16
        def point_3d(self, pid): return np.array([0.01 * pid, 0.02 * pid, 0.0])
        def min_points_for_pose(self): return 2
        def detect(self, image):
            base = float(image[0, 0])
            order = [5, 0, 9] if int(base) % 2 == 0 else [2, 7]
            return [Observation(point_id=p,
                                pixel=np.array([base + p + 0.25, 2.0 * p + 0.5]),
                                row=int(2.0 * p + 0.5)) for p in order]

    class StubLoader:
        def stream(self):
            for i in range(3):
                yield 0, np.full((6, 9), 2 * i, np.uint8), 60.0 + 0.1 * i
                yield 1, np.full((6, 9), 2 * i + 1, np.uint8), 60.05 + 0.1 * i

    legacy = DetectionPipeline(ApriltagBoard())._stream_detect_parallel(
        StubLoader(), [0, 1])
    swept = DetectionPipeline(ApriltagBoard())._stream_detect_parallel_via_sweep(
        StubLoader(), [0, 1])
    assert legacy[1] == swept[1] and legacy[2] == swept[2]
    assert set(legacy[0]) == set(swept[0])
    for cid in legacy[0]:
        assert len(legacy[0][cid]) == len(swept[0][cid])
        for x, y in zip(legacy[0][cid], swept[0][cid]):
            assert x.timestamp == y.timestamp
            assert tuple(x.image_size) == tuple(y.image_size)
            for ox, oy in zip(x.observations, y.observations):
                assert ox.point_id == oy.point_id
                assert ox.pixel.tobytes() == oy.pixel.tobytes()
                assert ox.row == oy.row


def test_real_board_classnames_are_mapped():
    """Guard against the silent-fallback bug: every REAL geometry-full board
    class must be inferable by board_from_geometry (the misspelled
    'ApriltagBoard' let every real AprilGrid skip the sweep path silently)."""
    from mvis.mvisdb.bridge import _TYPE_BY_CLASSNAME
    from mvis.detection.apriltag_board import AprilTagBoard
    from mvis.detection.aruco_board import ArucoBoard
    from mvis.detection.chessboard import Chessboard
    for cls in (AprilTagBoard, ArucoBoard, Chessboard):
        assert cls.__name__ in _TYPE_BY_CLASSNAME, cls.__name__


def test_board_geometry_split_and_facades():
    """BoardGeometry is the target contract; the legacy Board
    extends it with detect(); real boards satisfy both; the old lie (feature
    'boards' with no geometry) is documented, not hidden."""
    from mvis.detection.boards import (
        AprilTagBoard, ArucoBoard, Board, BoardGeometry, Chessboard, create_board)
    from mvis.detection.quality import frame_quality, quality_gate
    from mvis.detection.apriltag_features import AprilTagFeatureDetector
    for cls in (AprilTagBoard, ArucoBoard, Chessboard):
        assert issubclass(cls, Board) and issubclass(cls, BoardGeometry)
    assert issubclass(Board, BoardGeometry)
    # the structural lie is fixed: feature detectors are
    # FeatureFrontends, not Boards — no more geometry methods that only raise
    from mvis.detection import FeatureFrontend
    assert not issubclass(AprilTagFeatureDetector, Board)
    assert issubclass(AprilTagFeatureDetector, FeatureFrontend)
    assert callable(create_board) and hasattr(frame_quality, "__file__")


def test_subsample_flags_are_authority_and_lists_identical():
    """I3 step 2c-2: with the sweep-built db present, detection-stage hz +
    diversity selection is expressed as db flags, and the emitted lists are
    EXACTLY the flag query (same objects, same order) — identical to what the
    legacy surgery produced."""
    from mvis.io.detection_pipeline import DetectionPipeline

    class ApriltagBoard:
        _rows = 2; _cols = 2; _spacing = 0.03; _tag_size = 0.1; _start_id = 0
        def num_points(self): return 16
        def point_3d(self, pid): return np.array([0.01 * pid, 0.02 * pid, 0.0])
        def min_points_for_pose(self): return 3
        def detect(self, image):
            base = float(image[0, 0])
            return [Observation(point_id=p,
                                pixel=np.array([base + p + 0.25, 2.0 * p + 0.5]),
                                row=int(2.0 * p + 0.5))
                    for p in ([5, 0, 9] if int(base) % 2 == 0 else [2, 7, 11, 3])]

    class StubLoader:
        # 40 Hz stream so target_hz=5 actually thins
        def stream(self):
            for i in range(20):
                yield 0, np.full((6, 9), i % 7, np.uint8), 100.0 + 0.025 * i

    pipe = DetectionPipeline(ApriltagBoard())
    all_detected, sizes, _ = pipe._stream_detect_via_sweep(StubLoader())
    assert pipe._last_db is not None
    res = pipe._subsample_and_return(all_detected, sizes, {}, {}, 0, False, 5.0)

    db = res.db
    assert db is not None
    frames = db.camera(0).frames
    # every db frame carries a truthful flag + reason
    n_sel = sum(1 for f in frames if f.is_selected_to_use)
    assert n_sel == len(res.all_frames[0])
    reasons = {f.exclude_reason for f in frames if not f.is_selected_to_use}
    assert reasons <= {"detection:hz_subsample", "detection:diversity"}
    # full-rate tier = selected + diversity-dropped (hz-dropped are out)
    full_ts = {f.timestamp for f in res.all_frames_full_rate[0]}
    for f in frames:
        in_full = f.is_selected_to_use or f.exclude_reason == "detection:diversity"
        assert (f.timestamp in full_ts) == in_full
    # the emitted lists are the same objects the selectors kept, in order
    sel_ts = [f.timestamp for f in frames if f.is_selected_to_use]
    assert [fo.timestamp for fo in res.all_frames[0]] == sel_ts
    # and a legacy pipeline (no db) yields identical lists
    pipe2 = DetectionPipeline(ApriltagBoard())
    det2, sizes2, _ = pipe2._stream_detect_sequential(StubLoader())
    res2 = pipe2._subsample_and_return(det2, sizes2, {}, {}, 0, False, 5.0)
    assert res2.db is None
    assert [f.timestamp for f in res2.all_frames[0]] == \
           [f.timestamp for f in res.all_frames[0]]
    assert [f.timestamp for f in res2.all_frames_full_rate[0]] == \
           [f.timestamp for f in res.all_frames_full_rate[0]]
