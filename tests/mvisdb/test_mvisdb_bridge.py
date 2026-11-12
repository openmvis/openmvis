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
"""T0c bridge tests: today's pipeline shapes flow through MVISDatabase
BIT-IDENTICALLY (the harness that every later migration gate reuses).
"""
from __future__ import annotations

import numpy as np

from mvis.core.types import FrameObservations, Observation
from mvis.mvisdb.bridge import arrays_to_legacy, board_from_geometry, db_from_frame_observations
from mvis.mvisdb.mvisdatabase import BoardType, Corner, observation_fingerprint


class StubGeom:
    """Duck-typed board geometry: 2x2 grid of 4-corner tags, 16 points."""
    _rows = 2
    _cols = 2
    _spacing = 0.03
    _tag_size = 0.1
    _start_id = 0

    def num_points(self) -> int:
        return 16

    def point_3d(self, pid: int) -> np.ndarray:
        return np.array([0.01 * pid, 0.02 * pid, 0.0])


def obs(pid: int, u: float, v: float) -> Observation:
    return Observation(point_id=pid, pixel=np.array([u, v]))


def make_frames() -> dict[int, list[FrameObservations]]:
    return {
        0: [FrameObservations(camera_id=0, timestamp=10.0,
                              observations=[obs(5, 3.5, 7.25), obs(0, 1.0, 2.0),
                                            obs(9, 100.125, 200.5)],
                              image_size=(640, 480)),
            FrameObservations(camera_id=0, timestamp=10.1,
                              observations=[obs(2, 9.0, 8.0)],
                              image_size=(640, 480)),
            FrameObservations(camera_id=0, timestamp=10.2,
                              observations=[], image_size=(640, 480))],
        1: [FrameObservations(camera_id=1, timestamp=10.05,
                              observations=[obs(8, 5.0, 6.0)],
                              image_size=(640, 480))],
    }


def test_board_from_geometry_verbatim():
    b = board_from_geometry(StubGeom(), board_type=BoardType.APRILGRID)
    assert len(b.pts) == 16
    assert b.pts[7].tag_id == 1 and b.pts[7].corner_id is Corner.TL
    np.testing.assert_array_equal(b.point_3d(7), StubGeom().point_3d(7))
    assert b.rows == 2 and b.cols == 2 and b.size == 0.1


def test_legacy_round_trip_bit_identical():
    """frames -> db -> arrays_to_legacy == io.frame_detector.frames_to_arrays,
    bit for bit — the T0c contract."""
    from mvis.io.frame_detector import frames_to_arrays as legacy_arrays
    frames = make_frames()
    db = db_from_frame_observations(frames, StubGeom(),
                                    board_type=BoardType.APRILGRID,
                                    image_sizes={0: (640, 480), 1: (640, 480)})
    for cam in (0, 1):
        t_ref, ids_ref, uv_ref = legacy_arrays(frames[cam])
        t_db, ids_db, uv_db = arrays_to_legacy(db, cam)
        np.testing.assert_array_equal(t_ref, t_db)
        assert len(ids_ref) == len(ids_db)
        for a, b in zip(ids_ref, ids_db):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(uv_ref, uv_db):
            assert a.tobytes() == b.tobytes()       # raw IEEE754 bits


def test_fingerprint_stable_across_bridge_rebuild():
    frames = make_frames()
    kw = dict(board_type=BoardType.APRILGRID)
    fp1 = observation_fingerprint(db_from_frame_observations(frames, StubGeom(), **kw))
    fp2 = observation_fingerprint(db_from_frame_observations(make_frames(), StubGeom(), **kw))
    assert fp1 == fp2
    # perturb one uv by one ulp -> different capture
    frames[0][0].observations[0].pixel = np.array([np.nextafter(3.5, np.inf), 7.25])
    fp3 = observation_fingerprint(db_from_frame_observations(frames, StubGeom(), **kw))
    assert fp3 != fp1


def test_emission_order_preserved():
    frames = make_frames()
    db = db_from_frame_observations(frames, StubGeom(), board_type=BoardType.APRILGRID)
    g = db.camera(0).frames[0].board_detections[0]
    # detector emitted pid 5 first, then 0, then 9 — the replay column keeps that
    assert [d.point_id for d in g.detections] == [5, 0, 9]
    assert [d.detector_emission_order for d in g.detections] == [0, 1, 2]
    # legacy export follows emission order, not point_id order
    _, ids, _ = arrays_to_legacy(db, 0)
    assert list(ids[0]) == [5, 0, 9]


def test_bridge_db_validates():
    db = db_from_frame_observations(make_frames(), StubGeom(),
                                    board_type=BoardType.APRILGRID)
    db.validate()


def test_legacy_arrays_round_trip_bit_identical():
    """(t, ids, uv) -> db -> (t, ids, uv): the simulator/cam-IMU shape survives
    the structure bit for bit — closing the T0c loop from both directions."""
    from mvis.mvisdb.bridge import db_from_legacy_arrays
    rng = np.random.RandomState(3)
    t_cam = np.array([5.0, 5.1, 5.2])
    ids_list = [np.array([9, 2, 5]), np.array([0]), np.array([], int)]
    uv_list = [rng.rand(3, 2) * 640, rng.rand(1, 2) * 640, np.empty((0, 2))]
    db = db_from_legacy_arrays({0: (t_cam, ids_list, uv_list)}, StubGeom(),
                               board_type=BoardType.APRILGRID)
    t2, ids2, uv2 = arrays_to_legacy(db, 0)
    np.testing.assert_array_equal(t_cam, t2)
    for a, b in zip(ids_list, ids2):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(uv_list, uv2):
        assert np.asarray(a, float).tobytes() == b.tobytes()
    db.validate()
    # and the empty frame is honestly invalid (no groups with detections... it
    # HAS a group with zero detections -> detect_status no_detection)
    g = db.camera(0).frames[2].board_detections[0]
    assert g.detect_status == "no_detection"
