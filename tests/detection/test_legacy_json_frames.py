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
"""detection_cache.legacy_json_frames: the read-only legacy detections.json reader."""
from __future__ import annotations

import json

import numpy as np

from mvis.core.types import FrameObservations, Observation
from mvis.io.detection_cache import DetectionCache, legacy_json_frames


def _frames(rng):
    out = {}
    for cid in (0, 2):
        out[cid] = [FrameObservations(
            camera_id=cid, timestamp=float(10 + k * 0.05 + rng.normal(0, 1e-4)),
            observations=[Observation(point_id=int(p), pixel=rng.uniform(0, 640, 2), row=int(rng.integers(0, 480)))
                          for p in rng.choice(144, int(rng.integers(0, 30)), replace=False)],
            image_size=(640, 480)) for k in range(12)]
    return out


def test_reads_what_detection_cache_wrote(tmp_path):
    frames = _frames(np.random.default_rng(0))
    DetectionCache(str(tmp_path / "detections.json")).save(frames, "src", {"b": 1})
    got = legacy_json_frames(tmp_path / "detections.json")
    assert sorted(got) == sorted(frames)
    for cid in frames:
        for f, g in zip(frames[cid], got[cid]):
            assert (g.camera_id, g.timestamp, g.image_size) == (f.camera_id, f.timestamp, f.image_size)
            assert [o.point_id for o in g.observations] == [o.point_id for o in f.observations]
            assert [o.row for o in g.observations] == [o.row for o in f.observations]
            assert all(np.asarray(o.pixel).tobytes() == np.asarray(p.pixel, float).tobytes()
                       for o, p in zip(g.observations, f.observations))


def test_missing_bad_and_sparse_files(tmp_path):
    assert legacy_json_frames(tmp_path / "none.json") == {}
    (tmp_path / "bad.json").write_text("{not json")
    assert legacy_json_frames(tmp_path / "bad.json") == {}
    (tmp_path / "sparse.json").write_text(json.dumps(
        {"cameras": {"1": {"frames": [{"camera_id": 1, "timestamp": 2.5,
                                       "observations": [{"point_id": 3, "pixel": [1, 2]}]}]}}}))
    f = legacy_json_frames(tmp_path / "sparse.json")[1][0]
    assert f.image_size == (0, 0) and f.observations[0].row == 0
