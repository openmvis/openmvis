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
"""ensure_capture lifecycle + the SqliteDetectionCache shim (I1/I2a).

The shim's contract is A/B bit-identity against the legacy JSON cache: same
inputs, byte-equal outputs, same validity semantics — proven here on synthetic
data before any pipeline call site is touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from mvis.core.types import FrameObservations, Observation
from mvis.io.detection_cache import DetectionCache
from mvis.mvisdb.manager import DatabaseManager, SealedError
from mvis.mvisdb.shim import SqliteDetectionCache
from mvis.mvisdb.spec import CaptureSpec, SourceConfig, CameraAddress
from mvis.mvisdb.mvisdatabase import ClockKind, SourceType, observation_fingerprint
from tests.helpers.mvisdb_fixtures import make_db


# ── ensure_capture: hit / miss / stale ──────────────────────────────────────

def _spec(path="/data/b.bag", mtime=1) -> CaptureSpec:
    return CaptureSpec(SourceConfig(
        type=SourceType.ROS1BAG, path=path, clock=ClockKind.HEADER,
        cameras=(CameraAddress(0, "/c0"), CameraAddress(1, "/c1")),
        source_mtime_ns=mtime, source_size=10))


def test_ensure_capture_miss_hit_stale(tmp_path):
    mgr = DatabaseManager()
    path = tmp_path / "bag.mvisdb"
    calls = []

    def build(spec):
        calls.append(spec)
        return make_db()

    db1 = mgr.ensure_capture(path, _spec(), build)          # MISS -> builds
    assert len(calls) == 1 and path.exists()
    db2 = mgr.ensure_capture(path, _spec(), build)          # HIT -> no build
    assert len(calls) == 1
    assert observation_fingerprint(db2) == observation_fingerprint(db1)
    with pytest.raises(SealedError, match="identity mismatch"):
        mgr.ensure_capture(path, _spec(mtime=2), build)     # STALE -> refuse
    assert len(calls) == 1


# ── the shim: A/B against the legacy JSON cache ─────────────────────────────

def _frames():
    rng = np.random.RandomState(11)
    out = {}
    for cid in (0, 1):
        frames = []
        for i in range(3):
            n = rng.randint(0, 5)
            pts = [rng.rand(2) * 640 for _ in range(n)]
            obs = [Observation(point_id=int(p), pixel=px, row=int(px[1]))
                   for p, px in zip(rng.choice(64, size=n, replace=False), pts)]
            frames.append(FrameObservations(camera_id=cid, timestamp=100.0 + i * 0.1,
                                            observations=obs, image_size=(640, 480)))
        out[cid] = frames
    return out


HIST = {0: [[(100.0, 3, 0.5), (100.1, 2, 0.4)]]}
SIZES = {0: (640, 480), 1: (640, 480)}
CFG = {"rows": 6, "cols": 6, "tag_size": 0.088, "family": "t36h11", "spacing": 0.3}


def test_shim_ab_bit_identical(tmp_path):
    frames = _frames()
    legacy = DetectionCache(str(tmp_path / "cacheA"))
    shim = SqliteDetectionCache(str(tmp_path / "cacheB"))
    legacy.save(frames, "src.bag", CFG, HIST, SIZES)
    shim.save(frames, "src.bag", CFG, HIST, SIZES)

    fa, ha, sa = legacy.load("src.bag", CFG)
    fb, hb, sb = shim.load("src.bag", CFG)
    assert set(fa) == set(fb) and ha == hb and sa == sb
    for cid in fa:
        assert len(fa[cid]) == len(fb[cid])
        for x, y in zip(fa[cid], fb[cid]):
            assert x.timestamp == y.timestamp
            assert tuple(x.image_size) == tuple(y.image_size)
            assert [o.point_id for o in x.observations] == \
                   [o.point_id for o in y.observations]        # emission order
            for ox, oy in zip(x.observations, y.observations):
                assert ox.pixel.tobytes() == oy.pixel.tobytes()  # exact bits
                assert ox.row == oy.row


def test_shim_validity_semantics_match_legacy(tmp_path):
    frames = _frames()
    legacy = DetectionCache(str(tmp_path / "cA"))
    shim = SqliteDetectionCache(str(tmp_path / "cB"))
    for c in (legacy, shim):
        assert c.is_valid("src.bag", CFG) is False             # nothing yet
    legacy.save(frames, "src.bag", CFG)
    shim.save(frames, "src.bag", CFG)
    for c in (legacy, shim):
        assert c.is_valid("src.bag", CFG) is True
        assert c.is_valid("src.bag", {**CFG, "spacing": 0.4}) is False
        assert c.is_valid("other.bag", CFG) is False
        with pytest.raises(ValueError):
            c.load("other.bag", CFG)


def test_shim_overwrite_like_legacy(tmp_path):
    shim = SqliteDetectionCache(str(tmp_path / "c"))
    shim.save(_frames(), "src.bag", CFG)
    frames2 = _frames()
    frames2[0][0].observations[0].pixel = np.array([1.0, 2.0])
    frames2[0][0].observations[0].row = 2        # D38: row == int(v), always
    shim.save(frames2, "src.bag", CFG)                         # legacy overwrites
    fb, _, _ = shim.load("src.bag", CFG)
    np.testing.assert_array_equal(fb[0][0].observations[0].pixel, [1.0, 2.0])


def test_shim_cam_imu_loader_call_shape(tmp_path):
    """cam_imu_loader passes its SIGNATURE string as source_path and {} as
    board_config — pipes, slashes and all. The shim must round-trip that shape
    exactly like the legacy cache (miss -> save -> valid -> load)."""
    sig = "/abs/path/bag01.bag|cams=['/cam0','/cam1']|t0=5.0|w=60|min=8|Board(...)"
    frames = _frames()
    shim = SqliteDetectionCache(str(tmp_path / "cam_imu_bag01.bag_ab12cd34"))
    assert shim.is_valid(sig, {}) is False
    shim.save(frames, sig, {})
    assert shim.is_valid(sig, {}) is True
    fb, _, _ = shim.load(sig, {})
    for cid in frames:
        for x, y in zip(frames[cid], fb[cid]):
            for ox, oy in zip(x.observations, y.observations):
                assert ox.pixel.tobytes() == oy.pixel.tobytes()
    # a DIFFERENT sig on the same (colliding) path is invalid, never wrong data
    assert shim.is_valid(sig + "|w=30", {}) is False


def test_cli_detect_all_frames_cached(tmp_path, monkeypatch):
    """The seventh site: mvis cam's inline detection loop now caches — the
    second call must never touch the data source."""
    from types import SimpleNamespace
    import mvis.cli.main as cli_main
    from mvis.core.types import Observation as Obs

    class StubBoard:
        """The subject is the CACHE seam, not the detector — a deterministic
        detect() keeps the test fast and vision-free."""
        def detect(self, image):
            base = float(image[0, 0])           # varies per frame
            return [Obs(point_id=i, pixel=np.array([10.0 + i + base, 20.0 + i]),
                        row=20 + i) for i in range(4)]

    board = StubBoard()
    imgs = [np.full((480, 640), k, np.uint8) for k in range(3)]

    reads = {"n": 0}

    def fake_source(cameras):
        for i, im in enumerate(imgs):
            reads["n"] += 1
            yield 0, im, float(i)

    monkeypatch.setattr(cli_main, "stream_cameras", fake_source)
    cfg = SimpleNamespace(
        cameras=[SimpleNamespace(camera_id=0, source_type="images",
                                 source_path="/fake")],
        output_file=str(tmp_path / "out" / "calibration.yaml"))
    (tmp_path / "out").mkdir()

    f1, s1 = cli_main._detect_all_frames(cfg, board)
    n_after_first = reads["n"]
    assert n_after_first == 3 and len(f1[0]) > 0
    f2, s2 = cli_main._detect_all_frames(cfg, board)
    assert reads["n"] == n_after_first          # source NEVER re-read
    assert s2 == s1 and len(f2[0]) == len(f1[0])
    for x, y in zip(f1[0], f2[0]):
        for ox, oy in zip(x.observations, y.observations):
            assert ox.pixel.tobytes() == oy.pixel.tobytes()


def test_cli_detect_all_frames_redetects_a_replaced_source(tmp_path, monkeypatch):
    """mvis cam's key starts with the source path, so the cache identity stats that
    file: a source re-recorded at the same path re-detects instead of reusing."""
    import os
    from types import SimpleNamespace
    import mvis.cli.main as cli_main
    from mvis.core.types import Observation as Obs

    class StubBoard:
        def detect(self, image):
            return [Obs(point_id=i, pixel=np.array([10.0 + i, 20.0 + i]), row=20 + i)
                    for i in range(4)]

    src = tmp_path / "x.bag"
    src.write_bytes(b"first")
    reads = {"n": 0}

    def fake_source(cameras):
        for i in range(2):
            reads["n"] += 1
            yield 0, np.zeros((48, 64), np.uint8), float(i)

    monkeypatch.setattr(cli_main, "stream_cameras", fake_source)
    cfg = SimpleNamespace(
        cameras=[SimpleNamespace(camera_id=0, source_type="ros1", source_path=str(src),
                                 topic="/cam0/image_raw")],
        output_file=str(tmp_path / "out" / "calibration.yaml"))
    (tmp_path / "out").mkdir()

    cli_main._detect_all_frames(cfg, StubBoard())
    cli_main._detect_all_frames(cfg, StubBoard())
    assert reads["n"] == 2                      # cached: the source was read once
    src.write_bytes(b"re-recorded, longer")     # same path, new size and mtime
    os.utime(src, ns=(1, 1))
    cli_main._detect_all_frames(cfg, StubBoard())
    assert reads["n"] == 4                      # re-detected


def test_identity_v2_stats_the_source(tmp_path):
    """The campaign key: touching the source file invalidates the cache —
    the exact staleness class the legacy key could not see (MIC08)."""
    import os, time
    src = tmp_path / "bag.bag"
    src.write_bytes(b"x" * 100)
    shim = SqliteDetectionCache(str(tmp_path / "c"))
    shim.save(_frames(), str(src), CFG)
    assert shim.is_valid(str(src), CFG) is True
    os.utime(src, ns=(time.time_ns(), time.time_ns()))     # content-change signal
    assert shim.is_valid(str(src), CFG) is False
    src2 = tmp_path / "bag2.bag"; src2.write_bytes(b"x" * 101)   # size change too
    shim.save(_frames(), str(src2), CFG)
    assert shim.is_valid(str(src2), CFG) is True


def test_identity_v2_invalidates_legacy_keyed_caches(tmp_path):
    """Every pre-campaign cache mismatches by construction (cs3: prefix)."""
    from mvis.mvisdb.manager import DatabaseManager
    from tests.helpers.mvisdb_fixtures import make_db
    path = tmp_path / "c.mvisdb"
    DatabaseManager().save(make_db(), path,
                           identity_hash="ab12cd34ef56",    # legacy 12-hex md5
                           spec_json='{"shim": {"board_config": {}}}')
    shim = SqliteDetectionCache(str(tmp_path / "c.json"))
    assert shim.is_valid("anything", {}) is False


def test_full_board_config_flips_on_detector_knobs():
    """min_tag_px / spacing / start_id now flip the key — the plan §3 holes."""
    from mvis.io.detection_pipeline import DetectionPipeline
    def board(**over):
        class B: pass
        b = B()
        vals = dict(_rows=6, _cols=6, _tag_size=0.088, _tag_family="t36h11",
                    _spacing=0.3, _start_id=0, _min_tag_px=12,
                    _refine_subpixel=True)
        vals.update(over)
        for k, v in vals.items():
            setattr(b, k, v)        # instance attrs, like real boards' __init__
        return b
    a = DetectionPipeline(board())._board_config()
    assert a == DetectionPipeline(board())._board_config()
    assert DetectionPipeline(board(_min_tag_px=8))._board_config() != a
    assert DetectionPipeline(board(_spacing=0.25))._board_config() != a
    assert DetectionPipeline(board(_start_id=100))._board_config() != a
    assert DetectionPipeline(board(_refine_subpixel=False))._board_config() != a


def test_shim_load_exposes_db(tmp_path):
    """I3 step 2c-3: a cache hit hands out the reconstructed MVISDatabase, so
    the carrier is present on every sqlite-backed run, warm or cold."""
    import numpy as np
    from mvis.core.types import FrameObservations, Observation
    from mvis.mvisdb.shim import SqliteDetectionCache
    src = tmp_path / "bag.mp4"; src.write_bytes(b"x" * 64)
    cache = SqliteDetectionCache(str(tmp_path / "detections.mvisdb"))
    frames = {0: [FrameObservations(camera_id=0, timestamp=1.0, observations=[
        Observation(point_id=3, pixel=np.array([4.0, 5.0]), row=5)])]}
    cfg = {"board_type": "AprilTagBoard", "rows": 2}
    cache.save(frames, str(src), cfg, {}, {0: (9, 6)})
    c2 = SqliteDetectionCache(str(tmp_path / "detections.mvisdb"))
    assert not hasattr(c2, "last_db")          # only a successful load sets it
    out, _, _ = c2.load(str(src), cfg)
    assert c2.last_db is not None
    st = c2.last_db.camera(0)
    assert len(st.frames) == 1 and st.frames[0].timestamp == 1.0
    assert st.frames[0].is_selected_to_use     # flags arrive clean


def test_shim_save_with_imus_declares_and_stores(tmp_path):
    """Capture-completeness: shim.save(imus=...) puts IMU streams in the core,
    so a cam-IMU run's result_imu rows have their FK target."""
    import numpy as np
    from mvis.core.types import FrameObservations, Observation
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.shim import SqliteDetectionCache
    src = tmp_path / "bag.bag"; src.write_bytes(b"y" * 64)
    cache = SqliteDetectionCache(str(tmp_path / "cap.mvisdb"))
    frames = {0: [FrameObservations(camera_id=0, timestamp=1.0, observations=[
        Observation(point_id=3, pixel=np.array([4.0, 5.0]), row=5)])]}
    imus = [dict(name="imu0", topic="/imu0",
                 t=np.array([0.9, 1.0, 1.1]),
                 w=np.zeros((3, 3)), a=np.ones((3, 3)) * 9.8)]
    cache.save(frames, str(src), {"board_type": "AprilTagBoard"}, {}, {0: (9, 6)},
               imus=imus)
    db = DatabaseManager().load(tmp_path / "cap.mvisdb")
    assert len(db.imus) == 1
    st = db.imus[0]
    assert st.imu_id == 0 and st.info.name == "imu0" and st.info.topic == "/imu0"
    np.testing.assert_array_equal(st.measurements.t, [0.9, 1.0, 1.1])
    np.testing.assert_array_equal(st.measurements.accel, np.ones((3, 3)) * 9.8)

    # and the previously FK-blocked attach now works: IMU estimate persists
    from mvis.mvisdb.estimates import estimates_from_cam_imu_result
    from mvis.mvisdb.mvisdatabase import CalibChoice, Run, SensorType
    from tests.helpers.mvisdb_fixtures import _imu_result
    r = _imu_result(); r.imus = []
    db.run = Run(CalibChoice.CAM_IMU, status="ok")
    es = estimates_from_cam_imu_result(r, state_hz=20.0)
    # capture declares cam0 only — keep the states the capture can hold
    es.sensor_states = [ss for ss in es.sensor_states
                        if ss.sensor in {(SensorType.IMU, 0), (SensorType.CAM, 0)}]
    db.estimate_states = es
    m = DatabaseManager()
    rid = m.save(db, tmp_path / "cap.mvisdb")
    got = m.state(tmp_path / "cap.mvisdb", rid)
    kinds = {s.sensor for s in got.sensor_states}
    assert (SensorType.IMU, 0) in kinds          # result_imu row landed


def test_cache_save_attaches_no_placeholder_run(tmp_path):
    """Empty-run cleanup: a capture is not an experiment — cache saves leave
    zero runs; the first real solve attach gets run_id 1."""
    import numpy as np
    from mvis.core.types import FrameObservations, Observation
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.mvisdatabase import CalibChoice, Run
    from mvis.mvisdb.shim import SqliteDetectionCache
    src = tmp_path / "bag.mp4"; src.write_bytes(b"z" * 64)
    cache = SqliteDetectionCache(str(tmp_path / "cap.mvisdb"))
    frames = {0: [FrameObservations(camera_id=0, timestamp=1.0, observations=[
        Observation(point_id=3, pixel=np.array([4.0, 5.0]), row=5)])]}
    cache.save(frames, str(src), {"board_type": "AprilTagBoard"}, {}, {0: (9, 6)})
    m = DatabaseManager()
    assert m.runs(tmp_path / "cap.mvisdb") == []
    db = m.load(tmp_path / "cap.mvisdb")     # load with no runs works
    db.run = Run(CalibChoice.CAM_ONLY, status="ok", description="first real")
    rid = m.save(db, tmp_path / "cap.mvisdb")
    assert rid == 1                          # the first run is a REAL one


def test_shim_concurrent_saves_never_fail(tmp_path):
    """Parallel rows share one cache file; a save must survive another save replacing it."""
    import threading
    frames, errors = _frames(), []
    start = threading.Barrier(8)

    def worker(i):
        cache = SqliteDetectionCache(str(tmp_path / "shared"))
        moved = {c: [FrameObservations(camera_id=f.camera_id, timestamp=f.timestamp + i,
                                       observations=f.observations, image_size=f.image_size)
                     for f in fs] for c, fs in frames.items()}
        start.wait()
        try:
            for _ in range(10):
                cache.save(moved, "/data/b.bag", CFG, HIST, SIZES)
        except Exception as exc:  # the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    cache = SqliteDetectionCache(str(tmp_path / "shared"))
    assert cache.is_valid("/data/b.bag", CFG)
    assert cache.load("/data/b.bag", CFG)[0]


def test_shim_identity_folds_in_the_source_stat_of_a_capture_key(tmp_path):
    """F4 keys are 'source|topics=[...]|...', not a path: the source part must still be
    stat'ed, so rewriting an image directory or bag invalidates its cache."""
    import os

    from mvis.io.capture_key import capture_key
    src = tmp_path / "cams"
    src.mkdir()
    key = capture_key(src, [src / "cam0", src / "cam1"])
    shim = SqliteDetectionCache(str(tmp_path / "c"))
    shim.save(_frames(), key, CFG)
    assert shim.is_valid(key, CFG) is True
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert shim.is_valid(key, CFG) is False
    # a key whose source part is not a path keeps the stat-free identity
    assert shim.is_valid("0:rosbag:x|topics=['/c']", CFG) is False
