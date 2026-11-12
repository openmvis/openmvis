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
import numpy as np, pytest
from mvis.imu import ImuModel
from mvis.recalib.calib_spec import RigCalib, ImuCalib, MVIS
from mvis.recalib.sensor_map import build_topic_plan, derive_mapping

def _rig():
    intr = ImuModel.from_tangent(np.zeros(24), "rpng")
    base = ImuCalib("imu0", "rpng", intr, np.zeros(3), np.zeros(3), 0.0, "base")
    aux = ImuCalib("imu1", "rpng", intr, np.zeros(3), np.zeros(3), 0.004, "aux")
    cam0 = MVIS("cam0", 0.002, 0.0, "radtan", np.zeros(4), np.zeros(4))
    return RigCalib(base, [aux], [cam0])

def test_plan_assigns_td_and_kinds():
    rig = _rig()
    mapping = {"/imu0": "imu0", "/imu1": "imu1", "/cam0/image_raw": "cam0"}
    plan = build_topic_plan(rig, mapping, base_imu_topic="/imu0")
    assert plan["/imu0"].kind == "imu" and plan["/imu0"].td == 0.0
    assert plan["/imu1"].td == 0.004
    assert plan["/cam0/image_raw"].kind == "cam" and plan["/cam0/image_raw"].td == 0.002

def test_unmapped_topic_absent_from_plan():
    rig = _rig()
    plan = build_topic_plan(rig, {"/imu0": "imu0"}, base_imu_topic="/imu0")
    assert "/velodyne" not in plan   # caller treats absent topics as passthrough

def test_bad_base_topic_raises():
    rig = _rig()
    with pytest.raises(ValueError):
        build_topic_plan(rig, {"/imu1": "imu1"}, base_imu_topic="/imu0")

def test_base_topic_maps_to_non_base_raises():
    rig = _rig()
    with pytest.raises(ValueError):
        build_topic_plan(rig, {"/imu0": "imu1", "/imu1": "imu1"},
                         base_imu_topic="/imu0")  # /imu0 maps to aux imu1, not base


# ---------------------------------------------------------------------------
# derive_mapping tests
# ---------------------------------------------------------------------------

def _rig_with_topics():
    """Like _rig() but every sensor carries a topic."""
    intr = ImuModel.from_tangent(np.zeros(24), "rpng")
    base = ImuCalib("imu0", "rpng", intr, np.zeros(3), np.zeros(3), 0.0, "base",
                    topic="/imu0")
    aux = ImuCalib("imu1", "rpng", intr, np.zeros(3), np.zeros(3), 0.004, "aux",
                   topic="/imu1")
    cam0 = MVIS("cam0", 0.002, 0.0, "radtan", np.zeros(4), np.zeros(4),
                    topic="/cam0/image_raw")
    cam1 = MVIS("cam1", 0.003, 0.0, "radtan", np.zeros(4), np.zeros(4),
                    topic="/cam1/image_raw")
    return RigCalib(base, [aux], [cam0, cam1])


def test_derive_mapping_from_topics():
    rig = _rig_with_topics()
    base_imu_topic, mapping = derive_mapping(rig)

    assert base_imu_topic == "/imu0"
    assert mapping == {
        "/imu0": "imu0",
        "/imu1": "imu1",
        "/cam0/image_raw": "cam0",
        "/cam1/image_raw": "cam1",
    }

    # Verify composition: feed into build_topic_plan and check result
    plan = build_topic_plan(rig, mapping, base_imu_topic)
    assert plan["/imu0"].kind == "imu" and plan["/imu0"].td == 0.0
    assert plan["/imu1"].kind == "imu" and plan["/imu1"].td == 0.004
    assert plan["/cam0/image_raw"].kind == "cam" and plan["/cam0/image_raw"].td == 0.002
    assert plan["/cam1/image_raw"].kind == "cam" and plan["/cam1/image_raw"].td == 0.003


def test_derive_mapping_missing_topic_raises():
    intr = ImuModel.from_tangent(np.zeros(24), "rpng")
    # cam1 has no topic — should trigger ValueError naming it
    base = ImuCalib("imu0", "rpng", intr, np.zeros(3), np.zeros(3), 0.0, "base",
                    topic="/imu0")
    cam0 = MVIS("cam0", 0.002, 0.0, "radtan", np.zeros(4), np.zeros(4),
                    topic="/cam0/image_raw")
    cam1 = MVIS("cam1", 0.003, 0.0, "radtan", np.zeros(4), np.zeros(4),
                    topic=None)
    rig = RigCalib(base, [], [cam0, cam1])

    with pytest.raises(ValueError, match="cam1"):
        derive_mapping(rig)


def test_derive_mapping_duplicate_topic_raises():
    import dataclasses
    rig = _rig_with_topics()
    rig.cams[1] = dataclasses.replace(rig.cams[1], topic=rig.base_imu.topic)
    with pytest.raises(ValueError):
        derive_mapping(rig)


# ---- resolve_mapping: one rule for mvis apply-calib and scripts/recalib_batch.py ----------------

def _apply_calib_mapping_before(rig, map_path, cfg):
    """The mapping block `mvis apply-calib` had inline before resolve_mapping."""
    import yaml
    if map_path:
        with open(map_path) as f:
            m = yaml.safe_load(f)
        return m["base_imu_topic"], m["mapping"]
    elif "mapping" in cfg and "base_imu_topic" in cfg:
        return cfg["base_imu_topic"], cfg["mapping"]
    return derive_mapping(rig)


def _recalib_batch_mapping_before(rig, map_path):
    """The mapping block scripts/recalib_batch.py had inline (it reads no recalib config)."""
    import yaml
    if map_path:
        with open(map_path) as f:
            m = yaml.safe_load(f)
        return m["base_imu_topic"], m["mapping"]
    return derive_mapping(rig)


def _map_file(tmp_path):
    import yaml
    p = tmp_path / "map.yaml"
    p.write_text(yaml.safe_dump({"base_imu_topic": "/a", "mapping": {"/a": "imu0", "/c": "cam0"}}))
    return str(p)


def test_resolve_mapping_precedence(tmp_path):
    from mvis.recalib.sensor_map import resolve_mapping
    rig = _rig_with_topics()
    inline = {"base_imu_topic": "/i", "mapping": {"/i": "imu0"}}
    assert resolve_mapping(rig, _map_file(tmp_path), inline) == ("/a", {"/a": "imu0", "/c": "cam0"})
    assert resolve_mapping(rig, None, inline) == ("/i", {"/i": "imu0"})
    assert resolve_mapping(rig, None, {"mapping": {"/i": "imu0"}}) == derive_mapping(rig)   # both keys needed
    assert resolve_mapping(rig) == derive_mapping(rig)
    with pytest.raises(ValueError, match="cam0"):
        resolve_mapping(_rig())                                  # no stored topics


def _recalib_configs():
    """Every recalib config in the repo (top-level `calib` + `input`), as `mvis apply-calib -c` reads it."""
    import pathlib
    import yaml
    root = pathlib.Path(__file__).resolve().parents[2]
    out = []
    for p in sorted((root / "examples").rglob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(d, dict) and "calib" in d and "input" in d:
            out.append((p, d or {}))
    return out


def test_resolve_mapping_matches_both_callers_on_every_repo_recalib_config(tmp_path):
    from mvis.recalib.sensor_map import resolve_mapping
    configs = _recalib_configs()
    assert configs                                               # recalib_example.yaml at least
    rig = _rig_with_topics()
    for map_path in (None, _map_file(tmp_path)):
        for p, cfg in configs:
            got = resolve_mapping(rig, map_path, cfg)
            assert got == _apply_calib_mapping_before(rig, map_path, cfg), p
            # recalib_batch reads no config: it agrees as long as the config sets no inline mapping
            if not ("mapping" in cfg and "base_imu_topic" in cfg):
                assert got == _recalib_batch_mapping_before(rig, map_path), p
        assert resolve_mapping(rig, map_path) == _recalib_batch_mapping_before(rig, map_path)


def test_resolve_mapping_flags_only_derive_failures(tmp_path):
    """Callers print 'cannot auto-derive mapping' for MappingDeriveError only: a --map file that
    yaml rejects with a ValueError (bad date) raises that plain ValueError, as before."""
    from mvis.recalib.sensor_map import MappingDeriveError, resolve_mapping
    bad = tmp_path / "bad_map.yaml"
    bad.write_text("when: 2026-13-01\nbase_imu_topic: /a\nmapping: {/a: imu0}\n")
    with pytest.raises(ValueError) as ei:
        resolve_mapping(_rig_with_topics(), str(bad))
    assert not isinstance(ei.value, MappingDeriveError)
    with pytest.raises(MappingDeriveError, match="cam0") as ei:
        resolve_mapping(_rig())
    try:
        derive_mapping(_rig())
    except ValueError as e:
        assert str(ei.value) == str(e)


def test_cli_callers_word_the_derive_message_only_for_derive_failures():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    for f in ("src/mvis/cli/main.py", "scripts/recalib_batch.py"):
        s = (root / f).read_text()
        i = s.index("cannot auto-derive mapping")
        assert "except MappingDeriveError as e:" in s[max(0, i - 300):i], f
