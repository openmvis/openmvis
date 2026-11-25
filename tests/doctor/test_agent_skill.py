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
"""The mvis-calibrate skill (.claude/) and doc/agent.md stay consistent with the code:
every relative link resolves, every `mvis <command>` they name exists, the config templates
parse through the real loaders without an unknown-key warning, and the defaults they spell
out are the code's defaults."""
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mvis.core import config as C
from mvis.datasets.catalog import DatasetCatalog

ROOT = Path(__file__).resolve().parents[2]
CLAUDE = ROOT / ".claude"
SKILL = CLAUDE / "skills" / "mvis-calibrate"
TRIAGE = CLAUDE / "agents" / "mvis-triage.md"
SETTINGS = CLAUDE / "settings.json"
AGENT_DOC = ROOT / "doc" / "agent.md"
DOCS = [p for p in [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md")), TRIAGE, AGENT_DOC]
        if p.exists()]

needs_skill = pytest.mark.skipif(not SKILL.is_dir(), reason="skill not shipped in this tree")
needs_settings = pytest.mark.skipif(not SETTINGS.exists(), reason="no .claude/settings.json in this tree")

# Per-camera override entries of camera_input.cameras are free-form dicts; the parser does not
# validate them (see the comment on CamImuCameraInputConfig.cameras).
CAM_IMU_CAMERA_OVERRIDE_KEYS = {"factor_hz", "estimate_readout", "rolling_mode", "free_intrinsics",
                                "retime_sensor_clock", "retime_map", "vision_pose"}


def _link_target(raw):
    """Path part of a Markdown link target: drops a `"title"`, angle brackets and an anchor."""
    target = raw.strip().split()[0].strip("<>")
    return target.split("#")[0]


def _filled(template, tmp_path):
    """Write the template with its `<...>` placeholders replaced so it parses; numbers where
    the schema wants numbers."""
    text = re.sub(r"<[^>\n]+>", "x", template.read_text())
    for key, value in (("rows", "6"), ("cols", "6"), ("tag_size", "0.088"), ("tag_spacing", "0.3"),
                       ("gravity", "9.8013")):
        text = re.sub(rf"^(\s*{key}:) x", rf"\1 {value}", text, flags=re.M)
    path = tmp_path / template.name
    path.write_text(text)
    return path


def _unknown_keys(caplog):
    return [r.getMessage() for r in caplog.records if "unknown key" in r.getMessage()]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    for raw in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
        target = _link_target(raw)
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (doc.parent / target).resolve().exists(), f"{doc.name}: broken link {raw}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_named_commands_exist(doc):
    out = subprocess.run([sys.executable, "-m", "mvis.cli.main", "--help"],
                         capture_output=True, text=True, check=True).stdout
    known = set(re.search(r"\{([a-z0-9,-]+)\}", out).group(1).split(","))
    named = set(re.findall(r"`mvis ([a-z][a-z0-9-]*)", doc.read_text()))
    assert named <= known, f"{doc.name} names unknown commands: {sorted(named - known)}"


@needs_skill
def test_cam_template_parses_without_unknown_keys_and_states_the_defaults(tmp_path, caplog):
    path = _filled(SKILL / "templates" / "cam.yaml", tmp_path)
    raw = yaml.safe_load(path.read_text())
    assert raw["mode"] == "cam"
    with caplog.at_level(logging.WARNING, logger="mvis.core.config_parser"):
        cfg = C.CamRunConfig.from_yaml(str(path))
    assert not _unknown_keys(caplog), _unknown_keys(caplog)
    assert raw["solve"]["corner_subsample"] == C.CamRunConfig().corner_subsample
    assert cfg.corner_subsample == raw["solve"]["corner_subsample"]
    assert raw["estimate"]["board_points"] == C.OptimizerConfig().optimize_board_points


@needs_skill
def test_cam_imu_template_parses_without_unknown_keys_and_states_the_defaults(tmp_path, caplog):
    path = _filled(SKILL / "templates" / "cam_imu.yaml", tmp_path)
    raw = yaml.safe_load(path.read_text())
    assert raw["mode"] == "cam-imu"
    with caplog.at_level(logging.WARNING, logger="mvis.core.config_parser"):
        C.CamImuRunConfig.from_yaml(str(path))
    assert not _unknown_keys(caplog), _unknown_keys(caplog)
    for entry in raw["camera_input"]["cameras"]:
        assert set(entry) <= CAM_IMU_CAMERA_OVERRIDE_KEYS, entry
    solve, inp, cams = C.CamImuCalibConfig(), C.CamImuInputConfig(), C.CamImuCameraInputConfig()
    assert raw["solve"]["estimate_t_ga"] == solve.estimate_t_ga
    assert raw["solve"]["vision_pose"] == solve.vision_pose
    assert raw["solve"]["corner_subsample"] == solve.corner_subsample
    assert raw["estimate"]["imu"]["model"] == solve.imu_intrinsics_model
    assert raw["estimate"]["imu"]["intrinsics"] == solve.imu_intrinsics
    assert raw["estimate"]["camera"]["intrinsics"] == cams.free_intrinsics
    assert raw["estimate"]["board_points"] == solve.estimate_board_heights
    assert raw["input"]["window_mode"] == inp.window_mode
    assert raw["input"]["segment_length"] == inp.segment_length
    # the Full preset is the code default for non-base cameras
    assert raw["camera_input"]["cameras"][1]["factor_hz"] == solve.factor_hz


@needs_skill
def test_sample_dataset_keys_are_catalogued():
    text = (SKILL / "references" / "datasets.md").read_text()
    named = set(re.findall(r"^\| `([a-z0-9_]+)`", text, re.M))
    named |= set(re.findall(r"--dataset ([a-z0-9_]+)", text))
    named |= set(re.findall(r"--select ([a-z0-9_]+)/", text))
    assert named, "no catalog keys named in datasets.md"
    missing = named - set(DatasetCatalog.load().keys())
    assert not missing, f"datasets.md names unknown catalog keys: {sorted(missing)}"


@needs_settings
def test_settings_allow_the_commands_the_skill_runs():
    allow = set(json.loads(SETTINGS.read_text())["permissions"]["allow"])
    expected = {"Bash(mvis --version)", "Bash(mvis inspect:*)", "Bash(mvis check:*)",
                "Bash(mvis verdict:*)", "Bash(mvis run:*)", "Bash(mvis roundtrip:*)",
                "Bash(mvis datasets list:*)", "Bash(which mvis)",
                "Bash(mkdir -p results/agent_runs/:*)", "Bash(mkdir -p mvis_runs/:*)"}
    assert expected <= allow, sorted(expected - allow)
