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
"""Permanent guarantee: EVERY shipped + regression input config parses.

Globs `examples/configs/**` and the regression input configs
(`examples/regression/<rig>/<bag>.yaml`), classifies each (camera-only
CalibrationConfig, sectioned CamRunConfig, or cam-IMU CamImuRunConfig), and asserts
it loads. Guards every future config-schema change against silently breaking the
shipped configs (config-cleanup repo-hygiene H1). Parsing touches only the YAML, not
the referenced datasets.

Below that, every example run config at any depth is linted with `mvis check`'s
config lint (no unknown key, a valid `mode:`), and the public example folders are
checked for home-directory paths.
"""

import glob
import os

# Built from parts so this guard does not trip the export leak scan itself.
_HOME_MARKERS = ("/" + "home/", "/" + "Users/")

import pytest
import yaml

from mvis.core.config import CalibrationConfig, CamImuRunConfig, CamRunConfig

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))

_CONFIG_KEYS = ("board", "cameras", "optimizer", "input", "solve", "camera_input")


def _input_configs():
    paths = (glob.glob(os.path.join(_ROOT, "examples", "configs", "*.yaml"))
             + glob.glob(os.path.join(_ROOT, "examples", "configs", "*", "*.yaml"))
             # regression INPUT configs: examples/regression/<rig>/<bag>.yaml (depth 2)
             + glob.glob(os.path.join(_ROOT, "examples", "regression", "*", "*.yaml")))
    out = []
    for p in sorted(set(paths)):
        if os.path.basename(p) == "manifest.yaml" or os.sep + "gt" + os.sep in p:
            continue
        try:
            raw = yaml.safe_load(open(p)) or {}
        except Exception:
            out.append(p)            # malformed YAML -> let the test flag it
            continue
        if isinstance(raw, dict) and any(k in raw for k in _CONFIG_KEYS):
            out.append(p)            # config-like (skips GT/result/target yamls)
    return out


_CONFIGS = _input_configs()


def _kind(raw: dict):
    """Route a raw config dict to its loader (matches the dual-shape parsers)."""
    if "camera_input" in raw or "imu_topics" in raw:
        return CamImuRunConfig
    inp = raw.get("input")
    if isinstance(inp, dict) and ("imu_topics" in inp or "format" in inp):
        return CamImuRunConfig
    if "solve" in raw and isinstance(inp, dict) and "cameras" in inp:
        return CamRunConfig                      # sectioned camera-only
    if "solve" in raw and "cameras" not in raw and "input" not in raw:
        return CamImuRunConfig
    return CalibrationConfig                      # legacy flat camera-only


def test_configs_present():
    # guard against a glob that silently finds nothing (vacuous suite)
    assert len(_CONFIGS) >= 50, f"expected the shipped+regression configs, found {len(_CONFIGS)}"


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: os.path.relpath(p, _ROOT))
def test_config_parses(path):
    raw = yaml.safe_load(open(path)) or {}
    cfg = _kind(raw).from_yaml(path)
    assert cfg is not None
    assert cfg.board is not None


# --- strict lint of the example run configs ------------------------------------------
# Every run config (a YAML with a top-level `board:`) under these example folders,
# at any depth, must pass `mvis check`'s config lint without an unknown key, a
# missing or bad `mode:`, or a parse failure. Missing datasets are not checked
# here. Folders that are not shipped are skipped.
_EXAMPLE_DIRS = ("configs", "experiments", "agent", "frequency_study", "omni_radtan_study")
_STRICT_IDS = {"config.unknown_key", "config.mode", "config.parse"}


def _example_run_configs():
    out = []
    for d in _EXAMPLE_DIRS:
        for p in sorted(glob.glob(os.path.join(_ROOT, "examples", d, "**", "*.yaml"),
                                  recursive=True)):
            try:
                raw = yaml.safe_load(open(p))
            except Exception:
                out.append(p)                # malformed YAML -> flagged by the test
                continue
            if isinstance(raw, dict) and "board" in raw:
                out.append(p)
    return out


_RUN_CONFIGS = _example_run_configs()


def test_example_run_configs_present():
    # the shipped public families alone hold more than 60 run configs
    assert len(_RUN_CONFIGS) >= 60, f"found only {len(_RUN_CONFIGS)} example run configs"


@pytest.mark.parametrize("path", _RUN_CONFIGS, ids=lambda p: os.path.relpath(p, _ROOT))
def test_example_run_config_lints_clean(path):
    from mvis.doctor.check_config import check_config
    bad = [f"{f.id}: {f.message}" for f in check_config(path)
           if f.status == "fail" and f.id in _STRICT_IDS]
    assert not bad, bad


# --- no maintainer home paths in the shipped example folders ---------------------------
# The folders on the public export allow-list (examples/configs/example_*.yaml, the
# recalib example and the public dataset families) must not name a home directory;
# dataset paths go through ${MVIS_DATASETS}.
_PUBLIC_EXAMPLES = ("example_*.yaml", "recalib_example.yaml", "tga_mvis/*", "tga_t265/*",
                    "tga_t265_4cams/*", "tga_t265_5imus/*", "tga_tumvi/*", "tumvi_cam_imu/*",
                    "uzh_fpv/*")


def _public_example_files():
    out = []
    for g in _PUBLIC_EXAMPLES:
        out += [p for p in glob.glob(os.path.join(_ROOT, "examples", "configs", g))
                if os.path.isfile(p)]
    return sorted(out)


def test_public_examples_have_no_home_paths():
    files = _public_example_files()
    assert len(files) >= 60, f"found only {len(files)} public example files"
    hits = [f"{os.path.relpath(p, _ROOT)}:{i}"
            for p in files for i, line in enumerate(open(p, errors="replace"), 1)
            if any(m in line for m in _HOME_MARKERS)]
    assert not hits, hits
