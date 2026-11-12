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
"""Subprocess smoke test for the ``mvis apply-calib`` CLI subcommand."""
import shutil
import subprocess
import sys

import yaml


# Calib with rostopic/topic fields — used by tests 1, 2, 3.
d_with_topics = {
    "cameras": [{"name": "cam0",
                 "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1},
                 "distortion": {"model": "radtan", "coeffs": [0, 0, 0, 0]},
                 "td": 0.0, "rolling_shutter_readout": 0.0,
                 "rostopic": "/cam0/image_raw"}],
    "imus": [{"name": "imu0", "role": "base", "model": "rpng",
              "b_g": [0, 0, 0], "b_a": [0, 0, 0],
              "imu_intr_xi": [0.0] * 24,
              "topic": "/imu0"}],
}


def test_cli_apply_calib_runs(tiny_bag, tmp_path):
    # minimal identity calib YAML
    d = {"cameras": [{"name": "cam0",
                       "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1},
                       "distortion": {"model": "radtan", "coeffs": [0, 0, 0, 0]},
                       "td": 0.0, "rolling_shutter_readout": 0.0}],
         "imus": [{"name": "imu0", "role": "base", "model": "rpng",
                   "b_g": [0, 0, 0], "b_a": [0, 0, 0],
                   "imu_intr_xi": [0.0] * 24}]}
    calib = tmp_path / "calib.yaml"
    calib.write_text(yaml.safe_dump(d))

    mp = tmp_path / "map.yaml"
    mp.write_text(yaml.safe_dump({"base_imu_topic": "/imu0",
                                   "mapping": {"/imu0": "imu0",
                                               "/cam0/image_raw": "cam0"}}))
    out = tmp_path / "out.bag"
    r = subprocess.run(
        [sys.executable, "-m", "mvis.cli.main", "apply-calib",
         "--calib", str(calib), "--in", str(tiny_bag), "--out", str(out),
         "--map", str(mp)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "out.bag.recalib.yaml").exists()


def test_apply_calib_auto_map(tiny_bag, tmp_path):
    """No --map given; calib has topics → auto-derive succeeds."""
    calib = tmp_path / "calib_topics.yaml"
    calib.write_text(yaml.safe_dump(d_with_topics))
    out = tmp_path / "out_auto.bag"
    r = subprocess.run(
        [sys.executable, "-m", "mvis.cli.main", "apply-calib",
         "--calib", str(calib), "--in", str(tiny_bag), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "out_auto.bag.recalib.yaml").exists()


def test_apply_calib_config_file(tiny_bag, tmp_path):
    """--config YAML supplies calib/input/output; CLI auto-derives mapping."""
    calib = tmp_path / "calib_topics.yaml"
    calib.write_text(yaml.safe_dump(d_with_topics))
    out = tmp_path / "out_cfg.bag"
    cfg = {"calib": str(calib), "input": str(tiny_bag), "output": str(out)}
    cfg_file = tmp_path / "recalib.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    r = subprocess.run(
        [sys.executable, "-m", "mvis.cli.main", "apply-calib",
         "-c", str(cfg_file)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "out_cfg.bag.recalib.yaml").exists()


def test_apply_calib_config_expands_dataset_var(tiny_bag, tmp_path):
    """A config `input: ${MVIS_DATASETS}/...` (as in examples/configs/recalib_example.yaml) is expanded."""
    import os
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(tiny_bag, data / "in.bag")
    calib = tmp_path / "calib_topics.yaml"
    calib.write_text(yaml.safe_dump(d_with_topics))
    out = tmp_path / "out_var.bag"
    cfg_file = tmp_path / "recalib.yaml"
    cfg_file.write_text(yaml.safe_dump({"calib": str(calib), "input": "${MVIS_DATASETS}/in.bag",
                                        "output": str(out)}))
    r = subprocess.run(
        [sys.executable, "-m", "mvis.cli.main", "apply-calib", "-c", str(cfg_file)],
        capture_output=True, text=True, env={**os.environ, "MVIS_DATASETS": str(data)})
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_apply_calib_default_output(tiny_bag, tmp_path):
    """No --out given → output defaults to <input>.corrected.bag."""
    calib = tmp_path / "calib_topics.yaml"
    calib.write_text(yaml.safe_dump(d_with_topics))
    seq_bag = tmp_path / "seq.bag"
    shutil.copy(tiny_bag, seq_bag)
    r = subprocess.run(
        [sys.executable, "-m", "mvis.cli.main", "apply-calib",
         "--calib", str(calib), "--in", str(seq_bag)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "seq.corrected.bag").exists()


def test_apply_calib_missing_topics_needs_map(tiny_bag, tmp_path):
    """Old calib without rostopic/topic + no --map → non-zero exit + helpful message."""
    d_no_topics = {
        "cameras": [{"name": "cam0",
                     "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1},
                     "distortion": {"model": "radtan", "coeffs": [0, 0, 0, 0]},
                     "td": 0.0, "rolling_shutter_readout": 0.0}],
        "imus": [{"name": "imu0", "role": "base", "model": "rpng",
                  "b_g": [0, 0, 0], "b_a": [0, 0, 0],
                  "imu_intr_xi": [0.0] * 24}],
    }
    calib = tmp_path / "calib_no_topics.yaml"
    calib.write_text(yaml.safe_dump(d_no_topics))
    out = tmp_path / "should_not_exist.bag"
    r = subprocess.run(
        [sys.executable, "-m", "mvis.cli.main", "apply-calib",
         "--calib", str(calib), "--in", str(tiny_bag), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode != 0
    stderr = r.stderr + r.stdout
    assert "map" in stderr.lower() or "re-export" in stderr.lower(), stderr
