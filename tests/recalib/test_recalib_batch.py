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
"""Smoke test for scripts/recalib_batch.py.

Exercises the end-to-end batch runner on SYNTHETIC data (no real bag).
Uses the `tiny_bag` fixture from conftest.py as the source bag, writes
a minimal valid calib YAML and a map YAML into a temp directory, then
invokes the script as a subprocess and checks:
  - returncode 0
  - corrected bag exists in out_dir
  - .recalib.yaml sidecar exists
  - summary line prints "1/1 bags corrected."
"""
import os
import shutil
import subprocess
import sys

import pytest
import yaml

# Repo root derived from this file's location (tests/recalib/ -> root) so the
# subprocess cwd survives repo renames/moves.
_REPO_ROOT = __import__('pathlib').Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_calib_yaml(path: str) -> None:
    """Write a minimal valid calib YAML accepted by load_rig_calib."""
    doc = {
        "imus": [
            {
                "name": "imu0",
                "role": "base",
                "model": "rpng",
                "b_g": [0.0, 0.0, 0.0],
                "b_a": [0.0, 0.0, 0.0],
                # 24 zeros: Dw(6) + Da(6) + Tg(9) + R(3 Lie-algebra)
                "imu_intr_xi": [0.0] * 24,
                "td": 0.0,
            }
        ],
        "cameras": [
            {
                "name": "cam0",
                "intrinsics": {"fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0},
                "distortion": {"model": "radtan", "coeffs": [0.0, 0.0, 0.0, 0.0]},
                "td": 0.0,
                "rolling_shutter_readout": 0.0,
            }
        ],
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f)


def _write_map_yaml(path: str) -> None:
    """Write a minimal topic-map YAML matching the tiny_bag fixture."""
    doc = {
        "base_imu_topic": "/imu0",
        "mapping": {
            "/imu0": "imu0",
            "/cam0/image_raw": "cam0",
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recalib_batch_smoke(tiny_bag, tmp_path):
    """Script processes one bag, exits 0, writes corrected bag + sidecar."""
    # --- set up directory layout -------------------------------------------
    in_dir = tmp_path / "in_bags"
    out_dir = tmp_path / "out_bags"
    in_dir.mkdir()

    # copy/link the tiny_bag into in_dir as seq01.bag
    src_bag = in_dir / "seq01.bag"
    shutil.copy2(tiny_bag, src_bag)

    calib_path = str(tmp_path / "calib.yaml")
    map_path = str(tmp_path / "map.yaml")
    _write_calib_yaml(calib_path)
    _write_map_yaml(map_path)

    # --- invoke the script -------------------------------------------------
    result = subprocess.run(
        [
            sys.executable,
            "scripts/recalib_batch.py",
            "--calib", calib_path,
            "--map", map_path,
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )

    # --- assertions --------------------------------------------------------
    stdout = result.stdout
    stderr = result.stderr

    assert result.returncode == 0, (
        f"Script exited {result.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )

    corrected_bag = out_dir / "seq01.bag"
    assert corrected_bag.exists(), (
        f"Corrected bag not found at {corrected_bag}\nSTDOUT:\n{stdout}"
    )

    sidecar = out_dir / "seq01.bag.recalib.yaml"
    assert sidecar.exists(), (
        f"Provenance sidecar not found at {sidecar}\nSTDOUT:\n{stdout}"
    )

    assert "1/1 bags corrected." in stdout, (
        f"Expected '1/1 bags corrected.' in stdout:\n{stdout}"
    )


def test_recalib_batch_continue_on_error(tiny_bag, tmp_path):
    """Script continues on a broken bag, exits 1, prints failure tally."""
    in_dir = tmp_path / "in_bags"
    out_dir = tmp_path / "out_bags"
    in_dir.mkdir()

    # one valid bag
    shutil.copy2(tiny_bag, in_dir / "good.bag")
    # one corrupt bag (not a real rosbag)
    (in_dir / "bad.bag").write_bytes(b"not a real bag")

    calib_path = str(tmp_path / "calib.yaml")
    map_path = str(tmp_path / "map.yaml")
    _write_calib_yaml(calib_path)
    _write_map_yaml(map_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/recalib_batch.py",
            "--calib", calib_path,
            "--map", map_path,
            "--in-dir", str(in_dir),
            "--out-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )

    assert result.returncode == 1, (
        f"Expected exit 1 (one failure), got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # "FAIL bad.bag" in stderr
    assert "FAIL" in result.stderr, (
        f"Expected FAIL line in stderr:\n{result.stderr}"
    )

    # Summary: 1 out of 2 succeeded
    assert "1/2 bags corrected." in result.stdout, (
        f"Expected '1/2 bags corrected.' in stdout:\n{result.stdout}"
    )
