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
"""Unit tests for the `mvis datasets` CLI plumbing (no downloads)."""
import argparse

import pytest

from mvis.datasets.cli import add_datasets_subparser


def _parse(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_datasets_subparser(sub)
    return parser.parse_args(argv)


def test_actions_and_flags_parse():
    args = _parse(["datasets", "download", "--suite", "camera-imu",
                   "--select", "tum_mic/MIC05", "--select", "tum_vi_raw/imu1"])
    assert args.action == "download"
    assert args.select == ["tum_mic/MIC05", "tum_vi_raw/imu1"]
    args = _parse(["datasets", "convert", "--mcap", "--selection", "smoke"])
    assert args.action == "convert" and args.mcap and args.selection == "smoke"


def test_unknown_action_rejected(capsys):
    with pytest.raises(SystemExit):
        _parse(["datasets", "explode"])


def test_list_runs_without_network(capsys):
    args = _parse(["datasets", "list", "--dataset", "tum_vi_standard_512"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "tum_vi_standard_512" in out and "imu1" in out


def test_verify_missing_sequence_is_fast_and_reported(capsys, monkeypatch,
                                                     tmp_path):
    # Point DATASETS_ROOT at an empty tmp dir so imu2 is missing in any
    # checkout (it used to rely on imu2 not being under the real ~/Datasets).
    # Exercises the verify path without hashing gigabytes.
    from mvis.datasets import catalog as catalog_module
    monkeypatch.setattr(catalog_module, "DATASETS_ROOT", tmp_path)
    args = _parse(["datasets", "verify", "--select",
                   "tum_vi_standard_512/imu2"])
    assert args.func(args) == 1
    out = capsys.readouterr().out
    assert "missing" in out
