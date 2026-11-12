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
"""One cam-IMU config can drive a SET of bags (input.bags), each calibrated to
<out>/bag<NN>/cam_imu.json so mvis.recalib.result_metrics.discover_batch_results finds
them — no per-bag config copies."""
from mvis.cli.main import _bag_name, batch_plan
from mvis.core.config import CamImuRunConfig

CFG = "examples/regression/mvis_4imus_3cams/cam_imu_april_01.yaml"


def test_bag_name_uses_trailing_number_else_enumerates():
    assert _bag_name("/d/virig_4imus_4cams_09.bag", 0) == "bag09"   # trailing number kept
    assert _bag_name("/d/foo.bag", 3) == "bag04"                    # no number -> enumerate (1-based)


def test_input_bags_defaults_empty():
    assert CamImuRunConfig.from_yaml(CFG).input.bags == []


def test_batch_plan_templates_discoverable_output():
    cfg = CamImuRunConfig.from_yaml(CFG)
    cfg.input.bags = ["/d/x_01.bag", "/d/y_02.bag"]
    plan = batch_plan(cfg, "results/sweep")
    assert plan == [("bag01", "/d/x_01.bag", "results/sweep/bag01/cam_imu.json"),
                    ("bag02", "/d/y_02.bag", "results/sweep/bag02/cam_imu.json")]


def test_batch_plan_falls_back_to_single_path():
    cfg = CamImuRunConfig.from_yaml(CFG)            # no bags -> the single input.path, named by its number
    plan = batch_plan(cfg, "out")
    assert len(plan) == 1 and plan[0][1] == cfg.input.path


def test_bag_name_euroc_mav0_uses_parent_sequence_number():
    # euroc entries are <...>/MIC04_imucalib1/mav0 — the identity is the PARENT dir, not 'mav0'
    assert _bag_name("/d/mic_calib/MIC04_imucalib1/mav0", 0) == "bag04"
    assert _bag_name("/d/mic_calib/MIC15_imucalib7/mav0", 5) == "bag15"
    assert _bag_name("/d/x_09.bag", 0) == "bag09"   # ros1 trailing-number unchanged
