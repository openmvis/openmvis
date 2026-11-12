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
"""Path variables in run configs (opensource-prep): ``${MVIS_DATASETS}`` and ``~``.

Shipped example configs point at datasets as ``${MVIS_DATASETS}/<name>`` so they
work on any machine; ``MVIS_DATASETS`` defaults to ``~/Datasets``.
"""

import os

from mvis.core.config import CamImuRunConfig, CamRunConfig
from mvis.core.config_parser import expand_path_vars
from mvis.doctor.check_config import check_config

HOME = os.path.expanduser("~")


def test_mvis_datasets_defaults_to_home_datasets(monkeypatch):
    monkeypatch.delenv("MVIS_DATASETS", raising=False)
    assert expand_path_vars("${MVIS_DATASETS}/tum_vi") == f"{HOME}/Datasets/tum_vi"


def test_mvis_datasets_env_wins(monkeypatch):
    monkeypatch.setenv("MVIS_DATASETS", "/data")
    assert expand_path_vars("${MVIS_DATASETS}/tum_vi") == "/data/tum_vi"


def test_tilde_and_nested_values(monkeypatch):
    monkeypatch.setenv("MVIS_DATASETS", "/data")
    tree = {"a": ["~/x", {"b": "${MVIS_DATASETS}/y"}], "n": 3, "s": "plain"}
    assert expand_path_vars(tree) == {"a": [f"{HOME}/x", {"b": "/data/y"}], "n": 3, "s": "plain"}


def test_unset_other_variable_left_as_written(monkeypatch):
    monkeypatch.delenv("MVIS_NO_SUCH_VAR", raising=False)
    assert expand_path_vars("${MVIS_NO_SUCH_VAR}/x") == "${MVIS_NO_SUCH_VAR}/x"


def test_cam_imu_config_expands_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MVIS_DATASETS", "/data")
    p = tmp_path / "c.yaml"
    p.write_text("input:\n  format: euroc\n  path: ${MVIS_DATASETS}/tum_vi/mav0\n"
                 "camera_input:\n  path: ~/cam.yaml\n")
    cfg = CamImuRunConfig.from_yaml(str(p))
    assert cfg.input.path == "/data/tum_vi/mav0"
    assert cfg.camera_input.path == f"{HOME}/cam.yaml"


def test_cam_run_config_expands_source_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MVIS_DATASETS", "/data")
    p = tmp_path / "c.yaml"
    p.write_text("input:\n  cameras:\n  - camera_id: 0\n    source_path: ${MVIS_DATASETS}/v.bag\n")
    cfg = CamRunConfig.from_yaml(str(p))
    assert cfg.input.cameras[0].source_path == "/data/v.bag"


def test_doctor_checks_the_expanded_path(tmp_path, monkeypatch):
    (tmp_path / "frames").mkdir()
    monkeypatch.setenv("MVIS_DATASETS", str(tmp_path))
    p = tmp_path / "c.yaml"
    p.write_text("mode: cam\ninput:\n  cameras:\n  - camera_id: 0\n"
                 "    source_type: folder\n    source_path: ${MVIS_DATASETS}/frames\n")
    assert not [f for f in check_config(str(p)) if f.id == "camera.source_path"]
