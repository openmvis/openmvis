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
"""mvis.core.paths.datasets_root: one definition of where the datasets live."""
import os

from mvis.core import paths
from mvis.core.config_parser import expand_path_vars


def test_default_is_expanded_home_datasets(monkeypatch):
    monkeypatch.delenv("MVIS_DATASETS", raising=False)
    assert paths.datasets_root() == os.path.expanduser("~/Datasets")


def test_env_is_used_as_written(monkeypatch):
    monkeypatch.setenv("MVIS_DATASETS", "/data")
    assert paths.datasets_root() == "/data"
    monkeypatch.setenv("MVIS_DATASETS", "~/elsewhere")       # not expanded, as before
    assert paths.datasets_root() == "~/elsewhere"


def test_config_expansion_uses_the_same_default(monkeypatch):
    monkeypatch.delenv("MVIS_DATASETS", raising=False)
    assert expand_path_vars("${MVIS_DATASETS}/x") == os.path.join(paths.datasets_root(), "x")
    assert expand_path_vars("a:${MVIS_DATASETS}") == "a:~/Datasets"   # only a leading ~ expands


def test_sim_config_root_matches(monkeypatch):
    from mvis.simulation import config as sc
    assert sc.DATASET_ROOT == os.environ.get("MVIS_DATASETS", os.path.expanduser("~/Datasets"))
