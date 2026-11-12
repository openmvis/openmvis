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
"""The VI-rig 4-cam catalog entries must describe the bags on the Hub, or downloads fail their size check."""
import json
from pathlib import Path

import mvis.datasets

CATALOG = Path(mvis.datasets.__file__).with_name("dataset_catalog.json")


def _sources():
    data = json.loads(CATALOG.read_text())
    entry = next(d for d in data["datasets"] if d["key"] == "mvis_4imus_4cams")
    return {s["sequence"]: s for s in entry["sources"]}


def test_bag_09_is_the_repaired_bag_on_the_hub():
    bag = _sources()["bag_09"]
    assert bag["size_bytes"] == 7381133022
    assert bag["sha256"].startswith("858df19e")
    assert bag["sha256"].endswith("fb839bae")


def test_bag_08_matches_the_hub():
    assert _sources()["bag_08"]["size_bytes"] == 7719451418


def test_entries_point_at_the_corrected_bags_not_raw():
    urls = [s["url"] for s in _sources().values()]
    assert len(urls) == 15
    assert not [u for u in urls if "_raw" in u]
