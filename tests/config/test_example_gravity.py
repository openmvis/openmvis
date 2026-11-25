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
"""Every published cam-IMU example config sets `gravity:` to its dataset site's normal gravity
(4 dp), so a copied config solves at the local value (gravity-first, 2026-11-25)."""
import glob
import os

import pytest
import yaml

from mvis.core.gravity import normal_gravity

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
NEWARK, GARCHING, DUEBENDORF, WUHAN = (39.68, 40), (48.26, 482), (47.40, 440), (30.59, 25)
SITES = {                                         # published example dir -> (latitude_deg, altitude_m)
    "configs/example_*.yaml": NEWARK,
    "configs/tga_mvis/*.yaml": NEWARK,
    "configs/tga_t265/*.yaml": NEWARK,
    "configs/tga_t265_4cams/*.yaml": NEWARK,
    "configs/tga_t265_5imus/*.yaml": NEWARK,
    "configs/tumvi_cam_imu/*.yaml": GARCHING,
    "configs/tga_tumvi/*.yaml": GARCHING,
    "configs/uzh_fpv/*.yaml": DUEBENDORF,
    "configs/whuvid_zed2/*.yaml": WUHAN,
    "agent/virig_4imus_3cams/04_configs/cam_imu.yaml": NEWARK,
    "agent/tumvi_raw/04_configs/cam_imu.yaml": GARCHING,
}


def _cam_imu_configs():
    for pattern, site in SITES.items():
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and data.get("mode") == "cam-imu":
                yield pytest.param(path, site, id=os.path.relpath(path, ROOT))


CONFIGS = list(_cam_imu_configs())


def test_the_published_examples_are_found():
    assert len(CONFIGS) >= 84, len(CONFIGS)       # 82 example configs + 2 agent examples (2026-11-25)


@pytest.mark.parametrize("path, site", CONFIGS)
def test_example_config_sets_the_local_gravity(path, site):
    from mvis.core.config_parser import parse_cam_imu_config
    c = parse_cam_imu_config(path)
    assert c.solve.gravity_source == "config"
    assert c.solve.gravity == round(normal_gravity(*site), 4)
    assert c.solve.gravity_place
