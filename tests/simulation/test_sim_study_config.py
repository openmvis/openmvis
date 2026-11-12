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
"""Unit tests for the package SimStudyConfig.

SimStudyConfig lives in mvis.simulation.config so sim runs are first-class
(the experiment scripts import it from there). Covers defaults, the
dataset-root default, from_env, the rate-coupling validate() guard, and the
calib_cfg() -> CamImuCalibConfig mapping.
"""

from mvis.simulation.config import SimStudyConfig, DATASET_ROOT
from mvis.simulation.noise import MVIS_IMU_NOISE
from mvis.core.config import CamImuCalibConfig


def test_defaults_and_dataset_root():
    c = SimStudyConfig()
    assert c.detector == "aruco"
    assert (c.state_hz, c.cam_hz, c.imu_hz) == (8.0, 20.0, 400.0)
    # noise defaults come from the centralized constant (M1)
    assert c.sigma_w == MVIS_IMU_NOISE.sigma_w
    assert c.sigma_ba == MVIS_IMU_NOISE.sigma_ba
    # dataset paths default under the configurable root, not a hardcoded abs path
    assert c.bag.startswith(DATASET_ROOT)
    assert c.camchain.startswith(DATASET_ROOT)


def test_calib_cfg_maps_noise_and_flags():
    cfg = SimStudyConfig().calib_cfg()
    assert isinstance(cfg, CamImuCalibConfig)
    assert cfg.sigma_g == MVIS_IMU_NOISE.sigma_w      # sigma_w -> sigma_g bridge
    assert cfg.sigma_a == MVIS_IMU_NOISE.sigma_a
    assert cfg.imu_intrinsics == "dw_da"              # pin_imu False -> freed
    assert cfg.imu_intrinsics_model == "calyx"        # matches project-wide default
    assert cfg.state_hz == 8.0
    assert cfg.sigma_td_prior == 0.0


def test_calib_cfg_imu_model_configurable():
    assert SimStudyConfig(imu_model="rpng").calib_cfg().imu_intrinsics_model == "rpng"


def test_calib_cfg_pin_and_state_override():
    cfg = SimStudyConfig(pin_imu=True).calib_cfg(pin_td=True, state_hz=15.0)
    assert cfg.imu_intrinsics == "pinned"             # pin_imu True
    assert cfg.sigma_td_prior == 1e-6                 # pin_td True
    assert cfg.state_hz == 15.0                       # override wins


def test_validate_rate_coupling_guard():
    # cam_hz must be an integer multiple of state_hz (else aliasing)
    assert SimStudyConfig(cam_hz=20.0, state_hz=8.0).validate() is False   # 2.5x
    assert SimStudyConfig(cam_hz=24.0, state_hz=8.0).validate() is True    # 3x, imu ok
    # imu_hz must be >= 10x max state_hz
    assert SimStudyConfig(cam_hz=80.0, state_hz=8.0, imu_hz=50.0).validate() is False


def test_from_env_april_branch(monkeypatch):
    monkeypatch.setenv("DETECTOR", "april")
    monkeypatch.setenv("BAGNUM", "03")
    monkeypatch.setenv("STATEHZ", "15")
    monkeypatch.setenv("TDOFF", "1")
    monkeypatch.setenv("PININTR", "1")
    c = SimStudyConfig.from_env()
    assert c.detector == "april"
    assert "april_03" in c.bag
    assert c.sfm_mode == "planar"      # april -> homography SfM
    assert c.state_hz == 15.0
    assert c.td_inject == 0.0          # TDOFF=1
    assert c.pin_intr is True


def test_from_env_aruco_default(monkeypatch):
    for k in ("DETECTOR", "TDOFF", "PININTR", "STATEHZ", "STATEHZLIST", "TDSWEEP"):
        monkeypatch.delenv(k, raising=False)
    c = SimStudyConfig.from_env()
    assert c.detector == "aruco"
    assert c.sfm_mode == "general"
    assert c.td_inject is None         # default: inject GT td
    assert c.state_hz == 8.0
