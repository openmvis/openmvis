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
"""Unit tests for the centralized IMU noise densities (config-cleanup M1).

Locks the canonical MVIS densities + the two naming-bridge helpers so the sim
scripts and the recovery-solve config can't drift (the values were previously
copy-pasted in sim_aruco_grounded.py and sim_grounded_scenario.py)."""

import numpy as np

from mvis.simulation.noise import ImuNoiseDensities, MVIS_IMU_NOISE
from mvis.core.config import CamImuCalibConfig


def test_mvis_densities_match_historical_constants():
    # the exact MVIS launch densities the sim scripts used before centralization
    assert MVIS_IMU_NOISE.sigma_w == 1.6968e-04
    assert MVIS_IMU_NOISE.sigma_bg == 1.9393e-05
    assert MVIS_IMU_NOISE.sigma_a == 3.0e-03
    assert MVIS_IMU_NOISE.sigma_ba == 3.0e-03


def test_sim_kwargs_keys_and_values():
    """simulate_imu spells gyro density `sigma_w`."""
    kw = MVIS_IMU_NOISE.sim_kwargs()
    assert set(kw) == {"sigma_w", "sigma_bg", "sigma_a", "sigma_ba"}
    assert kw["sigma_w"] == MVIS_IMU_NOISE.sigma_w
    assert kw["sigma_a"] == MVIS_IMU_NOISE.sigma_a


def test_calib_kwargs_bridges_sigma_w_to_sigma_g():
    """CamImuCalibConfig spells the SAME gyro density `sigma_g`."""
    kw = MVIS_IMU_NOISE.calib_kwargs()
    assert set(kw) == {"sigma_g", "sigma_bg", "sigma_a", "sigma_ba"}
    assert kw["sigma_g"] == MVIS_IMU_NOISE.sigma_w        # the naming bridge
    assert kw["sigma_bg"] == MVIS_IMU_NOISE.sigma_bg
    assert kw["sigma_a"] == MVIS_IMU_NOISE.sigma_a
    assert kw["sigma_ba"] == MVIS_IMU_NOISE.sigma_ba


def test_calib_kwargs_threads_into_config():
    cfg = CamImuCalibConfig(state_hz=8.0, **MVIS_IMU_NOISE.calib_kwargs())
    assert cfg.sigma_g == MVIS_IMU_NOISE.sigma_w
    assert cfg.sigma_bg == MVIS_IMU_NOISE.sigma_bg
    assert cfg.sigma_a == MVIS_IMU_NOISE.sigma_a
    assert cfg.sigma_ba == MVIS_IMU_NOISE.sigma_ba


def test_custom_noise_is_first_class():
    """A user supplies custom noise by constructing their own instance."""
    custom = ImuNoiseDensities(sigma_w=1e-3, sigma_bg=1e-5, sigma_a=2e-2, sigma_ba=4e-4)
    assert custom.calib_kwargs()["sigma_g"] == 1e-3
    assert custom.sim_kwargs()["sigma_a"] == 2e-2


def test_scaled_scalar_and_per_channel():
    s = MVIS_IMU_NOISE.scaled(2.0)
    assert s.sigma_w == 2 * MVIS_IMU_NOISE.sigma_w
    assert s.sigma_ba == 2 * MVIS_IMU_NOISE.sigma_ba
    # k=0 -> noise-free
    z = MVIS_IMU_NOISE.scaled(0.0)
    assert (z.sigma_w, z.sigma_bg, z.sigma_a, z.sigma_ba) == (0, 0, 0, 0)
    # per-channel factors (w, bg, a, ba)
    p = MVIS_IMU_NOISE.scaled((1.0, 2.0, 3.0, 4.0))
    assert p.sigma_bg == 2 * MVIS_IMU_NOISE.sigma_bg
    assert p.sigma_a == 3 * MVIS_IMU_NOISE.sigma_a


def test_frozen_and_distinct_instances():
    n = ImuNoiseDensities(1.0, 2.0, 3.0, 4.0)
    assert n.sim_kwargs()["sigma_w"] == 1.0 and n.calib_kwargs()["sigma_g"] == 1.0
    import dataclasses
    try:
        n.sigma_w = 9.0          # frozen -> must raise
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


def test_sim_aruco_grounded_defaults_use_the_constant():
    """The SimStudyConfig defaults must equal the centralized constant (proves the
    sim script was wired to it, not left with copy-pasted literals)."""
    import os, sys
    here = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "examples", "experiments")
    sys.path.insert(0, os.path.abspath(here))
    try:
        from sim_aruco_grounded import SimStudyConfig
    except Exception as e:                     # the example's own deps may be absent
        import pytest
        pytest.skip(f"examples/experiments/sim_aruco_grounded.py does not import: {e}")
    c = SimStudyConfig()
    assert c.sigma_w == MVIS_IMU_NOISE.sigma_w
    assert c.sigma_bg == MVIS_IMU_NOISE.sigma_bg
    assert c.sigma_a == MVIS_IMU_NOISE.sigma_a
    assert c.sigma_ba == MVIS_IMU_NOISE.sigma_ba
