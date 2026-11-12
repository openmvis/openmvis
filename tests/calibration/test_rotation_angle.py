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
"""mvis.core.rotation.rotation_angle and the sites that use it."""
import gtsam
import numpy as np
import pytest

from mvis.calibration.state_dump import Tolerances, _gate_block
from mvis.core.rotation import rotation_angle
from mvis.mvisdb.estimates import _so3_log
from mvis.simulation.report import _rot_err_deg
from mvis.simulation.run import _rotdeg


def _rot(axis, th):
    return gtsam.Rot3.Expmap(np.asarray(axis, float) / np.linalg.norm(axis) * th).matrix()


@pytest.mark.parametrize("th", [0.0, 1e-9, 1e-5, 0.3, 2.0, 3.1415])
def test_rotation_angle_recovers_the_angle(th):
    assert np.isclose(rotation_angle(_rot([1, -2, 0.5], th)), th, rtol=1e-9, atol=1e-16)


def test_rotation_angle_keeps_tiny_angles_that_arccos_loses():
    R = _rot([0, 0, 1], 1e-9)
    assert np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) == 0.0
    assert np.isclose(rotation_angle(R), 1e-9, rtol=1e-6)


def test_rotation_angle_matches_arccos_on_normal_inputs():
    """What the arccos copies in the scripts computed, within the digits they print."""
    rng = np.random.default_rng(0)
    for th in rng.uniform(1e-3, 3.1, 50):
        R = _rot(rng.normal(size=3), th)
        assert np.isclose(rotation_angle(R), np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)), atol=1e-11)


def test_state_dump_gate_reports_the_relative_angle():
    a, b = np.eye(4), np.eye(4)
    a[:3, :3] = _rot([1, 0, 0], 0.2)
    b[:3, :3] = _rot([1, 0, 0], 0.2 + 1e-7)
    r = _gate_block("values/x0", a, b, Tolerances(), None, None)
    assert np.isclose(r.max_abs, 1e-7, rtol=1e-3)


def test_report_and_run_rotation_errors_agree():
    Ra, Rb = gtsam.Rot3.Rx(0.1), gtsam.Rot3.Rx(0.3)
    assert np.isclose(_rot_err_deg(Ra, Rb), np.rad2deg(0.2))
    Pa, Pb = gtsam.Pose3(Ra, gtsam.Point3(1, 2, 3)), gtsam.Pose3(Rb, gtsam.Point3(0, 0, 0))
    assert _rotdeg(Pa, Pb) == _rot_err_deg(Ra, Rb)


def test_estimates_so3_log_matches_the_old_formula_on_normal_inputs():
    def old(R):
        th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
        return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    rng = np.random.default_rng(1)
    for th in rng.uniform(1e-3, 3.0, 50):
        R = _rot(rng.normal(size=3), th)
        assert np.allclose(_so3_log(R), old(R), atol=1e-13, rtol=0)
    assert np.array_equal(_so3_log(np.eye(3)), np.zeros(3))
