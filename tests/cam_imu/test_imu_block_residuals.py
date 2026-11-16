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
"""IMU Residuals per state block: the pipeline keeps each interval's rotation / position / velocity /
bias-step residual (raw and whitened) and the report shows one row per block plus a per-block table,
instead of one whole-state number per interval."""
import numpy as np

from mvis.calibration.pipeline_cam_imu import IMU_BLOCKS, O0, _imu_block_residuals
from tests.cam_imu.test_imu_chi2_tga import _values


class _Fac:
    Dt = 1 / 15

    def __init__(self, k):
        self.e = np.concatenate([np.full(3, 1e-3 * (k + 1)), np.full(3, 2e-4), np.full(3, 3e-3),
                                 np.full(3, 1e-5), np.full(3, 2e-4)])

    def _common(self, *a):
        return self.e, {}

    def whitened_error(self, *a):
        return self.e * 1e3


def test_block_residuals_raw_and_whitened_per_interval():
    out = _imu_block_residuals([_Fac(0), _Fac(1)], _values(2, 0.0), O0)
    assert out["blocks"] == [b for b, _ in IMU_BLOCKS] == ["theta", "p", "v", "bg", "ba"]
    assert out["raw"].shape == out["white"].shape == (2, 5)
    np.testing.assert_allclose(out["raw"][:, 0], [np.sqrt(3) * 1e-3, np.sqrt(3) * 2e-3])
    np.testing.assert_allclose(out["white"], out["raw"] * 1e3)
    np.testing.assert_allclose(out["dt"], [1 / 15, 1 / 15])


def test_a_factor_without_raw_residual_leaves_raw_nan():
    class Bare:
        def whitened_error(self, *a):
            return np.ones(15)
    out = _imu_block_residuals([Bare()], _values(1))
    assert np.isnan(out["raw"]).all() and np.allclose(out["white"], np.sqrt(3))


def test_report_section_has_one_row_per_block_and_the_table():
    from mvis.analysis.cam_imu_report import _imu_residual_body
    rng = np.random.default_rng(0)
    W = np.abs(rng.normal(1.6, 0.4, (40, 5)))
    d = {"imu_block_residuals": {"blocks": ["theta", "p", "v", "bg", "ba"], "white": W, "raw": W * 1e-3,
                                 "dt": np.full(40, 1 / 15)},
         "imu_cost_split": {"theta": 10.0, "p": 5.0, "v": 3.0, "bg": 1.0, "ba": 1.0}}
    html = _imu_residual_body(d)
    for label in ("rotation θ", "position p", "velocity v", "gyro bias step Δb_g", "accel bias step Δb_a"):
        assert label in html
    assert "Per state block over the run" in html and "50.0%" in html      # theta's cost share
    assert "Per-interval cost" not in html


def test_report_falls_back_to_the_whole_state_for_older_results():
    from mvis.analysis.cam_imu_report import _imu_residual_body
    html = _imu_residual_body({"imu_interval_cost": [1.0, 2.0], "imu_cost_split": {"theta": 1.0}})
    assert "Per-interval cost" in html and "Cost split" in html
