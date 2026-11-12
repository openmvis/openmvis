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
"""A polled camera through the whole sensor-clock path, about 30 s: pass 1 per-frame offsets, the
lattice fit, re-stamping, pass 2, the sidecar written and read back, and a later solve from that map.

Simulated, as no twin has a polled camera: free-running on an 11 Hz sensor clock, stamped by a 9 Hz
polling loop that takes the latest finished frame (the ELP / usb_cam pattern).
"""
import numpy as np
import pytest

from mvis.calibration.pipeline_cam_imu import CamImuCalibConfig
from mvis.io.sensor_clock import load_retime_map, read_sidecar, write_sensor_clock_sidecar
from mvis.simulation.synth import recover
from tests.helpers.sim_scenarios import _polled_scenario

pytestmark = pytest.mark.integration


def test_polled_camera_two_pass_sidecar_and_reuse(tmp_path):
    gt, cams, t_imu, w_m, a_m, aux, T_s = _polled_scenario()
    res = recover([dict(cams[0], retime_sensor_clock=True)], gt.board_points_local, t_imu, w_m, a_m, aux,
                  CamImuCalibConfig(vision_pose="imu"))
    assert res.ok and res.diagnostics["iterations"] > 0 and "lm_no_step" not in res.diagnostics
    assert res.diagnostics["reproj_rms_px"] < 1.05 * 0.5 * np.sqrt(2.0)    # sigma_px 0.5 per axis
    sc = res.diagnostics["sensor_clock"][0]
    assert sc["retimed"], sc
    assert abs(sc["rate_hz"] - 1.0 / T_s) < 0.01
    assert sc["n_retimed"] >= 0.9 * len(cams[0]["t_cam"])

    # the capture times sit on the sensor lattice: within a fraction of a period of phase + n T
    fitted, T = np.asarray(sc["retime_map"])[:, 1], sc["period_s"]
    assert np.abs(np.remainder(fitted - fitted[0] + T / 2, T) - T / 2).max() < 1e-6

    # sidecar: written compact + maps, and the map comes back as it went in
    path = tmp_path / "cam_imu_sensor_clock.json"
    compact = write_sensor_clock_sidecar(res.diagnostics["sensor_clock"], str(path), ["/cam0/image_raw"])
    assert compact["0"]["sidecar"] == str(path) and "retime_map" not in compact["0"]
    rec = read_sidecar(str(path), check_format=True)["cameras"]["0"]
    assert rec["topic"] == "/cam0/image_raw" and rec["rate_hz"] == sc["rate_hz"]
    back = load_retime_map(str(path), 0)
    np.testing.assert_array_equal(back, np.asarray(sc["retime_map"]))

    # a later solve from that map skips pass 1 and re-stamps the same frames
    again = recover([dict(cams[0], retime_map=back)], gt.board_points_local, t_imu, w_m, a_m, aux,
                    CamImuCalibConfig(vision_pose="imu", max_iterations=1))
    r = again.diagnostics["sensor_clock"][0]
    assert r["source"] == "retime_map" and r["n_retimed"] == sc["n_retimed"]
