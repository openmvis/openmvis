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
"""End-to-end solves on digital twins of two real cam-IMU recordings, no images needed.

Each twin (tests/data/twins/, exported by the maintainers from a real solve) is 8 s of that solve: its
calibration, perturbed with a fixed seed (`mvis.simulation.twin.perturb`), is the truth, with the real frame times, corner ids and solve settings. These run the
whole pipeline in about 20 s per solve and catch estimator bias that unit tests cannot.

  twin_gs  global shutter, 470 Hz IMU, t_ga +5.3 ms
  twin_rs  rolling shutter with negative readout (-26 ms), 200 Hz IMU, t_ga -7.4 ms

The no-noise checks are the precise ones: any error left is bias. Limits sit above the errors
measured on these windows (perturbed fixtures, 2026-11-10). The seeded checks are loose, as 8 s is short.

    scripts/test.sh integration
"""
import numpy as np
import pytest

from mvis.simulation.twin import Twin, solve

pytestmark = pytest.mark.integration

GS, RS = "twin_gs", "twin_rs"

# no noise: (name, measured 2026-11-10)
LIMITS_NO_NOISE = {
    GS: dict(rms_px=0.012, t_ga_ms=0.15, lever_mm=0.6, td_ms=0.05, tr_ms=0.05),     # 0.0058, 0.030, 0.44, 0.002, 0
    RS: dict(rms_px=0.02, t_ga_ms=0.20, lever_mm=1.2, td_ms=0.05, tr_ms=0.05),      # 0.0069, 0.047, 0.39, 0.007, 0.009
}
# real noise, seed 0
LIMITS_NOISE = {
    GS: dict(rms_px=0.45, t_ga_ms=1.0, lever_mm=2.0, td_ms=0.5),                    # 0.376, 0.48, 1.10, 0.26
    RS: dict(rms_px=0.90, t_ga_ms=0.5, lever_mm=1.5, td_ms=0.5, tr_ms=1.0),         # 0.806, 0.12, 0.37, 0.02, 0.14
}
_cache = {}


def _twin(name):
    from pathlib import Path
    return Twin.load(str(Path(__file__).parents[1] / "data" / "twins" / f"{name}.npz"))


def run(name, vision_pose="imu", noise=False, **kw):
    """One solve per setting across this module (about 20 s each)."""
    key = (name, vision_pose, noise, tuple(sorted(kw.items())))
    if key not in _cache:
        _cache[key] = solve(_twin(name), vision_pose=vision_pose, seed=0, noise=noise, **kw)
    return _cache[key]


def lever(r):
    return np.array([c["lever_mm"] for c in r["cams"]])


def check_limits(r, lim):
    assert r["ok"]
    assert r["iterations"] > 0, "LM made no step (ok alone does not show that it solved)"
    assert r["rms_px"] < lim["rms_px"]
    assert abs(r["t_ga_ms"]) < lim["t_ga_ms"]
    assert np.abs(lever(r)).max() < lim["lever_mm"]
    assert max(abs(c["td_ms"]) for c in r["cams"]) < lim["td_ms"]
    if "tr_ms" in lim:
        assert max(abs(c["tr_ms"]) for c in r["cams"]) < lim["tr_ms"]


@pytest.mark.parametrize("name", [GS, RS])
def test_imu_factor_recovers_truth_without_noise(name):
    check_limits(run(name), LIMITS_NO_NOISE[name])


@pytest.mark.parametrize("name", [GS, RS])
def test_imu_factor_recovers_truth_with_noise(name):
    check_limits(run(name, noise=True), LIMITS_NOISE[name])


def test_t_ga_term_removes_the_gs_bias():
    """Without t_ga in the vision-IMU factor the GS estimate is pulled by about 1 ms."""
    assert run(GS, vision_imu_t_ga=False)["t_ga_ms"] < -0.7
    assert abs(run(GS)["t_ga_ms"]) < 0.15


@pytest.mark.parametrize("name,low,high", [(GS, -5.0, -2.5), (RS, -18.0, -11.0)])
def test_interp_factor_keeps_its_known_lever_bias(name, low, high):
    """The chord model reads lever z low (2026-11-10: GS -4.2, RS -14.2 mm); a change here is a
    change in the solver or the twin."""
    z = lever(run(name, vision_pose="interp"))[:, 2]
    assert np.all((z > low) & (z < high)), z


def test_aux_imu_is_recovered_without_noise():
    """An aux IMU added to the twin at known values comes back, so the multi-IMU rows can be trusted."""
    import dataclasses
    twin = _twin(GS)
    T = np.eye(4)
    T[:3, 3] = [0.1, -0.05, 0.02]
    aux = dict(name="aux", xi24=twin.truth["xi24"], T_IstoI=T.tolist(), td=0.005, t_ga=0.003,
               b_g=twin.truth["b_g"], b_a=twin.truth["b_a"], gyro_only=False)
    twin = dataclasses.replace(twin, truth={**twin.truth, "aux": [aux]}, aux_t_imu=[twin.t_imu.copy()])
    r = solve(twin, seed=0, noise=False)["aux"][0]
    assert np.linalg.norm(r["lever_mm"]) < 2.0
    assert abs(r["td_ms"]) < 0.1 and abs(r["t_ga_ms"]) < 0.5
