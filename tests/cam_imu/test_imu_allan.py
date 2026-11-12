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
"""mvis.imu.allan: Allan deviation and static-span gravity from raw IMU streams."""
import numpy as np

from mvis.imu.allan import GRAV, allan_dev, read_imu, static_gravity


def _write_euroc(path, t, w, a):
    rows = np.column_stack([np.asarray(t) * 1e9, w, a])
    np.savetxt(path, rows, delimiter=",", header="t,wx,wy,wz,ax,ay,az")


def test_read_imu_converts_ns_and_splits_columns(tmp_path):
    p = tmp_path / "data.csv"
    w, a = np.arange(6.0).reshape(2, 3), 10 + np.arange(6.0).reshape(2, 3)
    _write_euroc(p, [1.0, 2.0], w, a)
    t, w2, a2 = read_imu(p)
    assert np.allclose(t, [1.0, 2.0]) and np.allclose(w2, w) and np.allclose(a2, a)


def test_allan_dev_of_white_noise_is_sigma_over_sqrt_tau():
    """White rate noise of density N: sigma(tau) = N / sqrt(tau)."""
    fs, density = 100.0, 0.01
    x = np.random.default_rng(0).normal(0.0, density * np.sqrt(fs), 400_000)
    taus = np.array([0.1, 1.0, 10.0])
    assert np.allclose(allan_dev(x, fs, taus), density / np.sqrt(taus), rtol=0.05)


def test_allan_dev_is_zero_for_a_constant_and_nan_out_of_range():
    d = allan_dev(np.full(1000, 2.5), 100.0, np.array([0.001, 1.0, 100.0]))
    assert np.isnan(d[0]) and d[1] < 1e-9 and np.isnan(d[2])   # m < 1, ok, 2m >= n


def test_static_gravity_uses_only_static_windows(tmp_path):
    fs = 100
    n = 10 * fs
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.001, (n, 3))
    w[:2 * fs] = rng.normal(0, 1.0, (2 * fs, 3))               # moving for 2 s
    a = np.tile([0.0, 0.0, 9.7], (n, 1))
    a[:2 * fs] = [0.0, 0.0, 30.0]                              # must not count
    _write_euroc(tmp_path / "data.csv", t, w, a)
    out = static_gravity(tmp_path)
    assert out["data.raw.csv"] is None                         # absent file
    assert out["data.csv"]["static_s"] == 8
    assert np.isclose(out["data.csv"]["grav_median"], 9.7)
    assert out["data.csv"]["pct"] == round((9.7 / GRAV - 1) * 100, 2)


def test_static_gravity_needs_five_static_seconds(tmp_path):
    fs = 100
    n = 10 * fs
    t = np.arange(n) / fs
    w = np.random.default_rng(2).normal(0, 1.0, (n, 3))
    _write_euroc(tmp_path / "data.csv", t, w, np.tile([0.0, 0.0, 9.8], (n, 1)))
    assert static_gravity(tmp_path)["data.csv"] is None
