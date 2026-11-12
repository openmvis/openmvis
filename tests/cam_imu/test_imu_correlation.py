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
"""mvis.analysis.imu_correlation: solver-free gyro->accel offset from the raw streams."""
import numpy as np
import pytest

from mvis.analysis.imu_correlation import bandpass, peak, tier1_curve, tier1_lever_curve, tier2_curve


def test_bandpass_keeps_the_band_and_removes_dc_and_high():
    fs = 200.0
    t = np.arange(0, 20, 1 / fs)
    inband, low, high = np.sin(2 * np.pi * 5 * t), np.ones_like(t), np.sin(2 * np.pi * 60 * t)
    y = bandpass(inband + low + high, fs, 1.0, 20.0)
    mid = len(t) // 4, 3 * len(t) // 4
    assert np.abs(y[mid[0]:mid[1]] - inband[mid[0]:mid[1]]).max() < 0.02


def test_bandpass_caps_the_upper_edge_below_nyquist():
    x = np.random.default_rng(0).normal(size=(500, 3))
    assert np.array_equal(bandpass(x, 100.0, 1.0, 1000.0), bandpass(x, 100.0, 1.0, 45.0))


def test_peak_refines_an_interior_maximum_with_a_parabola():
    lags = np.arange(-5, 6) * 0.1
    lag, val = peak(lags, -(lags - 0.13) ** 2)
    assert np.isclose(lag, 0.13) and val == -(0.1 - 0.13) ** 2


def test_peak_at_the_edge_is_not_refined():
    lags = np.array([0.0, 1.0, 2.0])
    assert peak(lags, np.array([3.0, 2.0, 1.0])) == (0.0, 3.0)
    assert peak(lags, np.array([1.0, 2.0, 3.0])) == (2.0, 3.0)


def _rotating_gravity(t_ga, fs=400.0, seconds=30.0):
    """Body-frame gravity under a smooth 3-axis rotation (da/dt = -w x a), with the accel stamps shifted
    by t_ga relative to the gyro's."""
    t = np.arange(0, seconds, 1 / fs)
    w = np.column_stack([1.5 * np.sin(2 * np.pi * 1.3 * t), 1.1 * np.sin(2 * np.pi * 2.1 * t + 1),
                         0.9 * np.sin(2 * np.pi * 3.1 * t + 2)])
    a = np.empty_like(w)
    a[0] = [0.0, 0.0, 9.81]
    for k in range(len(t) - 1):
        a[k + 1] = a[k] - np.cross(0.5 * (w[k] + w[k + 1]), a[k]) / fs   # midpoint, no half-sample lag
    return t, w, t + t_ga, a


def test_tier1_curve_peaks_at_the_injected_offset():
    fs = 400.0
    t_ga = 0.005
    tg, w, ta, a = _rotating_gravity(t_ga, fs)
    lags = np.arange(-15, 15.01, 1.0) * 1e-3
    c = tier1_curve(tg, w, ta, a, lags, (0.5, 20.0), fs)
    assert c.shape == lags.shape and c.max() > 0.9
    assert abs(peak(lags, c)[0] + t_ga) < 1e-3   # curve lag: gyro at accel stamp + lag, so -t_ga here


def test_tier2_curve_is_a_correlation_per_lag():
    fs = 200.0
    tg, w, ta, a = _rotating_gravity(0.0, fs, seconds=20.0)
    lags = np.array([-0.05, 0.0, 0.05])
    c = tier2_curve(tg, w, ta, a, lags, (0.5, 20.0), fs)
    assert c.shape == (3,) and np.all(np.abs(c) <= 1.0)


def _old_lever_run(csv, lo, hi, lags_ms, t0=None, t1=None):
    """studies/tga_correlation_lever.run as it was before the move (verbatim arithmetic)."""
    from scipy import signal
    d = np.genfromtxt(csv, delimiter=",", skip_header=1)
    t = d[:, 0] * 1e-9; t -= t[0]; w = d[:, 1:4]; a = d[:, 4:7]
    if t0 is not None:
        m = (t >= t0) & (t <= t1); t, w, a = t[m], w[m], a[m]
    fs = 1.0 / np.median(np.diff(t))
    tt = np.arange(t[0], t[-1], 1.0 / fs)
    a_u = np.column_stack([np.interp(tt, t, a[:, k]) for k in range(3)])
    sos = signal.butter(2, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    a_f = signal.sosfiltfilt(sos, a_u, axis=0)
    dadt = np.gradient(a_f, 1.0 / fs, axis=0)
    out = []
    for lag_ms in lags_ms:
        lag = lag_ms * 1e-3
        w_u = np.column_stack([np.interp(tt + lag, t, w[:, k]) for k in range(3)])
        wd = np.gradient(w_u, 1.0 / fs, axis=0)
        base = -np.cross(w_u, a_u)
        cols = []
        for e in np.eye(3):
            L = np.cross(wd, e[None, :] * np.ones_like(w_u)) + np.cross(w_u, np.cross(w_u, e[None, :] * np.ones_like(w_u)))
            L = signal.sosfiltfilt(sos, L, axis=0)
            cols.append(np.gradient(L, 1.0 / fs, axis=0))
        base_f = signal.sosfiltfilt(sos, base, axis=0)
        y = (dadt - base_f).reshape(-1)
        A = np.column_stack([c.reshape(-1) for c in cols])
        r_fit, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        model = base_f + sum(r_fit[j] * cols[j] for j in range(3))
        num = np.sum(dadt * model); den = np.sqrt(np.sum(dadt**2) * np.sum(model**2))
        out.append((lag_ms, num / den, r_fit.copy()))
    arr = np.array([(l, c) for l, c, _ in out])
    i = int(arr[:, 1].argmax())
    if 0 < i < len(arr) - 1:
        y0, y1, y2 = arr[i - 1, 1], arr[i, 1], arr[i + 1, 1]
        fr = (y0 - y2) / (2 * (y0 - 2 * y1 + y2)) if (y0 - 2 * y1 + y2) != 0 else 0
        pk = arr[i, 0] + fr * (arr[1, 0] - arr[0, 0])
    else:
        pk = arr[i, 0]
    return pk, arr[i, 1], out[i][2]


def _lever_csv(tmp_path, t_ga, lever, fs=400.0, seconds=20.0, seed=0):
    """A rotating-gravity stream plus the translation accel of an IMU `lever` (m) off the rotation
    centre and a little noise, accel shifted by t_ga, written as a EuRoC imu csv."""
    tg, w, ta, a = _rotating_gravity(t_ga, fs, seconds)
    wd = np.gradient(w, 1 / fs, axis=0)
    r = np.asarray(lever, float)[None, :] * np.ones_like(w)
    a = a + np.cross(wd, r) + np.cross(w, np.cross(w, r))
    a = a + np.random.default_rng(seed).normal(0, 0.01, a.shape)
    # one stream, accel delayed: sample the shifted accel back on the gyro stamps
    a_on_g = np.column_stack([np.interp(tg, ta, a[:, k]) for k in range(3)])
    p = tmp_path / "imu.csv"
    rows = np.column_stack([np.round(tg * 1e9 + 1.5e12), w, a_on_g])
    np.savetxt(p, rows, delimiter=",", header="t,wx,wy,wz,ax,ay,az", comments="",
               fmt=["%d"] + ["%.9f"] * 6)
    return p


def _lever_inputs(p):
    d = np.genfromtxt(p, delimiter=",", skip_header=1)
    t = d[:, 0] * 1e-9
    t -= t[0]
    return t, d[:, 1:4], d[:, 4:7], 1.0 / np.median(np.diff(t))


@pytest.mark.parametrize("t_ga, lever, band", [(0.004, (0.1, -0.05, 0.2), (0.5, 10.0)),
                                               (-0.002, (0.0, 0.0, 0.0), (2.0, 25.0)),
                                               (0.0, (-0.3, 0.1, 0.05), (20.0, 80.0))])
def test_tier1_lever_curve_is_the_old_study_run_bitwise(tmp_path, t_ga, lever, band):
    p = _lever_csv(tmp_path, t_ga, lever)
    lags_ms = np.arange(-15, 15.01, 0.5)
    pk_old, c_old, r_old = _old_lever_run(p, band[0], band[1], lags_ms)
    t, w, a, fs = _lever_inputs(p)
    curve, r_fits = tier1_lever_curve(t, w, t, a, lags_ms * 1e-3, band, fs)
    pk, c = peak(lags_ms, curve)                     # ms, as the study refined it
    assert pk == pk_old and c == c_old
    assert np.array_equal(r_fits[int(np.argmax(curve))], r_old)
    pk_s = peak(lags_ms * 1e-3, curve)[0] * 1e3      # seconds, as the CLI refines it
    assert abs(pk_s - pk_old) < 0.05


def test_tier1_lever_curve_peaks_near_the_offset_and_fits_the_lever(tmp_path):
    lags = np.arange(-10, 10.01, 0.5) * 1e-3
    t, w, a, fs = _lever_inputs(_lever_csv(tmp_path, 0.004, (0.0, 0.0, 0.0)))
    curve, r_fits = tier1_lever_curve(t, w, t, a, lags, (0.5, 10.0), fs)
    assert curve.shape == lags.shape and r_fits.shape == (len(lags), 3)
    assert abs(peak(lags, curve)[0] + 0.004) < 5e-4
    lever = (0.1, -0.05, 0.2)
    t, w, a, fs = _lever_inputs(_lever_csv(tmp_path, 0.004, lever))
    curve, r_fits = tier1_lever_curve(t, w, t, a, lags, (0.5, 10.0), fs)
    assert np.allclose(r_fits[int(np.argmax(curve))], lever, atol=0.01)


@pytest.mark.parametrize("t_ga, lever, band, window", [
    (0.004, (0.1, -0.05, 0.2), (0.5, 10.0), (None, None)),
    (-0.002, (0.0, 0.0, 0.0), (2.0, 25.0), (None, None)),
    (0.0, (-0.3, 0.1, 0.05), (20.0, 80.0), (None, None)),
    (0.003, (0.05, 0.2, -0.1), (0.5, 10.0), (3.0, 15.0)),
    (-0.006, (0.2, 0.0, 0.1), (2.0, 25.0), (0.0, 8.5))])
def test_studies_tga_correlation_lever_shim_is_the_old_run_bitwise(tmp_path, t_ga, lever, band,
                                                                   window):
    """studies/tga_correlation_lever.run stays importable for external callers and returns
    the old (pk, corr, r_fit) bit for bit (maintainer-only: needs studies/, see conftest)."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "studies" / "tga_correlation_lever.py"
    spec = importlib.util.spec_from_file_location("tga_correlation_lever", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = _lever_csv(tmp_path, t_ga, lever, seed=abs(hash(lever)) % 1000)
    lags_ms = np.arange(-15, 15.01, 0.5)
    old = _old_lever_run(p, band[0], band[1], lags_ms, *window)
    new = mod.run(p, band[0], band[1], lags_ms, *window)
    assert np.asarray(new[0]).tobytes() == np.asarray(old[0]).tobytes()
    assert np.asarray(new[1]).tobytes() == np.asarray(old[1]).tobytes()
    assert new[2].tobytes() == old[2].tobytes()
