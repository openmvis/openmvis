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
"""Slow IMUs are upsampled before integration; fast ones are left exactly as they are."""
import numpy as np
import pytest

from mvis.imu.resample import upsample_imu


def _sig(hz=100.0, n=200):
    t = 10.0 + np.arange(n) / hz
    w = np.column_stack([np.sin(t), np.cos(2 * t), np.sin(3 * t)])
    a = np.column_stack([np.cos(t), np.sin(2 * t), 9.8 + np.cos(3 * t)])
    return t, w, a


@pytest.mark.parametrize("hz", [200.0, 199.0, 200.1, 500.0, 1000.0])
def test_fast_enough_is_the_same_arrays(hz):
    t, w, a = _sig(hz)
    out = upsample_imu(t, w, a, 200.0)
    assert out[0] is t and out[1] is w and out[2] is a


def test_off_is_the_same_arrays():
    t, w, a = _sig(50.0)
    assert upsample_imu(t, w, a, 0.0)[0] is t


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_100hz_gains_one_midpoint_and_keeps_real_samples(method):
    t, w, a = _sig(100.0)
    tn, wn, an = upsample_imu(t, w, a, 200.0, method)
    assert len(tn) == 2 * len(t) - 1
    assert np.all(np.diff(tn) > 0)
    np.testing.assert_allclose(np.diff(tn), 0.005, atol=1e-9)
    np.testing.assert_allclose(tn[::2], t)
    np.testing.assert_allclose(wn[::2], w, atol=1e-12)
    np.testing.assert_allclose(an[::2], a, atol=1e-12)


def test_rate_that_needs_more_than_one_new_sample():
    t, w, a = _sig(60.0)
    tn, _, _ = upsample_imu(t, w, a, 200.0)
    assert 1.0 / np.median(np.diff(tn)) >= 200.0
    assert len(tn) == 4 * (len(t) - 1) + 1


def test_linear_midpoint_is_the_average():
    t, w, a = _sig(100.0)
    _, wn, _ = upsample_imu(t, w, a, 200.0, "linear")
    np.testing.assert_allclose(wn[1], 0.5 * (w[0] + w[1]))


def test_cubic_recovers_a_smooth_signal_better_than_linear():
    t, w, a = _sig(100.0)
    tn, _, al = upsample_imu(t, w, a, 200.0, "linear")
    _, _, ac = upsample_imu(t, w, a, 200.0, "cubic")
    true = np.column_stack([np.cos(tn), np.sin(2 * tn), 9.8 + np.cos(3 * tn)])
    assert np.abs(ac - true).max() < 0.01 * np.abs(al - true).max()


def test_no_samples_inside_a_dropout():
    t, w, a = _sig(100.0)
    keep = np.r_[0:50, 80:200]
    tn, _, _ = upsample_imu(t[keep], w[keep], a[keep], 200.0)
    gap = (tn > t[49] + 1e-9) & (tn < t[80] - 1e-9)
    assert not gap.any()


def test_unknown_method_raises():
    t, w, a = _sig(100.0)
    with pytest.raises(ValueError):
        upsample_imu(t, w, a, 200.0, "quintic")


def test_pipeline_helper_upsamples_only_the_slow_imu_and_off_is_untouched():
    from types import SimpleNamespace as C
    from mvis.calibration.pipeline_cam_imu import _upsampled_imus
    t, w, a = _sig(400.0)
    ts, ws, as_ = _sig(100.0)
    aux = [dict(t_imu=ts, w_m=ws, a_m=as_, name="slow", seed=1), dict(t_imu=t, w_m=w, a_m=a, name="fast")]
    t2, _, _, out = _upsampled_imus(C(imu_min_rate_hz=0.0), t, w, a, aux)
    assert t2 is t and out is aux
    t2, _, _, out = _upsampled_imus(C(imu_min_rate_hz=200.0, imu_upsample_method="linear"), t, w, a, aux)
    assert t2 is t
    assert len(out[0]["t_imu"]) == 2 * len(ts) - 1 and out[0]["seed"] == 1
    assert out[1]["t_imu"] is t
    assert len(aux[0]["t_imu"]) == len(ts)
