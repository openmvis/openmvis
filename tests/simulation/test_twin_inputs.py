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
"""simulation.run.simulate_twin_inputs: the closed-loop and t_ga-study seed schemes reproduce the
inline code they replaced bit for bit; gt_vs_recovered keeps the old private name."""
import numpy as np
import gtsam
import pytest

from mvis.simulation.from_result import GroundTruth
from mvis.simulation.imu_measurements import imu_times, simulate_imu
from mvis.simulation.multicam import simulate_cameras, simulate_imus, traj_from_gt
from mvis.simulation.noise import MVIS_IMU_NOISE, ImuNoiseDensities
from mvis.simulation.run import (_gt_vs_recovered, gt_vs_recovered, simulate_twin_inputs,
                                 sims_to_cams)


def _gt(rng):
    """Two cameras (one rolling) looking +z at a board 2 m away, a wobbling 3 s path, a base and an
    aux IMU with random intrinsics."""
    g = np.linspace(-0.2, 0.2, 5)
    board = np.array([[x, y, 0.0] for x in g for y in g])
    ts = np.linspace(0.0, 3.0, 90)
    amp, fr = rng.uniform(0.02, 0.08, 3), rng.uniform(1.0, 3.0, 3)
    T_states = [gtsam.Pose3(gtsam.Rot3.Rodrigues(*(0.05 * np.sin(fr * t))),
                            gtsam.Point3(*(amp * np.sin(fr * t + 1.0)))).matrix().tolist() for t in ts]
    intr = [400., 400, 320, 240, 0, 0, 0, 0]
    cams = [dict(intr8=intr, T_CtoI=gtsam.Pose3().matrix().tolist(), td=0.0, readout=0.0,
                 is_fisheye=False, image_size=[640, 480]),
            dict(intr8=intr, T_CtoI=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.05, 0, 0)).matrix().tolist(),
                 td=float(rng.uniform(-0.01, 0.01)), readout=0.01, is_fisheye=False, image_size=[640, 480])]
    T_aux = gtsam.Pose3(gtsam.Rot3.Rz(0.1), gtsam.Point3(0.1, -0.05, 0)).matrix().tolist()
    imus = [dict(name="imu0", is_base=True, xi24=list(rng.normal(0, 1e-3, 24)), T_IstoI=None, td=0.0),
            dict(name="aux1", is_base=False, xi24=list(rng.normal(0, 1e-3, 24)), td=0.001, T_IstoI=T_aux)]
    gt = GroundTruth(cams=cams, imus=imus, b_g=[0, 0, 0], b_a=[0, 0, 0],
                     T_BinW=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 2)).matrix().tolist(),
                     t_states=ts.tolist(), T_states=T_states, imu_model="calyx", g_inW=[0, 0, -9.81])
    return gt, board


def _closed_loop_reference(gt, board, rates, sigma_px, seed, imu_hz, noise):
    """The inline body digital_twin_closed_loop and mc_repeatability.run_seed had."""
    sims = simulate_cameras(gt, board, rates=rates, sigma_px=sigma_px, seed=seed)
    cams = sims_to_cams(sims)
    t_imu = imu_times(float(gt.t_states[0]) + 0.5, float(gt.t_states[-1]) - 0.5, imu_hz)
    base, aux = simulate_imus(gt, t_imu, noise=noise, seed=seed + 100)
    return cams, base.t, base.w_m, base.a_m, aux


def _tga_reference(gt, board, noise, t_ga, seed):
    """The inline body tga_sim_recovery.run_one had (with its simulate_base_imu_tga)."""
    from mvis.imu import ImuModel, parse_imu_model
    sims = simulate_cameras(gt, board, rates=[20.0] * len(gt.cams), sigma_px=0.5, seed=seed)
    cams = sims_to_cams(sims)
    t_imu = imu_times(float(gt.t_states[0]) + 0.5, float(gt.t_states[-1]) - 0.5, 400.0)
    traj = traj_from_gt(gt)
    intr = ImuModel.from_tangent(np.asarray(gt.imus[0]['xi24'], float), parse_imu_model(gt.imu_model))
    g = np.asarray(gt.g_inW, float)
    nk = noise.sim_kwargs()
    A = simulate_imu(traj, t_imu, intr, g_inG=g, seed=seed + 50, **nk)
    B = simulate_imu(traj, t_imu, intr, g_inG=g, td=-t_ga, seed=seed + 51, **nk)
    return cams, A.t, A.w_m, B.a_m, []


def _assert_bitwise(a, b):
    def same(x, y):
        if isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                same(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for u, v in zip(x, y):
                same(u, v)
        elif isinstance(x, np.ndarray):
            assert x.dtype == np.asarray(y).dtype and x.shape == np.asarray(y).shape
            assert x.tobytes() == np.asarray(y).tobytes()
        else:
            assert x == y
    same(a, b)


@pytest.mark.parametrize("k", range(4))
def test_closed_loop_scheme_is_bitwise_the_old_inline_code(k):
    rng = np.random.default_rng(k)
    gt, board = _gt(rng)
    seed = int(rng.integers(0, 1000))
    noise = MVIS_IMU_NOISE if k % 2 == 0 else ImuNoiseDensities(sigma_w=2e-3, sigma_bg=1e-4,
                                                                 sigma_a=2e-2, sigma_ba=1e-3)
    rates = None if k == 0 else [float(rng.choice([10, 20, 30]))] * 2
    got = simulate_twin_inputs(gt, board, rates=rates, sigma_px=0.5, seed=seed, imu_hz=200.0 + 100 * k,
                               imu_noise=noise)
    _assert_bitwise(got, _closed_loop_reference(gt, board, rates, 0.5, seed, 200.0 + 100 * k, noise))
    assert len(got[4]) == 1 and all(len(c['t_cam']) > 10 for c in got[0])


@pytest.mark.parametrize("t_ga", [0.0, -2e-3, 1e-3, 5e-3])
def test_tga_scheme_is_bitwise_the_old_study_code(t_ga):
    rng = np.random.default_rng(int(abs(t_ga) * 1e4))
    gt, board = _gt(rng)
    noise = ImuNoiseDensities(sigma_w=1.6e-3, sigma_bg=2e-5, sigma_a=1.4e-2, sigma_ba=3e-4)
    seed = 1000 + 7 * int(rng.integers(0, 4))
    got = simulate_twin_inputs(gt, board, rates=[20.0] * 2, sigma_px=0.5, seed=seed, imu_hz=400.0,
                               imu_noise=noise, imu_seed=seed + 50, include_aux=False, t_ga=t_ga)
    _assert_bitwise(got, _tga_reference(gt, board, noise, t_ga, seed))


def test_t_ga_only_moves_the_accel_and_aux_follows_include_aux():
    gt, board = _gt(np.random.default_rng(9))
    a = simulate_twin_inputs(gt, board, imu_noise=None, t_ga=0.0)
    b = simulate_twin_inputs(gt, board, imu_noise=None, t_ga=0.004)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert not np.array_equal(a[3], b[3])
    assert len(a[4]) == 1 and simulate_twin_inputs(gt, board, include_aux=False)[4] == []
    aux_ref = simulate_imus(gt, a[1], noise=None, seed=100)[1]
    _assert_bitwise(a[4], aux_ref)


def test_gt_vs_recovered_keeps_the_private_alias():
    assert _gt_vs_recovered is gt_vs_recovered
