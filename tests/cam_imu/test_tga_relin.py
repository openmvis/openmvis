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
"""Re-integrating an IMU at its current t_ga once t_ga leaves a threshold around the last point.

The factors correct t_ga linearly around a point, like a bias. Around 0 that is too coarse for a
-6 ms offset (T265): the fix moves the point, not the factor's structure.
"""
import gtsam
import numpy as np
import pytest

from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.imu_factor_multi import ImuFactorACI_Multi_In_V
from mvis.calibration.pipeline_cam_imu import _slice_imu
from mvis.calibration.tga_relin import TgaRelinOptimizer, TgaStream, shifted_accel
from mvis.imu.imu_model import CALYX, create_imu_model

SIG = (1e-3, 1e-5, 1e-2, 1e-4)
G = np.array([0.0, 0.0, -9.81])
T0, T1 = 0.10, 0.20
KEY = gtsam.symbol("o", 0)


def _stream():
    t = np.arange(0.0, 0.4, 1 / 200.0)
    w = np.column_stack([0.4 * np.sin(5 * t), 0.3 * np.cos(4 * t), 0.2 * np.sin(3 * t)])
    a = np.column_stack([np.sin(9 * t), np.cos(7 * t), 9.8 + 0.5 * np.sin(6 * t)])
    return t, w, a


def _pre(t, w, a):
    p = ACIPreintegrator(*SIG, model=CALYX, avg_meas=True)
    p.reset(np.zeros(3), np.zeros(3))
    p.set_intrinsics(create_imu_model(CALYX))
    ts, ws, as_ = _slice_imu(t, w, a, T0, T1)
    p.feed_imu_interval(ts, ws, as_)
    return p


def _args():
    pose0 = gtsam.Pose3()
    pose1 = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.01, 0.02, 0.03), gtsam.Point3(0.01, 0.02, 0.03))
    vb = np.zeros(9)
    return pose0, pose1, vb, vb, np.zeros(24)


def test_default_point_is_zero_and_bitwise():
    t, w, a = _stream()
    f = ImuFactorACI_In.from_preintegrator(_pre(t, w, a), G)
    assert f.t_lin == 0.0
    e = f.whitened_error(*_args(), -0.002)
    f.t_lin = 0.0
    np.testing.assert_array_equal(e, f.whitened_error(*_args(), -0.002))


def test_refresh_equals_a_factor_built_on_the_shifted_accel():
    t, w, a = _stream()
    tau = -0.006
    f = ImuFactorACI_In.from_preintegrator(_pre(t, w, a), G)
    s = TgaStream(key=KEY, t=t, w=w, a=a, sigmas=SIG, model=CALYX)
    s.add(f, T0, T1)
    s.relinearize(tau)
    assert f.t_lin == tau and s.t_lin == tau
    fresh = ImuFactorACI_In.from_preintegrator(_pre(t, w, shifted_accel(t, a, tau)), G)
    # At the new point the linear term is zero, so t_ga = tau reads the shifted integration.
    np.testing.assert_allclose(f.whitened_error(*_args(), tau), fresh.whitened_error(*_args(), 0.0), atol=1e-12)
    # Around the point it moves linearly in t_ga - t_lin.
    np.testing.assert_allclose(f.whitened_error(*_args(), tau + 1e-3),
                               fresh.whitened_error(*_args(), 1e-3), atol=1e-12)


def test_refresh_keeps_a_closure_built_before_it():
    t, w, a = _stream()
    f = ImuFactorACI_In.from_preintegrator(_pre(t, w, a), G)
    fg = f.make_factor(gtsam.symbol("x", 0), gtsam.symbol("x", 1), gtsam.symbol("v", 0),
                       gtsam.symbol("v", 1), gtsam.symbol("i", 0), k_tga=KEY)
    v = gtsam.Values()
    p0, p1, vb0, vb1, xi = _args()
    v.insert(gtsam.symbol("x", 0), p0); v.insert(gtsam.symbol("x", 1), p1)
    v.insert(gtsam.symbol("v", 0), vb0); v.insert(gtsam.symbol("v", 1), vb1)
    v.insert(gtsam.symbol("i", 0), xi); v.insert(KEY, np.array([-0.006]))
    before = fg.error(v)
    s = TgaStream(key=KEY, t=t, w=w, a=a, sigmas=SIG, model=CALYX)
    s.add(f, T0, T1)
    s.relinearize(-0.006)
    assert fg.error(v) != before
    assert fg.error(v) == pytest.approx(0.5 * np.sum(f.whitened_error(*_args(), -0.006) ** 2))


def test_aux_factor_refresh_updates_the_inner_factor_and_end_rates():
    t, w, a = _stream()
    from mvis.calibration.factors.imu_calib import ImuCalib  # noqa: F401  (import check)
    p = _pre(t, w, a)
    f = ImuFactorACI_Multi_In_V.from_preintegrator(p, G, gtsam.Pose3(), 0.0, window_td=0.0)
    s = TgaStream(key=KEY, t=t, w=w, a=a, sigmas=SIG, model=CALYX)
    s.add(f, T0, T1)
    s.relinearize(-0.006)
    fresh = _pre(t, w, shifted_accel(t, a, -0.006))
    assert f.base.t_lin == -0.006
    np.testing.assert_allclose(f.base.D_p, fresh.D_p, atol=1e-15)
    np.testing.assert_allclose(f.w_inIs0, fresh.w_m_s, atol=1e-15)


class _Counting:
    """A stream without factors: records when the optimizer asks to move the point."""
    def __init__(self, key, threshold_moves=None):
        self.key, self.t_lin, self.moves = key, 0.0, []

    def relinearize(self, tau):
        self.moves.append(tau)
        self.t_lin = tau


def _prior_graph(target, sigma=1e-3):
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorVector(KEY, np.array([target]), gtsam.noiseModel.Isotropic.Sigma(1, sigma)))
    v = gtsam.Values()
    v.insert(KEY, np.zeros(1))
    return g, v


def test_no_move_inside_the_threshold():
    g, v = _prior_graph(-0.002)
    s = _Counting(KEY)
    TgaRelinOptimizer(g, v, gtsam.LevenbergMarquardtParams(), [s], 0.003).optimize()
    assert s.moves == []


def test_move_once_when_the_threshold_is_crossed():
    g, v = _prior_graph(-0.006)
    s = _Counting(KEY)
    opt = TgaRelinOptimizer(g, v, gtsam.LevenbergMarquardtParams(), [s], 0.003)
    r = opt.optimize()
    assert s.moves == pytest.approx([-0.006], abs=1e-6)
    assert r.atVector(KEY)[0] == pytest.approx(-0.006, abs=1e-6)
    assert opt.relinearizations and opt.relinearizations[0]["t_ga"] == pytest.approx(-0.006, abs=1e-6)


def test_moves_again_only_after_another_threshold():
    # Heavily damped first steps walk t_ga out in stages.
    g, v = _prior_graph(-0.010)
    s = _Counting(KEY)
    p = gtsam.LevenbergMarquardtParams()
    p.setlambdaInitial(5e6)
    p.setlambdaFactor(1.0)
    p.setMaxIterations(60)
    TgaRelinOptimizer(g, v, p, [s], 0.003).optimize()
    assert len(s.moves) >= 2
    pts = [0.0] + s.moves
    assert all(abs(b - a) > 0.003 for a, b in zip(pts[:-1], pts[1:]))


def test_stepping_matches_gtsam_optimize_when_nothing_moves():
    x = gtsam.symbol("x", 0)
    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorPose3(x, gtsam.Pose3(gtsam.Rot3.Rodrigues(0.3, -0.2, 0.1), gtsam.Point3(1, 2, 3)),
                                 gtsam.noiseModel.Isotropic.Sigma(6, 0.1)))
    g.add(gtsam.PriorFactorVector(KEY, np.array([-0.001]), gtsam.noiseModel.Isotropic.Sigma(1, 1e-3)))
    v = gtsam.Values()
    v.insert(x, gtsam.Pose3())
    v.insert(KEY, np.zeros(1))
    ref_opt = gtsam.LevenbergMarquardtOptimizer(g, v, gtsam.LevenbergMarquardtParams())
    ref = ref_opt.optimize()
    opt = TgaRelinOptimizer(g, v, gtsam.LevenbergMarquardtParams(), [_Counting(KEY)], 0.003)
    res = opt.optimize()
    assert opt.iterations() == ref_opt.iterations()
    assert opt.error() == ref_opt.error()
    assert res.atPose3(x).equals(ref.atPose3(x), 0.0)


def test_segment_refresh_equals_a_segment_built_on_the_shifted_accel():
    from mvis.calibration.factors.imu_segment import ImuSegment
    t, w, a = _stream()
    tau = -0.006
    seg = ImuSegment(t, w, a, CALYX)
    fresh = ImuSegment(t, w, shifted_accel(t, a, tau), CALYX)
    s = TgaStream(key=KEY, t=t, w=w, a=a, sigmas=SIG, model=CALYX)
    s.segments = [seg]
    s.relinearize(tau)
    assert seg.t_lin == tau
    np.testing.assert_array_equal(seg.a, fresh.a)


def test_vision_factor_passes_only_what_the_segment_lacks():
    from mvis.calibration.factors.imu_segment import ImuSegment
    t, w, a = _stream()
    seg = ImuSegment(t, w, a, CALYX)

    class F:                # the one method under test, on a stand-in factor
        tga, seg_ = True, seg
    from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor as C
    f = F()
    f.seg = seg
    assert C._tga_value(f, np.array([-0.002])) == -0.002     # untouched until the point moves
    seg.relinearize_tga(shifted_accel(t, a, -0.006), -0.006)
    assert C._tga_value(f, np.array([-0.007])) == pytest.approx(-0.001)
    f.tga = False
    assert C._tga_value(f, np.array([-0.007])) is None


def _walk(target, lam, iters=60):
    g, v = _prior_graph(target)
    s = _Counting(KEY)
    p = gtsam.LevenbergMarquardtParams()
    p.setlambdaInitial(lam)
    p.setlambdaFactor(1.0)
    p.setMaxIterations(iters)
    opt = TgaRelinOptimizer(g, v, p, [s], 0.003)
    opt.optimize()
    return s, opt


def test_no_move_in_the_first_two_iterations():
    s, opt = _walk(-0.010, 3e5)
    assert opt.relinearizations
    assert all(r["iteration"] >= 3 for r in opt.relinearizations)


def test_a_transient_excursion_in_the_first_steps_never_moves_the_point():
    # Reaches past the threshold only on step 1 or 2, then the check sees it back inside.
    g, v = _prior_graph(-0.002)
    s = _Counting(KEY)
    TgaRelinOptimizer(g, v, gtsam.LevenbergMarquardtParams(), [s], 0.003).optimize()
    assert s.moves == []


def test_a_solve_converging_inside_the_skip_still_ends_at_a_fresh_point():
    g, v = _prior_graph(-0.006)
    s = _Counting(KEY)
    opt = TgaRelinOptimizer(g, v, gtsam.LevenbergMarquardtParams(), [s], 0.003)
    opt.optimize()
    assert s.moves == pytest.approx([-0.006], abs=1e-6)
    assert abs(opt._opt.values().atVector(KEY)[0] - s.t_lin) <= 0.003
