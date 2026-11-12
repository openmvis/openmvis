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
"""Speed-up K8 (init, BITWISE) gates — every comparison is `np.array_equal`
against the frozen S0 copies (`tests/_reference/ref_diagnostics`, `ref_aci`)
or against the pre-K8 call pattern:

  * K8a `_slice_imu` (searchsorted) == the mask copy on random windows incl.
    values exactly equal to stamps, duplicated stamps, empty / inverted
    windows, and the non-monotonic fallback;
  * K8b `aci.correct_measurements` == the inline `feed_imu_single` expressions;
    `aci.state_step` walk == `ref_aci.ACIPreintegrator` D_R / D_v / D_p / Dt
    (incl. the `d_t <= 0` skip);
  * K8c `_densify_velocities` (shared prefix, copied snapshots) == the frozen
    per-state copy on the `test_aux_decimation` fixture with states on a
    keyframe, before the first keyframe, in the last interval, between raw
    samples and exactly on a raw sample; one state alone == its row in the
    full batch; non-monotonic stream (fallback) == the copy;
  * K8d the seed is the same with and without `_precomp`
    (`refine_time_offset(return_precomp=True)` → `estimate_rotation_gyro`),
    the whole `rep_rot` dict included;
  * K8e `ACIPreintegrator(track_intrinsics=False)` gives bitwise `D_*`, `J_b`
    and `P_meas`; `linear_alignment` on it == with the full tracking.

Runs at any BLAS thread count: everything here is 3x3 / 9x9 scalar numpy.
"""

from __future__ import annotations

import numpy as np
import gtsam
import pytest

from mvis.imu import CALYX, KALIBR, RPNG, ImuModel, create_imu_model
from mvis.calibration.factors import aci as A
from mvis.calibration.factors.aci import (ACIPreintegrator, _IntrPack,
                                          correct_measurements, state_step)
from mvis.calibration import cam_imu_init as CI
from mvis.calibration.cam_imu_init import (
    CamImuSeed, _slice_imu, _is_monotonic, refine_time_offset,
    estimate_rotation_gyro, estimate_time_offset_coarse, initialize_cam_imu,
    linear_alignment, pnp_pose_track, _stride_for_baseline)
from mvis.calibration import pipeline_cam_imu as P
from mvis.calibration.pipeline_cam_imu import CamImuCalibConfig, _decimate

from tests._reference import ref_aci, ref_diagnostics
from tests.helpers.speedup import imu_signal, imu_intr, _BG, _BA, _G
from tests.helpers.imu_signals import _make_sequence
from tests.helpers.speedup import same_as_oracle


def _eq(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


# ---------------------------------------------------------------------------
# K8a — _slice_imu
# ---------------------------------------------------------------------------
def _windows(rng, t, n=300):
    """(t0, t1) pairs: exact stamps, stamps ± tiny, before / after the stream,
    t0 == t1, t1 < t0, and plain random."""
    out = []
    pick = lambda: float(rng.choice(t))
    for _ in range(n):
        kind = rng.integers(0, 8)
        if kind == 0:
            t0, t1 = pick(), pick()
        elif kind == 1:
            t0, t1 = pick() + 1e-12, pick() - 1e-12
        elif kind == 2:
            t0, t1 = t[0] - 0.1, pick()
        elif kind == 3:
            t0, t1 = pick(), t[-1] + 0.1
        elif kind == 4:
            t0 = pick(); t1 = t0
        elif kind == 5:
            t0, t1 = pick() + 0.01, pick() - 0.01             # may be inverted
        elif kind == 6:
            t0, t1 = t[0], t[-1]
        else:
            t0, t1 = sorted(rng.uniform(t[0] - 0.05, t[-1] + 0.05, 2))
        out.append((t0, t1))
    return out


def test_slice_imu_matches_oracle_random_and_on_stamps():
    rng = np.random.default_rng(80)
    t, w, a = imu_signal()
    # duplicated stamps keep the stream monotonic (non-strict) and exercise the
    # 'right' / 'left' sides on ties
    t = t.copy(); t[40] = t[39]; t[41] = t[39]
    assert _is_monotonic(t)
    n_on = 0
    for t0, t1 in _windows(rng, t):
        n_on += int(np.any(t == t0) or np.any(t == t1))
        live = _slice_imu(t, w, a, t0, t1)
        ref = ref_diagnostics._slice_imu(t, w, a, t0, t1)
        for x, y in zip(live, ref):
            assert _eq(x, y), (t0, t1)
    assert n_on > 50


def test_slice_imu_non_monotonic_keeps_mask():
    rng = np.random.default_rng(81)
    t, w, a = imu_signal()
    t = t.copy(); t[[10, 70]] = t[[70, 10]]            # two stamps out of order
    assert not _is_monotonic(t)
    for t0, t1 in _windows(rng, t, 60):
        live = _slice_imu(t, w, a, t0, t1)
        ref = ref_diagnostics._slice_imu(t, w, a, t0, t1)
        for x, y in zip(live, ref):
            assert _eq(x, y), (t0, t1)
    tn = t.copy(); tn[5] = np.nan
    assert not _is_monotonic(tn)


# ---------------------------------------------------------------------------
# K8b — correct_measurements / state_step
# ---------------------------------------------------------------------------
def _inline_correction(pack, b_g, b_a, w_m_0, a_m_0, w_m_1, a_m_1, avg_meas):
    """The pre-K8 `feed_imu_single` lines, verbatim."""
    Tg = pack.Tg
    Ma, Mw = pack.Ma, pack.Mw
    a0_ina = a_m_0 - b_a
    a1_ina = a_m_1 - b_a
    a0_inI = Ma @ a0_ina
    a1_inI = Ma @ a1_ina
    w0_inw = w_m_0 - b_g - Tg @ a0_inI
    w0_inI = Mw @ w0_inw
    w1_inw = w_m_1 - b_g - Tg @ a1_inI
    w1_inI = Mw @ w1_inw
    a_ina, a_hat = a0_ina, a0_inI
    w_inw, w_hat = w0_inw, w0_inI
    if avg_meas:
        a_ina = (a0_ina + a1_ina) / 2.0
        a_hat = (a0_inI + a1_inI) / 2.0
        w_inw = (w0_inw + w1_inw) / 2.0
        w_hat = (w0_inI + w1_inI) / 2.0
    return a_ina, a_hat, w_inw, w_hat


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
@pytest.mark.parametrize("avg_meas", [True, False])
def test_correct_measurements_matches_inline(model, avg_meas):
    rng = np.random.default_rng(82)
    for intr in (create_imu_model(model), imu_intr(model, rng)):
        pack = _IntrPack(intr)
        for _ in range(50):
            args = (rng.normal(size=3) * 0.01, rng.normal(size=3) * 0.1,
                    rng.normal(size=3), rng.normal(size=3) + [0, 9.8, 0],
                    rng.normal(size=3), rng.normal(size=3) + [0, 9.8, 0])
            live = correct_measurements(pack, *args, avg_meas)
            ref = _inline_correction(pack, *args, avg_meas)
            for x, y in zip(live, ref):
                assert _eq(x, y)


def _walk(t, w, a, b_g, b_a, pack):
    D_R, D_v, D_p, Dt = np.eye(3), np.zeros(3), np.zeros(3), 0.0
    for i in range(len(t) - 1):
        _a_ina, a_hat, _w_inw, w_hat = correct_measurements(
            pack, b_g, b_a, w[i], a[i], w[i + 1], a[i + 1], True)
        D_R, D_v, D_p, Dt = state_step(D_R, D_v, D_p, Dt, t[i + 1] - t[i], w_hat, a_hat)
    return D_R, D_v, D_p, Dt


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
def test_state_step_walk_matches_oracle_preintegrator(model):
    rng = np.random.default_rng(83)
    t, w, a = _make_sequence(seed=3)
    # a duplicated stamp (d_t == 0 → the step is skipped on both sides) and a
    # near-still stretch so integrate_R's small branches are on the walk
    t = t.copy(); t[12] = t[11]
    w = w.copy(); w[20:25] = 1e-9
    bg = np.array([0.01, -0.02, 0.015]); ba = np.array([0.1, -0.05, 0.2])
    for intr in (create_imu_model(model), imu_intr(model, rng)):
        ref = ref_aci.ACIPreintegrator(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=True)
        ref.reset(bg, ba)
        ref.set_intrinsics(intr)
        ref.feed_imu_interval(t, w, a)
        D_R, D_v, D_p, Dt = _walk(t, w, a, bg, ba, _IntrPack(intr))
        assert _eq(D_R, ref.D_R) and _eq(D_v, ref.D_v) and _eq(D_p, ref.D_p)
        assert Dt == ref.Dt


def test_state_step_skips_non_positive_dt_and_never_mutates():
    rng = np.random.default_rng(84)
    D_R = gtsam.Rot3.Expmap(rng.normal(size=3)).matrix()
    D_v, D_p, Dt = rng.normal(size=3), rng.normal(size=3), 0.37
    snap = (D_R.copy(), D_v.copy(), D_p.copy())
    w_hat, a_hat = rng.normal(size=3), rng.normal(size=3)
    for d_t in (0.0, -1e-3):
        out = state_step(D_R, D_v, D_p, Dt, d_t, w_hat, a_hat)
        assert out[0] is D_R and out[1] is D_v and out[2] is D_p and out[3] == Dt
    out = state_step(D_R, D_v, D_p, Dt, 2.5e-3, w_hat, a_hat)
    assert out[3] == Dt + 2.5e-3
    for x, y in zip((D_R, D_v, D_p), snap):
        assert _eq(x, y)                                   # inputs untouched
    assert all(o is not i for o, i in zip(out[:3], (D_R, D_v, D_p)))


# ---------------------------------------------------------------------------
# K8e — track_intrinsics=False
# ---------------------------------------------------------------------------
_STATE_KEYS = ("D_R", "D_v", "D_p", "Dt", "J_R_bg", "J_R_ba", "J_p_bg", "J_p_ba",
               "J_v_bg", "J_v_ba", "P_meas", "a_m_s", "a_m_e", "w_m_s", "w_m_e")


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
@pytest.mark.parametrize("avg_meas", [True, False])
def test_track_intrinsics_false_is_bitwise_on_state_bias_and_covariance(model, avg_meas):
    rng = np.random.default_rng(85)
    t, w, a = _make_sequence(seed=5)
    bg = np.array([0.01, -0.02, 0.015]); ba = np.array([0.1, -0.05, 0.2])
    for intr in (create_imu_model(model), imu_intr(model, rng)):
        outs = []
        for cls, kw in ((ACIPreintegrator, dict(track_intrinsics=False)),
                        (ACIPreintegrator, {}), (ref_aci.ACIPreintegrator, {})):
            pre = cls(0.01, 0.001, 0.02, 0.002, model=model, avg_meas=avg_meas, **kw)
            pre.reset(bg, ba)
            pre.set_intrinsics(intr)
            pre.feed_imu_interval(t, w, a)
            outs.append(pre)
        lite, full, ref = outs
        for k in _STATE_KEYS:
            assert _eq(getattr(lite, k), getattr(full, k)), k
            assert same_as_oracle(k, getattr(full, k), getattr(ref, k)), k
        for k in ("J_R_in", "J_p_in", "J_v_in", "J_R_tga", "J_p_tga", "J_v_tga"):
            assert not np.any(getattr(lite, k))            # stays at reset zeros
        for k in ("J_p_in", "J_v_in", "J_p_tga", "J_v_tga"):
            assert np.any(getattr(full, k))                # ...while full tracks it
            # (J_R_tga is exactly 0 with Tg = 0: the gyro row only sees a-dot through Tg)
    assert ACIPreintegrator(0.01, 0.001, 0.02, 0.002).track_intrinsics is True


# ---------------------------------------------------------------------------
# fixtures: the test_aux_decimation scenario (1 cam + base IMU) and the
# test_cam_imu_init one (40 s window)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aux_scenario():
    from tests.helpers.sim_scenarios import _make_aux_scenario, _RADTAN
    sd = _make_aux_scenario()
    cfg = CamImuCalibConfig(estimate_t_ga=False, state_hz=0.0, sigma_px=0.5,
                            huber_px=2.0, imu_intrinsics="pinned")
    t_kept, T_kept, _ = pnp_pose_track(sd['t_cam'], list(sd['cam'].ids),
                                       list(sd['cam'].uv), sd['board_pts'], _RADTAN)
    seed = initialize_cam_imu(t_kept, T_kept, sd['s'].t, sd['s'].w_m, sd['s'].a_m,
                              **cfg.init_kwargs)
    return dict(sd=sd, cfg=cfg, t_kept=np.asarray(t_kept, float), T_kept=T_kept, seed=seed)


@pytest.fixture(scope="module")
def init_scenario():
    from tests.helpers.sim_scenarios import scenario_for_init as _scenario, data_for_init as _data, _RADTAN
    sc = _scenario()
    t_cam, cam, s = _data(sc, win=12.0)
    pts_B = np.asarray(sc.board.all_points_3d(), float)
    t_kept, poses, _n = pnp_pose_track(t_cam, cam.ids, cam.uv, pts_B, _RADTAN)
    return dict(t_kept=np.asarray(t_kept, float), poses=poses, s=s)


# ---------------------------------------------------------------------------
# K8c — _densify_velocities
# ---------------------------------------------------------------------------
def _edge_states(sx):
    """The plan's K8 cases on top of the pipeline's own decimated states."""
    seed, s = sx['seed'], sx['sd']['s']
    t_imu = np.asarray(s.t, float)
    t_key = np.asarray(seed.t_key, float)
    keep = _decimate(sx['t_kept'], sx['cfg'].state_hz)
    base = sx['t_kept'][keep]
    k_on = int(np.searchsorted(t_imu, t_key[0])) + 7          # a raw sample after key 0
    k_mid = int(np.searchsorted(t_imu, t_key[1])) + 3
    dt_imu = float(np.median(np.diff(t_imu)))
    extra = np.array([
        t_key[1],                          # exactly on a keyframe
        t_key[2] + 5e-10,                  # within the 1e-9 snap
        t_key[0] - 0.1,                    # before the first keyframe
        t_key[-1] + 0.2,                   # in the last (open) interval
        t_imu[k_mid] + 0.3 * dt_imu,       # between two raw samples
        t_imu[k_on],                       # exactly on a raw sample
        t_imu[k_on + 1],                   # ...and the next one (prefix reuse)
        t_imu[k_on] + 1e-12,               # just past a raw sample
        t_imu[k_on] - 1e-12,               # just before one
        t_key[1] + 1e-9,                   # first state past the snap window
    ])
    t_states = np.sort(np.concatenate([base, extra]))
    assert np.any(np.isin(t_states, t_imu))                  # on-sample cases present
    assert np.any(t_states < t_key[0]) and np.any(t_states > t_key[-1])
    return t_states


def test_densify_matches_oracle_on_aux_decimation_fixture(aux_scenario):
    sx = aux_scenario
    s, seed, cfg = sx['sd']['s'], sx['seed'], sx['cfg']
    t_states = _edge_states(sx)
    assert _is_monotonic(np.asarray(s.t, float))
    live = P._densify_velocities(t_states, None, seed, s.t, s.w_m, s.a_m, cfg, _G)
    ref = ref_diagnostics._densify_velocities(t_states, None, seed, s.t, s.w_m, s.a_m, cfg, _G)
    assert _eq(live, ref)
    assert np.isfinite(live).all()
    # the per-state oracle path (_propagate_state) is itself unchanged
    for i in (3, 11, 25):
        Ta, va = P._propagate_state(seed.T_ItoW[0], seed.v_inW[0], seed.t_key[0], t_states[i],
                                    s.t, s.w_m, s.a_m, seed.b_g, seed.b_a, cfg, _G)
        Tb, vb = ref_diagnostics._propagate_state(seed.T_ItoW[0], seed.v_inW[0], seed.t_key[0],
                                                  t_states[i], s.t, s.w_m, s.a_m, seed.b_g,
                                                  seed.b_a, cfg, _G)
        assert _eq(Ta.matrix(), Tb.matrix()) and _eq(va, vb)


def test_densify_one_state_alone_equals_its_row_in_the_batch(aux_scenario):
    """The shared prefix must not leak between states: each state alone (its
    own prefix) gives the same v as the full batch, in any order."""
    sx = aux_scenario
    s, seed, cfg = sx['sd']['s'], sx['seed'], sx['cfg']
    t_states = _edge_states(sx)
    full = P._densify_velocities(t_states, None, seed, s.t, s.w_m, s.a_m, cfg, _G)
    rng = np.random.default_rng(86)
    perm = rng.permutation(len(t_states))
    shuffled = P._densify_velocities(t_states[perm], None, seed, s.t, s.w_m, s.a_m, cfg, _G)
    assert _eq(shuffled, full[perm])
    for i in rng.choice(len(t_states), 12, replace=False):
        one = P._densify_velocities(t_states[i:i + 1], None, seed, s.t, s.w_m, s.a_m, cfg, _G)
        assert _eq(one[0], full[i]), i


def test_densify_non_monotonic_stream_takes_the_per_state_path(aux_scenario, monkeypatch):
    sx = aux_scenario
    s, seed, cfg = sx['sd']['s'], sx['seed'], sx['cfg']
    t_states = _edge_states(sx)[::5]
    t = np.asarray(s.t, float).copy()
    i0 = int(np.searchsorted(t, seed.t_key[0])) + 2
    t[[i0, i0 + 1]] = t[[i0 + 1, i0]]                      # two stamps swapped
    assert not _is_monotonic(t)
    calls = []
    orig = P._propagate_state

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(P, "_propagate_state", spy)
    live = P._densify_velocities(t_states, None, seed, t, s.w_m, s.a_m, cfg, _G)
    ref = ref_diagnostics._densify_velocities(t_states, None, seed, t, s.w_m, s.a_m, cfg, _G)
    assert _eq(live, ref)
    assert len(calls) == int(np.sum(t_states - seed.t_key[0] >= 1e-9))


def test_densify_synthetic_edge_cases_match_oracle():
    """Synthetic signal (hits integrate_R's small branches), 3 keyframes, states
    on / before / between / exactly on raw samples / after the last keyframe —
    the test_speedup_equivalence case with more on-sample states and a
    duplicated raw stamp inside a window."""
    rng = np.random.default_rng(87)
    t, w, a = imu_signal()
    t = t.copy(); t[66] = t[65]                              # duplicate stamp (d_t = 0)
    cfg = CamImuCalibConfig()
    t_key = np.array([t[20] + 0.001, t[50] + 0.002, t[80] + 0.0015])
    T_key = [gtsam.Pose3(gtsam.Rot3.Rodrigues(*rng.normal(size=3) * 0.1),
                         gtsam.Point3(*rng.normal(size=3))) for _ in t_key]
    v_key = rng.normal(size=(3, 3))
    seed = CamImuSeed(T_CtoI=gtsam.Pose3(), td=0.0, s=1.0, b_g=_BG, b_a=_BA,
                      g_inB=np.array([0, 0, -9.81]), R_BtoW=np.eye(3), T_BinW=gtsam.Pose3(),
                      t_key=t_key, T_ItoW=T_key, v_inW=v_key)
    t_states = np.concatenate([
        [t[10], t_key[0], t_key[0] + 1e-10, t_key[0] + 1e-9, t[21], t[22], t[30] + 0.0012,
         t[35], t[49], t_key[1], t[51], t[60], t[65], t[66], t[67], t[72] + 0.004, t_key[2],
         t[81], t[100] + 0.003, t[110], t[-1], t[-1] + 0.01],
        t[25:45] + 1e-13, t[85:95]])
    for cfgk in (cfg, CamImuCalibConfig(imu_intrinsics_model="kalibr"),
                 CamImuCalibConfig(imu_intrinsics_model="rpng")):
        A_ = P._densify_velocities(t_states, None, seed, t, w, a, cfgk, _G)
        B_ = ref_diagnostics._densify_velocities(t_states, None, seed, t, w, a, cfgk, _G)
        assert _eq(A_, B_)


# ---------------------------------------------------------------------------
# K8d — precomp
# ---------------------------------------------------------------------------
def _dict_eq(a, b):
    assert set(a) == set(b)
    for k in a:
        assert _eq(a[k], b[k]), k


def test_refine_and_rotation_equal_with_and_without_precomp(init_scenario):
    sx = init_scenario
    R_list = [T.rotation().matrix() for T in sx['poses']]
    s = sx['s']
    stride = _stride_for_baseline(sx['t_kept'], 0.3)
    td0, _ = estimate_time_offset_coarse(sx['t_kept'], R_list, s.t, s.w_m, 0.1)
    td_a, rep_a = refine_time_offset(sx['t_kept'], R_list, s.t, s.w_m, td0, stride=stride)
    td_b, rep_b, pre = refine_time_offset(sx['t_kept'], R_list, s.t, s.w_m, td0,
                                          stride=stride, return_precomp=True)
    assert td_a == td_b
    _dict_eq(rep_a, rep_b)
    assert len(pre) == 3
    R_a, bg_a, rot_a = estimate_rotation_gyro(sx['t_kept'], R_list, s.t, s.w_m, td_b, stride=stride)
    R_b, bg_b, rot_b = estimate_rotation_gyro(sx['t_kept'], R_list, s.t, s.w_m, td_b,
                                              stride=stride, _precomp=pre)
    assert _eq(R_a, R_b) and _eq(bg_a, bg_b)
    _dict_eq(rot_a, rot_b)                                   # the whole rep_rot dict
    # the precomp IS what estimate_rotation_gyro would have built itself
    pairs, w_C, R_cum = pre
    _t_mid, w_C2, pairs2 = CI.camera_angular_velocity(sx['t_kept'], R_list, stride)
    R_cum2 = CI._cumulative_rotation(np.asarray(s.t, float), np.asarray(s.w_m, float))
    assert _eq(pairs, pairs2) and _eq(w_C, w_C2) and _eq(R_cum, R_cum2)


def _seed_fields(seed):
    out = {"T_CtoI": seed.T_CtoI.matrix(), "td": seed.td, "s": seed.s, "b_g": seed.b_g,
           "b_a": seed.b_a, "g_inB": seed.g_inB, "R_BtoW": seed.R_BtoW,
           "T_BinW": seed.T_BinW.matrix(), "t_key": seed.t_key,
           "T_ItoW": np.stack([T.matrix() for T in seed.T_ItoW]), "v_inW": seed.v_inW}
    for k, v in seed.report.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"report.{k}.{kk}"] = vv
        elif v is not None:
            out[f"report.{k}"] = v
    return out


def test_initialize_cam_imu_seed_equal_with_and_without_precomp(init_scenario, monkeypatch):
    sx = init_scenario
    s = sx['s']
    kw = CamImuCalibConfig().init_kwargs
    seed_new = initialize_cam_imu(sx['t_kept'], sx['poses'], s.t, s.w_m, s.a_m, **kw)
    orig = CI.estimate_rotation_gyro
    seen = []

    def drop_precomp(*a, _precomp=None, **k):
        if a[0] is None:                                     # refine's own td scan
            return orig(*a, _precomp=_precomp, **k)
        seen.append(_precomp is not None)
        return orig(*a, **k)                                 # the pre-K8 call

    monkeypatch.setattr(CI, "estimate_rotation_gyro", drop_precomp)
    seed_old = initialize_cam_imu(sx['t_kept'], sx['poses'], s.t, s.w_m, s.a_m, **kw)
    assert seen == [True]                                    # init passes the precomp
    fa, fb = _seed_fields(seed_new), _seed_fields(seed_old)
    assert set(fa) == set(fb)
    for k in fa:
        assert _eq(fa[k], fb[k]), k


# ---------------------------------------------------------------------------
# K8e — linear_alignment on the lighter preintegrator
# ---------------------------------------------------------------------------
def test_linear_alignment_equal_with_full_tracking(init_scenario, monkeypatch):
    sx = init_scenario
    s = sx['s']
    from tests.helpers.sim_scenarios import _T_CtoI, _BG as BG_
    R_CtoI = _T_CtoI.rotation().matrix()
    idx = np.arange(0, len(sx['t_kept']), 4)
    args = (sx['t_kept'][idx], [sx['poses'][i] for i in idx], R_CtoI, BG_, s.t, s.w_m, s.a_m)
    flags = []

    def full(*a, **k):
        flags.append(k.get("track_intrinsics", True))
        k["track_intrinsics"] = True
        return ACIPreintegrator(*a, **k)

    outs = []
    for estimate_ba in (False, True):
        lite = linear_alignment(*args, estimate_ba=estimate_ba)
        monkeypatch.setattr(CI, "ACIPreintegrator", full)
        ref = linear_alignment(*args, estimate_ba=estimate_ba)
        monkeypatch.setattr(CI, "ACIPreintegrator", ACIPreintegrator)
        outs.append((lite, ref))
    assert flags and not any(flags)                          # init asked for track_intrinsics=False
    for lite, ref in outs:
        for k in ("s", "g_inB", "v_inB", "p_IinC", "b_a", "g_norm_ls", "residual_rms",
                  "svals", "estimated_ba", "t_key", "n_skipped_intervals", "sigma_p",
                  "sigma_s", "sigma_ba"):
            x, y = getattr(lite, k), getattr(ref, k)
            if x is None or y is None:
                assert x is None and y is None, k
            else:
                assert _eq(x, y), k
