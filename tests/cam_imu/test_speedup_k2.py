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
"""Speed-up step K2 (per-camera calib memo + base-IMU error-only branch) —
gates beyond the frozen-oracle fixture of `test_speedup_equivalence.py`:

  * `_calib_memo.get_calib_setup`: a hit returns the same setup object for
    the same (base, xi); one entry per base object (two cameras interleaved
    both stay warm); a one-ulp change of xi misses; mutating the base
    (intr / T / td / tr / model) misses and the rebuilt setup equals a fresh
    `base.retract(xi)`; `base.copy()` is a separate entry;
  * the memoized setup equals the direct build bitwise: calib fields,
    `Ad(T_C^-1)`, `Jr_se3(xi_T)` (lazy, built once) and the projector on both
    paths;
  * all three CameraCalib vision factors equal their pre-K2 oracles bitwise
    with a COLD and a WARM memo (`CameraCalibFactor` against the new
    `ref_camera_calib_factor`, the other two against the S0 oracles), and a
    factor whose base is mutated between calls follows the mutation;
  * a gtsam graph of factors sharing one base hits the memo for every factor
    but the first per Values (the pattern the gain relies on);
  * base IMU factor: `whitened_error == error_and_jacobians()[0]` bitwise
    with t_ga None and set (K2(b)); live == `ref_imu_factor` on the
    whitened error and every block; `so3_left_jacobian` evaluated once per
    `error_and_jacobians`; the CustomFactor's `H is None` path takes
    `whitened_error` (error_and_jacobians patched to raise) and returns the
    same vector;
  * gyro-only factor: the same four checks against `ref_imu_factor_gyro`
    (plus `base_calib.retract` once per `error_and_jacobians`, was twice).
"""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors import _calib_memo as CM
from mvis.calibration.factors import imu_factor as IF
from mvis.calibration.factors import imu_factor_gyro as IG
from mvis.calibration.factors._calib_memo import (get_calib_setup, clear_calib_memo,
                                                  calib_memo_stats)
from mvis.calibration.factors.camera_calib import CameraCalib
from tests.helpers.camera_calib_factor import CameraCalibFactor
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.imu_factor_gyro import ImuFactorACI_Multi_In_SO3
from mvis.calibration.factors.lie_groups import se3_right_jacobian
from mvis.calibration.factors.projection import make_projector
from mvis.imu import CALYX, KALIBR, RPNG

from tests._reference import (ref_camera_calib_factor, ref_camera_imu_factor,
                              ref_camera_interp_factor, ref_imu_factor,
                              ref_imu_factor_gyro)
from tests.helpers.speedup import (assert_close, oracle_error, oracle_norms, _eq, cam_base, board_corners,
                                            camimu_case, interp_case,
                                            _CAM_MODELS, _TRS, _CAMIMU_VARIANTS,
                                            _W, _H, _T_BOARD, _T_CTOI, _BG, _BA)
from tests.helpers.imu_factor_cases import _build_factor, _consistent_state
from tests.helpers.imu_factor_cases import (_build as _build_gyro, _consistent as
                                            _consistent_gyro)

_POSE = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.2, 0.3), gtsam.Point3(0.1, 0.2, -0.5))


@pytest.fixture(autouse=True)
def _fresh_memo():
    clear_calib_memo()
    yield
    clear_calib_memo()


def _calib_fields(c: CameraCalib):
    return c.T.matrix(), c.td, c.tr, c.intr, c.model


def _same_calib(a: CameraCalib, b: CameraCalib):
    Ta, tda, tra, ia, ma = _calib_fields(a)
    Tb, tdb, trb, ib, mb = _calib_fields(b)
    return _eq(Ta, Tb) and tda == tdb and tra == trb and _eq(ia, ib) and ma == mb


# ---------------------------------------------------------------------------
# memo semantics
# ---------------------------------------------------------------------------
def test_memo_hit_on_same_inputs_one_entry_per_base():
    rng = np.random.default_rng(1)
    base = cam_base("equidistant", 0.02)
    xi = rng.normal(size=base.DIM) * 1e-3
    s1 = get_calib_setup(base, xi)
    assert get_calib_setup(base, xi.copy()) is s1        # same bytes -> hit
    assert get_calib_setup(base, list(xi)) is s1         # any array-like of the same floats
    st = calib_memo_stats()
    assert (st["misses"], st["hits"], st["entries"]) == (1, 2, 1)

    base2 = cam_base("radtan", 0.0)                      # a second camera, interleaved
    xi2 = rng.normal(size=base2.DIM) * 1e-3
    t1 = get_calib_setup(base2, xi2)
    assert t1 is not s1
    for _ in range(3):
        assert get_calib_setup(base, xi) is s1
        assert get_calib_setup(base2, xi2) is t1
    st = calib_memo_stats()
    assert (st["misses"], st["hits"], st["entries"]) == (2, 8, 2)


def test_memo_misses_on_xi_change():
    rng = np.random.default_rng(2)
    base = cam_base("radtan", 1e-13)
    xi = rng.normal(size=base.DIM) * 1e-3
    s = get_calib_setup(base, xi)
    xi_b = xi.copy()
    xi_b[3] = np.nextafter(xi_b[3], 1.0)                 # one ulp (~2e-19): the
    s_b = get_calib_setup(base, xi_b)                    # retract is numerically
    assert s_b is not s                                  # the same, the KEY is not
    assert _same_calib(s_b.calib, base.retract(xi_b))
    assert calib_memo_stats()["misses"] == 2
    xi_c = xi.copy()
    xi_c[8] += 1e-6                                      # fx by 1e-6 px: differs
    s_c = get_calib_setup(base, xi_c)
    assert s_c is not s_b and not _same_calib(s_c.calib, s.calib)
    assert _same_calib(s_c.calib, base.retract(xi_c))
    # single entry per base: the old xi is a miss again, with equal numbers
    s2 = get_calib_setup(base, xi)
    assert s2 is not s and _same_calib(s2.calib, s.calib)
    assert calib_memo_stats()["misses"] == 4


@pytest.mark.parametrize("field", ["intr", "T", "td", "tr", "model"])
def test_memo_misses_when_the_base_is_mutated(field):
    rng = np.random.default_rng(3)
    base = cam_base("equidistant", 0.02)
    xi = rng.normal(size=base.DIM) * 1e-3
    before = get_calib_setup(base, xi)
    assert get_calib_setup(base, xi) is before
    if field == "intr":
        base.intr[0] += 1e-9                              # in-place
    elif field == "T":
        base.T = base.T.retract(np.array([0, 0, 1e-12, 0, 0, 0.0]))
    elif field == "td":
        base.td += 1e-12
    elif field == "tr":
        base.tr = 0.0
    else:
        base.model = "radtan"                             # same intr length (8)
    after = get_calib_setup(base, xi)
    assert after is not before, field
    assert not _same_calib(after.calib, before.calib), field
    assert _same_calib(after.calib, base.retract(xi)), field
    assert after.proj.model == base.model
    if field == "T":
        assert not _eq(after.Ad_TCinv, before.Ad_TCinv)
    assert get_calib_setup(base, xi) is after            # warm again
    assert calib_memo_stats()["misses"] == 2


def test_memo_copy_of_base_is_a_separate_entry():
    rng = np.random.default_rng(4)
    base = cam_base("kb4", 0.0)
    xi = rng.normal(size=base.DIM) * 1e-3
    s = get_calib_setup(base, xi)
    c = base.copy()
    t = get_calib_setup(c, xi)
    assert t is not s and _same_calib(t.calib, s.calib)
    assert _eq(t.Ad_TCinv, s.Ad_TCinv) and _eq(t.Jr_xiT, s.Jr_xiT)
    assert calib_memo_stats()["entries"] == 2
    del c                                                 # weak key: entry goes with it
    assert calib_memo_stats()["entries"] == 1


# ---------------------------------------------------------------------------
# setup == direct build
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cam_model", list(_CAM_MODELS))
@pytest.mark.parametrize("tr", _TRS)
def test_setup_equals_direct_build_bitwise(cam_model, tr):
    rng = np.random.default_rng(5)
    base = cam_base(cam_model, tr)
    xi = rng.normal(size=base.DIM) * 1e-3
    xi_orig = xi.copy()
    s = get_calib_setup(base, xi)
    calib = base.retract(xi)
    assert _same_calib(s.calib, calib)
    assert s.calib.DIM == calib.DIM and s.calib.n_intr == calib.n_intr
    assert _eq(s.Ad_TCinv, calib.T.inverse().AdjointMap())
    assert _eq(s.Jr_xiT, se3_right_jacobian(xi[0:6]))
    assert s.Ad_TCinv is s.Ad_TCinv and s.Jr_xiT is s.Jr_xiT       # built once
    # the memo must not alias the caller's xi (the lazy Jr reads its own copy)
    xi[0] += 1.0
    assert _eq(s.Jr_xiT, se3_right_jacobian(xi_orig[0:6]))
    assert get_calib_setup(base, xi) is not s           # and the mutated xi misses

    proj = make_projector(calib.model, calib.intr)
    assert s.proj.model == proj.model and _eq(s.proj.cal_vec, proj.cal_vec)
    T_CtoW = _POSE.compose(calib.T)
    pts = [np.array([0.1, -0.05, 1.2]), np.array([0.6, 0.4, 0.9]),
           np.array([0.0, 0.0, -1.0]), np.array([2.5, -2.0, 0.3])]
    for p_c in pts:
        p_W = np.asarray(T_CtoW.transformFrom(gtsam.Point3(*p_c)), float)
        for jac in (False, True):
            outs = []
            for pr in (s.proj, proj):
                try:
                    outs.append(pr.project(T_CtoW, p_W, jac=jac))
                except RuntimeError:
                    outs.append("guard")
            a, b = outs
            if isinstance(a, str) or isinstance(b, str):
                assert a == b
                continue
            for x, y in zip(a, b):
                assert (x is None and y is None) or _eq(x, y)


# ---------------------------------------------------------------------------
# vision factors == oracle, cold and warm memo
# ---------------------------------------------------------------------------
def _check_three_entry_points(live, ref, args, tag, cold_misses=1):
    """error_and_jacobians / error / corner_norms (when present) twice: the
    first pass fills the memo, the second reads it; both bitwise == oracle."""
    clear_calib_memo()
    has_cn = hasattr(live, "corner_norms")
    for rep in ("cold", "warm"):
        e_l, H_l = live.error_and_jacobians(*args)
        e_r, H_r = ref.error_and_jacobians(*args)
        assert_close((e_l, H_l), (e_r, H_r), exact=True, name=f"{tag} {rep}")
        model = live.base.model
        assert_close(live.error(*args), oracle_error(model, ref.error(*args), e_r), exact=True,
                     name=f"{tag} {rep} error")
        if has_cn:
            assert_close(live.corner_norms(*args), oracle_norms(model, ref.corner_norms(*args), e_r), exact=True,
                         name=f"{tag} {rep} corner_norms")
    st = calib_memo_stats()
    n_calls = (3 if has_cn else 2) * 2
    assert st["misses"] == cold_misses and st["hits"] == n_calls - cold_misses, (tag, st)


def calibfactor_case(cam_model, tr, n, rmode, rng):
    base = cam_base(cam_model, tr)
    p_B = board_corners(n, rng)
    uv = np.column_stack([rng.uniform(0, _W, n), rng.uniform(0, _H, n)])
    w_m = np.array([0.3, -0.2, 0.1])
    live = CameraCalibFactor(uv, p_B, w_m, base, (_W, _H), rolling_mode=rmode)
    ref = ref_camera_calib_factor.CameraCalibFactor(uv, p_B, w_m, base, (_W, _H),
                                                    rolling_mode=rmode)
    vb = np.concatenate([rng.normal(size=3) * 0.3, _BG, _BA])
    xi = rng.normal(size=base.DIM) * 1e-3
    return live, ref, (_POSE, vb, _T_BOARD, xi)


@pytest.mark.parametrize("cam_model", list(_CAM_MODELS))
@pytest.mark.parametrize("tr", _TRS)
def test_camera_calib_factor_matches_oracle_cold_and_warm(cam_model, tr):
    rng = np.random.default_rng(6)
    n_zero = 0
    for n, rmode in ((1, 0), (12, 1), (120, 2)):
        live, ref, args = calibfactor_case(cam_model, tr, n, rmode, rng)
        _check_three_entry_points(live, ref, args, f"calibfactor {cam_model} tr={tr} n={n}")
        _, H = live.error_and_jacobians(*args)
        n_zero += int((~np.any(H[0] != 0.0, axis=1)).sum() // 2)
    assert n_zero > 0                                    # the guarded corners are exercised


@pytest.mark.parametrize("cam_model", list(_CAM_MODELS))
@pytest.mark.parametrize("tr", _TRS)
def test_camera_imu_factor_matches_oracle_cold_and_warm(cam_model, tr):
    rng = np.random.default_rng(7)
    for variant in _CAMIMU_VARIANTS:
        live, ref, args = camimu_case(cam_model, tr, 12, variant, rng)
        _check_three_entry_points(live, ref, args, f"camimu {cam_model} tr={tr} {variant[:5]}")


@pytest.mark.parametrize("cam_model", list(_CAM_MODELS))
@pytest.mark.parametrize("phi_case", ["zero", "tiny", "near_pi", "normal"])
def test_camera_interp_factor_matches_oracle_cold_and_warm(cam_model, phi_case):
    rng = np.random.default_rng(8)
    cases = [(tr, lam, n, heights, rmode)
             for tr in _TRS for lam in ("inside", "above")
             for (n, heights, rmode) in ((1, False, 0), (12, True, 1))]
    for tr, lam, n, heights, rmode in cases:
        live, ref, args = interp_case(cam_model, tr, phi_case, lam, n, heights, rmode, rng)
        _check_three_entry_points(live, ref, args,
                                  f"interp {cam_model} φ={phi_case} tr={tr} λ={lam} n={n}")


def test_factor_follows_a_base_mutated_between_calls():
    """A stale memo would keep the old intrinsics; the oracle re-reads the
    base every call, so live must equal it after the mutation too."""
    rng = np.random.default_rng(9)
    live, ref, args = calibfactor_case("equidistant", 0.02, 12, 0, rng)
    e0, _ = live.error_and_jacobians(*args)
    assert_close(live.error_and_jacobians(*args), ref.error_and_jacobians(*args), exact=True)
    live.base.intr[0] += 2.0                             # fx of the SHARED base
    live.base.td -= 1e-3
    e1, H1 = live.error_and_jacobians(*args)
    assert not _eq(e0, e1)
    assert_close((e1, H1), ref.error_and_jacobians(*args), exact=True, name="after mutation")
    e_r = ref.error_and_jacobians(*args)[0]
    assert_close(live.error(*args), oracle_error(live.base.model, ref.error(*args), e_r), exact=True,
                 name="after mutation error")
    assert calib_memo_stats()["misses"] == 2


def test_graph_of_factors_sharing_a_base_hits_the_memo():
    rng = np.random.default_rng(10)
    base = cam_base("equidistant", 0.0)
    X, B, C = gtsam.symbol_shorthand.X, gtsam.symbol_shorthand.B, gtsam.symbol_shorthand.C
    pose_k, pose_k1 = _POSE, _POSE.retract(np.array([0.01, -0.02, 0.03, 0.02, 0.01, -0.01]))
    graph = gtsam.NonlinearFactorGraph()
    n_fac = 6
    for j in range(n_fac):
        n = 10
        p_B = board_corners(n, rng, nan_corner=False)
        uv = np.column_stack([rng.uniform(0, _W, n), rng.uniform(0, _H, n)])
        f = CameraCalibInterpFactor(uv, p_B, 0.01 + 0.002 * j, 0.05, base, (_W, _H))
        graph.add(f.make_factor(X(0), X(1), B(0), C(0), sigma_px=1.0, huber_px=1.345))
    values = gtsam.Values()
    values.insert(X(0), pose_k); values.insert(X(1), pose_k1); values.insert(B(0), _T_BOARD)
    values.insert(C(0), rng.normal(size=base.DIM) * 1e-3)

    clear_calib_memo()
    graph.error(values)                                   # error-only pass
    st = calib_memo_stats()
    assert (st["misses"], st["hits"]) == (1, n_fac - 1)
    graph.linearize(values)                               # Jacobian pass, same xi
    st = calib_memo_stats()
    assert (st["misses"], st["hits"]) == (1, 2 * n_fac - 1)
    values.update(C(0), values.atVector(C(0)) + 1e-6)     # new xi -> one miss per pass
    graph.error(values)
    st = calib_memo_stats()
    assert (st["misses"], st["hits"]) == (2, 3 * n_fac - 2)


# ---------------------------------------------------------------------------
# base IMU factor (K2 b)
# ---------------------------------------------------------------------------
def _perturbed_state(f, seed):
    """A state far from consistency: non-zero residual and a sizeable d_th."""
    rng = np.random.RandomState(seed)
    pose0, pose1, vb0, vb1, xi_in = _consistent_state(f, seed=seed)
    pose1 = pose1.retract(rng.randn(6) * 0.05)
    vb0 = vb0 + rng.randn(9) * np.r_[0.2, 0.2, 0.2, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1]
    vb1 = vb1 + rng.randn(9) * 0.05
    return pose0, pose1, vb0, vb1, xi_in + rng.randn(24) * 0.05


def _counting(fn, box):
    def wrapped(*a, **k):
        box[0] += 1
        return fn(*a, **k)
    return wrapped


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
def test_imu_factor_whitened_error_equals_eaj_and_oracle(model, monkeypatch):
    f, pre = _build_factor(model)
    ref = ref_imu_factor.ImuFactorACI_In.from_preintegrator(pre, f.g_inG)
    n_jl = [0]
    monkeypatch.setattr(IF, "so3_left_jacobian", _counting(IF.so3_left_jacobian, n_jl))
    for s in (_consistent_state(f), _perturbed_state(f, 11), _perturbed_state(f, 12)):
        for t_ga in (None, 0.0015, -0.0007):
            t = 0.0 if t_ga is None else t_ga
            e = f.whitened_error(*s, t)
            n_jl[0] = 0
            e2, H = f.error_and_jacobians(*s, t_ga)
            assert n_jl[0] == 1                           # Jl once, shared with W
            assert _eq(e, e2), (model, t_ga)              # K2(b): same _common + W path
            assert len(H) == (6 if t_ga is not None else 5)
            assert _eq(e, ref.whitened_error(*s, t))
            e_r, H_r = ref.error_and_jacobians(*s, t_ga)
            assert_close((e2, H), (e_r, H_r), exact=True, name=f"{model} t_ga={t_ga}")
            assert np.max(np.abs(e)) > 1e-6 or s is not None    # perturbed states are non-zero


@pytest.mark.parametrize("model", [KALIBR, RPNG, CALYX])
@pytest.mark.parametrize("with_tga", [False, True])
def test_imu_factor_custom_factor_error_only_path(model, with_tga, monkeypatch):
    f, _ = _build_factor(model)
    s = _perturbed_state(f, 13)
    X, V = gtsam.symbol_shorthand.X, gtsam.symbol_shorthand.V
    keys = [X(0), X(1), V(0), V(1), gtsam.symbol('Y', 0)]
    t_ga = 0.0011 if with_tga else None
    factor = f.make_factor(*keys, k_tga=gtsam.symbol('T', 0) if with_tga else None)
    values = gtsam.Values()
    for k, v in zip(keys, s):
        values.insert(k, v)
    if with_tga:
        values.insert(gtsam.symbol('T', 0), np.array([t_ga]))
    e_jac = f.error_and_jacobians(*s, t_ga)[0]
    e_w = f.whitened_error(*s, 0.0 if t_ga is None else t_ga)
    # H is None -> whitened_error only: the Jacobian path must not run
    monkeypatch.setattr(f, "error_and_jacobians",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("jac path taken")))
    e_cf = factor.unwhitenedError(values)
    assert _eq(e_cf, e_w) and _eq(e_cf, e_jac)
    # (gtsam sums the squares in its own order: not a bitwise claim)
    assert np.isclose(factor.error(values), 0.5 * float(e_w @ e_w), rtol=1e-12)
    monkeypatch.undo()
    jf = factor.linearize(values)                         # H path still works
    assert jf is not None and _eq(-np.asarray(jf.getb()), e_jac)


# ---------------------------------------------------------------------------
# gyro-only factor (same pattern)
# ---------------------------------------------------------------------------
def _gyro_ref(f):
    return ref_imu_factor_gyro.ImuFactorACI_Multi_In_SO3(
        f.D_R, f.P_meas, f.Dt, f.bg_init, f.intr_init, f.J_R_bg, f.J_R_in,
        f.w_inIs0, f.w_inIs1, f.base_calib)


def _gyro_states(f, model):
    rng = np.random.RandomState(31)
    out = [_consistent_gyro(f, model)[:5]]
    for _ in range(2):
        pose0, pose1, bg0, bg1, xi_calib, _ = _consistent_gyro(f, model, seed=int(rng.randint(100)))
        out.append((pose0, pose1.retract(rng.randn(6) * 0.05), bg0 + rng.randn(3) * 0.05,
                    bg1 + rng.randn(3) * 0.02, xi_calib + rng.randn(31) * 0.02))
    return out


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_gyro_factor_whitened_error_equals_eaj_and_oracle(model, monkeypatch):
    f = _build_gyro(model)
    ref = _gyro_ref(f)
    n_jl, n_retract = [0], [0]
    monkeypatch.setattr(IG, "so3_left_jacobian", _counting(IG.so3_left_jacobian, n_jl))
    monkeypatch.setattr(f.base_calib, "retract", _counting(f.base_calib.retract, n_retract))
    for s in _gyro_states(f, model):
        e = f.whitened_error(*s)
        n_jl[0] = n_retract[0] = 0
        e2, H = f.error_and_jacobians(*s)
        assert n_jl[0] == 1 and n_retract[0] == 1        # was 3 Jl / 2 retracts
        assert _eq(e, e2)
        assert _eq(e, ref.whitened_error(*s))
        assert_close((e2, H), ref.error_and_jacobians(*s), exact=True, name=f"gyro {model}")
    assert np.max(np.abs(e)) > 1e-3                      # the last state is far from consistent


@pytest.mark.parametrize("model", [KALIBR, RPNG])
def test_gyro_factor_custom_factor_error_only_path(model, monkeypatch):
    f = _build_gyro(model)
    s = _gyro_states(f, model)[2]
    X, G, K = gtsam.symbol_shorthand.X, gtsam.symbol_shorthand.G, gtsam.symbol_shorthand.K
    keys = [X(0), X(1), G(0), G(1), K(0)]
    factor = f.make_factor(*keys)
    values = gtsam.Values()
    for k, v in zip(keys, s):
        values.insert(k, v)
    e_jac = f.error_and_jacobians(*s)[0]
    e_w = f.whitened_error(*s)
    monkeypatch.setattr(f, "error_and_jacobians",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("jac path taken")))
    e_cf = factor.unwhitenedError(values)
    assert _eq(e_cf, e_w) and _eq(e_cf, e_jac)
    monkeypatch.undo()
    jf = factor.linearize(values)
    assert jf is not None and _eq(-np.asarray(jf.getb()), e_jac)
