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
"""Frozen-oracle equivalence fixture for the cam-IMU speed-up (S0).

Every kernel the speed-up rewrites is compared against its verbatim pre-change
copy in `tests/_reference/` on ONE seeded fixture that hits every branch the
plan lists:

  * IMU models CALYX / KALIBR / RPNG (RPNG had no coverage before), gyro raw
    (`intr=None`) and calibrated, `jac` on and off;
  * 150 propagation targets of mixed sign spanning >= 8 checkpoints per
    direction, with Δ = 0, Δ exactly on a raw sample (both directions),
    Δ within 1e-13 of a checkpoint and ±1e-13 from the anchor;
  * ω samples that hit `integrate_RAJ`'s |ω| < 1e-8, `small_w` and `small_th`
    branches (hits are counted on the oracle and asserted);
  * vision factors over equidistant / radtan / kb4 with 1, 12 and 120 corners
    including a corner behind the camera and a NaN (out-of-range) corner,
    tr ∈ {0, 1e-13, 0.02}, rolling_mode 0 / 1, heights on / off, `dw_column`
    and `frame_dt` on / off;
  * interp factor at φ = 0, |φ| = 1e-11 and near π, λ inside and outside [0, 1];
  * the diagnostics helpers (`_reproj_rms`, `_reproj_corner_norms`, with a
    frame whose corners are all guarded) and the init helpers (`_slice_imu`,
    `_propagate_state`, `_densify_velocities`) on states on / before / between /
    exactly on raw samples.

At S0 every comparison is `exact=True` (bitwise): that is the proof that the
oracles ARE the live code. A BITWISE step (D1, K1, K2, K8) keeps `exact=True`;
a ROUNDING step (K3-K7, D2) switches the kernels it touches to
`assert_close(..., exact=False)` (residual atol 1e-12 px, H blocks rtol 1e-12
of the block max, exactly-zero rows and NaN patterns preserved) and says so in
its commit. GOLDEN_EXACT-style bitwise claims are machine-specific (SIMD
sin/cos), so the rounding mode is what CI relies on.
"""

import numpy as np
import gtsam
import pytest

from mvis.calibration.factors import lie_groups as L
from mvis.calibration.factors.aci import integrate_RAJ, ACIPreintegrator, _SMALL
from mvis.calibration.factors.imu_segment import ImuSegment
from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.core.shutter import row_offset
from mvis.calibration.factors.projection import make_projector
from mvis.calibration.cam_imu_init import _slice_imu, CamImuSeed
from mvis.calibration import pipeline_cam_imu as P
from mvis.calibration.pipeline_cam_imu import (CamImuCalibConfig, X, B0, _Cc, H0, I0,
                                               _Vk, _Fdt)
from mvis.imu import CALYX, KALIBR, RPNG

from tests._reference import (ref_lie_groups, ref_aci, ref_imu_segment,
                              ref_camera_imu_factor, ref_camera_interp_factor,
                              ref_camera_factor, ref_projection, ref_diagnostics)
from tests.helpers.speedup import (oracle_error, oracle_norms,
                                   _G, _BG, _BA, _W, _CAM_MODELS,
                                   _TRS, _N_BP, _eq, assert_close, imu_signal, imu_intr,
                                   propagation_targets, board_corners, _T_BOARD,
                                   cam_base, _CAMIMU_VARIANTS, camimu_case,
                                   interp_poses, interp_case, _H)
from tests.helpers.speedup import same_as_oracle

# S0: live == oracle bitwise. Rounding steps flip this per kernel (see module doc).
EXACT = True


# ---------------------------------------------------------------------------
# fixture generators
# ---------------------------------------------------------------------------


def on_checkpoint_hits(t, t_anchor, deltas):
    """How many targets take the `delta < 1e-12` (on-checkpoint) branch, by the
    same arithmetic as `ImuSegment._propagate_dir`."""
    hits = 0
    for sign in (1, -1):
        sel = deltas >= 0.0 if sign > 0 else deltas < 0.0
        tau = sign * deltas[sel]
        if not tau.size:
            continue
        tau_max = float(tau.max())
        if sign > 0:
            inside = np.nonzero((t > t_anchor) & (t < t_anchor + tau_max))[0]
        else:
            inside = np.nonzero((t > t_anchor - tau_max) & (t < t_anchor))[0][::-1]
        tau_cp = np.concatenate([[0.0], sign * (t[inside] - t_anchor)])
        for tq in tau:
            m = int(np.searchsorted(tau_cp, float(tq), side="right")) - 1
            hits += int(float(tq) - tau_cp[m] < 1e-12)
    return hits


class RajBranchCounter:
    """Wraps `integrate_RAJ` and tallies which branch each call takes."""

    def __init__(self, fn):
        self.fn = fn
        self.hits = {"tiny": 0, "small_w": 0, "small_th": 0, "full": 0}

    def __call__(self, d_t, w_hat, a_hat):
        wn = float(np.linalg.norm(w_hat))
        if wn < 1e-8:
            self.hits["tiny"] += 1
        elif wn < _SMALL:
            self.hits["small_w"] += 1
        elif wn * d_t < _SMALL:
            self.hits["small_th"] += 1
        else:
            self.hits["full"] += 1
        return self.fn(d_t, w_hat, a_hat)


# ---------------------------------------------------------------------------
# leaves: lie groups, projection, _rs_row_offset
# ---------------------------------------------------------------------------
def test_lie_group_leaves_match_oracle():
    rng = np.random.default_rng(1)
    axes = rng.normal(size=(9, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    mags = [0.0, 1e-12, 1e-11, 1e-9, 1e-6, 1e-3, 1.0, np.pi - 1e-9, 2.5]
    for ax, mg in zip(axes, mags):
        phi = mg * ax
        assert _eq(L.skew(phi), ref_lie_groups.skew(phi))
        assert _eq(L.exp_so3(phi), ref_lie_groups.exp_so3(phi))
        assert _eq(L.log_so3(L.exp_so3(phi)), ref_lie_groups.log_so3(L.exp_so3(phi)))
        assert _eq(L.so3_right_jacobian(phi), ref_lie_groups.so3_right_jacobian(phi))
        assert _eq(L.so3_left_jacobian_inv(phi), ref_lie_groups.so3_left_jacobian_inv(phi))
        assert _eq(L.so3_right_jacobian_inv(phi), ref_lie_groups.so3_right_jacobian_inv(phi))
        xi = np.concatenate([phi, rng.normal(size=3) * 0.1])
        assert _eq(L.se3_right_jacobian(xi), ref_lie_groups.se3_right_jacobian(xi))


def test_rs_row_offset_matches_oracle():
    for mode in (0, 1, 2):
        for row in (0.0, 123.4, 479.0, 480.0):
            assert row_offset(row, _H, mode) == ref_camera_factor._rs_row_offset(row, _H, mode)


@pytest.mark.parametrize("model", ["radtan", "equidistant", "fov", "double_sphere",
                                   "eucm", "omni_radtan", "kb4"])
def test_projector_matches_oracle(model):
    """`make_projector(...).project` per model, both paths, on in-range,
    FOV-edge, behind-camera and NaN points: same uv / Jacobians / raise."""
    cv = {"radtan": [350., 350, 512, 512, -0.05, 0.01, 1e-3, -1e-3],
          "equidistant": [350., 350, 512, 512, 0.01, -0.001, 1e-4, -1e-5],
          "fov": [350., 350, 512, 512, 0.9],
          "double_sphere": [350., 350, 512, 512, 0.3, 0.6],
          "eucm": [350., 350, 512, 512, 0.6, 1.1],
          "omni_radtan": [350., 350, 512, 512, 1.2, -0.05, 0.01, 1e-3, -1e-3],
          "kb4": [350., 350, 512, 512, 0.02, -0.01, 3e-3, -5e-4]}[model]
    live = make_projector(model, np.array(cv))
    ref = ref_projection.make_projector(model, np.array(cv))
    T = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, -0.2, 0.3), gtsam.Point3(0.2, -0.1, 0.4))
    pts = [[0.1, 0.05, 1.0], [1.4, -1.3, 1.0], [3.0, 0.0, 0.1], [0.0, 0.0, -1.0],
           [np.nan, 0.0, 1.0], [0.3, 0.2, 0.0]]
    n_raise = 0
    for pc in pts:
        p_W = T.transformFrom(gtsam.Point3(*pc))      # camera-frame point -> world
        for jac in (False, True):
            try:
                a = live.project(T, p_W, jac=jac)
            except RuntimeError:
                a = "raise"
            try:
                b = ref.project(T, p_W, jac=jac)
            except RuntimeError:
                b = "raise"
            if model in ("equidistant", "radtan") and not jac and pc[2] <= 0.0:
                b = "raise"     # the oracle mirrors z <= 0 here; the guard came later (#25, F1)
            if a == "raise" or b == "raise":
                assert a == b, f"{model} {pc} jac={jac}: raise mismatch"
                n_raise += 1
                continue
            for x, y in zip(a, b):
                assert (x is None and y is None) or _eq(x, y), f"{model} {pc} jac={jac}"
    assert n_raise >= 2, f"{model}: the fixture must exercise the guard"


# ---------------------------------------------------------------------------
# integrate_RAJ and the preintegrator
# ---------------------------------------------------------------------------
def test_integrate_raj_matches_oracle_all_branches():
    rng = np.random.default_rng(2)
    counter = RajBranchCounter(ref_aci.integrate_RAJ)
    n = 0
    for _ in range(2500):
        d_t = rng.choice([1e-4, 2.5e-3, 5e-3, 0.02, 0.05]) * rng.uniform(0.5, 1.5)
        scale = rng.choice([0.0, 5e-9, 1e-8, 1e-4, 5e-3, _SMALL, 0.05, 1.0, 4.0])
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * scale
        a = rng.normal(size=3) * 5.0
        A = integrate_RAJ(d_t, w, a)
        B = counter(d_t, w, a)
        for x, y in zip(A, B):
            assert _eq(x, y)
        n += 1
    assert n == 2500
    assert all(v > 0 for v in counter.hits.values()), counter.hits


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
def test_aci_preintegrator_matches_oracle(model):
    t, w, a = imu_signal()
    outs = []
    for cls in (ACIPreintegrator, ref_aci.ACIPreintegrator):
        pre = cls(0.01, 1e-3, 0.02, 2e-3, model=model, avg_meas=True)
        pre.reset(_BG, _BA)
        pre.set_intrinsics(imu_intr(model))
        ts, ws, as_ = _slice_imu(t, w, a, t[55] + 0.0013, t[101] + 0.002)
        pre.feed_imu_interval(ts, ws, as_)
        outs.append(pre)
    for k in ("D_R", "D_v", "D_p", "Dt", "J_R_bg", "J_R_ba", "J_p_bg", "J_p_ba", "J_v_bg",
              "J_v_ba", "J_R_in", "J_p_in", "J_v_in", "J_R_tga", "J_p_tga", "J_v_tga",
              "P_meas", "a_m_s", "a_m_e", "w_m_s", "w_m_e"):
        assert same_as_oracle(k, getattr(outs[0], k), getattr(outs[1], k)), k


# ---------------------------------------------------------------------------
# ImuSegment.propagate
# ---------------------------------------------------------------------------
_PROP_FIELDS = ("D_R", "D_v", "D_p", "J_R_b", "J_p_b", "J_R_in", "J_p_in",
                "dtheta_dDelta", "w_inI")


@pytest.mark.parametrize("model", [CALYX, KALIBR, RPNG])
@pytest.mark.parametrize("gyro", ["calibrated", "raw"])
@pytest.mark.parametrize("jac", [True, False])
def test_imu_segment_propagate_matches_oracle(model, gyro, jac, monkeypatch):
    rng = np.random.default_rng(3)
    t, w, a = imu_signal()
    t0 = t[70] + 0.0021
    deltas = propagation_targets(t, t0, rng)
    intr = None if gyro == "raw" else imu_intr(model)
    counter = RajBranchCounter(ref_aci.integrate_RAJ)
    monkeypatch.setattr(ref_imu_segment, "integrate_RAJ", counter)
    live = ImuSegment(t, w, a, model).propagate(t0, deltas, _BG, _BA, intr, jac=jac)
    ref = ref_imu_segment.ImuSegment(t, w, a, model).propagate(t0, deltas, _BG, _BA,
                                                               intr, jac=jac)
    for i, (p, q) in enumerate(zip(live, ref)):
        for k in _PROP_FIELDS:
            assert_close(getattr(p, k), getattr(q, k), exact=EXACT, name=f"Δ[{i}].{k}")
    # branch coverage (asserted on the oracle; the live run saw the same inputs)
    assert on_checkpoint_hits(t, t0, deltas) >= 5
    n_cp = ((t > t0) & (t < t0 + 0.16)).sum() + ((t > t0 - 0.05) & (t < t0)).sum()
    assert n_cp >= 16                                  # >= 8 checkpoints per direction
    assert counter.hits["small_th"] > 0 and counter.hits["full"] > 0, counter.hits
    if gyro == "raw":
        # identity intrinsics (Tg = 0): the near-still stretch gives |ω| = 1e-3
        # (small_w) and w_m == b_g gives |ω| = 0 exactly (the |ω| < 1e-8 branch).
        # With Tg != 0 the g-sensitivity term keeps |ω| above _SMALL there.
        assert counter.hits["small_w"] > 0 and counter.hits["tiny"] > 0, counter.hits


def test_imu_segment_jac_false_states_equal_jac_true():
    """Gap closed by the plan: jac=False and jac=True give the same D_*."""
    rng = np.random.default_rng(4)
    t, w, a = imu_signal()
    t0 = t[70] + 0.0021
    deltas = propagation_targets(t, t0, rng)
    seg = ImuSegment(t, w, a, CALYX)
    A = seg.propagate(t0, deltas, _BG, _BA, imu_intr(CALYX), jac=True)
    B = seg.propagate(t0, deltas, _BG, _BA, imu_intr(CALYX), jac=False)
    for p, q in zip(A, B):
        for k in ("D_R", "D_v", "D_p", "dtheta_dDelta", "w_inI"):
            assert _eq(getattr(p, k), getattr(q, k)), k


# ---------------------------------------------------------------------------
# CameraCalibImuFactor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cam_model", list(_CAM_MODELS))
@pytest.mark.parametrize("tr", _TRS)
@pytest.mark.parametrize("n", [1, 12, 120])
def test_camera_imu_factor_matches_oracle(cam_model, tr, n):
    rng = np.random.default_rng(5)
    n_zero_rows = n_nan = 0
    variants = _CAMIMU_VARIANTS
    for variant in variants:
        live, ref, args = camimu_case(cam_model, tr, n, variant, rng)
        tag = f"{cam_model} tr={tr} n={n} {variant[:5]}"
        e_l, H_l = live.error_and_jacobians(*args)
        e_r, H_r = ref.error_and_jacobians(*args)
        assert_close((e_l, H_l), (e_r, H_r), exact=EXACT, name=tag)
        assert_close(live.error(*args), oracle_error(cam_model, ref.error(*args), e_r), exact=EXACT,
                     name=tag + " error")
        cn_l, cn_r = live.corner_norms(*args), ref.corner_norms(*args)
        assert_close(cn_l, oracle_norms(cam_model, cn_r, e_r), exact=EXACT, name=tag + " corner_norms")
        # error() and error_and_jacobians() take different gtsam entry points
        # (Cal3.uncalibrate vs PinholeCamera.project): same residual only to
        # rounding. Both pinhole error paths have the z check (#25, F1), so
        # their two paths agree on every corner.
        assert np.allclose(live.error(*args), e_l, atol=1e-6) or (n >= 12 and cam_model not in ("equidistant", "radtan"))
        n_zero_rows += int((~np.any(H_l[0] != 0.0, axis=1)).sum() // 2)
        n_nan += int(np.isnan(cn_l).sum())
        if n >= 12 and cam_model in ("equidistant", "radtan"):
            assert np.isnan(cn_l[5]) and np.isfinite(cn_r[5]), tag      # the oracle mirrors it
    if n >= 12:
        assert n_nan >= len(variants), "NaN corner must be NaN in corner_norms"
        if cam_model in ("radtan", "equidistant"):
            # jac path: gtsam rejects z <= 0 -> the behind corner AND the NaN
            # corner give exactly-zero rows
            assert n_zero_rows >= 2 * len(variants), "behind + NaN corners must zero rows"
        else:                                   # kb4 projects behind-camera points (θ > 90°)
            assert n_zero_rows >= len(variants), "NaN corner must zero rows"


# ---------------------------------------------------------------------------
# CameraCalibInterpFactor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cam_model", list(_CAM_MODELS))
@pytest.mark.parametrize("phi_case", ["zero", "tiny", "near_pi"])
def test_camera_interp_factor_matches_oracle(cam_model, phi_case):
    rng = np.random.default_rng(6)
    cases = [(tr, lam, n, heights, rmode)
             for tr in _TRS for lam in ("inside", "below", "above")
             for (n, heights, rmode) in ((1, False, 0), (12, True, 1))]
    cases.append((0.02, "inside", 120, True, 0))
    n_all = len(cases)
    n_zero_rows = n_nan = 0
    for tr, lam, n, heights, rmode in cases:
        live, ref, args = interp_case(cam_model, tr, phi_case, lam, n, heights, rmode, rng)
        tag = f"{cam_model} φ={phi_case} tr={tr} λ={lam} n={n} h={heights} rm={rmode}"
        e_l, H_l = live.error_and_jacobians(*args)
        e_r, H_r = ref.error_and_jacobians(*args)
        assert_close((e_l, H_l), (e_r, H_r), exact=EXACT, name=tag)
        assert_close(live.error(*args), oracle_error(cam_model, ref.error(*args), e_r), exact=EXACT,
                     name=tag + " error")
        cn_l, cn_r = live.corner_norms(*args), ref.corner_norms(*args)
        assert_close(cn_l, oracle_norms(cam_model, cn_r, e_r), exact=EXACT, name=tag + " corner_norms")
        n_zero_rows += int((~np.any(H_l[0] != 0.0, axis=1)).sum() // 2)
        n_nan += int(np.isnan(cn_l).sum())
    if len(cases) == n_all:          # the guarded corners live in the 12/120-corner cases
        assert n_zero_rows > 0 and n_nan > 0


def test_interp_pose_matches_oracle():
    from mvis.calibration.factors.camera_interp_factor import _interp_pose
    rng = np.random.default_rng(7)
    for phi_case in ("zero", "tiny", "near_pi", "normal"):
        pk, pk1 = interp_poses(phi_case, rng)
        for lam in (0.0, 0.3, 1.0, -0.2, 1.4):
            a = _interp_pose(pk, pk1, lam)
            b = ref_camera_interp_factor._interp_pose(pk, pk1, lam)
            assert _eq(a[0].matrix(), b[0].matrix())
            for x, y in zip(a[1:], b[1:]):
                assert _eq(x, y)


# ---------------------------------------------------------------------------
# diagnostics + init helpers
# ---------------------------------------------------------------------------
def _graph_fixture(rng):
    """A tiny mixed factor list (interp + IMU-propagated, with heights and a
    frame whose corners are ALL guarded) and a Values holding their keys."""
    t, w, a = imu_signal()
    seg_l = ImuSegment(t, w, a, CALYX)
    seg_r = ref_imu_segment.ImuSegment(t, w, a, CALYX)
    values = gtsam.Values()
    values.insert(X(0), gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.2, 0.3), gtsam.Point3(0.1, 0.2, -0.5)))
    values.insert(X(1), gtsam.Pose3(gtsam.Rot3.Rodrigues(0.12, 0.19, 0.31), gtsam.Point3(0.12, 0.19, -0.52)))
    values.insert(_Vk(0), np.concatenate([rng.normal(size=3) * 0.3, _BG, _BA]))
    values.insert(B0, _T_BOARD)
    values.insert(I0, rng.normal(size=24) * 1e-2)
    values.insert(H0, rng.normal(0.0, 0.004, _N_BP))
    values.insert(_Fdt(0), np.array([1.3e-3]))
    live_f, ref_f = [], []
    for c, (cm, n, all_nan) in enumerate((("radtan", 12, False), ("equidistant", 12, True),
                                          ("kb4", 120, False))):
        base = cam_base(cm, 0.02)
        values.insert(_Cc(c), rng.normal(size=base.DIM) * 1e-3)
        p_B = board_corners(n, rng)
        if all_nan:
            p_B[:] = np.nan
        uv = np.column_stack([rng.uniform(0, _W, n), rng.uniform(0, _H, n)])
        pids = rng.permutation(_N_BP)[:n]
        fi = CameraCalibInterpFactor(uv, p_B, 0.013, 0.05, base, (_W, _H), rolling_mode=1,
                                     board_pids=pids, n_board_points=_N_BP)
        fi.make_factor(X(0), X(1), B0, _Cc(c), k_heights=H0)
        gi = ref_camera_interp_factor.CameraCalibInterpFactor(
            uv, p_B, 0.013, 0.05, base, (_W, _H), rolling_mode=1, board_pids=pids,
            n_board_points=_N_BP)
        gi.make_factor(X(0), X(1), B0, _Cc(c), k_heights=H0)
        fm = CameraCalibImuFactor(uv, p_B, 0.012, seg_l, t[70] + 0.0021, _G, base, (_W, _H),
                                  frame_dt=True, board_pids=pids, n_board_points=_N_BP)
        fm.make_factor(X(0), _Vk(0), B0, _Cc(c), k_imu=I0, k_dt=_Fdt(0), k_heights=H0)
        gm = ref_camera_imu_factor.CameraCalibImuFactor(
            uv, p_B, 0.012, seg_r, t[70] + 0.0021, _G, base, (_W, _H), frame_dt=True,
            board_pids=pids, n_board_points=_N_BP)
        gm.make_factor(X(0), _Vk(0), B0, _Cc(c), k_imu=I0, k_dt=_Fdt(0), k_heights=H0)
        live_f += [(c, 1, fi), (c, 1, fm)]
        ref_f += [(c, 1, gi), (c, 1, gm)]
    return live_f, ref_f, values


def _guard_radtan_oracle(monkeypatch):
    """The frozen radtan projection mirrored a z <= 0 point into the image on its jac=False
    path; since F1 it gives NaN, as the jac=True path always did. Give the oracle that guard
    (and nothing else) so the diagnostics can still be compared exactly."""
    frozen = ref_projection._RefCameraModel.project

    def guarded(self, point_3d, jac=False):
        p = np.asarray(point_3d, dtype=np.float64)
        if self.model == "radtan" and not jac and not p[2] > 0.0:
            return np.array([np.nan, np.nan])
        return frozen(self, point_3d, jac)
    monkeypatch.setattr(ref_projection._RefCameraModel, "project", guarded)


def test_reproj_diagnostics_match_oracle(monkeypatch):
    _guard_radtan_oracle(monkeypatch)
    rng = np.random.default_rng(8)
    live_f, ref_f, values = _graph_fixture(rng)
    rms_l, cam_l = P._reproj_rms(live_f, values)
    rms_r, cam_r = ref_diagnostics._reproj_rms(ref_f, values)
    assert_close(rms_l, rms_r, exact=EXACT, name="rms")
    assert _eq(cam_l, cam_r)
    assert np.isnan(rms_l[2]) and np.isnan(rms_l[3])     # the all-guarded camera
    px_l, pc_l = P._reproj_corner_norms(live_f, values)
    px_r, pc_r = ref_diagnostics._reproj_corner_norms(ref_f, values)
    assert_close(px_l, px_r, exact=EXACT, name="corner_px")
    assert _eq(pc_l, pc_r) and np.isfinite(px_l).all()
    assert (pc_l == 1).sum() == 0                          # all-guarded corners dropped
    assert _eq(P._reproj_corner_norms([], values)[0], ref_diagnostics._reproj_corner_norms([], values)[0])


def test_slice_imu_matches_oracle():
    rng = np.random.default_rng(9)
    t, w, a = imu_signal()
    for _ in range(40):
        t0 = rng.choice([t[20], t[20] + 1e-4, t[31] - 1e-7, t[0]])
        t1 = rng.choice([t[40], t[40] + 2e-3, t[45] - 1e-9, t[-1]])
        A = _slice_imu(t, w, a, t0, t1)
        B = ref_diagnostics._slice_imu(t, w, a, t0, t1)
        for x, y in zip(A, B):
            assert _eq(x, y)


def test_densify_velocities_matches_oracle():
    """States on a keyframe, before the first keyframe, between raw samples,
    exactly on a raw sample and in the last interval (K8c verification)."""
    rng = np.random.default_rng(10)
    t, w, a = imu_signal()
    cfg = CamImuCalibConfig()
    t_key = np.array([t[20] + 0.001, t[50] + 0.002, t[80] + 0.0015])
    T_key = [gtsam.Pose3(gtsam.Rot3.Rodrigues(*rng.normal(size=3) * 0.1),
                         gtsam.Point3(*rng.normal(size=3))) for _ in t_key]
    v_key = rng.normal(size=(3, 3))
    seed = CamImuSeed(T_CtoI=gtsam.Pose3(), td=0.0, s=1.0, b_g=_BG, b_a=_BA,
                      g_inB=np.array([0, 0, -9.81]), R_BtoW=np.eye(3), T_BinW=gtsam.Pose3(),
                      t_key=t_key, T_ItoW=T_key, v_inW=v_key)
    t_states = np.array([t[10], t_key[0], t_key[0] + 1e-10, t[30] + 0.0012, t[35],
                         t_key[1], t[60], t[72] + 0.004, t_key[2], t[100] + 0.003, t[110]])
    A = P._densify_velocities(t_states, None, seed, t, w, a, cfg, _G)
    B = ref_diagnostics._densify_velocities(t_states, None, seed, t, w, a, cfg, _G)
    assert_close(A, B, exact=EXACT, name="v_init")
    assert _eq(A[0], v_key[0]) and _eq(A[1], v_key[0]) and _eq(A[2], v_key[0])
    Ta, va = P._propagate_state(T_key[0], v_key[0], t_key[0], t[35], t, w, a, _BG, _BA, cfg, _G)
    Tb, vb = ref_diagnostics._propagate_state(T_key[0], v_key[0], t_key[0], t[35], t, w, a,
                                              _BG, _BA, cfg, _G)
    assert _eq(Ta.matrix(), Tb.matrix())
    assert_close(va, vb, exact=EXACT, name="v1")
