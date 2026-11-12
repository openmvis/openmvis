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
"""Bitwise proofs for the merged duplicate helpers.

Each section freezes, inline, a verbatim copy of every old variant of one
duplicated block, and checks that the shared helper the callers now use gives
the same bytes (`tobytes` equality, NaN patterns included) on realistic and
random inputs. A failure here means a merge changed numbers: it is not a
refactor any more.
"""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.lie_groups import (
    exp_so3, log_so3, so3_right_jacobian, so3_right_jacobian_inv, so3_left_jacobian_inv,
)


def _same(a, b):
    """Same dtype, shape and bytes."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _rand_rot(rng, mag=None):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    m = rng.uniform(0.0, np.pi) if mag is None else mag
    return gtsam.Rot3.Expmap(m * axis)


# ---------------------------------------------------------------------------
# 1. IMU-state pose interpolation + its tangent blocks
#    (camera_interp_factor._interp_pose and the inline Jacobian block that
#    camera_interp_factor and camera_feature_factor each carried)
# ---------------------------------------------------------------------------
def _frozen_interp_pose(pose_k, pose_k1, lam):
    R_k = pose_k.rotation().matrix()
    p_k = np.asarray(pose_k.translation(), float)
    p_k1 = np.asarray(pose_k1.translation(), float)
    phi = log_so3(R_k.T @ pose_k1.rotation().matrix())
    A = exp_so3(lam * phi)
    R_lam = R_k @ A
    p_lam = (1.0 - lam) * p_k + lam * p_k1
    return gtsam.Pose3(gtsam.Rot3(R_lam), gtsam.Point3(*p_lam)), phi, A, R_lam


def _frozen_interp_block(pose_k, pose_k1, lam):
    """The inline block of both error_and_jacobians (identical in the two factors)."""
    p_k = np.asarray(pose_k.translation(), float)
    p_k1 = np.asarray(pose_k1.translation(), float)
    dp = p_k1 - p_k
    T_I, phi, A, R_lam = _frozen_interp_pose(pose_k, pose_k1, lam)
    Jr_lp = so3_right_jacobian(lam * phi)
    M_k = np.zeros((6, 6))
    M_k[0:3, 0:3] = A.T - lam * Jr_lp @ so3_left_jacobian_inv(phi)
    M_k[3:6, 3:6] = (1.0 - lam) * A.T
    M_k1 = np.zeros((6, 6))
    M_k1[0:3, 0:3] = lam * Jr_lp @ so3_right_jacobian_inv(phi)
    M_k1[3:6, 3:6] = lam * exp_so3((1.0 - lam) * phi)
    zeta_lam = np.concatenate([phi, R_lam.T @ dp])
    return T_I, phi, A, R_lam, M_k, M_k1, zeta_lam


def _interp_pose_cases():
    rng = np.random.default_rng(20261107)
    cases = []
    # realistic: 200 Hz states of a hand-held rig (|phi| up to ~0.05 rad, mm steps)
    for mag in (0.0, 1e-11, 1e-6, 0.01, 0.05, np.pi - 1e-6):
        R_k = _rand_rot(rng)
        pk = gtsam.Pose3(R_k, gtsam.Point3(*rng.normal(size=3)))
        pk1 = gtsam.Pose3(R_k.compose(_rand_rot(rng, mag)),
                          gtsam.Point3(*(np.asarray(pk.translation()) + 0.005 * rng.normal(size=3))))
        cases.append((pk, pk1))
    # random: any rotations, any translations
    for _ in range(60):
        cases.append((gtsam.Pose3(_rand_rot(rng), gtsam.Point3(*rng.normal(scale=3, size=3))),
                      gtsam.Pose3(_rand_rot(rng), gtsam.Point3(*rng.normal(scale=3, size=3)))))
    lams = [0.0, 1.0, 0.5, -0.2, 1.4, 1e-13, 1 - 1e-13] + list(rng.uniform(-0.5, 1.5, 6))
    return [(a, b, float(lam)) for a, b in cases for lam in lams]


def test_interp_pose_and_tangents_match_the_frozen_inline_blocks():
    from mvis.calibration.factors.pose_interpolation import interp_pose, interp_pose_tangents
    from mvis.calibration.factors.camera_interp_factor import _interp_pose
    assert _interp_pose is interp_pose               # old private name kept
    for pk, pk1, lam in _interp_pose_cases():
        ref = _frozen_interp_block(pk, pk1, lam)
        T_I, phi, A, R_lam = interp_pose(pk, pk1, lam)
        dp = np.asarray(pk1.translation(), float) - np.asarray(pk.translation(), float)
        new = (T_I, phi, A, R_lam) + interp_pose_tangents(phi, A, R_lam, lam, dp)
        assert _same(new[0].matrix(), ref[0].matrix())
        for x, y in zip(new[1:], ref[1:]):
            assert _same(x, y)


# ---------------------------------------------------------------------------
# 2. Corner bookkeeping of the three reprojection factors (CornerReprojMixin)
#    and the three hand-built corner-norm call sites it replaces
#    (pipeline_cam_imu feature solve x2, multi_board_pipeline._reproj)
# ---------------------------------------------------------------------------
def _frozen_norms(fac, *args):
    """The old corner_norms body, identical in the three factors."""
    err, valid = fac._residuals(*args)
    norms = np.linalg.norm(err.reshape(-1, 2), axis=1)
    norms[~valid] = np.nan
    return norms


def _frozen_interp_norms_values(fac, values):
    """The old CameraCalibInterpFactor.corner_norms_values."""
    k = fac.keys
    h = values.atVector(k[4]) if fac.uses_heights_key else None
    return _frozen_norms(fac, values.atPose3(k[0]), values.atPose3(k[1]),
                         values.atPose3(k[2]), values.atVector(k[3]), h)


def _frozen_feature_call_site(fac, vals, c, jj):
    """pipeline_cam_imu feature solve, both _fcorner_norms (before E1)."""
    from mvis.calibration.pipeline_cam_imu import X, _Cc, _L
    P = np.array([vals.atPoint3(_L(i)) for i in fac.ids])
    return _frozen_norms(fac, vals.atPose3(X(jj - 1)), vals.atPose3(X(jj)),
                         vals.atVector(_Cc(c)), P)


def _frozen_multi_board_reproj(cam_factors, board_ids, vals):
    """multi_board_pipeline._reproj (before E1)."""
    from mvis.calibration.pipeline_cam_imu import X, _B, _Cc
    from mvis.calibration.reproj_stats import corner_stats
    norms = [_frozen_norms(fac, vals.atPose3(X(j - 1)), vals.atPose3(X(j)),
                           vals.atPose3(_B(int(bid))), vals.atVector(_Cc(c)))
             for (c, j, fac), bid in zip(cam_factors, board_ids)]
    s = corner_stats(np.concatenate(norms) if norms else np.empty(0))
    return s['rms'] if s['n_corners'] else np.inf


def _reproj_graph(seed, n_frames=6):
    """Interp (+heights), IMU-propagated and feature factors on 3 camera models and
    2 boards, built through make_factor as build_joint_graph does, plus their Values."""
    from mvis.calibration.factors.camera_feature_factor import CameraFeatureInterpFactor
    from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor
    from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
    from mvis.calibration.factors.imu_segment import ImuSegment
    from mvis.calibration.pipeline_cam_imu import X, _B, _Cc, _L, _Vk, I0, H0
    from mvis.imu import CALYX
    from tests.helpers.speedup import (board_corners, cam_base, imu_signal, _T_BOARD,
                                       _G, _BG, _BA, _W, _H, _N_BP)
    rng = np.random.default_rng(seed)
    t, w, a = imu_signal()
    seg = ImuSegment(t, w, a, CALYX)
    vals = gtsam.Values()
    for k in range(n_frames + 1):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3.Rodrigues(*(0.1 * rng.normal(size=3))),
                                      gtsam.Point3(*(0.05 * rng.normal(size=3)))))
        vals.insert(_Vk(k), np.concatenate([rng.normal(size=3) * 0.3, _BG, _BA]))
    vals.insert(_B(0), _T_BOARD)
    vals.insert(_B(1), _T_BOARD.compose(gtsam.Pose3(gtsam.Rot3.Rodrigues(0.0, 0.05, 0.1),
                                                    gtsam.Point3(0.2, -0.1, 0.05))))
    vals.insert(I0, rng.normal(size=24) * 1e-2)
    vals.insert(H0, rng.normal(0.0, 0.004, _N_BP))
    interp, imu, feat, bids = [], [], [], []
    lid = 0
    for c, cm in enumerate(("radtan", "equidistant", "kb4")):
        base = cam_base(cm, (0.0, 0.02, 1e-13)[c])
        vals.insert(_Cc(c), rng.normal(size=base.DIM) * 1e-3)
        for j in range(1, n_frames + 1):
            n = int(rng.choice([1, 12, 40]))
            p_B = board_corners(n, rng)
            uv = np.column_stack([rng.uniform(0, _W, n), rng.uniform(0, _H, n)])
            bid = int(rng.integers(0, 2))
            heights = bool(rng.integers(0, 2))
            hk = (dict(board_pids=rng.permutation(_N_BP)[:n], n_board_points=_N_BP)
                  if heights else {})
            fi = CameraCalibInterpFactor(uv, p_B, float(rng.uniform(-0.01, 0.06)), 0.05,
                                         base, (_W, _H), rolling_mode=j % 2, **hk)
            fi.make_factor(X(j - 1), X(j), _B(bid), _Cc(c), k_heights=H0 if heights else None)
            interp.append((c, j, fi)); bids.append(bid)
            fm = CameraCalibImuFactor(uv, p_B, 0.012, seg, t[70] + 0.0021, _G, base,
                                      (_W, _H), rolling_mode=j % 2, **hk)
            fm.make_factor(X(j - 1), _Vk(j - 1), _B(bid), _Cc(c), k_imu=I0,
                           k_heights=H0 if heights else None)
            imu.append((c, j, fm))
            ids = list(range(lid, lid + n)); lid += n
            for i, p in zip(ids, p_B):
                q = vals.atPose3(_B(bid)).transformFrom(gtsam.Point3(*np.nan_to_num(p)))
                vals.insert(_L(i), q)
            ff = CameraFeatureInterpFactor(uv, ids, float(rng.uniform(-0.01, 0.06)), 0.05,
                                           base, (_W, _H), rolling_mode=j % 2,
                                           z_min=float(rng.choice([0.0, 0.5])))
            ff.make_factor(X(j - 1), X(j), _Cc(c), [_L(i) for i in ids])
            feat.append((c, j, ff))
    return interp, imu, feat, bids, vals


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_corner_reproj_mixin_matches_the_frozen_methods_and_call_sites(seed):
    from mvis.calibration.factors.corner_reproj import CornerReprojMixin
    from mvis.calibration.pipeline_cam_imu import _reproj_corner_norms
    from mvis.calibration.reproj_stats import corner_stats
    interp, imu, feat, bids, vals = _reproj_graph(seed)
    n_nan = 0
    for c, j, fac in interp:
        assert isinstance(fac, CornerReprojMixin)
        new = fac.corner_norms_values(vals)
        assert _same(new, _frozen_interp_norms_values(fac, vals))
        n_nan += int(np.isnan(new).sum())
    for c, j, fac in imu:
        args = fac.args_from_values(vals)
        assert _same(fac.corner_norms(*args), _frozen_norms(fac, *args))
        assert _same(fac.corner_norms_values(vals), _frozen_norms(fac, *args))
    for c, j, fac in feat:
        assert _same(fac.corner_norms_values(vals), _frozen_feature_call_site(fac, vals, c, j))
    assert n_nan > 0                                   # guarded corners exercised
    # multi-board _reproj: corner_stats over the pooled finite norms. build_joint_graph refuses
    # board heights with several boards, so that graph only holds factors without them.
    mb = [(f, b) for f, b in zip(interp, bids) if not f[2].uses_heights_key]
    mb_f, mb_b = [f for f, _ in mb], [b for _, b in mb]
    assert len(mb_f) > 3 and len(set(mb_b)) == 2
    s = corner_stats(_reproj_corner_norms(mb_f, vals)[0])
    new_rms = s['rms'] if s['n_corners'] else np.inf
    assert _same(new_rms, _frozen_multi_board_reproj(mb_f, mb_b, vals))
    s = corner_stats(_reproj_corner_norms([], vals)[0])
    assert (s['rms'] if s['n_corners'] else np.inf) == _frozen_multi_board_reproj([], [], vals)


# ---------------------------------------------------------------------------
# 3. Rotation-corrected IMU whitener (ImuFactorACI_In / ImuFactorACI_Multi_In_SO3
#    ._cov_sqrt_inv) -> aci.whitener_rot_corrected
# ---------------------------------------------------------------------------
def _frozen_cov_sqrt_inv_15(P_meas, d_th_total, Jl=None):
    from mvis.calibration.factors.lie_groups import so3_left_jacobian
    if Jl is None:
        Jl = so3_left_jacobian(d_th_total)
    P = P_meas.copy()
    P[0:3, :] = Jl @ P_meas[0:3, :]
    P[:, 0:3] = P[:, 0:3] @ Jl.T
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P)
    return np.linalg.solve(L, np.eye(15))


def _frozen_cov_sqrt_inv_6(P_meas, d_th, Jl=None):
    from mvis.calibration.factors.lie_groups import so3_left_jacobian
    if Jl is None:
        Jl = so3_left_jacobian(d_th)
    P = P_meas.copy()
    P[0:3, :] = Jl @ P_meas[0:3, :]
    P[:, 0:3] = P[:, 0:3] @ Jl.T
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P)
    return np.linalg.solve(L, np.eye(6))


def _imu_covariances():
    """Real ACI preintegration covariances (15x15 full, 6x6 gyro-only) plus random SPD ones."""
    from mvis.calibration.factors.aci import ACIPreintegrator, ACIGyroPreintegrator
    from tests.helpers.speedup import imu_signal
    t, w, a = imu_signal()
    out15, out6 = [], []
    for n in (2, 10, 40):
        pre = ACIPreintegrator(1.6e-4, 2e-5, 2e-3, 3e-4)
        pre.reset(np.zeros(3), np.zeros(3))
        pre.feed_imu_interval(t[:n], w[:n], a[:n])
        out15.append(pre.P_meas)
        g = ACIGyroPreintegrator(1.6e-4, 2e-5)
        g.reset(np.zeros(3))
        g.feed_imu_interval(t[:n], w[:n])
        out6.append(g.P_meas)
    rng = np.random.default_rng(3)
    for _ in range(20):
        for n, out in ((15, out15), (6, out6)):
            A = rng.normal(size=(n, n)) * 10.0 ** rng.uniform(-6, -2)
            out.append(A @ A.T + np.eye(n) * 1e-14)
    return out15, out6


def test_whitener_rot_corrected_matches_both_frozen_cov_sqrt_inv():
    from mvis.calibration.factors.aci import whitener_rot_corrected
    from mvis.calibration.factors.lie_groups import so3_left_jacobian
    rng = np.random.default_rng(4)
    out15, out6 = _imu_covariances()
    d_ths = [np.zeros(3), np.full(3, 1e-12), np.array([0.01, -0.02, 0.005])] + \
        [rng.normal(size=3) * s for s in (1e-4, 0.1, 1.0)]
    for d in d_ths:
        for P in out15:
            assert _same(whitener_rot_corrected(P, d), _frozen_cov_sqrt_inv_15(P, d))
            Jl = so3_left_jacobian(d)
            assert _same(whitener_rot_corrected(P, d, Jl), _frozen_cov_sqrt_inv_15(P, d, Jl))
        for P in out6:
            assert _same(whitener_rot_corrected(P, d), _frozen_cov_sqrt_inv_6(P, d))
            Jl = so3_left_jacobian(d)
            assert _same(whitener_rot_corrected(P, d, Jl), _frozen_cov_sqrt_inv_6(P, d, Jl))


# ---------------------------------------------------------------------------
# 4. ACI step blocks: imu_segment._step's state lines -> aci._state_recurrence,
#    and the phi_nn / H_wa block of _step and feed_imu_single -> aci._phi_hwa
# ---------------------------------------------------------------------------
def _frozen_step_state(D_R, D_v, D_p, d_t, dR, int_R, iint_R, a_hat):
    """imu_segment._step's own copy of the recurrence (before E1)."""
    R_prev = D_R
    D_p_new = D_p + D_v * d_t + R_prev @ iint_R @ a_hat
    D_v_new = D_v + R_prev @ int_R @ a_hat
    D_R_new = R_prev @ dR
    return D_R_new, D_v_new, D_p_new


def _frozen_phi_hwa(R_prev, dR, d_t, Jr, int_R, iint_R, int_RAJ, iint_RAJ, a_hat):
    from mvis.calibration.factors.lie_groups import skew
    _I3 = np.eye(3)
    phi_nn = np.eye(9)
    phi_nn[0:3, 0:3] = dR.T
    phi_nn[3:6, 0:3] = -R_prev @ skew(iint_R @ a_hat)
    phi_nn[3:6, 6:9] = _I3 * d_t
    phi_nn[6:9, 0:3] = -R_prev @ skew(int_R @ a_hat)
    H_wa = np.zeros((9, 6))
    H_wa[0:3, 0:3] = Jr * d_t
    H_wa[3:6, 0:3] = -R_prev @ iint_RAJ
    H_wa[6:9, 0:3] = -R_prev @ int_RAJ
    H_wa[3:6, 3:6] = R_prev @ iint_R
    H_wa[6:9, 3:6] = R_prev @ int_R
    return phi_nn, H_wa


def _aci_step_inputs():
    from mvis.calibration.factors.aci import integrate_RAJ
    from tests.helpers.speedup import imu_signal
    t, w, a = imu_signal()
    rng = np.random.default_rng(5)
    out = []
    D_R, D_v, D_p = np.eye(3), np.zeros(3), np.zeros(3)
    for k in range(len(t) - 1):                         # realistic: a 200 Hz walk
        d_t = float(t[k + 1] - t[k])
        w_hat, a_hat = 0.5 * (w[k] + w[k + 1]), 0.5 * (a[k] + a[k + 1])
        out.append((D_R, D_v, D_p, d_t, w_hat, a_hat))
        dR = integrate_RAJ(d_t, w_hat, a_hat)[0]
        D_R = D_R @ dR
        D_v = D_v + rng.normal(size=3) * 0.01
        D_p = D_p + D_v * d_t
    for _ in range(100):                                # random, incl. tiny and large rates
        w_hat = rng.normal(size=3) * 10.0 ** rng.uniform(-10, 1)
        out.append((_rand_rot(rng).matrix(), rng.normal(size=3), rng.normal(size=3),
                    float(10.0 ** rng.uniform(-5, -1)), w_hat, rng.normal(size=3) * 10))
    return out


def test_aci_step_helpers_match_the_frozen_blocks():
    from mvis.calibration.factors.aci import integrate_RAJ, _state_recurrence, _phi_hwa
    for D_R, D_v, D_p, d_t, w_hat, a_hat in _aci_step_inputs():
        dR, int_R, iint_R, Jr, int_RAJ, iint_RAJ = integrate_RAJ(d_t, w_hat, a_hat)
        for x, y in zip(_state_recurrence(D_R, D_v, D_p, d_t, dR, int_R, iint_R, a_hat),
                        _frozen_step_state(D_R, D_v, D_p, d_t, dR, int_R, iint_R, a_hat)):
            assert _same(x, y)
        args = (D_R, dR, d_t, Jr, int_R, iint_R, int_RAJ, iint_RAJ, a_hat)
        for x, y in zip(_phi_hwa(*args), _frozen_phi_hwa(*args)):
            assert _same(x, y)


# ---------------------------------------------------------------------------
# 5. CalibrationOptimizer's inline pose bracketing -> find_bracketing_poses
# ---------------------------------------------------------------------------
def _frozen_optimizer_bracket(t_lookup, bp_timeline_t, bp_timeline_idx):
    """optimizer.py, the bracketing block of build_graph (before E1); None = `continue`."""
    if t_lookup < bp_timeline_t[0] or t_lookup > bp_timeline_t[-1]:
        return None
    idx_right = np.searchsorted(bp_timeline_t, t_lookup)
    if idx_right >= len(bp_timeline_t):
        idx_left = len(bp_timeline_t) - 2
        idx_right = len(bp_timeline_t) - 1
    elif idx_right == 0:
        idx_left = 0
        idx_right = min(1, len(bp_timeline_t) - 1)
    elif abs(bp_timeline_t[idx_right] - t_lookup) < 1e-9:
        idx_left = idx_right
        if idx_right < len(bp_timeline_t) - 1:
            idx_right = idx_right + 1
        else:
            idx_left = idx_right - 1
    else:
        idx_left = idx_right - 1
    bp0_idx = bp_timeline_idx[idx_left]
    bp1_idx = bp_timeline_idx[idx_right]
    t0 = float(bp_timeline_t[idx_left])
    t1 = float(bp_timeline_t[idx_right])
    return bp0_idx, bp1_idx, t0, t1


def test_find_bracketing_poses_matches_the_frozen_optimizer_block():
    from mvis.calibration.factors.camera_projection import find_bracketing_poses
    rng = np.random.default_rng(6)
    n_checked = 0
    for n in (1, 2, 3, 20, 300):
        for _ in range(5):
            tl = np.sort(rng.uniform(0, 30, n)) if rng.integers(0, 2) else \
                np.arange(n) / 20.0 + 1700000000.0                  # realistic 20 Hz stamps
            idx = [int(i) for i in rng.permutation(10 * n)[:n]]
            qs = list(tl) + [tl[0] - 1e-3, tl[-1] + 1e-3, tl[0], tl[-1]] + \
                list(rng.uniform(tl[0] - 1, tl[-1] + 1, 50)) + list(tl + 5e-10) + list(tl - 5e-10)
            for q in qs:
                a = find_bracketing_poses(float(q), tl, idx)
                b = _frozen_optimizer_bracket(float(q), tl, idx)
                assert a == b and (a is None or [type(x) for x in a] == [type(x) for x in b])
                n_checked += 1
    assert n_checked > 1000


# ---------------------------------------------------------------------------
# 6. pipeline_cam's two inline re-indexings of solved board poses -> poses_by_frame
# ---------------------------------------------------------------------------
def _frozen_ds_phase1_poses(result_p1, n_frames, init_poses):
    """pipeline_cam, double-sphere phase 1 -> phase 2 warm start (before E1)."""
    p1_poses = [None] * n_frames
    vfi = result_p1.valid_frame_indices
    if not vfi:
        vfi = [i for i, p in enumerate(init_poses) if p is not None]
    if result_p1.board_poses_gtsam:
        for k, fi in enumerate(vfi):
            if k < len(result_p1.board_poses_gtsam):
                p1_poses[fi] = result_p1.board_poses_gtsam[k]
    return p1_poses


def _frozen_rerun_poses(result_r, n_frames, clean_poses, old):
    """pipeline_cam, single-camera re-solve after the diverged-pose cleanup (before E1)."""
    if result_r.board_poses_gtsam:
        vfi_r = result_r.valid_frame_indices
        if not vfi_r:
            vfi_r = [i for i, p in enumerate(clean_poses) if p is not None]
        new_p = [None] * n_frames
        for fi, pose in zip(vfi_r, result_r.board_poses_gtsam):
            if fi < len(new_p):
                new_p[fi] = pose
        return new_p
    return old


def test_poses_by_frame_matches_both_frozen_pipeline_cam_blocks():
    from types import SimpleNamespace
    from mvis.calibration.omni_schedule import poses_by_frame
    rng = np.random.default_rng(9)
    for _ in range(300):
        n = int(rng.integers(0, 30))
        init = [None if rng.random() < 0.3 else gtsam.Pose3() for _ in range(n)]
        cand = sorted(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist()) if n else []
        vfi = cand if rng.random() < 0.7 else []
        n_avail = len(vfi) if vfi else sum(p is not None for p in init)
        n_bp = int(rng.integers(0, n_avail + 2)) if rng.random() < 0.9 else 0
        bpg = [gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(k, 0, 0)) for k in range(n_bp)]
        res = SimpleNamespace(valid_frame_indices=vfi, board_poses_gtsam=bpg)
        new = poses_by_frame(res, n, init)
        assert new == _frozen_ds_phase1_poses(res, n, init)           # same objects, same slots
        old = object()
        ref = _frozen_rerun_poses(res, n, init, old)
        assert (poses_by_frame(res, n, init) if res.board_poses_gtsam else old) == ref


# ---------------------------------------------------------------------------
# 8. Corner masks x3 (filter_observations, pipeline_cam_imu._corner_mask,
#    plot_3d._corner_kept_mask) -> corner_subsample.corner_mask; timing dict x2
#    -> corner_subsample.corners_per_tag
# ---------------------------------------------------------------------------
_MODES = ["all", "top_left", "tl", "bl", "br", "tr", "bottom_left", "bottom_right",
          "top_right", "two_diag", "two_anti"]


def _frozen_filter_observations(obs, mode):
    from mvis.detection.corner_subsample import _SINGLE_CORNER, _MULTI_CORNER
    obs_list = list(obs)
    if mode == "all":
        return obs_list
    if mode in _SINGLE_CORNER:
        ci = _SINGLE_CORNER[mode]
        return [o for o in obs_list if o.point_id % 4 == ci]
    if mode in _MULTI_CORNER:
        keep = _MULTI_CORNER[mode]
        return [o for o in obs_list if (o.point_id % 4) in keep]
    raise ValueError(mode)


def _frozen_cam_imu_corner_mask(ids, mode):
    from mvis.detection.corner_subsample import _SINGLE_CORNER, _MULTI_CORNER
    if mode == "all":
        return np.ones(len(ids), bool)
    corner = np.asarray(ids, int) % 4
    if mode in _SINGLE_CORNER:
        return corner == _SINGLE_CORNER[mode]
    if mode in _MULTI_CORNER:
        return np.isin(corner, _MULTI_CORNER[mode])
    raise ValueError(mode)


def _frozen_plot_kept_mask(n_points, mode, has_tags):
    mask = np.ones(n_points, dtype=bool)
    if mode == "all" or not has_tags:
        return mask
    ids = np.arange(n_points)
    if mode in ("top_left", "tl"):
        mask = (ids % 4 == 3)
    elif mode == "bl":
        mask = (ids % 4 == 0)
    elif mode == "br":
        mask = (ids % 4 == 1)
    elif mode == "tr":
        mask = (ids % 4 == 2)
    elif mode == "two_diag":
        mask = (ids % 4 == 0) | (ids % 4 == 2)
    elif mode == "two_anti":
        mask = (ids % 4 == 1) | (ids % 4 == 3)
    return mask


def test_corner_mask_matches_the_three_frozen_masks():
    from mvis.core.types import Observation
    from mvis.detection.corner_subsample import corner_mask, filter_observations
    from mvis.calibration.pipeline_cam_imu import _corner_mask
    from mvis.analysis.plot_3d import _corner_kept_mask
    assert _corner_mask is corner_mask
    rng = np.random.default_rng(10)
    id_sets = [np.arange(0, 144), np.arange(0, 0), np.array([3]),
               rng.integers(0, 400, 77), rng.permutation(144)[:50]]
    for ids in id_sets:
        obs = [Observation(point_id=int(i), pixel=rng.uniform(0, 640, 2)) for i in ids]
        for mode in _MODES:
            assert _same(corner_mask(ids, mode), _frozen_cam_imu_corner_mask(ids, mode))
            assert _same(corner_mask(list(ids), mode), _frozen_cam_imu_corner_mask(list(ids), mode))
            new = filter_observations(obs, mode)
            old = _frozen_filter_observations(obs, mode)
            assert len(new) == len(old) and all(a is b for a, b in zip(new, old))
    for n in (0, 4, 144, 400):
        for mode in _MODES[:6] + ["two_diag", "two_anti"]:      # the modes the old plot copy knew
            for has_tags in (True, False):
                assert _same(_corner_kept_mask(n, mode, has_tags),
                             _frozen_plot_kept_mask(n, mode, has_tags))
    for bad in ("three", "TL", ""):
        with pytest.raises(ValueError):
            corner_mask(np.arange(8), bad)
        with pytest.raises(ValueError):
            filter_observations([], bad)
        with pytest.raises(ValueError):
            _corner_kept_mask(8, bad, True)                       # the plot copy now raises too
        assert _same(_corner_kept_mask(8, bad, False), np.ones(8, bool))


def test_corners_per_tag_matches_the_frozen_timing_dict():
    from mvis.detection.corner_subsample import corners_per_tag
    subsample_mult = {"top_left": 1.0, "two_diag": 2.0, "two_anti": 2.0, "all": 4.0}
    for mode in _MODES + ["three", "", "TL"]:
        a, b = corners_per_tag(mode), subsample_mult.get(mode, 1.0)
        assert a == b and type(a) is type(b)


# ---------------------------------------------------------------------------
# 9. Board-height gauge weights x2 (optimizer, pipeline_cam_imu) -> board_shape
# ---------------------------------------------------------------------------
def _frozen_gauge_w_optimizer(pts):
    pts = np.asarray(pts, float)
    x = pts[:, 0] - pts[:, 0].mean()
    y = pts[:, 1] - pts[:, 1].mean()
    N = len(pts)
    return np.column_stack([np.full(N, 1.0 / N), x / np.sum(x * x), y / np.sum(y * y)])


def _frozen_gauge_w_cam_imu(pts_B):
    pts_B = np.asarray(pts_B, float)
    N = len(pts_B)
    x = pts_B[:, 0] - pts_B[:, 0].mean()
    y = pts_B[:, 1] - pts_B[:, 1].mean()
    return np.column_stack([np.full(N, 1.0 / N), x / np.sum(x * x), y / np.sum(y * y)])


def test_board_height_gauge_weights_match_both_frozen_copies():
    from mvis.calibration.board_shape import board_height_gauge_weights
    from mvis.core.config import BoardConfig
    from mvis.detection.board_factory import create_board
    rng = np.random.default_rng(11)
    sets = [np.asarray(create_board(BoardConfig(board_type="apriltag", rows=6, cols=6,
                                                tag_size=0.088, tag_spacing=0.3)).all_points_3d(), float),
            np.asarray(create_board(BoardConfig(board_type="chessboard", rows=9, cols=6,
                                                square_size=0.03)).all_points_3d(), float)]
    sets += [np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                              rng.normal(0, 1e-3, n)]) for n in (3, 17, 144, 500)]
    assert len(sets) >= 4
    for pts in sets:
        W = board_height_gauge_weights(pts)
        assert _same(W, _frozen_gauge_w_optimizer(pts))
        assert _same(W, _frozen_gauge_w_cam_imu(list(map(list, pts))))


# ---------------------------------------------------------------------------
# 10. Velocity-jump gates x3 (detect_pnp_outliers, cam_imu_init, mono_sfm)
#     -> pose_alignment.velocity_jump_inliers
# ---------------------------------------------------------------------------
def _frozen_jump_loop_pnp(omegas, threshold, n):
    is_inlier = [True] * n
    for i in range(n):
        if i > 0 and omegas[i - 1] > threshold:
            is_inlier[i] = False
        if i < n - 1 and omegas[i] > threshold:
            is_inlier[i] = False
    return is_inlier


def _frozen_jump_loop_init(v, thr, n):
    ok = np.ones(n, bool)
    for i in range(n):
        if i > 0 and v[i - 1] > thr:
            ok[i] = False
        if i < n - 1 and v[i] > thr:
            ok[i] = False
    return ok


def _frozen_jump_loop_sfm(v, thr, n_poses):
    ok = np.ones(n_poses, bool)
    for k in range(n_poses):
        if k > 0 and v[k - 1] > thr:
            ok[k] = False
        if k < len(v) and v[k] > thr:
            ok[k] = False
    return ok


def test_velocity_jump_inliers_matches_the_three_frozen_loops():
    from mvis.calibration.pose_alignment import velocity_jump_inliers
    rng = np.random.default_rng(12)
    for _ in range(400):
        n = int(rng.integers(1, 60))
        v = np.abs(rng.normal(size=n - 1)) * 10.0 ** rng.uniform(-2, 1)
        if n > 3 and rng.random() < 0.5:
            v[rng.integers(0, n - 1)] = 50.0                  # a flip spike
        if n > 3 and rng.random() < 0.1:
            v[0] = np.nan
        thr = float(np.percentile(v, 95) * 3.0) if n > 1 else 1.0
        new = velocity_jump_inliers(v, thr)
        assert [bool(k) for k in new] == _frozen_jump_loop_pnp(v, thr, n)
        assert _same(new, _frozen_jump_loop_init(v, thr, n))
        assert _same(new, _frozen_jump_loop_sfm(v, thr, n))


def test_board_z_range_is_the_old_literal():
    from mvis.calibration.pose_alignment import _BOARD_Z_RANGE
    assert _BOARD_Z_RANGE == (0.05, 20.0)
    assert [type(x) for x in _BOARD_Z_RANGE] == [float, float]


# ---------------------------------------------------------------------------
# 11. pose_alignment: the two cross-correlation peaks -> _xcorr_offset, the two
#     nearest-time matchers -> _nearest_pairs
# ---------------------------------------------------------------------------
def _frozen_xcorr_block(s_ref, s_cam, n_grid, dt_sample, max_expected_dt):
    """Identical tail of estimate_time_offset_rotation / _homography (before E1):
    'weak' / 0.0 (no lag in range) / (dt_est, peak_corr)."""
    s_ref -= s_ref.mean()
    s_cam -= s_cam.mean()
    n_r = np.linalg.norm(s_ref)
    n_c = np.linalg.norm(s_cam)
    if n_r < 1e-10 or n_c < 1e-10:
        return "weak"
    s_ref /= n_r
    s_cam /= n_c
    corr = np.correlate(s_ref, s_cam, mode="full")
    lags = np.arange(-(n_grid - 1), n_grid) * dt_sample
    valid_mask = np.abs(lags) <= max_expected_dt
    if not np.any(valid_mask):
        return 0.0
    corr_valid = corr.copy()
    corr_valid[~valid_mask] = -np.inf
    peak_idx = np.argmax(corr_valid)
    return float(lags[peak_idx]), float(corr[peak_idx])


def test_xcorr_offset_matches_the_frozen_block():
    from mvis.calibration.pose_alignment import _xcorr_offset
    rng = np.random.default_rng(13)
    for _ in range(200):
        n = int(rng.integers(100, 3000))
        dt_sample = float(rng.choice([0.01, 0.0123]))
        t = np.arange(n) * dt_sample
        shift = rng.uniform(-0.3, 0.3)
        base = np.abs(np.sin(2.0 * t) + 0.3 * np.sin(7.1 * t + 1.0))
        s_ref = base + rng.normal(0, 0.05, n)
        s_cam = np.interp(t + shift, t, base) + rng.normal(0, 0.05, n)
        if rng.random() < 0.1:
            s_cam = np.full(n, 3.0)                            # too weak once zero-meaned
        max_dt = float(rng.choice([0.5, 1e-3, 0.0]))
        a1, a2, b1, b2 = s_ref.copy(), s_cam.copy(), s_ref.copy(), s_cam.copy()
        new = _xcorr_offset(a1, a2, dt_sample, max_dt)
        old = _frozen_xcorr_block(b1, b2, n, dt_sample, max_dt)
        assert _same(a1, b1) and _same(a2, b2)                 # same in-place effect
        if old == "weak":
            assert new is None
        elif old == 0.0:
            assert new == (0.0, None)
        else:
            assert new == old and all(type(x) is float for x in new)


def _frozen_match_relative_rotations(dR_ref, times_ref, dR_cam, times_cam, dt):
    all_dts = []
    if len(times_ref) > 1: all_dts.extend(np.diff(times_ref).tolist())
    if len(times_cam) > 1: all_dts.extend(np.diff(times_cam).tolist())
    max_td = np.median(all_dts) * 0.6 if all_dts else 0.1
    corrected = times_cam + dt
    matched, idx_ref, idx_cam = [], [], []
    for i, t_ref in enumerate(times_ref):
        diffs = np.abs(corrected - t_ref)
        j = np.argmin(diffs)
        if diffs[j] < max_td:
            matched.append((dR_ref[i], dR_cam[j]))
            idx_ref.append(i)
            idx_cam.append(j)
    return matched, idx_ref, idx_cam


def _frozen_translation_matches(ts_ref, ts_cam, dt):
    """The matching loop of estimate_translation_with_outliers (before E1)."""
    corrected = np.array(ts_cam) + dt
    all_dts = []
    if len(ts_ref) > 1: all_dts.extend(np.diff(ts_ref).tolist())
    if len(ts_cam) > 1: all_dts.extend(np.diff(ts_cam).tolist())
    max_td = np.median(all_dts) * 0.6 if all_dts else 0.1
    out = []
    for i, t_ref in enumerate(ts_ref):
        diffs = np.abs(corrected - t_ref)
        j = np.argmin(diffs)
        if diffs[j] > max_td:
            continue
        out.append((i, j))
    return out


def _stamp_sets(rng):
    out = []
    for _ in range(150):
        n_r, n_c = int(rng.integers(1, 80)), int(rng.integers(1, 80))
        hz_r, hz_c = rng.choice([20.0, 30.0, 4.0]), rng.choice([20.0, 30.0, 4.0])
        t_r = 1700000000.0 + np.arange(n_r) / hz_r + rng.normal(0, 1e-3, n_r)
        t_c = 1700000000.0 + np.arange(n_c) / hz_c + rng.normal(0, 1e-3, n_c)
        if rng.random() < 0.2:
            t_c[: n_c // 2] = t_r[0] + 0.25                    # ties: first index wins
        out.append((t_r, t_c, float(rng.uniform(-0.1, 0.1))))
    return out


def test_nearest_pairs_matches_both_frozen_matchers():
    from mvis.calibration.pose_alignment import _match_relative_rotations_with_indices, \
        _nearest_pairs
    rng = np.random.default_rng(14)
    for t_r, t_c, dt in _stamp_sets(rng):
        dR_r = [object() for _ in t_r]
        dR_c = [object() for _ in t_c]
        new = _match_relative_rotations_with_indices(dR_r, t_r, dR_c, t_c, dt)
        old = _frozen_match_relative_rotations(dR_r, t_r, dR_c, t_c, dt)
        assert all(a is c and b is d for (a, b), (c, d) in zip(new[0], old[0]))
        assert len(new[0]) == len(old[0])
        assert new[1] == old[1] and new[2] == old[2]
        assert [type(x) for x in new[2]] == [type(x) for x in old[2]]
        for ts_c in (t_c, list(t_c)):
            pairs, max_td = _nearest_pairs(t_r, ts_c, dt)
            assert [(i, j) for i, j, d in pairs if not d > max_td] == \
                _frozen_translation_matches(t_r, ts_c, dt)


# ---------------------------------------------------------------------------
# 12. Nearest reference stamp x3 (frame_selector._near_any, pipeline_cam's
#     co-frame gate, initializer's nearest-time pairing) -> frame_selector.nearest_ref
# ---------------------------------------------------------------------------
def _frozen_near_any(ts, ref_ts, tol):
    if len(ref_ts) == 0 or len(ts) == 0:
        return np.zeros(len(ts), dtype=bool)
    j = np.clip(np.searchsorted(ref_ts, ts), 1, max(len(ref_ts) - 1, 1))
    lo = ref_ts[np.maximum(j - 1, 0)]
    hi = ref_ts[np.minimum(j, len(ref_ts) - 1)]
    return np.minimum(np.abs(ts - lo), np.abs(ts - hi)) <= tol


def _frozen_pipeline_cam_keep(ts, ref_ts, tol):
    j = np.clip(np.searchsorted(ref_ts, ts), 1, max(len(ref_ts) - 1, 1))
    d = np.minimum(np.abs(ts - ref_ts[np.maximum(j - 1, 0)]),
                   np.abs(ts - ref_ts[np.minimum(j, len(ref_ts) - 1)]))
    return d <= tol


def _frozen_initializer_pair(ref_t, t):
    j = int(np.searchsorted(ref_t, t))
    near = [k for k in (j - 1, j) if 0 <= k < len(ref_t)]
    k = min(near, key=lambda k: abs(ref_t[k] - t))
    return k, abs(ref_t[k] - t)


def test_nearest_ref_matches_the_three_frozen_variants():
    from mvis.io.frame_selector import _near_any, nearest_ref
    rng = np.random.default_rng(15)
    for _ in range(300):
        n_r = int(rng.integers(1, 60))
        if rng.random() < 0.5:                                    # realistic 20 Hz rig stamps
            ref = 1700000000.0 + np.arange(n_r) / 20.0 + rng.normal(0, 2e-4, n_r)
            ts = 1700000000.0 + np.arange(int(rng.integers(0, 80))) / 30.0
        else:                                                     # random, with dup and ties
            ref = rng.integers(0, 40, n_r) * 0.5
            ts = rng.integers(-4, 90, int(rng.integers(0, 80))) * 0.25
        ref = np.sort(ref)
        tol = float(rng.choice([0.0, 0.01, 0.25, 0.5]))
        assert _same(_near_any(ts, ref, tol), _frozen_near_any(ts, ref, tol))
        if len(ts):
            assert _same(nearest_ref(ts, ref)[1] <= tol, _frozen_pipeline_cam_keep(ts, ref, tol))
        for t in ts:
            k, gap = int(nearest_ref(float(t), ref)[0]), abs(ref[int(nearest_ref(float(t), ref)[0])] - float(t))
            k0, gap0 = _frozen_initializer_pair(ref, float(t))
            assert ref[k] == ref[k0] and _same(gap, gap0)        # same reference stamp
    assert _same(_near_any(np.array([]), np.array([1.0]), 0.1), np.zeros(0, bool))
    assert _same(_near_any(np.array([1.0]), np.array([]), 0.1), np.zeros(1, bool))


# ---------------------------------------------------------------------------
# 13. The IMU stream methods of CalibScenario and MultiBoardScenario -> _ImuStreams
# ---------------------------------------------------------------------------
def _frozen_scenario_noise(sc):
    return dict(sigma_w=sc.sigma_w, sigma_a=sc.sigma_a, sigma_bg=sc.sigma_bg, sigma_ba=sc.sigma_ba)


def _frozen_base_imu(sc, times, seed=0):
    from mvis.simulation.scenario import simulate_base_imu
    return simulate_base_imu(sc.traj, times, sc.imu_intr, g_inW=sc.g_inW,
                             b_g=sc.b_g, b_a=sc.b_a, noise=_frozen_scenario_noise(sc), seed=seed)


def _frozen_aux_imu(sc, times, seed=0):
    from mvis.simulation.scenario import simulate_aux_imu
    return simulate_aux_imu(sc.traj, times, sc.aux, g_inW=sc.g_inW,
                            noise=_frozen_scenario_noise(sc), seed=seed)


def _same_stream(a, b):
    return all(_same(getattr(a, k), getattr(b, k)) for k in ("t", "w_m", "a_m"))


def test_scenario_imu_streams_match_the_frozen_methods():
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
    from mvis.detection.chessboard import Chessboard
    from mvis.imu import ImuModel, RPNG
    from mvis.simulation.imu_measurements import imu_times
    from mvis.simulation.multi_board import MultiBoardScenario, place_boards_ring, forward_T_CtoI
    from mvis.simulation.scenario import AuxImu, CalibScenario, _ImuStreams, make_scenario
    from mvis.simulation.trajectories import load_sine3d
    traj = load_sine3d(target_span=1.0)
    intr = ImuModel(x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
                    x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
                    x_tg=np.zeros(9), R=gtsam.Rot3.Expmap([0.02, -0.01, 0.03]).matrix(), model=RPNG)
    aux = AuxImu(T_IstoI=gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                                     gtsam.Point3(0.08, -0.04, 0.03)),
                 intr=ImuModel(model=RPNG), b_g=np.array([-0.008, 0.015, -0.01]),
                 b_a=np.array([0.02, -0.04, 0.03]), td=0.004)
    cal = _cal_from_vector(np.array([400., 400., 320., 240., -0.28, 0.07, 0.0005, -0.0004]), False)
    board = Chessboard(7, 7, 0.10)
    b_g, b_a = np.array([0.01, -0.02, 0.005]), np.array([0.05, -0.03, 0.02])
    t0 = traj.start_time + 0.5
    times = imu_times(t0, t0 + 2.0, 200.0)
    scs = []
    for noise in ({}, dict(sigma_w=1.6e-4, sigma_a=2e-3, sigma_bg=2e-5, sigma_ba=3e-4)):
        sc = make_scenario(traj, t0=t0, depth=3.0, board=board, cal=cal, T_CtoI=forward_T_CtoI(),
                           imu_intr=intr, b_g=b_g, b_a=b_a, image_size=(640, 480), aux=aux)
        for k, v in noise.items():
            setattr(sc, k, v)
        scs.append(sc)
        mb = MultiBoardScenario(traj=traj, boards=[board] * 4,
                                T_BinW_list=place_boards_ring(board, radius=1.5, directions=4),
                                cal=cal, is_fisheye=False, T_CtoI=forward_T_CtoI(),
                                image_size=(640, 480), imu_intr=intr, b_g=b_g, b_a=b_a,
                                aux=aux, **noise)
        scs.append(mb)
    for sc in scs:
        assert isinstance(sc, _ImuStreams)
        for seed in (0, 7):
            assert _same_stream(sc.base_imu(times, seed=seed), _frozen_base_imu(sc, times, seed))
            assert _same_stream(sc.aux_imu(times, seed=seed), _frozen_aux_imu(sc, times, seed))
    assert isinstance(scs[0], CalibScenario)
    scs[0].aux = None
    with pytest.raises(ValueError):
        scs[0].aux_imu(times)


# ---------------------------------------------------------------------------
# 14. camera_intrinsics._cal_to_vector: the serialize() parse -> np.delete(cal.vector(), 2)
# ---------------------------------------------------------------------------
def _frozen_cal_to_vector(cal):
    parts = cal.serialize().split()
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass
    fx = cal.fx()
    for i, v in enumerate(nums):
        if abs(v - fx) < 1e-10 and i + 8 < len(nums):
            params = nums[i:i + 9]
            return np.array([params[0], params[1], params[3], params[4],
                             params[5], params[6], params[7], params[8]])
    return np.array([cal.fx(), cal.fy(), cal.px(), cal.py(), cal.k1(), cal.k2(), 0.0, 0.0])


def test_cal_to_vector_matches_the_frozen_serialize_parse():
    """Bitwise on every realistic and random focal length. The old parse took the first
    serialized number equal to fx, so an fx equal to one of the archive header integers
    (0, 1, 15, 22 with this boost) read the header instead; no camera has such an fx."""
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector, _cal_to_vector
    rng = np.random.default_rng(16)
    vecs = [np.array([458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907,
                      0.00019359, 1.76187114e-05]),                       # EuRoC cam0
            np.array([190.97847715128717, 190.9733070521226, 254.93170605935475,
                      256.8974428996504, 0.0034823894022493434, 0.0007150348452162257,
                      -0.0020532361418706202, 0.00020293673591811182])]  # TUM-VI cam0
    for _ in range(500):
        f = 10.0 ** rng.uniform(1.5, 4)
        vecs.append(np.concatenate([[f, f * rng.uniform(0.98, 1.02)], rng.uniform(100, 2000, 2),
                                    rng.normal(size=4) * 10.0 ** rng.uniform(-6, 0, 4)]))
    for v in vecs:
        for fish in (False, True):
            cal = _cal_from_vector(v, fish)
            new, old = _cal_to_vector(cal), _frozen_cal_to_vector(cal)
            assert _same(new, old)
            assert _same(new, np.asarray(v, float))                     # and it round-trips


# ---------------------------------------------------------------------------
# 16. result_schema.calyx_dw_decomposed keeps its own element-wise QR sign fold.
#     Merging it into imu_model.decompose_dw (Q @ diag(s), diag(s) @ U) is NOT
#     bitwise: the matrix products drop the sign of exact zeros, and those reach the
#     read entries (U[0,1], U[0,2], U[1,2]) and the rotvec, e.g. diag(-1, 1, 1) prints
#     Dw_shear [-0.0, -0.0, 0.0] with the element-wise fold and [0.0, 0.0, 0.0] with
#     decompose_dw. The merge is a series-G candidate; this test guards the fold.
# ---------------------------------------------------------------------------
def _frozen_calyx_dw_decomposed(Dw, ndigits=5):
    A = np.asarray(Dw, float).reshape(3, 3)
    Q, U = np.linalg.qr(A)
    s = np.sign(np.diag(U))
    s[s == 0] = 1.0
    Q, U = Q * s[None, :], U * s[:, None]
    tr = float(np.trace(Q))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = float(np.arccos(c))
    if theta < 1e-12:
        rv = np.zeros(3)
    else:
        w = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
        rv = theta / (2.0 * np.sin(theta)) * w
    return {
        "Dw_scale": [round(float(U[i, i]), ndigits) for i in range(3)],
        "Dw_shear": [round(float(U[0, 1]), ndigits), round(float(U[0, 2]), ndigits),
                     round(float(U[1, 2]), ndigits)],
        "gyro_misalign_rotvec": [round(float(v), 6) for v in rv],
    }


def test_calyx_dw_decomposed_matches_the_frozen_fold():
    from mvis.io.result_schema import calyx_dw_decomposed
    rng = np.random.default_rng(17)
    th = 0.01
    rotz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    cases = [np.eye(3), np.diag([1.01, 0.99, 1.0]), -np.eye(3), np.triu(np.ones((3, 3))),
             np.zeros((3, 3)), np.diag([1.0, 0.0, 1.0]),
             # exact zeros in the read positions (signed-zero sensitive)
             np.diag([-1.0, 1.0, 1.0]),
             np.array([[1.01, 0.0, 0.0], [0.002, 0.99, 0.0], [0.0, 0.001, 1.0]]),
             rotz @ np.diag([1.01, 0.99, 1.0])]
    cases += [np.eye(3) + rng.normal(size=(3, 3)) * 10.0 ** rng.uniform(-6, -0.5) for _ in range(2000)]
    cases += [rng.normal(size=(3, 3)) for _ in range(200)]
    for A in cases:
        for nd in (5, 8):
            new, old = calyx_dw_decomposed(A, nd), _frozen_calyx_dw_decomposed(A, nd)
            assert repr(new) == repr(old)                         # signed zeros included
            assert repr(calyx_dw_decomposed(A.ravel().tolist(), nd)) == repr(old)


# ---------------------------------------------------------------------------
# 17. AprilTagBoard / ArucoBoard.all_points_3d -> the BoardGeometry default
# ---------------------------------------------------------------------------
def _frozen_all_points_3d(board):
    n = board.num_points()
    pts = np.zeros((n, 3), dtype=np.float64)
    for pid in range(n):
        pts[pid] = board._points_3d[pid]
    return pts


def test_all_points_3d_default_matches_the_frozen_board_copies():
    from mvis.core.config import BoardConfig
    from mvis.detection.board import BoardGeometry
    from mvis.detection.board_factory import create_board
    from mvis.detection.chessboard import Chessboard
    boards = [create_board(BoardConfig(board_type="apriltag", rows=r, cols=c, tag_size=s,
                                       tag_spacing=sp))
              for r, c, s, sp in ((6, 6, 0.088, 0.3), (7, 5, 0.024, 0.0072), (1, 1, 0.1, 0.25))]
    boards += [create_board(BoardConfig(board_type="aruco", rows=r, cols=c, marker_size=m,
                                        marker_separation=sp))
               for r, c, m, sp in ((5, 7, 0.04, 0.01), (2, 3, 0.1, 0.02))]
    for b in boards:
        assert type(b).all_points_3d is BoardGeometry.all_points_3d
        assert _same(b.all_points_3d(), _frozen_all_points_3d(b))
    ch = Chessboard(9, 6, 0.03)
    assert _same(ch.all_points_3d(), BoardGeometry.all_points_3d(ch))     # override agrees


# ---------------------------------------------------------------------------
# 15. Pixel -> normalized x3 (mono_sfm.normalize_detections, cam_imu_init's PnP
#     bearings, initializer._bearing_points) -> camera.normalized_points, each site
#     keeping its backend (cv2 / camera_model) and the initializer its 80 deg cap. G9 moved
#     the initializer's custom models from the unit-ray kernels to the camera_model
#     backend: same rays to rounding, so those compare to 1e-12, not bitwise
# ---------------------------------------------------------------------------
_NP_CAMS = {
    "radtan": [458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76e-05],
    "equidistant": [190.978, 190.973, 254.932, 256.897, 0.00348, 0.000715, -0.00205, 0.000203],
    "kb4": [190.978, 190.973, 254.932, 256.897, 0.00348, 0.000715, -0.00205, 0.000203],
    "double_sphere": [156.0, 155.9, 254.9, 256.7, -0.18, 0.59],
    "eucm": [161.0, 160.9, 254.9, 256.7, 0.6, 1.1],
    "ucm": [250.0, 250.0, 254.9, 256.7, 0.6],
    "fov": [190.0, 190.0, 254.9, 256.7, 0.92],
    "omni_radtan": [380.0, 380.0, 254.9, 256.7, 0.9, -0.1, 0.01, 1e-4, -2e-4],
}


def _frozen_normalize_detections(ids_list, uv_list, cal_vec, model):
    import cv2
    from mvis.core.camera_models import canonical_dist
    from mvis.camera.camera_model import build_camera_model
    cal_vec = np.asarray(cal_vec, float)
    K = np.array([[cal_vec[0], 0, cal_vec[2]], [0, cal_vec[1], cal_vec[3]], [0, 0, 1.0]])
    dist = np.asarray(canonical_dist(model, cal_vec[4:])) if model == "radtan" else None
    out = []
    for ids, uv in zip(ids_list, uv_list):
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if len(uv) == 0:
            out.append((np.array([], int), np.zeros((0, 2))))
            continue
        if model == "radtan":
            xy = cv2.undistortPoints(uv, K, dist).reshape(-1, 2)
            out.append((np.asarray(ids, int), xy))
        else:
            xy, valid = build_camera_model(model, cal_vec).unproject_to_normalized(uv.reshape(-1, 2))
            out.append((np.asarray(ids, int)[valid], xy[valid]))
    return out


def _frozen_bearing_points(img_pts, K, dist_coeffs, distortion_model):
    from mvis.calibration.initializer import _MAX_PNP_RAY_DEG, _full_cal_vec
    from mvis.camera.camera_model import build_camera_model
    from mvis.calibration.factors import custom_projections as cp
    kernel = {"fov": cp.unproject_fov, "double_sphere": cp.unproject_double_sphere,
              "eucm": cp.unproject_eucm, "ucm": cp.unproject_ucm,
              "omni_radtan": cp.unproject_omni_radtan, "kb4": cp.unproject_kb}
    n = len(img_pts)
    rays = np.full((n, 3), np.nan)
    if distortion_model == "equidistant":
        und, valid = build_camera_model(
            distortion_model, _full_cal_vec(K, dist_coeffs)).unproject_to_normalized(img_pts)
        rays[valid, :2], rays[valid, 2] = und[valid], 1.0
    else:
        cal_vec = _full_cal_vec(K, dist_coeffs)
        for i, pt in enumerate(img_pts):
            r = kernel[distortion_model](pt, cal_vec)
            if r is not None:
                rays[i] = r
    norm = np.linalg.norm(rays, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = rays[:, 2] / norm
    keep = np.isfinite(rays).all(axis=1) & (cos_t > np.cos(np.deg2rad(_MAX_PNP_RAY_DEG)))
    pts = rays[keep, :2] / rays[keep, 2:3]
    return pts.astype(np.float64), keep


def _np_pixel_sets(rng):
    grid = np.stack(np.meshgrid(np.linspace(-20, 532, 23), np.linspace(-20, 532, 23)), -1)
    return [grid.reshape(-1, 2), rng.uniform(0, 512, (300, 2)), rng.uniform(-200, 700, (50, 2)),
            np.zeros((0, 2)), np.array([[254.9, 256.7]])]


def test_normalized_points_matches_the_three_frozen_sites():
    from mvis.calibration.initializer import _bearing_points
    from mvis.calibration.mono_sfm import normalize_detections
    from mvis.camera.camera_model import build_camera_model
    from mvis.camera.normalized_points import normalized_points
    rng = np.random.default_rng(18)
    for model, cv in _NP_CAMS.items():
        cv = np.asarray(cv, float)
        sets = _np_pixel_sets(rng)
        ids = [np.arange(len(s)) for s in sets]
        new = normalize_detections(ids, sets, cv, model=model)
        old = _frozen_normalize_detections(ids, sets, cv, model)
        for (a, b), (c, d) in zip(new, old):
            assert _same(a, c) and _same(b, d)
        if model == "radtan":
            continue
        for s in sets:                                           # cam_imu_init PnP bearings
            a = normalized_points(s, model, cv)
            b = build_camera_model(model, cv).unproject_to_normalized(s)
            assert _same(a[0], b[0]) and _same(a[1], b[1])
        if model in ("radtan", "kb4"):
            continue
        K = np.array([[cv[0], 0, cv[2]], [0, cv[1], cv[3]], [0, 0, 1.0]])
        for s in sets[:3] + sets[4:]:                             # initializer, PnP rays
            a, b = _bearing_points(s, K, cv[4:], model), _frozen_bearing_points(s, K, cv[4:], model)
            if model == "equidistant":
                assert _same(a[0], b[0]) and _same(a[1], b[1])
            else:
                assert _same(a[1], b[1])
                np.testing.assert_allclose(a[0], b[0], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Cam-IMU result tails (board multi, feature multi, multi-board pipeline; the
#    single-feature path waits for G4) -> _marginal_sigmas, _tga_diagnostics, _bias_fields
# ---------------------------------------------------------------------------
def _frozen_marginals(graph, res, calib_bases, n_cams, cfg):
    from mvis.calibration.pipeline_cam_imu import _Cc, I0
    try:
        marg = gtsam.Marginals(graph, res)
        sigma_Cs = [np.sqrt(np.diag(marg.marginalCovariance(_Cc(c)))) for c in range(n_cams)]
        sigma_I = (np.sqrt(np.diag(marg.marginalCovariance(I0)))
                   if cfg.imu_intrinsics != "pinned" else None)
    except RuntimeError:
        marg = None
        sigma_Cs = [np.full(calib_bases[c].DIM, np.nan) for c in range(n_cams)]
        sigma_I = None
    return marg, sigma_Cs, sigma_I


def _frozen_tga(diagnostics, res, marg):
    from mvis.calibration.pipeline_cam_imu import O0
    if res.exists(O0):
        diagnostics['t_ga'] = float(res.atVector(O0)[0])
        try:
            diagnostics['t_ga_sigma'] = float(np.sqrt(marg.marginalCovariance(O0)[0, 0]))
        except Exception:
            diagnostics['t_ga_sigma'] = None


def _frozen_bias(res, n_states):
    from mvis.calibration.pipeline_cam_imu import _Vk
    vb0 = res.atVector(_Vk(0))
    vbN = res.atVector(_Vk(n_states - 1))
    return dict(b_g=np.asarray(vb0[3:6], float), b_a=np.asarray(vb0[6:9], float),
                b_g_drift=np.asarray(vbN[3:6] - vb0[3:6], float),
                b_a_drift=np.asarray(vbN[6:9] - vb0[6:9], float))


def _tail_graph(rng, n_cams, n_states, with_tga, determinate):
    from types import SimpleNamespace
    from mvis.calibration.pipeline_cam_imu import _Cc, I0, O0, _Vk
    graph, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    bases = []
    for c in range(n_cams):
        dim = int(rng.choice([12, 14, 16]))
        bases.append(SimpleNamespace(DIM=dim))
        A = rng.normal(size=(dim, dim))
        vals.insert(_Cc(c), rng.normal(size=dim))
        graph.add(gtsam.PriorFactorVector(_Cc(c), np.zeros(dim),
                                          gtsam.noiseModel.Gaussian.Covariance(A @ A.T + np.eye(dim))))
    keys = [I0] + ([O0] if with_tga else []) + [_Vk(k) for k in range(n_states)]
    for k in keys:
        d = 24 if k == I0 else 1 if k == O0 else 9
        vals.insert(k, rng.normal(size=d))
        if determinate or k != I0:
            graph.add(gtsam.PriorFactorVector(k, np.zeros(d),
                                              gtsam.noiseModel.Diagonal.Sigmas(rng.uniform(0.1, 2, d))))
        else:                                                  # I0 seen but unconstrained

            def _flat(this, values, H, d=d):
                if H is not None:
                    H[0] = np.zeros((1, d))
                return np.zeros(1)
            graph.add(gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(1), [k], _flat))
    return graph, vals, bases


def test_result_tail_helpers_match_the_frozen_blocks():
    from types import SimpleNamespace
    from mvis.calibration.pipeline_cam_imu import _marginal_sigmas, _tga_diagnostics, _bias_fields
    rng = np.random.default_rng(19)
    n_indet = 0
    for trial in range(24):
        n_cams, n_states = int(rng.integers(1, 4)), int(rng.integers(2, 6))
        graph, res, bases = _tail_graph(rng, n_cams, n_states, trial % 2 == 0, trial % 3 != 0)
        for mode in ("calibrated", "pinned"):
            cfg = SimpleNamespace(imu_intrinsics=mode)
            m_new, c_new, i_new = _marginal_sigmas(graph, res, bases, n_cams, cfg)
            m_old, c_old, i_old = _frozen_marginals(graph, res, bases, n_cams, cfg)
            assert (m_new is None) == (m_old is None)
            n_indet += m_new is None
            assert all(_same(a, b) for a, b in zip(c_new, c_old)) and len(c_new) == len(c_old)
            assert (i_new is None and i_old is None) or _same(i_new, i_old)
            d_new, d_old = {"x": 1}, {"x": 1}
            _tga_diagnostics(d_new, res, m_new)
            _frozen_tga(d_old, res, m_old)
            assert list(d_new) == list(d_old) and repr(d_new) == repr(d_old)
        b_new, b_old = _bias_fields(res, n_states), _frozen_bias(res, n_states)
        assert list(b_new) == list(b_old) and all(_same(b_new[k], b_old[k]) for k in b_old)
    assert n_indet > 0                                         # the indeterminate branch ran


# ---------------------------------------------------------------------------
# 18. RS row offsets x3 (the three reprojection factors' __init__) -> CornerReprojMixin._row_offsets
# ---------------------------------------------------------------------------
def _frozen_rho(fac):
    from mvis.core.shutter import row_offset
    return np.array([row_offset(v, fac.h_px, fac.rolling_mode) for v in fac.uv[:, 1]])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_offsets_match_the_three_frozen_init_lines(seed):
    interp, imu, feat, _, _ = _reproj_graph(seed)
    modes = set()
    for _, _, fac in interp + imu + feat:
        assert _same(fac.rho, _frozen_rho(fac))
        modes.add(fac.rolling_mode)
    assert modes == {0, 1}
    # edge rows (first, last, past the image) through the live helper on a factor
    from tests.helpers.speedup import _H
    fac = interp[0][2]
    for mode in (0, 1, 2):
        fac.uv = np.array([[1.0, 0.0], [1.0, _H - 1.0], [1.0, float(_H)], [1.0, -3.5]])
        fac.rolling_mode = mode
        assert _same(fac._row_offsets(), _frozen_rho(fac))
