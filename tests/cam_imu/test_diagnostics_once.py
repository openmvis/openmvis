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
"""Diagnostics computed once (cam-imu-speedup step D1, BITWISE).

`_run_cam_imu_calibration_multi_once` used to evaluate `graph.error` twice on
top of the optimizer's own evaluations and to run the per-corner projection
pass at `res` twice (per-frame RMS series, then the pooled Kalibr population).
D1 takes `err_init` / `err_final` from `LevenbergMarquardtOptimizer.error()`
and builds ONE corner-norm table per Values (`_corner_norm_table`) that both
helpers read. These tests pin the bitwise equivalences D1 relies on:

  1. `opt.error()` == `graph.error(values)` right after construction and
     == `graph.error(res)` after `optimize()`, exact float equality, on a graph
     of Huber-robust CustomFactors (the vision factors) + priors — with GTSAM's
     default params, the production `_lm_params`, and a start at the optimum;
  2. the fused table reproduces the frozen two-pass oracle
     (`tests/_reference/ref_diagnostics`) bitwise: per-frame RMS, camera per
     frame, pooled norms, camera per corner — including a frame whose corners
     are ALL guarded (NaN), a partially guarded frame, the legacy `keys is None`
     factors, factors built through `make_factor`, and the empty factor list;
  3. on the sim pipeline, `err_init` / `err_final` in the diagnostics equal
     `graph.error` at the captured initial Values / at the returned Values, and
     the reprojection series equal the oracle's two-pass output at those Values.
"""
import numpy as np
import gtsam
import pytest

from mvis.calibration import pipeline_cam_imu as P
from mvis.calibration.pipeline_cam_imu import (CamImuCalibConfig, X, B0, _Cc,
                                               _corner_norm_table, _reproj_rms,
                                               _reproj_corner_norms, CornerTable)
from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from tests._reference import ref_diagnostics

_POSES = [gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.05 + 0.04 * k),
                      gtsam.Point3(0.01 * k, 0, 0)) for k in range(4)]
_TB = gtsam.Pose3(gtsam.Rot3.Rodrigues(1.3, 0.1, 0), gtsam.Point3(0, 0, 1.5))
_CV = [350., 350, 512, 512, 0.0, 0.0, 0.0, 0.0]
_IMG = (1024, 1024)


def _eq(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def _board_corners(n, rng):
    p_B = np.zeros((n, 3))
    p_B[:, :2] = rng.uniform(-0.15, 0.15, (n, 2))
    return p_B


def _factor(c, j, n, rng, base, offset_px=0.4, nan_corners=()):
    """Interp board factor for camera `c` bracketed by states j-1, j whose
    residual at xi = 0 is -offset (plus noise); `nan_corners` get a NaN board
    point (guard-skipped: NaN corner norm)."""
    p_B = _board_corners(n, rng)
    f0 = CameraCalibInterpFactor(np.zeros((n, 2)), p_B, 0.01, 0.05, base, _IMG)
    uv = f0.error(_POSES[j - 1], _POSES[j], _TB, np.zeros(base.DIM)).reshape(-1, 2)
    uv = uv + offset_px * rng.standard_normal((n, 2))
    for k in nan_corners:
        p_B[k] = np.nan
    return CameraCalibInterpFactor(uv, p_B, 0.01, 0.05, base, _IMG)


def _fixture(make_factors):
    """cam_factors + Values: camera 0 radtan (12 corners; frame 2 has one NaN
    corner), camera 1 equidistant (8 corners; frame 3 ALL NaN). `make_factors`
    builds them through `make_factor` (own keys) instead of the legacy path."""
    rng = np.random.default_rng(3)
    bases = [CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.array(_CV), model="radtan"),
             CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.array(_CV), model="equidistant")]
    spec = [(0, 1, 12, ()), (0, 2, 12, (4,)), (0, 3, 12, ()),
            (1, 1, 8, ()), (1, 2, 8, (0, 7)), (1, 3, 8, tuple(range(8)))]
    cam_factors = []
    gfactors = []
    for c, j, n, nans in spec:
        fac = _factor(c, j, n, rng, bases[c], nan_corners=nans)
        if make_factors:
            gfactors.append(fac.make_factor(X(j - 1), X(j), B0, _Cc(c),
                                            sigma_px=1.0, huber_px=1.0))
        cam_factors.append((c, j, fac))
    values = gtsam.Values()
    for k, pose in enumerate(_POSES):
        values.insert(X(k), pose)
    values.insert(B0, _TB)
    for c, base in enumerate(bases):
        values.insert(_Cc(c), np.zeros(base.DIM))
    return cam_factors, values, bases, gfactors


# ---------------------------------------------------------------------------
# (2) fused table == two-pass oracle, bitwise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("make_factors", [False, True], ids=["legacy_keys", "make_factor"])
def test_corner_table_matches_two_pass_oracle(make_factors):
    cam_factors, values, _bases, _g = _fixture(make_factors)
    assert all((f.keys is not None) == make_factors for _c, _j, f in cam_factors)
    table = _corner_norm_table(cam_factors, values)
    assert isinstance(table, CornerTable)
    assert table.e.shape == (3 * 12 + 3 * 8,) and table.e.dtype == np.float64
    assert _eq(table.offsets, np.array([0, 12, 24, 36, 44, 52, 60]))
    assert _eq(table.frame_cam, np.array([0, 0, 0, 1, 1, 1]))
    assert _eq(table.corner_cam, np.repeat([0, 1], [36, 24]))
    # guard pattern: one NaN in frame 2, two in frame 5, ALL of frame 6
    assert np.isnan(table.e).sum() == 1 + 2 + 8
    assert np.isnan(table.e[52:60]).all() and np.isnan(table.e[12 + 4])

    rms_ref, cam_ref = ref_diagnostics._reproj_rms(cam_factors, values)
    px_ref, pc_ref = ref_diagnostics._reproj_corner_norms(cam_factors, values)
    assert np.isnan(rms_ref[5]) and np.isfinite(rms_ref[:5]).all()    # the all-guarded frame
    assert px_ref.shape == (60 - 11,) and np.isfinite(px_ref).all()

    rms, cam = table.frame_rms()
    px, pc = table.pooled()
    assert _eq(rms, rms_ref) and _eq(cam, cam_ref)
    assert _eq(px, px_ref) and _eq(pc, pc_ref)
    # the wrappers: with and without a prebuilt table
    for t in (None, table):
        r = _reproj_rms(cam_factors, values, table=t)
        q = _reproj_corner_norms(cam_factors, values, table=t)
        assert _eq(r[0], rms_ref) and _eq(r[1], cam_ref)
        assert _eq(q[0], px_ref) and _eq(q[1], pc_ref)
    # the live two-pass path (table=None) is what the oracle froze
    assert _eq(P._reproj_rms(cam_factors, values)[0], rms_ref)
    # frame_rms returns a fresh camera array (no aliasing with the table)
    assert cam is not table.frame_cam


def test_corner_table_empty_factor_list():
    values = gtsam.Values()
    table = _corner_norm_table([], values)
    assert table.e.shape == (0,) and table.e.dtype == np.float64
    assert _eq(table.offsets, np.zeros(1, int))
    assert table.frame_cam.shape == (0,) and table.corner_cam.shape == (0,)
    rms, cam = table.frame_rms()
    px, pc = table.pooled()
    rms_ref, cam_ref = ref_diagnostics._reproj_rms([], values)
    px_ref, pc_ref = ref_diagnostics._reproj_corner_norms([], values)
    assert _eq(rms, rms_ref) and _eq(cam, cam_ref)
    assert _eq(px, px_ref) and _eq(pc, pc_ref)
    assert _eq(_reproj_corner_norms([], values)[0], px_ref)


def test_corner_table_single_all_guarded_frame():
    """Only one frame and every corner guarded: rms NaN, pool empty."""
    cam_factors, values, _bases, _g = _fixture(False)
    sel = [cam_factors[5]]
    table = _corner_norm_table(sel, values)
    rms, cam = table.frame_rms()
    px, pc = table.pooled()
    assert rms.shape == (1,) and np.isnan(rms[0]) and _eq(cam, np.array([1]))
    assert px.shape == (0,) and pc.shape == (0,) and pc.dtype == np.int64
    rms_ref, _ = ref_diagnostics._reproj_rms(sel, values)
    px_ref, pc_ref = ref_diagnostics._reproj_corner_norms(sel, values)
    assert _eq(rms, rms_ref) and _eq(px, px_ref) and _eq(pc, pc_ref)


# ---------------------------------------------------------------------------
# (1) opt.error() == graph.error, exact
# ---------------------------------------------------------------------------
def _graph(perturb):
    cam_factors, values, bases, gfactors = _fixture(True)
    graph = gtsam.NonlinearFactorGraph()
    for f in gfactors:
        graph.add(f)
    for k, pose in enumerate(_POSES):
        graph.add(gtsam.PriorFactorPose3(X(k), pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.05)))
    graph.add(gtsam.PriorFactorPose3(B0, _TB, gtsam.noiseModel.Isotropic.Sigma(6, 0.05)))
    for c, base in enumerate(bases):
        graph.add(gtsam.PriorFactorVector(_Cc(c), np.zeros(base.DIM),
                                          gtsam.noiseModel.Isotropic.Sigma(base.DIM, 1.0)))
    if perturb:
        rng = np.random.default_rng(11)
        for k in range(len(_POSES)):
            values.update(X(k), values.atPose3(X(k)).retract(rng.normal(0, 0.01, 6)))
        values.update(B0, values.atPose3(B0).retract(rng.normal(0, 0.01, 6)))
    return graph, values


def _params(kind, max_iterations):
    if kind == "gtsam_default":
        p = gtsam.LevenbergMarquardtParams()
        p.setMaxIterations(max_iterations)
        return p
    return P._lm_params(CamImuCalibConfig(max_iterations=max_iterations))


@pytest.mark.parametrize("kind", ["gtsam_default", "production"])
@pytest.mark.parametrize("start,max_iterations", [("perturbed", 1), ("perturbed", 6),
                                                  ("optimum", 3)])
def test_optimizer_error_equals_graph_error(kind, start, max_iterations):
    graph, values = _graph(perturb=(start == "perturbed"))
    e0 = graph.error(values)
    assert np.isfinite(e0) and e0 > 0
    opt = gtsam.LevenbergMarquardtOptimizer(graph, values, _params(kind, max_iterations))
    assert opt.error() == e0                                   # exact
    assert opt.error() == graph.error(values)                  # the Values are untouched
    res = opt.optimize()
    e1 = graph.error(res)
    assert opt.error() == e1                                   # exact
    assert opt.error() == graph.error(opt.values())
    assert 1 <= opt.iterations() <= max_iterations
    if start == "perturbed":
        assert e1 < e0                                         # the LM did something
    assert graph.error(values) == e0                           # initial Values intact


# ---------------------------------------------------------------------------
# (3) the pipeline's diagnostics vs graph.error / the two-pass oracle
# ---------------------------------------------------------------------------
def test_once_pipeline_diagnostics_match_graph_and_oracle(monkeypatch):
    from mvis.detection.chessboard import Chessboard
    from mvis.calibration.pipeline_cam_imu import run_cam_imu_calibration
    from tests.helpers.sim_scenarios import scenario_for_pipeline as _scenario, data_for_pipeline as _data, _RADTAN, _IMG as _SIM_IMG

    sc = _scenario(sigma_px=0.5)
    t_cam, cam, s = _data(sc, win=12.0, seed=3)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)
    # interp: this compares repeated graph.error calls bitwise, and the linearized imu factor's
    # error depends on the point it last integrated at.
    cfg = CamImuCalibConfig(estimate_t_ga=False, state_hz=10.0, sigma_px=0.5, vision_pose="interp",
                            huber_px=2.0, imu_intrinsics="pinned", max_iterations=3)
    captured = {}
    orig = P.build_joint_graph

    def spy(*args, **kw):
        out = orig(*args, **kw)
        captured["graph"], captured["values"], captured["cam_factors"] = out[0], out[1], out[2]
        captured["err_init"] = out[0].error(out[1])          # before the pipeline touches it
        captured["pre"] = ref_diagnostics._reproj_rms(out[2], out[1])
        return out

    monkeypatch.setattr(P, "build_joint_graph", spy)
    r = run_cam_imu_calibration(t_cam, list(cam.ids), list(cam.uv), board_pts,
                                _RADTAN, s.t, s.w_m, s.a_m, _SIM_IMG, config=cfg)
    d = r.diagnostics
    graph, values, cam_factors = captured["graph"], captured["values"], captured["cam_factors"]
    assert d["iterations"] >= 1 and len(cam_factors) > 10
    assert d["err_init"] == captured["err_init"] == graph.error(values)
    assert d["err_final"] == graph.error(r.values)
    assert d["err_final"] < d["err_init"]
    # reprojection diagnostics == the frozen two-pass oracle at the same Values
    assert _eq(d["reproj_rms_pre_px"], captured["pre"][0])
    rms_ref, cam_ref = ref_diagnostics._reproj_rms(cam_factors, r.values)
    px_ref, _pc_ref = ref_diagnostics._reproj_corner_norms(cam_factors, r.values)
    assert _eq(d["reproj_rms_post_px"], rms_ref) and _eq(d["reproj_frame_cam"], cam_ref)
    from mvis.calibration.reproj_stats import corner_stats
    cs = corner_stats(px_ref)
    assert d["reproj_rms_px"] == cs["rms"] and d["reproj_median_px"] == cs["median"]
    assert d["reproj_p95_px"] == cs["p95"]
