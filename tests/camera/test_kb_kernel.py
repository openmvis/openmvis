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
"""The KB-N kernel at N = 4 (kb4), beyond tests/camera/test_kb4.py, and shared rules:

  * N = 4 is gtsam Cal3Fisheye (pixels + Jacobians); analytic Jacobians vs finite
    differences incl. near-axis and theta > pi/2 points, project/unproject round trip,
    fold (non-monotonic) -> None;
  * the camera-only unified factor, landmark block per custom model;
  * sanity rule (pass-2 guardrail == regression health), CLI camchain skip, mvisdb
    enum + schema CHECK, regression report columns.
"""
import logging
import sqlite3
from types import SimpleNamespace

import numpy as np
import gtsam
import pytest
import yaml

from mvis.calibration.factors.custom_projections import (
    project_kb, unproject_kb,
)
from mvis.camera import KannalaBrandt4, PinholeEquidistant, create_camera_model
from mvis.core import camera_models as cm

# kb4 GT, monotonic out to the 2.6 rad the points below reach (fx 250)
KB4_GT = np.array([250.0, 251.0, 501.0, 498.0, 0.02, -0.005, 5e-4, 1e-5])
KB4_EQ = np.array([400.0, 410.0, 320.0, 240.0, -0.01, 3e-3, -5e-4, 1e-4])
_IMG = (1000, 1000)


def _dirs(rng, n, th_lo, th_hi):
    """n unit directions with incidence theta in [th_lo, th_hi] (rad)."""
    th = rng.uniform(th_lo, th_hi, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)


# ── KB-N kernel: N = 4 == gtsam Cal3Fisheye ('equidistant') ──

def test_kb4_matches_gtsam_equidistant():
    rng = np.random.default_rng(0)
    eq = PinholeEquidistant(*KB4_EQ[:4], KB4_EQ[4:])
    pts = list(_dirs(rng, 200, 0.0, np.deg2rad(85.0)) * rng.uniform(0.2, 5, (200, 1)))
    pts += [np.array([1e-12, -2e-12, 0.7]), np.array([0.0, 0.0, 1.3]),
            np.array([3e-13, 0.0, 2.0])]                    # rho ~ 1e-12, on axis
    for P in pts:
        uv, Jp, Jc = project_kb(P, KB4_EQ, True)
        uv_g, Jp_g, Jc_g = eq.project(P, jac=True)
        assert np.abs(uv - uv_g).max() < 1e-9, P
        np.testing.assert_allclose(Jp, Jp_g, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(Jc, Jc_g, rtol=1e-9, atol=1e-9)   # 2 x 8
        ray = unproject_kb(uv, KB4_EQ)
        np.testing.assert_allclose(ray, P / np.linalg.norm(P), atol=1e-10)


def _kb_points():
    rng = np.random.default_rng(1)
    pts = list(_dirs(rng, 60, 0.0, np.pi / 2) * rng.uniform(0.3, 3, (60, 1)))
    pts += list(_dirs(rng, 40, np.pi / 2 + 0.05, 2.6) * rng.uniform(0.3, 3, (40, 1)))
    pts += [np.array([1e-12, 2e-12, 1.0]), np.array([-1e-12, 0.0, 0.4]),
            np.array([0.0, 0.0, 2.0])]
    return pts


def _fd_cols(fn, x, eps):
    cols = []
    for i in range(len(x)):
        d = np.zeros(len(x)); d[i] = eps[i]
        cols.append((fn(x + d) - fn(x - d)) / (2.0 * eps[i]))
    return np.stack(cols, 1)


@pytest.mark.parametrize("P", _kb_points())
def test_kb4_jacobians_match_fd(P):
    uv, Jp, Jc = project_kb(P, KB4_GT, True)
    assert uv is not None and Jc.shape == (2, 8)
    s = 1e-6 * np.linalg.norm(P)
    Jp_fd = _fd_cols(lambda x: project_kb(x, KB4_GT)[0], P, [s] * 3)
    np.testing.assert_allclose(Jp, Jp_fd, rtol=1e-6, atol=1e-4 * np.abs(Jp_fd).max())
    # the pixel is linear in every cal param, so tiny FD steps are exact; the
    # k-steps stay tiny so a high-theta point never crosses the fold
    Jc_fd = _fd_cols(lambda c: project_kb(P, c)[0], KB4_GT, [1e-5] * 4 + [1e-9] * 4)
    np.testing.assert_allclose(Jc, Jc_fd, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("P", _kb_points())
def test_kb4_round_trip_incl_beyond_90deg(P):
    uv, _, _ = project_kb(P, KB4_GT)
    ray = unproject_kb(uv, KB4_GT)
    assert ray is not None
    np.testing.assert_allclose(ray, P / np.linalg.norm(P), atol=1e-9)
    m = KannalaBrandt4(*KB4_GT[:4], KB4_GT[4:])
    if P[2] > 1e-6:                            # z-depth (D12); none past 90 deg
        np.testing.assert_allclose(m.unproject(uv, depth=2.0), 2.0 * P / P[2], rtol=1e-8,
                                   atol=1e-8)
    else:
        assert np.isnan(m.unproject(uv, depth=2.0)).all()


def test_kb4_invalid_points():
    assert project_kb(np.array([0.0, 0.0, -1.0]), KB4_GT)[0] is None   # theta = pi
    assert project_kb(np.zeros(3), KB4_GT)[0] is None
    m = create_camera_model("kb4", *KB4_GT[:4], KB4_GT[4:])
    assert np.isnan(m.project(np.array([0.0, 0.0, -1.0]))).all()
    assert m.project(np.array([0.0, 0.0, -1.0]), jac=True)[1] is None


def test_non_monotonic_coefficients_none_beyond_fold():
    """theta_d = theta - 0.3 theta^3 folds at theta = 1/sqrt(0.9) ~ 1.054 rad
    (max theta_d ~ 0.703): beyond it project AND unproject return None."""
    cv = np.array([300.0, 300.0, 320.0, 240.0, -0.3] + [0.0] * 3)
    fold = 1.0 / np.sqrt(0.9)
    for th in (0.3, 0.9, fold - 0.01):
        P = np.array([np.sin(th), 0.0, np.cos(th)])
        uv = project_kb(P, cv)[0]
        assert uv is not None
        np.testing.assert_allclose(unproject_kb(uv, cv), P, atol=1e-8)
    for th in (fold + 0.01, 1.3, 2.0):
        assert project_kb(np.array([np.sin(th), 0.0, np.cos(th)]), cv)[0] is None
    rd_max = fold - 0.3 * fold ** 3
    assert unproject_kb(np.array([320.0 + 300.0 * (rd_max + 0.05), 240.0]), cv) is None
    assert np.isnan(KannalaBrandt4(300, 300, 320, 240, cv[4:]).unproject(
        np.array([320.0 + 300.0 * 0.9, 240.0]))).all()


# folds at theta ~ 0.650 and rises again past ~ 1.256 rad (dtd -, then +)
_TWO_FOLD = np.array([300.0, 300.0, 320.0, 240.0, -1.0, 0.3] + [0.0] * 2)


def test_two_fold_polynomial_invalid_on_second_branch():
    """dtd > 0 again on the later rising branch is NOT valid: its pixels alias
    first-branch pixels (unproject cannot return them), so project is None
    for every theta past the FIRST fold."""
    from mvis.calibration.factors.custom_projections import _kb_fold
    fold = _kb_fold(tuple(_TWO_FOLD[4:]))
    np.testing.assert_allclose(fold, np.sqrt(1.0 - 1.0 / np.sqrt(3.0)), atol=1e-12)
    assert _kb_fold((0.0,) * 4) == np.pi
    for th in (0.8, 1.0, 1.3, 1.4, 1.5, 3.1):
        assert project_kb(np.array([np.sin(th), 0.0, np.cos(th)]), _TWO_FOLD)[0] is None
    for th in np.linspace(0.01, fold - 1e-3, 40):          # first branch round-trips
        P = np.array([np.sin(th), 0.0, np.cos(th)])
        np.testing.assert_allclose(unproject_kb(project_kb(P, _TWO_FOLD)[0], _TWO_FOLD),
                                   P, atol=1e-8)


def test_unproject_exact_newton_root_is_kept(monkeypatch):
    """An iterate landing exactly on the root (f == 0) ends Newton; it must not
    bisect away to a later root (two-fold: theta 0.3 came back as 1.4047) nor
    burn ~50 extra evaluations (zero k: theta 1.0 / 2.0)."""
    import mvis.calibration.factors.custom_projections as cp
    P = np.array([np.sin(0.3), 0.0, np.cos(0.3)])
    np.testing.assert_allclose(unproject_kb(project_kb(P, _TWO_FOLD)[0], _TWO_FOLD),
                               P, atol=1e-12)
    calls = []
    real = cp._kb_poly
    monkeypatch.setattr(cp, "_kb_poly", lambda th, ks: calls.append(th) or real(th, ks))
    cv = np.array([300.0, 300.0, 320.0, 240.0] + [0.0] * 4)
    for th in (1.0, 2.0):
        calls.clear()
        P = np.array([np.sin(th), 0.0, np.cos(th)])
        np.testing.assert_allclose(unproject_kb(project_kb(P, cv)[0], cv), P, atol=1e-12)
        assert len(calls) < 10


def _unified_values(pose, lm, cal):
    from mvis.calibration.optimizer import _sym
    v = gtsam.Values()
    v.insert(_sym("B", 0), pose); v.insert(_sym("B", 1), pose)
    v.insert(_sym("L", 0), gtsam.Point3(*lm)); v.insert(_sym("K", 0), np.asarray(cal, float))
    return v


def _linearize_unified(model, cal, pose, pt, uv):
    """(factor, A, b, column slices per key) of the camera-only unified factor;
    A is the whitened (sigma 1) Jacobian of the error, b = -error."""
    from mvis.calibration.optimizer import _make_unified_factor, _sym
    f, _ = _make_unified_factor(np.asarray(uv, float), pt, 0.0, 1000.0, 0.0, 0.0, 1.0,
                                gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
                                _sym("B", 0), _sym("B", 1), _sym("L", 0), _sym("K", 0),
                                distortion_model=model)
    A, b = f.linearize(_unified_values(pose, pt, cal)).jacobian()
    return f, A, b, {"B": slice(0, 12), "L": slice(12, 15), "K": slice(15, None)}


def _board_scene(rng):
    pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(*rng.uniform(-0.6, 0.6, 3)),
                       gtsam.Point3(*rng.uniform(-0.1, 0.1, 2), -rng.uniform(0.4, 0.8)))
    return pose, np.array([rng.uniform(0, 0.2), rng.uniform(0, 0.2), 0.0])


def test_unified_factor_kb4_matches_gtsam_equidistant():
    """Camera-only factor (not just the kernel): kb4 linearizes
    exactly like the gtsam Cal3Fisheye ('equidistant') factor in every block —
    residual, board poses, landmark, k1..k4 (pins J_point = J_pcam @ R_camᵀ)."""
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(40):
        pose, pt = _board_scene(rng)
        p_cam = np.asarray(pose.transformTo(gtsam.Point3(*pt)), float)
        if p_cam[2] <= 0.1:
            continue
        uv = project_kb(p_cam, KB4_EQ)[0] + rng.normal(0, 2.0, 2)
        _, A4, b4, c = _linearize_unified("equidistant", KB4_EQ, pose, pt, uv)
        _, Ak, bk, _ = _linearize_unified("kb4", KB4_EQ, pose, pt, uv)
        np.testing.assert_allclose(bk, b4, atol=1e-9)
        np.testing.assert_allclose(Ak[:, c["L"]], A4[:, c["L"]], atol=1e-8)
        np.testing.assert_allclose(Ak[:, 15:23], A4[:, 15:23], atol=1e-8)
        np.testing.assert_allclose(Ak[:, c["B"]], A4[:, c["B"]], atol=1e-8)  # analytic pose
        n += 1
    assert n >= 20


@pytest.mark.parametrize("model,dist", [
    ("fov", [0.9] + [0.0] * 3), ("double_sphere", [-0.2, 0.6, 0.0, 0.0]),
    ("eucm", [0.6, 1.0, 0.0, 0.0]), ("ucm", [0.6, 0.0, 0.0, 0.0]), ("omni_radtan", [1.0, -0.1, 0.02, 1e-3, -1e-3]),
    ("kb4", list(KB4_GT[4:]))])
def test_unified_factor_landmark_block_matches_fd(model, dist):
    """Every custom model's camera-only landmark block equals the FD Jacobian
    of the factor's own error w.r.t. the landmark."""
    rng = np.random.default_rng(1)
    cal = np.r_[400.0, 401.0, 320.0, 240.0, dist]
    n = 0
    for _ in range(10):
        pose, pt = _board_scene(rng)
        uv = np.array([320.0, 240.0]) + rng.normal(0, 20.0, 2)
        f, A, b, c = _linearize_unified(model, cal, pose, pt, uv)
        if not np.abs(A).any():                     # point outside the model
            continue
        fd = np.zeros((2, 3))
        for j in range(3):
            d = np.zeros(3); d[j] = 1e-6
            fd[:, j] = (f.unwhitenedError(_unified_values(pose, pt + d, cal))
                        - f.unwhitenedError(_unified_values(pose, pt - d, cal))) / 2e-6
        np.testing.assert_allclose(A[:, c["L"]], fd, rtol=1e-5, atol=1e-3)
        n += 1
    assert n >= 5


def test_registered_as_kb4():
    assert "kb4" in cm.CUSTOM


# ── names ──

def test_generic_fisheye_names_are_kb4():
    from mvis.core.config import CameraConfig
    from mvis.core.config_validator import validate_config
    from mvis.core.config import CalibrationConfig
    from mvis.io.camera_input import _norm_model
    for n in ("kannala_brandt", "fisheye", "Fisheye", "kannala-brandt"):
        assert _norm_model(n) == "kb4"
        assert cm.resolve_model_alias(n) == "kb4"
        assert CameraConfig(distortion_model=n).distortion_model == "kb4"
    assert _norm_model("pinhole equidistant") == "equidistant"     # a Kalibr camchain
    assert CameraConfig(distortion_model="equidistant").distortion_model == "equidistant"
    cfg = CalibrationConfig()
    cfg.cameras = [CameraConfig(camera_id=0, distortion_model="kannala_brandt4")]
    assert not [e for e in validate_config(cfg) if "distortion_model" in e]
    cfg.cameras = [CameraConfig(camera_id=0, distortion_model="kb9")]
    assert [e for e in validate_config(cfg) if "distortion_model" in e]


# ── distortion sanity rule (pass-2 guardrail == regression health) ──

def test_dist_sanity_rule_per_model():
    f = cm.dist_sanity_issue
    # existing models: unchanged plain |c| <= 2
    assert f("radtan", [2.1, 0, 0, 0]) == (0, None)
    assert f("equidistant", [0.1, 0.0, -2.5, 0.0]) == (2, None)
    assert f("double_sphere", [-0.2, 0.6]) is None
    # omni: xi in [0, 5]; k1, k2, p1, p2 keep |c| <= 2
    assert f("omni_radtan", [1.9, -0.1, 0.7, 0.0, 0.0]) is None
    assert f("omni_radtan", [2.5, 0, 0, 0, 0]) is None
    assert f("omni_radtan", [5.0, 0, 0, 0, 0]) is None
    assert f("omni_radtan", [0.0, 0, 0, 0, 0]) is None
    assert f("omni_radtan", [5.2, 0, 0, 0, 0]) == (0, "omni xi=5.200 outside [0, 5]")
    assert f("omni_radtan", [-0.1, 0, 0, 0, 0]) == (0, "omni xi=-0.100 outside [0, 5]")
    assert f("omni_radtan", [float("nan"), 0, 0, 0, 0])[0] == 0
    assert f("omni_radtan", [3.0, 0, 0, 0, 2.5]) == (4, None)
    assert f("omni_radtan", [3.0, -2.1, 0, 0]) == (1, None)        # legacy 4-coeff
    # the xi range is omni-only: double_sphere xi keeps |c| <= 2
    assert f("double_sphere", [2.5, 0.6]) == (0, None)
    # kb4: the plain |c| <= 2, as equidistant
    assert f("kb4", [0.1, 0.0, -2.5, 0.0]) == (2, None)


def test_pass2_guardrail_uses_model_rule():
    from mvis.calibration.pipeline_cam import _pass2_is_pathological
    from mvis.core.types import CalibrationResult, CameraParams

    def res(model, dist):
        return CalibrationResult(cameras=[CameraParams(
            0, np.array([300.0, 300.0, 320.0, 240.0]), np.asarray(dist, float), model,
            (640, 480))])
    p1 = res("radtan", [0, 0, 0, 0])
    assert _pass2_is_pathological(p1, res("radtan", [2.1, 0, 0, 0]))
    assert not _pass2_is_pathological(p1, res("omni_radtan", [1.9, -0.1, 0.7, 0, 0]))
    assert not _pass2_is_pathological(p1, res("omni_radtan", [2.5, -0.1, 0.7, 0, 0]))
    assert _pass2_is_pathological(p1, res("omni_radtan", [5.5, -0.1, 0.7, 0, 0]))
    assert _pass2_is_pathological(p1, res("omni_radtan", [-0.2, -0.1, 0.7, 0, 0]))
    assert _pass2_is_pathological(p1, res("kb4", [3.0, 0, 0, 0]))


def test_cli_writes_camchain_for_kb4():
    from mvis.cli.main import _kalibr_camchain_path
    out = SimpleNamespace(kalibr_yaml=None, path="/r/run.yaml")
    assert _kalibr_camchain_path(out, ["radtan", "omni_radtan"]) == \
        "/r/run_camchain-imucam.yaml"
    assert _kalibr_camchain_path(out, ["radtan", "kb4"]) == "/r/run_camchain-imucam.yaml"
    explicit = SimpleNamespace(kalibr_yaml="/r/chain.yaml", path=None)
    assert _kalibr_camchain_path(explicit, ["kb4"]) == "/r/chain.yaml"
    assert _kalibr_camchain_path(SimpleNamespace(kalibr_yaml=None, path=None),
                                 ["kb4"]) is None


# ── mvisdb ──

def test_mvisdb_accepts_kb4(tmp_path):
    from tests.helpers.mvisdb_fixtures import make_db
    from mvis.core.types import CalibrationResult, CameraParams
    from mvis.mvisdb.estimates import (
        calibration_result_from_estimates, estimates_from_calibration_result)
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.mvisdatabase import CameraInfo, CameraModel
    assert CameraModel("kb4") is CameraModel.KB4
    assert CameraInfo("c", camera_model=CameraModel.KB4).is_fisheye is True
    db = make_db()
    db.streams[1].info.camera_model = CameraModel.KB4
    path = tmp_path / "bag.mvisdb"
    DatabaseManager().save(db, path)
    assert DatabaseManager().load(path).streams[1].info.camera_model is CameraModel.KB4
    conn = sqlite3.connect(path)
    conn.execute("UPDATE camera SET camera_model='kb4' WHERE camera_id=0")    # CHECK ok
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE camera SET camera_model='kb9' WHERE camera_id=0")
    conn.close()
    es = estimates_from_calibration_result(CalibrationResult(cameras=[CameraParams(
        0, KB4_GT[:4].copy(), KB4_GT[4:].copy(), "kb4", _IMG)]))
    back = calibration_result_from_estimates(es).cameras[0]
    assert back.distortion_model == "kb4"
    np.testing.assert_allclose(back.distortion, KB4_GT[4:])


# ── reports ──

def test_reports_name_kb4_coefficients():
    from mvis.analysis.html_report import _distortion_table_rows
    from mvis.analysis.cam_imu_report import _cam_rows, _dist_label
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.core.types import CameraParams
    from mvis.simulation.report import _intr_names
    kb = CameraParams(0, KB4_GT[:4], KB4_GT[4:], "kb4", _IMG)
    html = "".join(_distortion_table_rows([kb]))
    assert all(f"<th>k{i}</th>" in html for i in range(1, 5))
    assert html.count("<th>") == 2 + 4
    cal = CameraCalib(gtsam.Pose3(), 0.0, 0.0, KB4_GT, model="kb4")
    assert _dist_label(_cam_rows(SimpleNamespace(cameras=[{"calib": cal}]))[0]) == \
        " (k1 k2 k3 k4)"
    assert _intr_names(8, "kb4") == ["fx", "fy", "cx", "cy"] + list(cm.DIST_NAMES["kb4"])
    assert _intr_names(8, "radtan") == ["fx", "fy", "cx", "cy", "d0", "d1", "d2", "d3"]


# ── regression stats (mvis.regression.cam_stats, read by the camera-only gate) ──

def _write_run(monkeypatch, tmp_path, dataset, model, dist):
    from mvis.regression import cam_stats
    from mvis.regression.manifest import RunSpec
    monkeypatch.setattr(cam_stats, "RESULTS_ROOT", tmp_path)
    run = RunSpec(mode="cam", dataset=dataset, config="cfg.yaml", collection="c0")
    d = run.output_dir
    d.mkdir(parents=True)
    (d / f"{run.dataset}_{run.slug}.yaml").write_text(yaml.safe_dump(dict(
        cameras=[dict(id=0, intrinsics=dict(fx=250.0, fy=251.0, cx=500.0, cy=498.0),
                      distortion=dict(model=model, coeffs=[float(c) for c in dist]))],
        reprojection_errors=dict(overall_rms=0.07, per_camera=[dict(
            camera=0, rms=0.07, median=0.06, p90=0.1, p95=0.12, max=0.3,
            num_observations=100)]))))
    return cam_stats.collect_stats(run)


@pytest.mark.parametrize("model,dist,tag,msg", [
    ("kb4", [0.02, -0.01, 3e-3, -5e-4], "ok", ""),
    ("kb4", [0.02, -2.5, 3e-3, -5e-4], "dist_blown", "|dist[1]|=2.50"),
    ("omni_radtan", [1.9, -0.1, 0.7, 0.0, 0.0], "ok", ""),
    ("omni_radtan", [2.5, 0.0, 0.0, 0.0, 0.0], "ok", ""),       # xi in [0, 5]
    ("omni_radtan", [5.5, 0.0, 0.0, 0.0, 0.0], "dist_blown", "omni xi=5.500"),
    ("omni_radtan", [3.0, -2.2, 0.0, 0.0, 0.0], "dist_blown", "|dist[1]|=2.20"),
    ("radtan", [2.1, 0.0, 0.0, 0.0], "dist_blown", "|dist[0]|=2.10")])
def test_regression_health(monkeypatch, tmp_path, model, dist, tag, msg):
    pc = _write_run(monkeypatch, tmp_path, "ds", model, dist)["per_camera"][0]
    assert pc["health"] == tag, (model, dist, pc["health_msg"])
    assert msg in pc["health_msg"]


def test_regression_stats_keep_every_distortion_coefficient(monkeypatch, tmp_path):
    """omni keeps its 5 named coefficients (xi k1 k2 p1 p2), radtan its 4, unrounded."""
    st = _write_run(monkeypatch, tmp_path, "omds", "omni_radtan", [1.9, -0.1, 0.07, 1e-3, -1e-3])
    assert st["per_camera"][0]["distortion"] == [1.9, -0.1, 0.07, 1e-3, -1e-3]
    st = _write_run(monkeypatch, tmp_path / "r", "rtds", "radtan", [-0.27957, 0.0797, 1e-6, -4e-4])
    assert st["per_camera"][0]["distortion"] == [-0.27957, 0.0797, 1e-6, -4e-4]
