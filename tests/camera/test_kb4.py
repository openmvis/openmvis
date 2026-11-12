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
"""kb4 kernel and wiring (plan kb4-kb8, gates G1, G2): the θ-direct 4-coefficient model."""
import json
import logging
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest
import yaml

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.custom_projections import (
    _kb_fold, project_kb, unproject_kb)
from mvis.camera import KannalaBrandt4, PinholeEquidistant, create_camera_model
from mvis.camera.camera_model import build_camera_model
from mvis.core import camera_models as cm
from tests._reference import ref_kb4

CAL = np.array([380.0, 381.5, 512.25, 510.75, 0.0067, -0.0013, 0.0026, -0.0006])
FOLDING = np.array([380.0, 381.5, 512.25, 510.75, -0.09, 0.004, -0.0005, 0.00002])
NOFOLD = np.array([380.0, 381.5, 512.25, 510.75, 0.02, -0.005, 5e-4, 1e-5])
# the golden kernel was frozen at a commit of the pre-release history (3850281); the
# fixture's name and its `commit` field carry that id
GOLDEN = Path(__file__).parents[1] / "fixtures/kb/kb_kernel_golden_3850281.npz"
BASALT = Path(__file__).parents[1] / "fixtures/basalt/tum_vi_raw_imu1_calibration.json"


def ray(deg, phi=0.7, r=1.0):
    t = math.radians(deg)
    return np.array([r * math.sin(t) * math.cos(phi), r * math.sin(t) * math.sin(phi),
                     r * math.cos(t)])


# ── G0: the kernel did not move ──

def test_kernel_matches_golden_of_3850281():
    g = np.load(GOLDEN)
    assert json.loads(GOLDEN.with_suffix(".json").read_text())["commit"].startswith("3850281")
    names = sorted({k.split(".")[0] for k in g.files if "." in k})
    assert len(names) == 7
    for name in names:
        cal = g[f"{name}.cal"]
        for i, P in enumerate(g["rays"]):
            uv, Jp, Jc = project_kb(P, cal, True)
            if np.isnan(g[f"{name}.uv"][i, 0]):
                assert uv is None
                continue
            assert np.abs(uv - g[f"{name}.uv"][i]).max() <= 1e-12
            np.testing.assert_allclose(Jp, g[f"{name}.J_point"][i], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(Jc, g[f"{name}.J_cal"][i], rtol=1e-12, atol=1e-12)
        for i, q in enumerate(g["pixels"]):
            r = unproject_kb(q, cal)
            if np.isnan(g[f"{name}.unproject"][i, 0]):
                assert r is None
            else:
                assert np.abs(r - g[f"{name}.unproject"][i]).max() <= 1e-12


# ── G1: the kernel against outside references ──

@pytest.mark.parametrize("deg", [1e-3, 1, 30, 60, 85, 89, 89.9])
@pytest.mark.parametrize("phi", [0.0, 0.7, 2.5, -1.9])
def test_matches_gtsam_cal3fisheye_below_90(deg, phi):
    P = ray(deg, phi, 1.7)
    uv, Jp, Jc = project_kb(P, CAL, True)
    uv_g, Jp_g, Jc_g = PinholeEquidistant(*CAL[:4], CAL[4:]).project(P, jac=True)
    assert np.abs(uv - uv_g).max() <= 1e-9
    assert np.abs(Jp - Jp_g).max() <= 1e-6 * max(1.0, np.abs(Jp_g).max())
    assert np.abs(Jc - Jc_g).max() <= 1e-6 * max(1.0, np.abs(Jc_g).max())


def _fd_point(cal, P):
    """h = 1e-6 min(1, rho) on X, Y (plan G1). On Z the step scales with the range: near the axis
    rho is ~1e-9 and such a step on Z ~ 1 is all rounding. cx = cy = 0, which J_point does not
    depend on, keeps the pixel's own rounding out of the difference."""
    rho = max(math.hypot(P[0], P[1]), 1e-30)
    cal = np.r_[cal[:2], 0.0, 0.0, cal[4:]]
    hs = [1e-6 * min(1.0, rho)] * 2 + [1e-6 * np.linalg.norm(P)]
    J = np.zeros((2, 3))
    for j, h in enumerate(hs):
        d = np.zeros(3)
        d[j] = h
        J[:, j] = (project_kb(P + d, cal)[0] - project_kb(P - d, cal)[0]) / (2 * h)
    return J


@pytest.mark.parametrize("cal,deg", [(CAL, d) for d in (1e-7, 1, 60, 89.9, 90, 91, 120)]
                         + [(NOFOLD, d) for d in (1e-7, 1, 60, 89.9, 90, 91, 120, 179.5)]
                         + [(FOLDING, math.degrees(_kb_fold(tuple(FOLDING[4:]))) - 1.0)])
@pytest.mark.parametrize("phi", [0.0, 0.7, 2.5])
def test_jacobians_match_central_difference(cal, deg, phi):
    P = ray(deg, phi)
    uv, Jp, Jc = project_kb(P, cal, True)
    assert uv is not None
    Jp_fd = _fd_point(cal, P)
    assert np.abs(Jp - Jp_fd).max() / np.abs(Jp).max() <= 1e-6
    Jc_fd = np.zeros((2, 8))
    for j, h in enumerate([1e-5] * 4 + [1e-6] * 4):     # linear in every cal param, small k step
        d = np.zeros(8)
        d[j] = h
        Jc_fd[:, j] = (project_kb(P, cal + d)[0] - project_kb(P, cal - d)[0]) / (2 * h)
    assert np.abs(Jc - Jc_fd).max() / np.abs(Jc).max() <= 1e-6


def test_jacobians_regular_at_exactly_90_deg():
    P = np.array([1.0, 0.0, 0.0])
    uv, Jp, Jc = project_kb(P, NOFOLD, True)
    assert np.isfinite(uv).all() and np.isfinite(Jp).all() and np.isfinite(Jc).all()


def _basalt_cals():
    """The kb4 camera of Basalt's own calibration file (the only one checked in)."""
    d = json.loads(BASALT.read_text())["value0"]
    for c in d["intrinsics"]:
        p = c["intrinsics"]
        yield [p[k] for k in ("fx", "fy", "cx", "cy")] + [p[k] for k in ("k1", "k2", "k3", "k4")]


def test_equals_basalt_on_in_domain_rays():
    cals = list(_basalt_cals())
    assert len(cals) == 2
    n = 0
    for cal in cals:
        fold = math.degrees(_kb_fold(tuple(cal[4:])))
        for deg in np.linspace(0.01, min(fold, 179.0) - 0.5, 60):
            for phi in (0.0, 1.1, 2.9, -2.0):
                P = ray(deg, phi, 2.3)
                uv = project_kb(P, np.array(cal))[0]
                assert np.abs(uv - ref_kb4.basalt_project(P, cal)).max() <= 1e-6
                n += 1
    assert n == 480


def test_equals_frozen_reference_formula():
    for deg in np.linspace(0.01, 179.0, 200):
        P = ray(deg, 0.7, 1.3)
        assert np.abs(project_kb(P, NOFOLD)[0] - ref_kb4.project(P, NOFOLD)).max() <= 1e-9


@pytest.mark.parametrize("deg", [1e-4, 1, 45, 80, 90, 95, 110, 130])
@pytest.mark.parametrize("phi", [0.3, -2.2])
def test_round_trip_through_90_deg(deg, phi):
    P = ray(deg, phi)
    r = unproject_kb(project_kb(P, NOFOLD)[0], NOFOLD)
    ang = math.degrees(math.atan2(np.linalg.norm(np.cross(P, r)), P @ r))
    assert ang <= 1e-5


def test_round_trip_at_rho_1e12():
    P = np.array([1e-12, 0.0, 1.0])
    r = unproject_kb(project_kb(P, NOFOLD)[0], NOFOLD)
    assert np.abs(r - P / np.linalg.norm(P)).max() <= math.radians(1e-5)


def test_invalid_inputs_return_none():
    fold = math.degrees(_kb_fold(tuple(FOLDING[4:])))
    assert fold < 179
    assert project_kb(np.array([0.0, 0.0, -1.0]), NOFOLD)[0] is None        # theta = pi
    assert project_kb(np.zeros(3), NOFOLD)[0] is None
    assert project_kb(ray(fold + 0.1), FOLDING)[0] is None                  # past the fold
    assert project_kb(ray(fold - 0.1), FOLDING)[0] is not None
    # second branch: past the fold's radius there is no ray, never a wrong one
    th = math.radians(fold)
    r_fold = FOLDING[0] * ref_kb4.theta_d_px(FOLDING, th) / FOLDING[0]
    assert unproject_kb(FOLDING[2:4] + [r_fold + 5.0, 0.0], FOLDING) is None
    assert project_kb(ray(min(fold + 20, 179)), FOLDING)[0] is None


@pytest.mark.parametrize("n", [4, 5, 7, 9, 12])
def test_any_other_length_raises(n):
    v = np.r_[CAL, np.zeros(4)][:n]
    with pytest.raises(ValueError, match="kb4"):
        build_camera_model("kb4", v)
    with pytest.raises(ValueError, match="kb4"):
        CameraCalib(gtsam.Pose3(), 0.0, 0.0, v, model="kb4")


@pytest.mark.parametrize("d", [[0.1], [0.1] * 3, [0.1] * 5, [0.1] * 8])
def test_class_refuses_other_coefficient_counts(d):
    with pytest.raises(ValueError, match="exactly 4"):
        KannalaBrandt4(1.0, 1.0, 0.0, 0.0, d)
    with pytest.raises(ValueError, match="exactly 4"):
        create_camera_model("kb4", 1.0, 1.0, 0.0, 0.0, d)


def test_class_contract():
    m = build_camera_model("kb4", CAL)
    assert isinstance(m, KannalaBrandt4)
    assert m.distortion_model_name() == "kb4" and m.num_params() == 8 and m.n_intr == 8
    assert m.get_gtsam_calibration() is None
    np.testing.assert_array_equal(m.cal_vec, CAL)
    np.testing.assert_allclose(m.project(ray(95)), ref_kb4.project(ray(95), CAL), atol=1e-9)
    assert create_camera_model("kb4", 1.0, 1.0, 0.0, 0.0).distortion().tolist() == [0.0] * 4
    # the class unprojects to z-depth (D12): no point past 90 deg; the kernel keeps the ray
    assert np.isnan(m.unproject(m.project(ray(95)))).all()
    np.testing.assert_allclose(unproject_kb(m.project(ray(95)), CAL), ray(95), atol=1e-7)
    np.testing.assert_allclose(m.unproject(m.project(ray(60)), depth=2.0),
                               ray(60) / ray(60)[2] * 2.0, atol=1e-7)
    assert np.isnan(build_camera_model("kb4", FOLDING).unproject(FOLDING[2:4] + [3000.0, 0.0])).all()


# ── G2: registry and spellings ──

def test_registry():
    assert "kb4" in cm.MODELS and cm.n_dist("kb4") == 4
    assert cm.DIST_NAMES["kb4"] == ("k1", "k2", "k3", "k4")
    assert cm.default_dist("kb4") == [0.0] * 4
    assert cm.is_custom("kb4") and "kb4" in cm.KB_THETA_MODELS
    assert cm.dist_sanity_issue("kb4", [0.1, -0.1, 0.1, -0.1]) is None
    assert cm.dist_sanity_issue("kb4", [0.1, 3.0, 0.1, -0.1]) == (1, None)     # as equidistant


@pytest.mark.parametrize("name", ["kb4", "KB4", "kb_4", "kannala_brandt4", "kannala-brandt4",
                                  "kannala_brandt_4", "kannala-brandt-4", "pinhole_kb4"])
def test_kb4_spellings(name):
    from mvis.io.camera_input import _norm_model
    assert _norm_model(name) == "kb4"
    if name != "pinhole_kb4":
        assert cm.resolve_model_alias(name) == "kb4"


@pytest.mark.parametrize("name", ["kannala_brandt", "fisheye", "pinhole kannala_brandt"])
def test_plain_fisheye_names_are_kb4(name):
    from mvis.io.camera_input import _norm_model
    assert _norm_model(name) == "kb4"
    if name != "pinhole kannala_brandt":
        assert cm.resolve_model_alias(name) == "kb4"
    assert _norm_model("equidistant") == "equidistant"


def test_config_takes_kb4_spelling_and_validates():
    from mvis.core.config import CalibrationConfig, CameraConfig
    from mvis.core.config_validator import validate_config
    c = CameraConfig(camera_id=0, distortion_model="kannala_brandt4")
    assert c.distortion_model == "kb4"
    errs = validate_config(CalibrationConfig(cameras=[c]))
    assert not [e for e in errs if "distortion_model" in e]


# ── G2: an unnamed 8-vector must not lose the name: a 95 deg ray through every entry point ──

def _board_cam_scene():
    """A pose and a point at 95 deg off axis in that camera, in the board frame."""
    T = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, -0.2, 0.3), gtsam.Point3(0.2, -0.1, 0.4))
    P_c = ray(95, 0.4, 0.8)
    return T, np.asarray(T.transformFrom(gtsam.Point3(*P_c))), P_c


def test_95deg_camera_calib():
    from mvis.calibration.factors.projection import make_projector
    c = CameraCalib(gtsam.Pose3(), 0.0, 0.0, NOFOLD, model="kb4")
    assert c.model == "kb4" and not c.is_fisheye and c.n_intr == 8
    T, p_B, P_c = _board_cam_scene()
    px = make_projector(c.model, c.intr).project(T, p_B, jac=False)[0]
    np.testing.assert_allclose(px, ref_kb4.project(P_c, NOFOLD), atol=1e-9)


def test_95deg_project_observation():
    from mvis.calibration.factors.camera_projection import observation_pose, project_observation
    T, p_B, P_c = _board_cam_scene()
    kw = dict(board_point=p_B, bp0=T, bp1=T, time_meas=0.0, time_0=0.0, time_1=1.0)
    np.testing.assert_allclose(project_observation(cal_vec=NOFOLD, distortion_model="kb4", **kw),
                               ref_kb4.project(P_c, NOFOLD), atol=1e-9)
    kw_pose = {k: v for k, v in kw.items() if k != "board_point"}
    np.testing.assert_allclose(np.asarray(observation_pose(**kw_pose).transformTo(gtsam.Point3(*p_B))),
                               P_c, atol=1e-12)
    # the unnamed form is the legacy equidistant one, which has no pixel for z < 0
    assert project_observation(cal_vec=NOFOLD, is_fisheye=True, **kw) is None


def test_95deg_normalize_detections_and_pnp_keep_the_model():
    from mvis.calibration.cam_imu_init import pnp_pose_track
    from mvis.calibration.mono_sfm import normalize_detections
    uv = np.array([ref_kb4.project(ray(95, 0.4), NOFOLD), ref_kb4.project(ray(30, 0.4), NOFOLD)])
    ids = np.array([3, 4])
    (ids_n, xy), = normalize_detections([ids], [uv], NOFOLD, model="kb4")
    # the 30 deg corner lands on x/z of its true ray; the 95 deg one has no x/z plane point
    assert list(ids_n) == [4]
    want = ray(30, 0.4)
    np.testing.assert_allclose(xy[0], want[:2] / want[2], atol=1e-7)
    # PnP: a board seen around the axis, through the kb4 bearings
    pts = np.array([[x, y, 0.0] for y in np.arange(7) * 0.05 for x in np.arange(7) * 0.05])
    T = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, -0.1, 0.05), gtsam.Point3(0.05, -0.02, -0.5))
    uvs = np.array([ref_kb4.project(T.transformTo(gtsam.Point3(*p)), NOFOLD) for p in pts])
    t_k, poses, _n = pnp_pose_track(np.array([0.0]), [np.arange(len(pts))], [uvs], pts, NOFOLD,
                                    model="kb4")
    assert len(t_k) == 1


def test_95deg_simulator_cal_keeps_the_model():
    from mvis.simulation.from_result import ModelCal, cal_from_vec
    cal = cal_from_vec(NOFOLD, True, "kb4")          # is_fisheye must not win over the name
    assert isinstance(cal, ModelCal) and cal.model == "kb4"
    assert isinstance(cal_from_vec(NOFOLD, True), gtsam.Cal3Fisheye)


def test_pipeline_cam_dict_keeps_the_model(monkeypatch):
    """pipeline_cam.py:2284 / 2318: the free-function API hands `model` to the pipeline."""
    import mvis.calibration.pipeline_cam as pc
    seen = {}

    class _P:
        def __init__(self, board, config):
            seen["models"] = config.distortion_models

        def run(self, frames, sizes):
            return "done"
    monkeypatch.setattr(pc, "CalibrationPipeline", _P)
    pc.run_cam_calibration([0.0], [np.array([0])], [np.zeros((1, 2))], None, (640, 480),
                           is_fisheye=True, model="kb4")
    assert seen["models"] == {0: "kb4"}


# ── G2: the legacy CLI path, stubbed ──

def test_cli_calibrate_single_solve_with_kb4(monkeypatch):
    import mvis.cli.main as cli
    from mvis.core.types import CalibrationResult, CameraParams
    calls = []

    class _Init:
        def __init__(self, board):
            pass

        def init_intrinsics_opencv(self, frames, size, dm):
            calls.append(("init", dm))
            return np.array([300.0, 300.0, 320.0, 240.0]), np.zeros(4)

        def init_board_poses(self, frames, intr, dist, dm, **kw):
            return [None] * len(frames)

    class _Opt:
        def __init__(self, board, opts):
            self.opts = opts

        def solve_single_camera(self, frames, intr, dist, poses, dm, size, camera_id=0):
            calls.append(("solve", dm))
            return CalibrationResult(cameras=[CameraParams(
                camera_id, np.array([300.0, 300.0, 320.0, 240.0]), np.asarray(dist), dm, size)])
    monkeypatch.setattr(cli, "create_board", lambda b: None)
    monkeypatch.setattr(cli, "Initializer", _Init)
    monkeypatch.setattr(cli, "Optimizer", _Opt)
    monkeypatch.setattr(cli, "_detect_all_frames", lambda c, b: ({0: [0, 1]}, {0: (640, 480)}))
    monkeypatch.setattr(cli, "print_report", lambda r: "")
    monkeypatch.setattr(cli.OutputWriter, "write", staticmethod(lambda *a, **k: None))
    cli.calibrate(SimpleNamespace(
        board=None, reference_camera=0, corner_subsample="all", output_file="u.yaml",
        output_format="yaml",
        cameras=[SimpleNamespace(camera_id=0, distortion_model="kb4", shutter_type="global")],
        optimizer=SimpleNamespace(
            max_init_frames=0, init_frame_selection="uniform", pnp_subprocess=False,
            optimize_rolling_shutter=False, max_iterations=10, pixel_sigma=1.0,
            huber_threshold=1.0, absolute_error_tol=1e-9, relative_error_tol=1e-9)))
    assert calls == [("init", "kb4"), ("solve", "kb4")]              # one solve, no phases


def test_initializer_routes_kb4_through_equidistant_init(monkeypatch):
    from mvis.calibration.initializer import Initializer
    ini = Initializer.__new__(Initializer)
    orig = Initializer.init_intrinsics_opencv
    seen = []

    def fake(self, frames, size, dm):
        if dm == "kb4":
            return orig(self, frames, size, dm)
        seen.append(dm)
        return np.array([300.0, 300.0, 320.0, 240.0]), np.array([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(Initializer, "init_intrinsics_opencv", fake)
    intr, d = ini.init_intrinsics_opencv([object()], (640, 480), "kb4")
    assert seen == ["equidistant"] and d.tolist() == [0.1, 0.2, 0.3, 0.4]


# ── G2: Kalibr export as pinhole-equi, the same polynomial ──

def _kalibr_equi_pixels(k_intr, k_coeffs, pts):
    """Kalibr's pinhole-equi, as aslam_cameras writes it: divide by z, theta = atan(r)."""
    fx, fy, cx, cy = k_intr
    k1, k2, k3, k4 = k_coeffs
    x, y = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
    r = np.hypot(x, y)
    th = np.arctan(r)
    thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    s = np.where(r > 1e-8, thd / np.where(r > 1e-8, r, 1.0), 1.0)
    return np.c_[fx * s * x + cx, fy * s * y + cy]


@pytest.mark.parametrize("cal", [NOFOLD, CAL, FOLDING, np.r_[CAL[:4], 0.0, 0.0, 0.0, 0.0]])
def test_kb4_kalibr_export_is_pinhole_equi_with_the_same_pixels(cal):
    from mvis.io.cam_imu_output import _kalibr_pack, kalibr_export_issue
    assert "kb4" in cm.KALIBR_EXPORTABLE and kalibr_export_issue("kb4", cal) is None
    cam_m, dist_m, k_intr, k_coeffs = _kalibr_pack("kb4", cal)
    assert (cam_m, dist_m) == ("pinhole", "equidistant")
    assert k_intr == cal[:4].tolist() and k_coeffs == cal[4:].tolist()
    rng = np.random.default_rng(3)
    th = rng.uniform(0.0, np.radians(89.0), 400)
    ph = rng.uniform(-np.pi, np.pi, 400)
    pts = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)] * rng.uniform(0.5, 3, (400, 1))
    m = build_camera_model("kb4", cal)
    fold = _kb_fold(tuple(cal[4:]))
    keep = th < min(fold, np.radians(89.0)) - 1e-3
    uv = np.array([m.project(q) for q in pts[keep]])
    np.testing.assert_allclose(_kalibr_equi_pixels(k_intr, k_coeffs, pts[keep]), uv, atol=1e-9)


def test_kb4_kalibr_warning_names_what_kalibr_loses():
    from mvis.camera.kb_domain import kb_valid_domain
    from mvis.io.cam_imu_output import kb4_kalibr_warning
    assert kb4_kalibr_warning(NOFOLD, (1024, 1024), {"max_theta_used_deg": 80.0}) is None
    assert kb4_kalibr_warning(NOFOLD) is None
    w = kb4_kalibr_warning(NOFOLD, (1024, 1024), {"max_theta_used_deg": 95.3})
    assert "95.3 deg" in w and "90 deg" in w
    info = kb_valid_domain(FOLDING, (1024, 1024))
    assert info["fold_theta_rad"] is not None and info["covers_image"] is False
    w = kb4_kalibr_warning(FOLDING, (1024, 1024))
    assert "folds at" in w and "inside the image" in w


def test_kb4_kalibr_camchain_path_is_written():
    from mvis.cli.main import _kalibr_camchain_path
    cfg = SimpleNamespace(kalibr_yaml=None, path="r/cam_imu.json")
    assert _kalibr_camchain_path(cfg, ["kb4", "eucm"],
                                 [NOFOLD, [1, 1, 1, 1, 0.6, 1.0]]).endswith("_camchain-imucam.yaml")
    assert _kalibr_camchain_path(cfg, ["equidistant"], [NOFOLD]).endswith("_camchain-imucam.yaml")


def test_kb4_cam_imu_writer_writes_both_files_and_warns(tmp_path, caplog):
    from mvis.io.cam_imu_output import to_kalibr_camchain_imucam, write_cam_imu_calibration
    from mvis.io.camera_input import load_camera_intrinsics
    cal = CameraCalib(gtsam.Pose3(), 0.001, 0.0, NOFOLD, model="kb4")
    r = SimpleNamespace(cameras=[dict(calib=cal, reproj_rms_px=0.1, image_size=(1024, 1024),
                                      corner_accounting={"max_theta_used_deg": 92.0})],
                        calib=cal, imus=[], b_g=np.zeros(3), b_a=np.zeros(3),
                        imu_intr_xi=None, diagnostics={}, ok=True)
    out, chain = tmp_path / "r_calib.yaml", tmp_path / "chain.yaml"
    with caplog.at_level(logging.WARNING, logger="mvis.io.cam_imu_output"):
        write_cam_imu_calibration(r, str(out), kalibr_path=str(chain))
    assert "cam0: kb4 written to Kalibr as pinhole-equi" in caplog.text and "92.0 deg" in caplog.text
    blk = yaml.safe_load(out.read_text())["cameras"][0]
    assert blk["model"] == "pinhole_kb4" and blk["distortion"]["model"] == "kb4"
    c0 = yaml.safe_load(chain.read_text())["cam0"]
    assert (c0["camera_model"], c0["distortion_model"]) == ("pinhole", "equidistant")
    assert c0["intrinsics"] == NOFOLD[:4].tolist() and c0["distortion_coeffs"] == NOFOLD[4:].tolist()
    assert to_kalibr_camchain_imucam(r)["cam0"] == c0
    back = load_camera_intrinsics(str(chain))[0]          # a camchain reads back as equidistant
    assert back["distortion_model"] == "equidistant"
    np.testing.assert_allclose(back["distortion_coeffs"], NOFOLD[4:])


def test_kb4_mvisdb_kalibr_export_writes_pinhole_equi(tmp_path):
    """The mvisdb/exports.py path: the db view of the same export, on a saved run."""
    from mvis.mvisdb.estimates import estimates_from_cam_imu_result
    from mvis.mvisdb.exports import export_kalibr_camchain
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.mvisdatabase import CalibChoice, CameraInfo, CameraModel, CameraStream
    from tests.helpers.mvisdb_fixtures import make_db
    base = CameraCalib(gtsam.Pose3(gtsam.Rot3.Rz(0.02), gtsam.Point3(0.01, 0.02, 0.03)),
                       0.004, 0.0, NOFOLD, model="kb4")
    r = SimpleNamespace(ok=True, calib=base, imu_model="rpng", imu_intr_xi=np.arange(24.0),
                        imu_intr_sigma=np.full(24, 0.5), imus=[],
                        cameras=[{"calib": base, "calib_sigma": None, "image_size": (1024, 1024)}])
    db = make_db(CalibChoice.CAM_IMU)
    db.streams[0] = CameraStream(0, CameraInfo("cam0", camera_model=CameraModel.KB4,
                                               resolution=(1024, 1024)))
    db.estimate_states = estimates_from_cam_imu_result(r, state_hz=20.0)
    path = tmp_path / "b.mvisdb"
    mgr = DatabaseManager()
    rid = mgr.save(db, path)
    out = tmp_path / "chain.yaml"
    export_kalibr_camchain(mgr, path, rid, out)
    c0 = yaml.safe_load(out.read_text())["cam0"]
    assert (c0["camera_model"], c0["distortion_model"]) == ("pinhole", "equidistant")
    np.testing.assert_allclose(c0["distortion_coeffs"], NOFOLD[4:])


def test_kb4_undistort_refused():
    from mvis.recalib.undistort import check_undistortable
    t = SimpleNamespace(kind="cam", cam=SimpleNamespace(model="kb4"))
    with pytest.raises(ValueError, match="kb4"):
        check_undistortable({"/a": t})


# ── G2: yaml, mvisdb, regenerate_html round trips; old dbs ──

def _result(model="kb4", cal=NOFOLD, size=(1024, 1024)):
    from mvis.core.types import CalibrationResult, CameraParams
    return CalibrationResult(cameras=[CameraParams(0, cal[:4], cal[4:], model, size)],
                             extrinsics=[], reference_camera=0,
                             board_poses=[np.eye(4)], board_pose_times=[1.0],
                             corner_accounting={0: {
                                 "camera": 0, "model": model, "n_corners": 12, "used": 10,
                                 "past_fold": 2, "other_invalid": 0, "no_pose_frames": 1,
                                 "no_pose_corners": 40, "outliers": 3, "max_theta_deg": 101.0,
                                 "max_theta_used_deg": 96.0, "fold_deg": 100.0,
                                 "fold_level": "alarm"}})


def test_yaml_round_trip_and_blocks(tmp_path):
    from mvis.io.camera_input import load_camera_intrinsics
    from mvis.io.output_writer import OutputWriter
    p = tmp_path / "m_calib.yaml"
    OutputWriter.write_calibration_yaml(_result(), str(p))
    doc = yaml.safe_load(p.read_text())
    cam = doc["cameras"][0]
    assert cam["model"] == "pinhole_kb4" and cam["distortion"]["model"] == "kb4"
    assert cam["valid_domain"]["covers_image"] is not None
    assert doc["corner_accounting"]["per_camera"][0]["past_fold"] == 2
    lc = load_camera_intrinsics(str(p))[0]
    assert lc["distortion_model"] == "kb4"
    np.testing.assert_allclose(lc["intrinsics"] + lc["distortion_coeffs"], NOFOLD, atol=1e-12)


def test_mvisdb_round_trip_matches_writer(tmp_path):
    """Fresh db (enum + schema CHECK); the I5 shadow's comparison holds, and the accounting is
    a writer-only diagnostic on both sides."""
    from mvis.io.output_writer import OutputWriter
    from mvis.mvisdb.estimates import estimates_from_calibration_result
    from mvis.mvisdb.exports import export_calib_yaml, normalise_calib_yaml_pair
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.mvisdatabase import CalibChoice, CameraInfo, CameraModel, CameraStream, Run
    from tests.helpers.mvisdb_fixtures import make_db
    result = _result()
    w = tmp_path / "run_calib.yaml"
    OutputWriter.write_calibration_yaml(result, str(w))
    db = make_db(CalibChoice.CAM_ONLY)
    db.streams = [CameraStream(0, CameraInfo("cam0", camera_model=CameraModel.KB4,
                                             resolution=(1024, 1024)))]
    db.run = Run(CalibChoice.CAM_ONLY, description="t", status="ok")
    db.estimate_states = estimates_from_calibration_result(result)
    path = tmp_path / "detections.mvisdb"
    mgr = DatabaseManager()
    rid = mgr.save(db, path)
    v = tmp_path / "view.yaml"
    export_calib_yaml(mgr, path, rid, v)
    a, b = yaml.safe_load(w.read_text()), yaml.safe_load(v.read_text())
    assert b["cameras"][0]["distortion"] == {"model": "kb4", "coeffs": NOFOLD[4:].tolist()}
    assert b["cameras"][0]["valid_domain"]["covers_image"] is not None
    normalise_calib_yaml_pair(a, b)
    assert a == b


def test_old_db_without_kb4_in_check_still_works(tmp_path):
    """No migration (56a426b): camera rows are written only at capture creation and result_camera
    has no CHECK, so an old cache keeps its rows and a kb4 result on it is stored."""
    from importlib.resources import files
    sql = files("mvis.mvisdb").joinpath("schema.sql").read_text()
    assert "'omni_radtan','kb4'" in sql
    old = sql.replace("'omni_radtan','kb4'", "'omni_radtan','kb8'")
    con = sqlite3.connect(tmp_path / "old.mvisdb")
    con.executescript(old)
    tables = [r[0] for r in con.execute("select name, sql from sqlite_master where type='table'")
              if r[1] and "CHECK (camera_model" in r[1]]
    assert tables
    rc = con.execute("select sql from sqlite_master where name='result_camera'").fetchone()
    assert rc is None or "camera_model IN" not in rc[0]
    con.close()


def test_regenerate_html_rows():
    from mvis.analysis.html_report import _corner_accounting_rows, _kb_domain_rows
    from mvis.analysis.report import print_report
    r = _result(cal=FOLDING)
    html = "".join(_kb_domain_rows(r.cameras))
    assert "KB4 Valid Domain" in html and "fold inside image" in html
    assert _kb_domain_rows(_result("equidistant").cameras) == []
    acc = "".join(_corner_accounting_rows(r))
    assert "Corner Accounting" in acc and "ALARM" in acc
    assert _corner_accounting_rows(SimpleNamespace(corner_accounting=None)) == []
    txt = print_report(r)
    assert "k1" in txt and "kb4 valid domain" in txt and "ALARM" in txt


def test_cam_imu_reports_name_kb4():
    from mvis.analysis.cam_imu_report import _dist_label, _kb_domain, _kb_domain_text
    cal = CameraCalib(gtsam.Pose3(), 0.0, 0.0, FOLDING, model="kb4")
    info = _kb_domain(cal, {"image_size": (1024, 1024)})
    row = {"model": "kb4", "valid_domain": info, "dist_names": cm.DIST_NAMES["kb4"]}
    assert _kb_domain_text(row).startswith("kb4 valid domain")
    assert "k1 k2 k3 k4" in _dist_label(row)


# ── the fold levels ──

def test_fold_levels():
    from mvis.camera.kb_domain import check_kb_valid_domain, fold_level, kb_valid_domain
    big = (1024, 1024)
    assert fold_level(kb_valid_domain(np.r_[100.0, 100.0, 50.0, 50.0, 0, 0, 0, 0], (100, 100))) == "none"
    folds = kb_valid_domain(FOLDING, (4000, 4000))
    assert folds["fold_theta_rad"] is not None and folds["covers_image"] is False
    assert fold_level(folds) == "warn"
    assert fold_level(folds, 1) == "alarm"
    # no fold, but theta reaches pi inside the image
    assert fold_level(kb_valid_domain(np.r_[100.0, 100.0, 2000.0, 2000.0, 0, 0, 0, 0], (4000, 4000))) == "warn"
    assert fold_level(kb_valid_domain(FOLDING)) == "none"       # no image size to judge by
    assert check_kb_valid_domain(NOFOLD, big)["covers_image"] is not None


def test_enum_and_schema_list_every_model():
    from importlib.resources import files
    from mvis.mvisdb.mvisdatabase import CameraModel
    assert {m.value for m in CameraModel} == set(cm.MODELS)
    sql = files("mvis.mvisdb").joinpath("schema.sql").read_text()
    for m in cm.MODELS:
        assert f"'{m}'" in sql
