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
"""Full Kalibr omni-radtan: cal_vec [fx, fy, cx, cy, xi, k1, k2, p1, p2] with p2
estimated (it used to be fixed at 0).

Pins, with p2 != 0 throughout (each would FAIL if p2 were silently dropped):
  * kernel FD-Jacobian parity wrt the point and all 9 cal params, the
    project -> unproject round trip (the undistortion residual sees p2) and an
    FD check of the undistortion Newton Jacobian's p2 terms;
  * OmniRadtan: legacy 4-coeff input -> p2 = 0 with pixels identical to the old
    8-long cal_vec, >5 coeffs raise;
  * camera-only synthetic end-to-end (initializer + the two-phase omni schedule)
    recovering p2 and the principal point — and the same data solved with p2
    forced to 0 (the old model) biasing cx;
  * io: Kalibr camchain import -> export round trip keeps p2, legacy 4-coeff
    MVIS / old MVIS-exported camchains load with p2 = 0, fov exports 1 coeff;
  * mvisdb legacy 4-coeff omni rows load padded; reports name the coefficients;
  * the cam-IMU regression gate fails on a key-set mismatch.
"""
import numpy as np
import gtsam
import pytest
import yaml

from mvis.calibration.factors.custom_projections import (
    project_omni_radtan, unproject_omni_radtan,
)
from mvis.camera.camera_model import build_camera_model, create_camera_model
from mvis.camera.omni_radtan import OmniRadtan
from mvis.core import camera_models as cm

# every coefficient non-zero; p2 well above the FD noise floor
CAL9 = np.array([300.0, 302.0, 322.0, 238.0, 1.0, -0.05, 0.01, 1e-3, 4e-3])


def _rand_points(n=60, seed=0):
    """Camera-frame points over a wide field (up to ~80 deg) at 0.3-3 m."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, np.deg2rad(80.0), n)
    ph = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(0.3, 3.0, n)
    return np.stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph),
                     r * np.cos(th)], 1)


# ── registry ──

def test_registry_omni_has_p2():
    assert cm.n_dist("omni_radtan") == 5
    assert cm.DIST_NAMES["omni_radtan"] == ("xi", "k1", "k2", "p1", "p2")
    assert cm.default_dist("omni_radtan") == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert cm.canonical_dist("omni_radtan", [1.1, .1, .2, .3]) == [1.1, .1, .2, .3, 0.0]
    assert cm.canonical_cal_vec("omni_radtan", CAL9[:8])[8] == 0.0


# ── kernel ──

def _fd(fn, x, eps):
    J = []
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        J.append((fn(xp) - fn(xm)) / (2 * eps))
    return np.array(J).T


def test_kernel_fd_jacobians_with_p2():
    n_ok = 0
    for P in _rand_points():
        px, J_pt, J_cal = project_omni_radtan(P, CAL9, True)
        if px is None:
            continue
        n_ok += 1
        assert J_cal.shape == (2, 9)
        Jp = _fd(lambda p: project_omni_radtan(p, CAL9)[0], P.copy(), 1e-6)
        Jc = _fd(lambda c: project_omni_radtan(P, c)[0], CAL9.copy(), 1e-7)
        np.testing.assert_allclose(J_pt, Jp, atol=1e-4 * max(1, np.abs(Jp).max()))
        np.testing.assert_allclose(J_cal, Jc, atol=1e-4 * max(1, np.abs(Jc).max()))
    assert n_ok > 50
    # the p2 column is a real derivative: du/dp2 = fx (r^2 + 2 mx^2)
    _, _, J = project_omni_radtan(np.array([0.4, -0.3, 1.0]), CAL9, True)
    assert abs(J[0, 8]) > 10.0 and abs(J[1, 8]) > 1.0


def test_kernel_p2_moves_pixels_and_legacy_8_is_p2_zero():
    P = np.array([0.4, -0.3, 1.0])
    px9 = project_omni_radtan(P, CAL9)[0]
    px8 = project_omni_radtan(P, CAL9[:8])[0]
    px9_0 = project_omni_radtan(P, np.r_[CAL9[:8], 0.0])[0]
    assert np.abs(px9 - px8).max() > 0.1                 # p2 is not ignored
    np.testing.assert_array_equal(px8, px9_0)            # 8-long == p2 = 0
    _, _, J8 = project_omni_radtan(P, CAL9[:8], True)
    assert J8.shape == (2, 8)                            # legacy width kept


def test_project_unproject_round_trip_with_p2():
    for P in _rand_points(seed=3):
        px = project_omni_radtan(P, CAL9)[0]
        if px is None:
            continue
        ray = unproject_omni_radtan(px, CAL9)
        assert ray is not None
        np.testing.assert_allclose(ray, P / np.linalg.norm(P), atol=1e-8)
    # dropping p2 in the undistortion would leave a visible ray error
    P = np.array([0.6, 0.5, 1.0])
    ray0 = unproject_omni_radtan(project_omni_radtan(P, CAL9)[0], CAL9[:8])
    assert np.linalg.norm(ray0 - P / np.linalg.norm(P)) > 1e-4


def test_undistort_newton_jacobian_matches_fd_with_p2():
    """The 2x2 Newton Jacobian in unproject_omni_radtan carries the p2 terms
    (a wrong Jacobian still converges for small p2, so the round trip alone
    would not catch it): FD-check it with a large p2 near the image edge."""
    from mvis.calibration.factors.custom_projections import _radtan_distort
    k1, k2, p1, p2 = -0.05, 0.01, 1e-3, 0.05
    eps = 1e-7
    for mx, my in [(0.9, -0.7), (-0.4, 0.8), (0.05, 0.02), (1.2, 0.6)]:
        J = np.array(_radtan_distort(mx, my, k1, k2, p1, p2)[2]).reshape(2, 2)
        Jn = np.zeros((2, 2))
        for j, (ex, ey) in enumerate([(eps, 0.0), (0.0, eps)]):
            fp = _radtan_distort(mx + ex, my + ey, k1, k2, p1, p2)[:2]
            fm = _radtan_distort(mx - ex, my - ey, k1, k2, p1, p2)[:2]
            Jn[:, j] = (np.array(fp) - np.array(fm)) / (2 * eps)
        np.testing.assert_allclose(J, Jn, atol=1e-7)


# ── CameraModel class ──

def test_omni_radtan_class_layout():
    m = OmniRadtan(*CAL9[:4], CAL9[4:])
    assert m.num_params() == m.n_intr == 9
    np.testing.assert_array_equal(m.cal_vec, CAL9)
    np.testing.assert_array_equal(m.distortion(), CAL9[4:])
    d = create_camera_model("omni_radtan", 300, 300, 320, 240)
    np.testing.assert_array_equal(d.distortion(), [1.0, 0, 0, 0, 0])
    uv, J_pt, J_intr = m.project(np.array([0.3, 0.2, 1.0]), jac=True)
    assert J_intr.shape == (2, 9) and np.isfinite(uv).all()
    with pytest.raises(ValueError, match="5 distortion"):
        OmniRadtan(300, 300, 320, 240, [1.0, 0, 0, 0, 0, 0.1])


def test_omni_radtan_legacy_4_coeff_is_p2_zero():
    legacy = CAL9[4:8]                                   # [xi, k1, k2, p1]
    m = OmniRadtan(*CAL9[:4], legacy)
    np.testing.assert_array_equal(m.distortion(), np.r_[legacy, 0.0])
    assert len(m.cal_vec) == 9
    for P in _rand_points(20, seed=5):
        # identical to the old 8-long (p2-less) kernel call
        old = project_omni_radtan(P, CAL9[:8])[0]
        new = m.project(P)
        if old is None:
            assert np.isnan(new).all()
        else:
            np.testing.assert_array_equal(new, old)
    # build_camera_model / CameraCalib canonicalize a legacy 8-long intr too
    assert len(build_camera_model("omni_radtan", CAL9[:8]).cal_vec) == 9
    from mvis.calibration.factors.camera_calib import CameraCalib
    c = CameraCalib(gtsam.Pose3(), 0.0, 0.0, CAL9[:8], model="omni_radtan")
    assert c.n_intr == 9 and c.DIM == 17 and c.intr[8] == 0.0


# ── camera-only end-to-end ──

_IMG = (640, 480)


def _omni_frames(board, n=30, seed=0, sigma=0.05):
    from mvis.core.types import FrameObservations, Observation
    rng = np.random.RandomState(seed)
    pts = board.all_points_3d().astype(float)
    frames, t, att = [], 0.0, 0
    while len(frames) < n and att < n * 50:
        att += 1
        T = gtsam.Pose3(gtsam.Rot3.Rodrigues(*rng.uniform(-0.6, 0.6, 3)),
                        gtsam.Point3(rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2),
                                     rng.uniform(0.35, 0.6)))
        obs = []
        for i, p in enumerate(pts):
            uv = project_omni_radtan(np.asarray(T.transformFrom(gtsam.Point3(*p))), CAL9)[0]
            if uv is None or not (5 <= uv[0] < _IMG[0] - 5 and 5 <= uv[1] < _IMG[1] - 5):
                continue
            obs.append(Observation(point_id=i, pixel=uv + rng.standard_normal(2) * sigma,
                                   timestamp=t))
        if len(obs) < 0.8 * len(pts):
            continue
        frames.append(FrameObservations(camera_id=0, timestamp=t, observations=obs,
                                        image_size=_IMG))
        t += 0.1
    return frames


def _solve_cam_only(board, frames, camera_configs=()):
    from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
    pc = PipelineConfig(camera_configs=list(camera_configs),
        max_iterations=30, max_iterations_single=30, pixel_sigma=1.0,
        huber_threshold=1.345, corner_subsample="all", optimize_intrinsics=True,
        optimize_time_offset=False, max_init_frames=100, init_frame_selection="uniform",
        reference_camera=None, rerun_after_outlier_rejection=False,
        distortion_models={0: "omni_radtan"}, verbose=False)
    return CalibrationPipeline(board, pc).run({0: frames}, {0: _IMG})


def test_cam_only_e2e_recovers_p2_and_principal_point(monkeypatch):
    """Initializer (5-coeff omni grid) + phase 1 (distortion frozen) + phase 2
    (xi frozen, k1/k2/p1/p2 free) on omni data with p2 = 4e-3. xi/fx/k stay
    coupled (doc/investigation_omni_radtan.md), but p2 and cx are recovered,
    and the old p2-less model fits worse. (How much the p2-less model biases cx
    depends on the noise draw, so that is not asserted.)"""
    from mvis.detection.chessboard import Chessboard
    import mvis.calibration.optimizer as opt_mod
    board = Chessboard(7, 9, 0.04)
    frames = _omni_frames(board)
    r = _solve_cam_only(board, frames)
    cam = r.cameras[0]
    assert cam.distortion_model == "omni_radtan" and len(cam.distortion) == 5
    p2 = float(cam.distortion[4])
    assert abs(p2 - CAL9[8]) < 5e-4, f"p2 {p2:.5f}"
    err_cx = float(cam.intrinsics[2] - CAL9[2])
    assert abs(err_cx) < 0.15, f"cx err {err_cx:+.3f}"   # seeds 0/2: +0.02/+0.07
    assert abs(float(cam.intrinsics[3] - CAL9[3])) < 0.2

    # the old model: K back to the p2-less 8-vector (kernel reads p2 = 0)
    k_full = opt_mod.k_cal_vec
    monkeypatch.setattr(opt_mod, "k_cal_vec", lambda m, i, d: (
        k_full(m, i, d)[:8] if m == "omni_radtan" else k_full(m, i, d)))
    r0 = _solve_cam_only(board, frames)
    assert r0.overall_rms > r.overall_rms


def _omni_stereo_frames(board, cal, n=30, seed=0, sigma=0.05):
    """Stereo omni rig, T_c1c0 = (Rodrigues(0, 0.05, 0), (-0.08, 0, 0)), both
    cameras `cal`; a pose is kept only if both cameras see >= 80% of corners."""
    from mvis.core.types import FrameObservations, Observation
    pts = board.all_points_3d().astype(float)
    rng = np.random.RandomState(seed)
    T_c1c0 = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.0, 0.05, 0.0), gtsam.Point3(-0.08, 0.0, 0.0))
    frames, t = {0: [], 1: []}, 0.0
    while len(frames[0]) < n:
        T = gtsam.Pose3(gtsam.Rot3.Rodrigues(*rng.uniform(-0.5, 0.5, 3)),
                        gtsam.Point3(rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15),
                                     rng.uniform(0.35, 0.6)))
        per = {}
        for cid, Tc in ((0, T), (1, T_c1c0.compose(T))):
            obs = []
            for k, p in enumerate(pts):
                uv = project_omni_radtan(np.asarray(Tc.transformFrom(gtsam.Point3(*p))), cal)[0]
                if uv is not None and 5 <= uv[0] < _IMG[0] - 5 and 5 <= uv[1] < _IMG[1] - 5:
                    obs.append(Observation(point_id=k, timestamp=t,
                                           pixel=uv + rng.standard_normal(2) * sigma))
            per[cid] = obs
        if min(len(o) for o in per.values()) < 0.8 * len(pts):
            continue
        for cid, obs in per.items():
            frames[cid].append(FrameObservations(camera_id=cid, timestamp=t,
                                                 observations=obs, image_size=_IMG))
        t += 0.1
    return frames


@pytest.mark.parametrize("xi,fx", [(1.0, 300.0), (2.5, 525.0)])
def test_omni_grid_init_chessboard_rows(xi, fx):
    """Unseeded omni init on a chessboard. Chessboard aliases _tag_size, so
    _is_tag_grid() is True for it and the omni grid grouped chessboard corners
    as tag rows (point_id // 4 ...): the line conics were fit to non-collinear
    points, gamma came out ~5x low (53 vs ~290 px) and the grid picked
    xi = 2.5, fx = 93 for truth xi = 1, fx = 300. (The grid's score is pinned
    in test_omni_free_xi.py.)"""
    from mvis.calibration.initializer import Initializer
    from mvis.detection.chessboard import Chessboard
    cal = np.array([fx, fx + 2, 322.0, 238.0, xi, -0.05, 0.01, 1e-3, 0.0])
    board = Chessboard(7, 9, 0.04)
    frames = _omni_stereo_frames(board, cal)[0]
    intr, dist = Initializer(board).init_intrinsics_opencv(frames, _IMG, "omni_radtan")
    # A seed, not a fit. The grid's fx(xi) = gamma * (1 + xi) / 2 keeps
    # fx / (1 + xi) = gamma / 2, which should match the truth's: ~4% / ~24% low
    # here for xi = 1 / 2.5 (the circle-fit gamma is exact only at xi = 1);
    # the old grouping gave 27 for both (truth 150).
    g = intr[0] / (1 + dist[0])
    assert abs(g - fx / (1 + xi)) < 0.3 * fx / (1 + xi), (intr, dist)


def test_unseeded_omni_stereo_converges():
    """Unseeded camera-only omni stereo rig (xi = 1): with the bad grid seed only
    15/30 (cam0) and 19/30 (cam1) frames got init poses, so Stage 4 attached
    cam1 frames lacking a cam0 pose to time-interpolated, unrelated cam0 poses
    (35-66 px residuals) and drove xi to ~ -0.4, RMS ~9 px."""
    from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
    from mvis.detection.chessboard import Chessboard
    cal = np.array([300.0, 302.0, 322.0, 238.0, 1.0, -0.05, 0.01, 1e-3, 0.0])
    board = Chessboard(7, 9, 0.04)
    frames = _omni_stereo_frames(board, cal)
    pc = PipelineConfig(
        max_iterations=30, max_iterations_single=30, pixel_sigma=1.0,
        huber_threshold=1.345, corner_subsample="all", optimize_intrinsics=True,
        optimize_time_offset=False, max_init_frames=100, init_frame_selection="uniform",
        reference_camera=0, rerun_after_outlier_rejection=False,
        distortion_models={0: "omni_radtan", 1: "omni_radtan"}, verbose=False)
    r = CalibrationPipeline(board, pc).run(frames, {0: _IMG, 1: _IMG})
    assert r.overall_rms < 0.2, r.overall_rms
    for cam in r.cameras:
        assert float(cam.distortion[0]) > 0.0, cam.distortion


def test_wide_omni_xi_above_2_keeps_pass_2(caplog):
    """A wide omni lens (xi = 2.5, the top of the initializer's xi grid) used
    to trip the pass-2 guard (|xi| <= 2), and the whole rig reverted to pass 1.
    xi in [0, 5] is now accepted, so the outlier rerun is kept (stereo rig:
    a single camera returns before pass 2). Seeded, so this pins only the
    pass-2 guard (the unseeded rig is test_unseeded_omni_stereo_converges)."""
    import logging
    from types import SimpleNamespace
    from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
    from mvis.detection.chessboard import Chessboard
    cal = np.array([525.0, 527.0, 322.0, 238.0, 2.5, -0.05, 0.01, 1e-3, 4e-3])
    board = Chessboard(7, 9, 0.04)
    frames = _omni_stereo_frames(board, cal)
    seeds = [SimpleNamespace(camera_id=c, shutter_type="global",
                             initial_intrinsics=[530.0, 530.0, 320.0, 240.0],
                             initial_distortion=[2.5, 0.0, 0.0, 0.0, 0.0]) for c in (0, 1)]
    pc = PipelineConfig(camera_configs=seeds,
        max_iterations=30, max_iterations_single=30, pixel_sigma=1.0,
        huber_threshold=1.345, corner_subsample="all", optimize_intrinsics=True,
        optimize_time_offset=False, max_init_frames=100, init_frame_selection="uniform",
        reference_camera=0, rerun_after_outlier_rejection=True,
        distortion_models={0: "omni_radtan", 1: "omni_radtan"}, verbose=False)
    with caplog.at_level(logging.INFO, logger="mvis.calibration.pipeline_cam"):
        r = CalibrationPipeline(board, pc).run(frames, {0: _IMG, 1: _IMG})
    assert "Rerun cam" in caplog.text                       # pass 2 ran
    assert "Pass-2 sanity check failed" not in caplog.text  # ... and was kept
    for cam in r.cameras:
        assert float(cam.distortion[0]) > 2.0, cam.distortion
    assert r.overall_rms < 0.2, r.overall_rms


def test_pipeline_legacy_4_value_omni_seed_gets_p2():
    """A config seed with the old 4-value omni initial_distortion
    [xi, k1, k2, p1] starts with p2 = 0 and still solves p2."""
    from types import SimpleNamespace
    from mvis.detection.chessboard import Chessboard
    board = Chessboard(7, 9, 0.04)
    seed = SimpleNamespace(camera_id=0, shutter_type="global",
                           initial_intrinsics=[310.0, 310.0, 320.0, 240.0],
                           initial_distortion=[1.0, 0.0, 0.0, 0.0])
    r = _solve_cam_only(board, _omni_frames(board, n=20, seed=1), [seed])
    cam = r.cameras[0]
    assert len(cam.distortion) == 5
    assert abs(float(cam.distortion[4]) - CAL9[8]) < 1e-3


# ── io ──

# TUM VI-style Kalibr omni camchain (p2 = 5.96e-4 is the real cam1 value)
_KALIBR_OMNI = dict(
    camera_model="omni", distortion_model="radtan",
    intrinsics=[1.7, 533.3, 533.1, 254.9, 256.9],
    distortion_coeffs=[-0.0123, 0.00987, 0.000712, 0.000596],
    resolution=[512, 512], rostopic="/cam1/image_raw")


def test_kalibr_omni_import_keeps_p2_and_exports_it(tmp_path):
    from mvis.io.camera_input import load_camera_intrinsics
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.io.cam_imu_output import to_kalibr_camchain_imucam
    from types import SimpleNamespace
    p = tmp_path / "camchain.yaml"
    p.write_text(yaml.safe_dump({"cam0": _KALIBR_OMNI}))
    s = load_camera_intrinsics(str(p))[0]
    assert s["distortion_model"] == "omni_radtan"
    cal_vec = list(s["intrinsics"]) + list(s["distortion_coeffs"])
    np.testing.assert_allclose(cal_vec, [533.3, 533.1, 254.9, 256.9, 1.7,
                                         -0.0123, 0.00987, 0.000712, 0.000596])
    cal = CameraCalib(gtsam.Pose3(), 0.0, 0.0, cal_vec, model="omni_radtan")
    out = to_kalibr_camchain_imucam(SimpleNamespace(
        cameras=[{"calib": cal, "image_size": (512, 512)}]))["cam0"]
    assert out["camera_model"] == "omni" and out["distortion_model"] == "radtan"
    np.testing.assert_allclose(out["intrinsics"], _KALIBR_OMNI["intrinsics"])
    np.testing.assert_allclose(out["distortion_coeffs"], _KALIBR_OMNI["distortion_coeffs"])


def test_legacy_omni_files_load_with_p2_zero(tmp_path):
    from mvis.io.camera_input import load_camera_intrinsics
    # MVIS _calib.yaml written before p2: 4 coeffs [xi, k1, k2, p1]
    mv = tmp_path / "old_calib.yaml"
    mv.write_text(yaml.safe_dump({"cameras": [dict(
        id=0, name="cam0", model="pinhole_omni_radtan",
        intrinsics=dict(fx=300.0, fy=302.0, cx=322.0, cy=238.0),
        distortion=dict(model="omni_radtan", coeffs=[1.0, -0.05, 0.01, 1e-3]),
        image_size=[640, 480], rolling_shutter_readout=0.0)]}))
    s = load_camera_intrinsics(str(mv))[0]
    assert s["distortion_coeffs"] == [1.0, -0.05, 0.01, 1e-3, 0.0]
    # camchain exported by the old MVIS writer: distortion_coeffs [k1, k2, p1, 0.0]
    kc = tmp_path / "old_camchain.yaml"
    kc.write_text(yaml.safe_dump({"cam0": dict(_KALIBR_OMNI,
                                               distortion_coeffs=[-0.0123, 0.00987, 0.000712, 0.0])}))
    s = load_camera_intrinsics(str(kc))[0]
    assert s["distortion_coeffs"][4] == 0.0 and len(s["distortion_coeffs"]) == 5


def test_kalibr_pack_legacy_8_long_omni_writes_p2_zero():
    from mvis.io.cam_imu_output import _kalibr_pack
    _, _, intr, coeffs = _kalibr_pack("omni_radtan", CAL9[:8])
    assert coeffs == [CAL9[5], CAL9[6], CAL9[7], 0.0]
    _, _, _, coeffs = _kalibr_pack("omni_radtan", CAL9)
    assert coeffs[3] == CAL9[8]


def test_fov_kalibr_export_is_single_omega():
    """Kalibr's pinhole-fov `distortion_coeffs` is the single [omega] — the
    camchain export writes exactly that (not zero-padded to 4), and it reads
    back to the same 5-long cal_vec."""
    from mvis.io.cam_imu_output import _kalibr_pack
    from mvis.io.camera_input import _from_kalibr
    cam_m, dist_m, intr, coeffs = _kalibr_pack("fov", [400., 401., 320., 240., 0.9])
    assert (cam_m, dist_m) == ("pinhole", "fov")
    assert intr == [400., 401., 320., 240.] and coeffs == [0.9]
    s = _from_kalibr({"cam0": dict(camera_model=cam_m, distortion_model=dist_m,
                                   intrinsics=intr, distortion_coeffs=coeffs,
                                   resolution=[640, 480], rostopic="/cam0")})[0]
    assert list(s["intrinsics"]) + list(s["distortion_coeffs"]) == [400., 401., 320., 240., 0.9]


def test_mvis_calib_yaml_writes_five_omni_coeffs(tmp_path):
    from mvis.core.types import CalibrationResult, CameraParams
    from mvis.io.output_writer import OutputWriter
    from mvis.io.camera_input import load_camera_intrinsics
    cp = CameraParams(0, CAL9[:4].copy(), CAL9[4:].copy(), "omni_radtan", _IMG)
    p = str(tmp_path / "r_calib.yaml")
    OutputWriter.write_calibration_yaml(CalibrationResult(cameras=[cp]), p)
    doc = yaml.safe_load(open(p))
    assert len(doc["cameras"][0]["distortion"]["coeffs"]) == 5
    s = load_camera_intrinsics(p)[0]
    np.testing.assert_allclose(s["distortion_coeffs"], CAL9[4:])


# ── mvisdb / reports ──

def test_mvisdb_legacy_omni_distortion_padded_on_load():
    from mvis.mvisdb.manager import _dist_f64
    from mvis.mvisdb.mvisdatabase import CameraModel
    raw = np.array([1.0, -0.05, 0.01, 1e-3]).tobytes()
    np.testing.assert_array_equal(_dist_f64(raw, "omni_radtan"),
                                  [1.0, -0.05, 0.01, 1e-3, 0.0])
    np.testing.assert_array_equal(_dist_f64(raw, CameraModel.OMNI_RADTAN),
                                  [1.0, -0.05, 0.01, 1e-3, 0.0])
    # never trims / pads other models' stored (possibly zero-padded) blobs
    ds = np.array([-0.2, 0.6, 0.0, 0.0]).tobytes()
    assert len(_dist_f64(ds, "double_sphere")) == 4
    assert _dist_f64(None, "omni_radtan") is None


def test_html_report_names_omni_coefficients():
    """The rendered distortion table shows omni's p2 column + value, and a
    mixed omni + radtan rig gets 5 generic d0..d4 columns (no coefficient cut)."""
    from mvis.analysis.html_report import _distortion_table_rows
    from mvis.core.types import CameraParams
    omni = CameraParams(0, CAL9[:4], CAL9[4:], "omni_radtan", _IMG)
    html = "".join(_distortion_table_rows([omni]))
    for h in ("xi", "k1", "k2", "p1", "p2"):
        assert f"<th>{h}</th>" in html
    assert html.count("<th>") == 2 + 5
    assert f"{CAL9[8]:.6f}" in html                       # the p2 value cell

    rt = CameraParams(1, CAL9[:4], [0.1, -0.2, 1e-3, 2e-3], "radtan", _IMG)
    rows = _distortion_table_rows([omni, rt])
    assert [f"<th>d{i}</th>" in rows[0] for i in range(6)] == [True] * 5 + [False]
    assert f"{CAL9[8]:.6f}" in rows[1]                     # omni row keeps p2
    assert rows[2].count("<td class='num'>") == 5          # radtan row zero-padded


def test_cam_imu_text_report_names_omni_coefficients():
    from types import SimpleNamespace
    from mvis.analysis.cam_imu_report import _cam_rows, _dist_label
    from mvis.calibration.factors.camera_calib import CameraCalib
    cal = CameraCalib(gtsam.Pose3(), 0.0, 0.0, CAL9, model="omni_radtan")
    rows = _cam_rows(SimpleNamespace(cameras=[{"calib": cal}]))
    assert _dist_label(rows[0]) == " (xi k1 k2 p1 p2)"
    cal_r = CameraCalib(gtsam.Pose3(), 0.0, 0.0, [300, 300, 320, 240, 0, 0, 0, 0],
                        model="radtan")
    assert _dist_label(_cam_rows(SimpleNamespace(cameras=[{"calib": cal_r}]))[0]) == ""


# ── regression gate ──

def test_cam_imu_gate_fails_on_key_set_mismatch():
    from mvis.regression.cam_imu_adapter import CamImuAdapter
    ad = CamImuAdapter()
    # every cam-IMU tolerance key; the test checks the key-set logic, not the values
    tol = {"reproj_rms_px": 0.1, "reproj_rms_rel": 0.05, "cam_reproj_px": 0.15,
           "td_ms": 1.0, "lever_mm": 5.0, "baseline_mm": 2.0, "dwda_diag": 0.01}
    ref = {"reproj_rms_px": 0.3, "cam0.td_ms": 1.0, "cam1.td_ms": 2.0}
    same = ad.compare_to_baseline(dict(ref), ref, tol)
    assert same and all(r.ok for r in same)
    fewer = ad.compare_to_baseline({"reproj_rms_px": 0.3, "cam0.td_ms": 1.0}, ref, tol)
    assert not all(r.ok for r in fewer)
    assert any("cam1.td_ms" in r.label and "missing in result" in r.label for r in fewer)
    more = ad.compare_to_baseline(dict(ref, **{"imu1.td_ms": 0.5}), ref, tol)
    assert any("imu1.td_ms" in r.label and "not in baseline" in r.label
               and not r.ok for r in more)
    # a label absent / None on both sides is not a mismatch
    nones = ad.compare_to_baseline(dict(ref, **{"cam2.baseline_mm": None}), ref, tol)
    assert all(r.ok for r in nones)
