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
"""kb4 / omni_radtan (9) beyond the solvers: the model-aware simulator (digital
twin + video overlay), the kb4 valid-domain metadata (first theta_d fold vs the
image corner), the Stage-1 init sanity rule and the recovery-table labels.
"""
import logging
from types import SimpleNamespace

import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.camera.camera_model import build_camera_model

KB4 = np.array([300.0, 301.0, 321.0, 239.0, 0.02, -0.01, 3e-3, -5e-4])
OMNI9 = np.array([330.0, 331.0, 321.0, 239.0, 1.1, -0.05, 0.01, 1e-3, 2e-3])
_IMG = (640, 480)


def _board_points():
    g = np.arange(7) * 0.1 - 0.3
    return np.array([[x, y, 0.0] for y in g for x in g])


def _stub_result(model, cal_vec):
    """A CamImuCalibResult-like stub: 3 s of slow motion, the board 1.5 m ahead."""
    X = gtsam.symbol_shorthand.X
    T_CtoI = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.01, -0.02, 0.03),
                         gtsam.Point3(0.01, -0.02, 0.03))
    cc = CameraCalib(T_CtoI, 0.004, 0.0, np.asarray(cal_vec, float), model=model)
    ts = 1.0 + np.arange(90) / 30.0
    vals = gtsam.Values()
    for k, t in enumerate(ts):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3.Rodrigues(0.05 * t, -0.03 * t, 0.1 * t),
                                      gtsam.Point3(0.2 * t - 0.3, 0.05 * t, 0.0)))
    return SimpleNamespace(
        cameras=[{"calib": cc, "image_size": _IMG}], imus=[], imu_intr_xi=np.zeros(24),
        b_g=np.zeros(3), b_a=np.zeros(3),
        T_BinW=gtsam.Pose3(gtsam.Rot3.Rx(0.1), gtsam.Point3(0.0, 0.0, 1.5)),
        values=vals, t_states=ts, imu_model="calyx")


# ── model-aware simulation ──

@pytest.mark.parametrize("model,cal", [("kb4", KB4), ("omni_radtan", OMNI9)])
def test_sim_from_result_reprojects_through_camera_model(model, cal, tmp_path):
    """capture_gt -> json -> simulate_cameras keeps the model, and every
    noise-free simulated corner is the CameraModel's projection of that board
    point at the frame pose (not a gtsam Cal3 of the first 4 coefficients)."""
    from mvis.simulation.from_result import capture_gt, gt_from_json, gt_to_json
    from mvis.simulation.multicam import simulate_cameras, traj_from_gt
    gt = capture_gt(_stub_result(model, cal))
    assert gt.cams[0]["model"] == model and len(gt.cams[0]["intr8"]) == len(cal)
    gt_to_json(gt, str(tmp_path / "gt.json"))
    gt = gt_from_json(str(tmp_path / "gt.json"))
    pts_B = _board_points()
    sim = simulate_cameras(gt, pts_B, rates=[30.0], sigma_px=0.0)[0]
    assert sim["model"] == model
    np.testing.assert_allclose(sim["cal_vec"], cal)
    cm = build_camera_model(model, cal)
    traj = traj_from_gt(gt)
    T_BinW = gtsam.Pose3(np.asarray(gt.T_BinW, float))
    T_CtoI = gtsam.Pose3(np.asarray(gt.cams[0]["T_CtoI"], float))
    td = gt.cams[0]["td"]
    n = 0
    for t, ids, uv in zip(sim["t_cam"], sim["ids_list"], sim["uv_list"]):
        R_GtoI, p = traj.get_pose(float(t) + td)
        T_CtoW = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(T_CtoI)
        for i, u in zip(ids, uv):
            pc = np.asarray(T_CtoW.transformTo(
                T_BinW.transformFrom(gtsam.Point3(*pts_B[i]))), float)
            np.testing.assert_allclose(u, cm.project(pc), atol=1e-9)
            n += 1
    assert n > 1000


def test_sim_cal_from_vec_model_dispatch():
    from mvis.simulation.from_result import ModelCal, cal_from_vec
    radtan = [300, 300, 320, 240, -0.1, 0.01, 0.001, 0.0]
    assert isinstance(cal_from_vec(radtan, False), gtsam.Cal3DS2)      # legacy flag
    assert isinstance(cal_from_vec(radtan, True), gtsam.Cal3Fisheye)
    assert isinstance(cal_from_vec(KB4, False, "kb4"), ModelCal)
    xy = np.array([0.3, -0.2])
    np.testing.assert_allclose(cal_from_vec(KB4, False, "kb4").uncalibrate(xy),
                               build_camera_model("kb4", KB4).project(np.r_[xy, 1.0]))
    with pytest.raises(ValueError, match="no distortion model"):
        cal_from_vec(OMNI9, False)                   # 9 long: must be named
    with pytest.raises(ValueError, match="Unknown distortion model"):
        cal_from_vec(KB4, False, "kb12")


def test_multi_board_sim_config_carries_model():
    from mvis.simulation.from_result import ModelCal
    from mvis.simulation.multi_board import MultiBoardSimConfig
    c = MultiBoardSimConfig(camera_model="kb4", cam_intrinsics=tuple(KB4))
    assert isinstance(c.cal(), ModelCal) and c.model_name() == "kb4"
    legacy = MultiBoardSimConfig()
    assert isinstance(legacy.cal(), gtsam.Cal3DS2) and legacy.model_name() == "radtan"


def test_video_overlay_matches_camera_model():
    """CalibScene.model drives _reproject_frame; an omni scene's overlay equals the
    CameraModel projection, and a 9-long cal_vec without a model raises."""
    from mvis.simulation.from_result import fit_bspline_from_states
    from mvis.simulation.video import CalibScene, _cam_pose, _reproject_frame
    r = _stub_result("omni_radtan", OMNI9)
    X = gtsam.symbol_shorthand.X
    traj, _ = fit_bspline_from_states(
        r.t_states, [r.values.atPose3(X(k)).matrix() for k in range(len(r.t_states))], 0.05)
    T_CtoI = r.cameras[0]["calib"].T
    scene = CalibScene(traj=traj, board_points={0: _board_points()},
                       board_poses={0: r.T_BinW}, cal_vec=OMNI9, is_fisheye=False,
                       image_size=_IMG, T_CtoI=T_CtoI, td=0.004, model="omni_radtan")
    t, ids = 2.0, np.arange(49)
    uv = _reproject_frame(scene, t, ids)
    cm = build_camera_model("omni_radtan", OMNI9)
    T_CtoW = _cam_pose(traj, T_CtoI, t + 0.004)
    want = np.array([cm.project(np.asarray(T_CtoW.transformTo(
        r.T_BinW.transformFrom(gtsam.Point3(*p))), float)) for p in _board_points()])
    np.testing.assert_allclose(uv, want, atol=1e-9)
    scene.model = None
    with pytest.raises(ValueError, match="no distortion model"):
        _reproject_frame(scene, t, ids)


# ── kb4 valid domain ──

def _kb4_fold_at_radius(r_px, f=300.0):
    """kb4 with only k1 = -a: dtd = 1 - 3 a th^2 folds at th_f = 1/sqrt(3a),
    where theta_d = 2/3 th_f; pick a so the fold radius f * theta_d = r_px."""
    th_f = 1.5 * r_px / f
    return np.array([f, f, 320.0, 240.0, -1.0 / (3 * th_f ** 2)] + [0.0] * 3)


def test_kb4_valid_domain_fold_inside_image_warns(caplog):
    from mvis.camera.kb_domain import check_kb_valid_domain, kb_valid_domain
    corner = 400.0                                    # 640x480 about (320, 240)
    inside = _kb4_fold_at_radius(390.0)
    info = kb_valid_domain(inside, _IMG)
    assert info["fold_theta_rad"] == pytest.approx(1.5 * 390 / 300)
    assert info["max_radius_px"] == pytest.approx(390.0)
    assert info["corner_radius_px"] == pytest.approx(corner)
    assert info["covers_image"] is False
    with caplog.at_level(logging.INFO):
        check_kb_valid_domain(inside, _IMG, label="cam0")
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "INSIDE the image" in caplog.text

    caplog.clear()
    # fold just outside the corner / KB4 folds at ~125 deg, far outside /
    # truly monotonic (k = 0: theta_d = theta up to 180 deg)
    mono = np.r_[KB4[:4], np.zeros(4)]
    for cal in (_kb4_fold_at_radius(410.0), KB4, mono):
        with caplog.at_level(logging.INFO):
            info = check_kb_valid_domain(cal, _IMG)
        assert info["covers_image"] is True
    assert all(r.levelno == logging.INFO for r in caplog.records)
    assert kb_valid_domain(mono, _IMG)["fold_theta_rad"] is None
    assert kb_valid_domain(KB4, None)["covers_image"] is None      # no image size


def test_kb4_valid_domain_monotonic_short_names_no_fold(caplog):
    """Monotonic theta_d whose 180 deg radius is inside the image: warn, but
    without claiming a fold; reports use a neutral tag."""
    from mvis.analysis.html_report import _kb_domain_rows
    from mvis.analysis.report import print_report
    from mvis.camera.kb_domain import check_kb_valid_domain, kb_domain_tag
    from mvis.core.types import CalibrationResult, CameraParams
    short = np.array([100.0, 100.0, 320.0, 240.0] + [0.0] * 4)  # f*pi = 314 < 400
    with caplog.at_level(logging.INFO):
        info = check_kb_valid_domain(short, _IMG)
    assert info["fold_theta_rad"] is None and info["covers_image"] is False
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "fold lies" not in caplog.text
    assert "180 deg INSIDE the image" in caplog.text
    assert kb_domain_tag(info) == "VALID DOMAIN ENDS INSIDE IMAGE"
    assert kb_domain_tag(check_kb_valid_domain(
        _kb4_fold_at_radius(390.0), _IMG)) == "FOLDS INSIDE IMAGE"
    res = CalibrationResult(cameras=[CameraParams(0, short[:4], short[4:], "kb4", _IMG)])
    txt = print_report(res)
    assert "[VALID DOMAIN ENDS INSIDE IMAGE]" in txt and "FOLDS" not in txt
    html = "".join(_kb_domain_rows(res.cameras))
    assert "domain ends inside image" in html and "fold inside image" not in html


def test_kb4_valid_domain_in_yaml_and_reports():
    from mvis.analysis.html_report import _kb_domain_rows
    from mvis.analysis.report import print_report
    from mvis.core.types import CalibrationResult, CameraParams
    from mvis.io.output_writer import OutputWriter
    from mvis.io.result_schema import camera_intrinsics_block
    blk = camera_intrinsics_block(0, "kb4", KB4[:4], KB4[4:], _IMG, 0.0)
    assert blk["valid_domain"]["covers_image"] is True
    assert "valid_domain" not in camera_intrinsics_block(
        0, "equidistant", KB4[:4], [0.0] * 4, _IMG, 0.0)          # others unchanged
    bad = _kb4_fold_at_radius(390.0)
    res = CalibrationResult(cameras=[CameraParams(0, bad[:4], bad[4:], "kb4", _IMG)])
    d = OutputWriter.to_calibration_dict(res)
    assert d["cameras"][0]["valid_domain"]["covers_image"] is False
    assert "kb4 valid domain: first fold" in print_report(res)
    assert "[FOLDS INSIDE IMAGE]" in print_report(res)
    html = "".join(_kb_domain_rows(res.cameras))
    assert "KB4 Valid Domain" in html and "fold inside image" in html
    radtan = CalibrationResult(cameras=[CameraParams(0, KB4[:4], [0.0] * 4, "radtan", _IMG)])
    assert _kb_domain_rows(radtan.cameras) == []
    assert "valid domain" not in print_report(radtan)


def test_kb4_valid_domain_yaml_round_trip(tmp_path):
    """The metadata key does not disturb the MVIS yaml reader."""
    from mvis.core.types import CalibrationResult, CameraParams
    from mvis.io.camera_input import load_camera_intrinsics
    from mvis.io.output_writer import OutputWriter
    res = CalibrationResult(cameras=[CameraParams(0, KB4[:4], KB4[4:], "kb4", _IMG)])
    p = tmp_path / "x_calib.yaml"
    OutputWriter.write_calibration_yaml(res, str(p))
    s = load_camera_intrinsics(str(p))[0]
    assert s["distortion_model"] == "kb4"
    np.testing.assert_allclose(s["distortion_coeffs"], KB4[4:])


# ── Stage-1 init sanity ──

def test_stage1_kb4_init_judged_like_equidistant():
    from mvis.calibration.pipeline_cam import _stage1_dist_extreme
    seed = [-0.6, 0, 0, 0]                 # folds at ~43 deg, fine for KB4 init
    assert _stage1_dist_extreme("equidistant", seed) is False
    assert _stage1_dist_extreme("kb4", seed) is False
    assert _stage1_dist_extreme("kb4", [6.0, 0, 0, 0]) is True        # plain bound
    assert _stage1_dist_extreme("equidistant", [6.0, 0, 0, 0]) is True


# ── recovery-table labels ──

def test_recovery_labels_named_for_kb4_only():
    from mvis.simulation.report import _intr_names
    assert _intr_names(8, "kb4")[4:] == [f"k{i}" for i in range(1, 5)]
    assert _intr_names(9, "omni_radtan") == ["fx", "fy", "cx", "cy",
                                             "d0", "d1", "d2", "d3", "d4"]


def test_recovery_scores_legacy_omni_gt_p2():
    """An 8-long (pre-p2) omni GT vs a 9-param estimate: the GT is padded with
    p2 = 0, so p2 is scored instead of silently dropped by min(len)."""
    from mvis.simulation.report import _gt_intr
    est = CameraCalib(gtsam.Pose3(), 0.0, 0.0, OMNI9, model="omni_radtan")
    gi = _gt_intr(dict(cam_intrinsics=OMNI9[:8]), est)
    np.testing.assert_array_equal(gi, np.r_[OMNI9[:8], 0.0])
    radtan = CameraCalib(gtsam.Pose3(), 0.0, 0.0, KB4, model="radtan")
    np.testing.assert_array_equal(_gt_intr(dict(cam_intrinsics=KB4), radtan), KB4)
