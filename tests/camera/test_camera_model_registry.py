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
"""Camera-model registry (`mvis.core.camera_models`) and the >4-coefficient
plumbing that consumes it.

Tests:
  - registry tables are consistent (MODELS / N_DIST / DIST_NAMES / defaults,
    GTSAM vs CUSTOM split) and agree with the CameraModel classes
  - canonical_dist / canonical_cal_vec pad, trim zero extras, raise otherwise
  - unknown model names RAISE everywhere a GTSAM-vs-custom choice is made
    (create_camera_model, build_camera_model, CameraCalib, project_observation,
    the optimizer's unified factor and K layout) instead of becoming radtan
  - custom kernels return J_cal of width max(8, len(cal_vec))
  - camera-only K dim = 4 + max(4, n_dist): 8 for every existing model
  - the legacy is_fisheye-only default (no model) raises for a non-8 cal_vec
    everywhere instead of padding a custom cal_vec into radtan
  - io / simulation / report Cal3 builders never treat custom models as gtsam
  - CamAdapter: a metric label on only one side is a gate failure
"""

import numpy as np
import pytest
import gtsam

from mvis.core import camera_models as cm
from mvis.camera.camera_model import build_camera_model, create_camera_model
from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.calibration.factors.camera_projection import k_cal_vec, project_observation
from mvis.calibration.factors.custom_projections import (
    CUSTOM_MODELS, project_double_sphere, project_eucm, project_fov,
    project_omni_radtan, project_ucm,
)

IMPLEMENTED = ("radtan", "equidistant", "fov", "double_sphere", "eucm", "ucm", "omni_radtan")


# ── registry tables ──

def test_registry_tables_consistent():
    assert set(cm.MODELS) == set(cm.N_DIST) == set(cm.DIST_NAMES)
    assert cm.GTSAM == {"radtan", "equidistant"}
    assert cm.CUSTOM == set(cm.MODELS) - cm.GTSAM
    assert "kb4" in cm.CUSTOM and "kb4" in cm.KALIBR_EXPORTABLE
    assert cm.KALIBR_EXPORTABLE == set(cm.MODELS)
    assert CUSTOM_MODELS is cm.CUSTOM          # custom_projections alias
    for m in cm.MODELS:
        assert len(cm.DIST_NAMES[m]) == cm.n_dist(m) == len(cm.default_dist(m))
        assert cm.is_custom(m) == (m in cm.CUSTOM)
    assert cm.n_dist("kb4") == 4
    assert cm.DIST_NAMES["kb4"] == tuple(f"k{i}" for i in range(1, 5))


@pytest.mark.parametrize("model", IMPLEMENTED)
def test_registry_matches_camera_classes(model):
    cam = create_camera_model(model, 300.0, 300.0, 320.0, 240.0)
    assert len(cam.distortion()) == cm.n_dist(model)
    assert cam.n_intr == 4 + cm.n_dist(model) == cam.num_params()
    np.testing.assert_array_equal(cam.distortion(), cm.default_dist(model))


def test_defaults_preserved():
    assert cm.default_dist("double_sphere") == [-0.2, 0.6]
    assert cm.default_dist("eucm") == [0.6, 1.0]
    assert cm.default_dist("ucm") == [0.6]
    assert cm.default_dist("fov") == [1.0]
    assert cm.default_dist("omni_radtan")[0] == 1.0
    assert cm.default_dist("radtan") == [0.0] * 4
    assert cm.default_dist("kb4") == [0.0] * 4
    d = cm.default_dist("eucm")
    d[0] = 9.0                                  # a copy, not the table
    assert cm.default_dist("eucm") == [0.6, 1.0]


# ── canonical_dist ──

def test_canonical_dist_pads_and_trims():
    assert cm.canonical_dist("kb4", [0.1, 0.2]) == [0.1, 0.2, 0.0, 0.0]
    assert cm.canonical_dist("double_sphere", [-0.2, 0.6, 0.0, 0.0]) == [-0.2, 0.6]
    assert cm.canonical_dist("fov", [0.9, 0, 0, 0]) == [0.9]
    assert cm.canonical_dist("radtan", None) == [0.0] * 4
    assert cm.canonical_dist("radtan", [1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]
    assert cm.canonical_cal_vec("eucm", [1, 2, 3, 4, 0.6, 1.0, 0, 0]) == [1, 2, 3, 4, 0.6, 1.0]
    assert cm.canonical_cal_vec("ucm", [1, 2, 3, 4, 0.6, 0, 0, 0]) == [1, 2, 3, 4, 0.6]


def test_canonical_dist_raises():
    with pytest.raises(ValueError, match="non-zero extras"):
        cm.canonical_dist("double_sphere", [-0.2, 0.6, 0.1, 0.0])
    with pytest.raises(ValueError, match="non-zero extras"):
        cm.canonical_dist("kb4", [0.0] * 4 + [1e-3])
    with pytest.raises(ValueError, match="Unknown distortion model"):
        cm.canonical_dist("bogus", [0.0])
    with pytest.raises(ValueError):
        cm.canonical_cal_vec("radtan", [1.0, 2.0])


@pytest.mark.parametrize("fn", [cm.n_dist, cm.default_dist, cm.is_custom, cm.check_model])
def test_registry_unknown_model_raises(fn):
    with pytest.raises(ValueError, match="Unknown distortion model"):
        fn("pinhole-radtan")


# ── unknown models raise (never silently radtan) ──

def test_create_camera_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown distortion model"):
        create_camera_model("bogus", 1.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="Unknown distortion model"):
        build_camera_model("bogus", np.ones(8))


def test_kb4_has_a_class():
    cm = create_camera_model("kb4", 1.0, 1.0, 0.0, 0.0)
    assert cm.distortion_model_name() == "kb4" and cm.n_intr == 8


def test_camera_calib_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown distortion model"):
        CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.ones(8), model="bogus")


def _project(model, cal_vec):
    bp = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, -1.0))  # T_board_cam
    return project_observation(
        board_point=np.array([0.05, 0.02, 0.0]), bp0=bp, bp1=bp,
        time_meas=0.0, time_0=0.0, time_1=1.0, cal_vec=cal_vec,
        distortion_model=model)


def test_project_observation_unknown_raises():
    cal = np.array([300.0, 300.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])
    assert _project("radtan", cal) is not None      # sanity: valid geometry
    with pytest.raises(ValueError, match="Unknown distortion model"):
        _project("bogus", cal)
    # kb4 projects through its own kernel (k = 0 -> equidistant), never Cal3DS2
    np.testing.assert_allclose(_project("kb4", cal),
                               _project("equidistant", cal), atol=1e-9)


def test_optimizer_factor_unknown_raises():
    from mvis.calibration.optimizer import _make_unified_factor
    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    args = (np.zeros(2), np.zeros(3), 0.0, 480.0, 0.0, 0.0, 1.0, noise,
            gtsam.symbol("B", 0), gtsam.symbol("B", 1), gtsam.symbol("L", 0),
            gtsam.symbol("K", 0))
    _make_unified_factor(*args, distortion_model="eucm")      # registered: fine
    with pytest.raises(ValueError, match="Unknown distortion model"):
        _make_unified_factor(*args, distortion_model="bogus")
    with pytest.raises(ValueError, match="Unknown distortion model"):
        k_cal_vec("bogus", [1, 1, 0, 0], [0, 0, 0, 0])


def test_cal_from_vector_rejects_non_gtsam_length():
    _cal_from_vector(np.ones(8), False)
    with pytest.raises(ValueError, match="8-element"):
        _cal_from_vector(np.ones(9), False)
    with pytest.raises(ValueError, match="8-element"):
        _cal_from_vector(np.ones(12), True)


# ── K layout / canonical lengths ──

def test_k_cal_vec_dims():
    intr = [300.0, 301.0, 320.0, 240.0]
    for m in IMPLEMENTED:          # <=4-coeff models: Vector8; omni (p2): 9
        v = k_cal_vec(m, intr, cm.default_dist(m))
        assert v.shape == ((9,) if m == "omni_radtan" else (8,))
        np.testing.assert_array_equal(v[:4], intr)
    # legacy 4-coeff omni [xi, k1, k2, p1] -> p2 = 0
    np.testing.assert_array_equal(k_cal_vec("omni_radtan", intr, [1, .1, .2, .01]),
                                  intr + [1, .1, .2, .01, 0.0])
    np.testing.assert_array_equal(k_cal_vec("fov", intr, [0.9]),
                                  intr + [0.9, 0.0, 0.0, 0.0])
    # legacy: radtan's OpenCV k3 tail is cut as the old Vector8 did (warned)
    with pytest.warns(UserWarning, match="radtan"):
        np.testing.assert_array_equal(k_cal_vec("radtan", intr, [1, 2, 3, 4, 5]),
                                      intr + [1, 2, 3, 4])
    np.testing.assert_array_equal(k_cal_vec("double_sphere", intr, [-0.2, 0.6, 0, 0]),
                                  intr + [-0.2, 0.6, 0.0, 0.0])   # zero-padded ds ok
    # every other model: a non-zero extra coefficient raises, never cut
    with pytest.raises(ValueError, match="non-zero extras"):
        k_cal_vec("double_sphere", intr, [-0.2, 0.6, 0.1, 0])
    with pytest.raises(ValueError, match="non-zero extras"):
        k_cal_vec("omni_radtan", intr, [1, .1, .2, .01, .02, .03])
    assert k_cal_vec("kb4", intr, [0.1] * 4).shape == (8,)
    with pytest.raises(ValueError, match="non-zero extras"):
        k_cal_vec("kb4", intr, [0.1] * 8)


def test_camera_calib_canonicalizes_padded_intr():
    ds8 = [300.0, 300.0, 320.0, 240.0, -0.2, 0.6, 0.0, 0.0]
    c = CameraCalib(gtsam.Pose3(), 0.0, 0.0, ds8, model="double_sphere")
    assert c.n_intr == 6 and c.DIM == 14
    np.testing.assert_array_equal(c.intr, ds8[:6])
    r = CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.arange(8.0) + 1)   # radtan untouched
    assert r.DIM == 16
    with pytest.raises(ValueError, match="non-zero extras"):
        CameraCalib(gtsam.Pose3(), 0.0, 0.0, ds8[:6] + [0.1, 0.0], model="double_sphere")
    with pytest.raises(ValueError, match="no gtsam Cal3"):
        c.cal
    assert build_camera_model("double_sphere", ds8).n_intr == 6


# ── kernel Jacobian width ──

KERNELS = [
    (project_fov, [0.9]),
    (project_double_sphere, [-0.2, 0.6]),
    (project_eucm, [0.6, 1.0]),
    (project_ucm, [0.64]),
    (project_omni_radtan, [1.0, -0.1, 0.01, 1e-3]),
]


@pytest.mark.parametrize("kernel,dist", KERNELS)
@pytest.mark.parametrize("n_pad", [0, 4, 8])
def test_kernel_jacobian_width(kernel, dist, n_pad):
    # native, zero-padded-to-8 and over-long cal_vecs
    cal = np.array([300.0, 300.0, 320.0, 240.0] + dist + [0.0] * n_pad)
    px, J_pt, J_cal = kernel(np.array([0.1, -0.05, 1.0]), cal, True)
    assert px is not None and J_pt.shape == (2, 3)
    assert J_cal.shape == (2, max(8, len(cal)))
    # the native 8-wide block is unchanged by padding
    _, _, J8 = kernel(np.array([0.1, -0.05, 1.0]),
                      np.r_[cal[:4 + len(dist)], np.zeros(8 - 4 - len(dist))], True)
    np.testing.assert_array_equal(J_cal[:, :8], J8)
    # past the native coefficients only zeros — except omni's p2 column 8,
    # which is a real (non-zero) derivative even at p2 = 0
    first_zero = 9 if kernel is project_omni_radtan else 8
    assert not J_cal[:, first_zero:].any()


@pytest.mark.parametrize("model", IMPLEMENTED)
def test_camera_model_project_jacobian_is_n_intr_wide(model):
    cam = create_camera_model(model, 300.0, 300.0, 320.0, 240.0)
    uv, J_pt, J_intr = cam.project(np.array([0.1, -0.05, 1.0]), jac=True)
    assert np.isfinite(uv).all() and J_intr.shape == (2, cam.n_intr)


# ── regression compare: key-set mismatch is a failure ──

def test_cam_adapter_key_mismatch_fails():
    from mvis.regression.cam_adapter import CamAdapter
    a = CamAdapter()
    # every cam tolerance key; the test checks the key-set logic, not the values
    tol = {"focal_pct": 0.005, "focal_abs": 1.0, "principal_px": 2.0, "dist_abs": 0.02,
           "reproj_px": 0.1, "reproj_rel": 0.1, "baseline_mm": 1.0}
    stats = {"overall_median": 0.3, "overall_p95": 5.0, "baseline_mm": None,
             "per_camera": [{"camera": 0, "fx": 445.0, "fy": 444.0, "cx": 360.0,
                             "cy": 280.0, "median": 0.3, "p95": 5.0,
                             "distortion": [1.0, -0.3, 0.1, 0.0]}]}
    ref = a.extract_metrics(stats)
    rows = a.compare_to_baseline(dict(ref), ref, tol)
    assert rows and all(r.ok for r in rows)                 # self: passes

    longer = dict(ref, **{"cam0.dist[4]": 0.002})           # e.g. omni gains p2
    bad = [r for r in a.compare_to_baseline(longer, ref, tol) if not r.ok]
    assert [r.label for r in bad] == ["cam0.dist[4] [not in baseline]"]

    shorter = {k: v for k, v in ref.items() if k != "cam0.dist[3]"}
    bad = [r for r in a.compare_to_baseline(shorter, ref, tol) if not r.ok]
    assert [r.label for r in bad] == ["cam0.dist[3] [missing in result]"]

    # a None-valued one-sided label carries no information: not a failure
    ref_nobl = {k: v for k, v in ref.items() if k != "baseline_mm"}
    rows = a.compare_to_baseline(dict(ref), ref_nobl, tol)
    assert all(r.ok for r in rows)


# ── legacy is_fisheye-only default: only 8-long radtan / equidistant ──

DS6 = np.array([300.0, 300.0, 320.0, 240.0, -0.2, 0.6])


def test_legacy_model():
    assert cm.legacy_model(False) == "radtan"
    assert cm.legacy_model(True, np.zeros(8)) == "equidistant"
    with pytest.raises(ValueError, match="no distortion model"):
        cm.legacy_model(False, DS6)


def test_no_model_custom_cal_vec_raises_not_radtan():
    """A 6-long ds cal_vec with no model used to be zero-padded into radtan
    [xi, alpha, 0, 0] = [k1, k2, p1, p2] and handed to cv2."""
    from mvis.calibration.mono_sfm import normalize_detections
    from mvis.calibration.cam_imu_init import pnp_pose_track
    uv = [np.array([[320.0, 240.0], [330.0, 250.0]])]
    with pytest.raises(ValueError, match="no distortion model"):
        CameraCalib(gtsam.Pose3(), 0.0, 0.0, DS6)
    with pytest.raises(ValueError, match="no distortion model"):
        normalize_detections([np.array([0, 1])], uv, DS6, False)
    with pytest.raises(ValueError, match="no distortion model"):
        pnp_pose_track([0.0], [np.array([0, 1])], uv, np.zeros((2, 3)), DS6)
    with pytest.raises(ValueError, match="no distortion model"):
        project_observation(
            board_point=np.zeros(3), bp0=gtsam.Pose3(), bp1=gtsam.Pose3(),
            time_meas=0.0, time_0=0.0, time_1=1.0, cal_vec=DS6, is_fisheye=False)
    # with the model named, the same cal_vec is fine
    assert CameraCalib(gtsam.Pose3(), 0.0, 0.0, DS6, model="double_sphere").DIM == 14
    assert len(normalize_detections([np.array([0, 1])], uv, DS6,
                                    model="double_sphere")[0][0]) == 2


def test_camera_input_fit_coeffs_never_drops_nonzero():
    from mvis.io.camera_input import _fit_coeffs
    assert _fit_coeffs("eucm", [0.6, 1.0, 0.0, 0.0]) == [0.6, 1.0]   # MVIS zero-pad
    assert _fit_coeffs("omni_radtan", [1.0, 0.1]) == [1.0, 0.1, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="non-zero extras"):
        _fit_coeffs("equidistant", [.1, .2, .3, .4, .5, .6, .7, .8])   # 8 coefficients


def test_sim_cal_from_vec_is_model_aware():
    from mvis.simulation.from_result import cal_from_vec, gt_cam_model
    omni8 = [300, 300, 320, 240, 1.0, -0.1, 0.01, 0.001]
    assert isinstance(cal_from_vec(omni8, False), gtsam.Cal3DS2)    # legacy: radtan
    # a named custom model projects through its CameraModel (8-long omni: p2 = 0)
    assert not isinstance(cal_from_vec(omni8, False, model="omni_radtan"),
                          (gtsam.Cal3DS2, gtsam.Cal3Fisheye))
    with pytest.raises(ValueError, match="no distortion model"):
        cal_from_vec(DS6, False)
    assert gt_cam_model({"intr8": omni8, "is_fisheye": True}) == "equidistant"
    assert gt_cam_model({"intr8": omni8, "model": "omni_radtan"}) == "omni_radtan"


def test_sim_cameras_simulate_custom_gt_as_its_model():
    """An omni GT camera (8 long, like radtan) is simulated as omni, not radtan."""
    from types import SimpleNamespace
    from mvis.simulation.multicam import simulate_cameras
    gt = SimpleNamespace(cams=[dict(intr8=[300, 300, 320, 240, 1.0, -0.1, 0.01, 0.001],
                                    model="omni_radtan", is_fisheye=False,
                                    T_CtoI=np.eye(4).tolist(), image_size=[640, 480])],
                         t_states=[0.0, 1.0, 2.0], T_states=[np.eye(4).tolist()] * 3,
                         T_BinW=np.eye(4).tolist())
    out = simulate_cameras(gt, np.zeros((4, 3)))
    assert out[0]["model"] == "omni_radtan"


def test_regenerate_html_cal3_only_for_gtsam_models():
    from mvis.analysis.regenerate_html import _build_calibrations
    from mvis.core.types import CalibrationResult, CameraParams
    cams = [CameraParams(camera_id=i, intrinsics=np.array([300.0, 300, 320, 240]),
                         distortion=np.array(d, float), distortion_model=m)
            for i, (m, d) in enumerate([("radtan", [0.1, 0, 0, 0]),
                                        ("equidistant", [0.1, 0, 0, 0]),
                                        ("double_sphere", [-0.2, 0.6])])]
    out = _build_calibrations(CalibrationResult(cameras=cams))
    assert isinstance(out[0], gtsam.Cal3DS2) and isinstance(out[1], gtsam.Cal3Fisheye)
    assert 2 not in out                                 # not faked as Cal3Fisheye
