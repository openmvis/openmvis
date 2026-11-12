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
"""D3 — camera-input / cam-IMU-output model round-trips for all distortion models.

Locks in that every supported model (radtan, equidistant, fov, double_sphere,
eucm, ucm, omni_radtan, kb4) survives:
  * `_norm_model` name resolution (MVIS + Kalibr aliases),
  * Kalibr pack (`_kalibr_pack`) -> unpack (`_from_kalibr`) (kb4 excepted:
    no Kalibr equivalent, the pack raises),
  * MVIS-format YAML write (`_cam_block`) -> read (`_from_mvis`),
with the flat cal_vec [fx,fy,cx,cy, dist...] preserved.
"""
import numpy as np
import pytest

from mvis.io import camera_input as ci
from mvis.io import cam_imu_output as co

MODELS = {
    "radtan":        [400., 401., 320., 240., -0.1, 0.05, 0.001, 0.002],
    "equidistant":   [400., 401., 320., 240., 0.1, 0.01, 0.001, 0.0001],
    "fov":           [400., 401., 320., 240., 0.9],
    "double_sphere": [400., 401., 320., 240., -0.2, 0.6],
    "eucm":          [400., 401., 320., 240., 0.6, 1.0],
    "ucm":           [400., 401., 320., 240., 0.6],
    "omni_radtan":   [400., 401., 320., 240., 1.2, 0.05, 0.001, 0.002, -0.0015],  # + p2
    "kb4":           [400., 401., 320., 240., 0.02, -0.01, 3e-3, -5e-4],
}
KALIBR_MODELS = [m for m in MODELS if m != "kb4"]


@pytest.mark.parametrize("name,expected", [
    ("radtan", "radtan"), ("plumb_bob", "radtan"),
    ("equidistant", "equidistant"), ("fisheye", "kb4"), ("kb4", "kb4"),
    ("pinhole equidistant", "equidistant"), ("fisheye equidistant", "equidistant"),
    ("fov", "fov"),
    ("double_sphere", "double_sphere"), ("ds none", "double_sphere"),
    ("eucm none", "eucm"),
    ("omni radtan", "omni_radtan"),
    ("pinhole_double_sphere", "double_sphere"),
    # ucm as a whole token / alias; eucm and bare omni are NOT ucm
    ("ucm", "ucm"), ("pinhole_ucm", "ucm"), ("unified", "ucm"), ("ucm_none", "ucm"),
    ("UCM", "ucm"), ("omni_none", "ucm"), ("omni-none", "ucm"),
    ("eucm", "eucm"), ("pinhole_eucm", "eucm"), ("omni", "omni_radtan"),
    ("omni_radtan", "omni_radtan"),
    # kb4 spellings resolve BEFORE the equidistant aliases they embed
    ("pinhole_kb4", "kb4"), ("kannala_brandt4", "kb4"), ("kb_4", "kb4"),
    ("kannala_brandt", "kb4"), ("pinhole kannala_brandt", "kb4"),
])
def test_norm_model(name, expected):
    assert ci._norm_model(name) == expected


def test_norm_model_unknown_raises():
    with pytest.raises(ValueError):
        ci._norm_model("totally_made_up")


@pytest.mark.parametrize("model", KALIBR_MODELS)
def test_kalibr_round_trip(model):
    intr = np.array(MODELS[model], float)
    cam_m, dist_m, k_intr, k_coeffs = co._kalibr_pack(model, intr)
    doc = {"cam0": dict(camera_model=cam_m, distortion_model=dist_m,
                        intrinsics=k_intr, distortion_coeffs=k_coeffs,
                        resolution=[640, 480], rostopic="/cam0/image_raw")}
    s = ci._from_kalibr(doc)[0]
    assert s["distortion_model"] == model
    cal_vec = list(s["intrinsics"]) + list(s["distortion_coeffs"])
    assert np.allclose(cal_vec, intr)
    # D19 (fixed in I0): every non-radtan model routes through the fisheye
    # path — this test asserted the pre-D19 semantics and sat outside every
    # selection until the I5 camchain work widened the filter
    assert s["is_fisheye"] == (model != "radtan")


@pytest.mark.parametrize("model,n", [
    ("radtan", 8), ("equidistant", 8), ("fov", 5),
    ("double_sphere", 6), ("eucm", 6), ("ucm", 5), ("omni_radtan", 9), ("kb4", 8)])
def test_calib_yaml_coeffs_trimmed_to_model(model, n):
    """Regression: the MVIS `_calib.yaml` zero-pads ds/eucm/fov distortion
    coeffs to 4. The input adapter must trim to the model's count so cal_vec length
    == model n_intr (else the cam-IMU CameraCalib reports n_intr=8 while the model's
    J_intr has 6/5 cols -> broadcast error in CameraCalibInterpFactor)."""
    intr = MODELS[model]
    padded = list(intr[4:]) + [0.0] * (4 - len(intr[4:]))   # pad coeffs to 4
    doc = {"cameras": [dict(
        id=0, name="cam0", model=f"pinhole_{model}",
        intrinsics=dict(fx=intr[0], fy=intr[1], cx=intr[2], cy=intr[3]),
        distortion=dict(model=model, coeffs=padded),
        image_size=[1024, 1024], rolling_shutter_readout=0.0)]}
    s = ci._from_mvis(doc, "/{name}/image_raw")[0]
    cal_vec = list(s["intrinsics"]) + list(s["distortion_coeffs"])
    assert len(cal_vec) == n, f"{model}: cal_vec len {len(cal_vec)} != {n}"
    assert np.allclose(cal_vec, intr)


@pytest.mark.parametrize("model", list(MODELS))
def test_mvis_round_trip(model):
    """A MVIS-format cam block (as R5 writes it) reads back to the same
    cal_vec + model via `_from_mvis`."""
    intr = MODELS[model]
    doc = {"cameras": [dict(
        id=0, name="cam0",
        model=f"pinhole_{model}",
        intrinsics=dict(fx=intr[0], fy=intr[1], cx=intr[2], cy=intr[3]),
        distortion=dict(model=model, coeffs=intr[4:]),
        image_size=[640, 480], rolling_shutter_readout=0.0)]}
    s = ci._from_mvis(doc, "/{name}/image_raw")[0]
    assert s["distortion_model"] == model
    cal_vec = list(s["intrinsics"]) + list(s["distortion_coeffs"])
    assert np.allclose(cal_vec, intr)


# Kalibr's native intrinsics layout (kalibr_common/ConfigReader.py): the
# projection params come FIRST for ds/eucm, before fu,fv,pu,pv. The round-trip
# test above cannot catch a wrong order (pack and unpack would agree), so pin
# the external layout against a Kalibr eucm camchain (a synthetic wide-FOV lens).
@pytest.mark.parametrize("model,kalibr_intr,cal_vec", [
    ("eucm", [0.66, 1.05, 455.0, 455.5, 630.0, 470.0],
             [455.0, 455.5, 630.0, 470.0, 0.66, 1.05]),
    ("double_sphere", [-0.2, 0.6, 400., 401., 320., 240.],
                      [400., 401., 320., 240., -0.2, 0.6]),
    # omni-none: [xi, gamma_u, gamma_v, cx, cy], xi = a/(1-a), gamma = f/(1-a)
    ("ucm", [1.5, 1000., 1002.5, 320., 240.],
            [400., 401., 320., 240., 0.6]),
])
def test_kalibr_native_layout(model, kalibr_intr, cal_vec):
    cam_m = {"eucm": "eucm", "double_sphere": "ds", "ucm": "omni"}[model]
    doc = {"cam0": dict(camera_model=cam_m, distortion_model="none",
                        intrinsics=kalibr_intr, distortion_coeffs=[],
                        resolution=[1280, 1024], rostopic="/cam0/image_raw")}
    s = ci._from_kalibr(doc)[0]
    assert np.allclose(list(s["intrinsics"]) + list(s["distortion_coeffs"]), cal_vec)
    _, _, k_intr, k_coeffs = co._kalibr_pack(model, np.array(cal_vec))
    assert np.allclose(k_intr, kalibr_intr) and k_coeffs == []


# ── ucm <-> Kalibr omni-none ──

def _kalibr_omni(X, xi, gu, gv, cx, cy):
    """Kalibr OmniProjection without distortion (the raw formula)."""
    d = np.linalg.norm(X)
    return np.array([gu * X[0] / (X[2] + xi * d) + cx, gv * X[1] / (X[2] + xi * d) + cy])


def test_ucm_kalibr_pack_round_trip_and_formula():
    from mvis.calibration.factors.custom_projections import project_ucm
    for a in (0.0, 0.3, 0.64, 0.9):
        cal = np.array([381.2, 380.7, 509.3, 514.8, a])
        cam_m, dist_m, k_intr, k_coeffs = co._kalibr_pack("ucm", cal)
        assert (cam_m, dist_m, k_coeffs) == ("omni", "none", [])
        doc = {"cam0": dict(camera_model=cam_m, distortion_model=dist_m, intrinsics=k_intr,
                            distortion_coeffs=k_coeffs, resolution=[1024, 1024],
                            rostopic="/cam0/image_raw")}
        s = ci._from_kalibr(doc)[0]
        assert s["distortion_model"] == "ucm"
        np.testing.assert_allclose(list(s["intrinsics"]) + list(s["distortion_coeffs"]),
                                   cal, atol=1e-12, rtol=0)
        for th in (10.0, 60.0, 85.0) + ((92.0,) if a > 0 else ()):   # inside every fold here
            X = np.array([np.sin(np.radians(th)) * 0.8, np.sin(np.radians(th)) * 0.6,
                          np.cos(np.radians(th))]) * 1.3
            np.testing.assert_allclose(_kalibr_omni(X, *k_intr), project_ucm(X, cal)[0],
                                       atol=1e-9, rtol=0)


@pytest.mark.parametrize("alpha", [1.0, 1.02, -0.1, float("nan")])
def test_ucm_kalibr_pack_refuses_alpha_out_of_range(alpha):
    with pytest.raises(ValueError, match="omni-none"):
        co._kalibr_pack("ucm", [400., 401., 320., 240., alpha])
    assert co.kalibr_export_issue("ucm", [400., 401., 320., 240., alpha])
    assert co.kalibr_export_issue("ucm", [400., 401., 320., 240., 0.6]) is None


def test_kalibr_omni_pairs():
    """Kalibr omni + none is ucm (D6); omni + radtan stays omni_radtan; an
    omni + none with non-zero coefficients is refused, not silently dropped."""
    base = dict(camera_model="omni", intrinsics=[1.5, 1000., 1002.5, 320., 240.],
                resolution=[640, 480], rostopic="/cam0/image_raw")
    for dm in ("none", ""):
        s = ci._from_kalibr({"cam0": {**base, "distortion_model": dm, "distortion_coeffs": []}})[0]
        assert s["distortion_model"] == "ucm" and s["is_fisheye"]
    s = ci._from_kalibr({"cam0": {**base, "distortion_model": "radtan",
                                  "distortion_coeffs": [0.01, 0.0, 0.0, 0.0]}})[0]
    assert s["distortion_model"] == "omni_radtan"
    with pytest.raises(ValueError, match="non-zero"):
        ci._from_kalibr({"cam0": {**base, "distortion_model": "none",
                                  "distortion_coeffs": [0.01]}})


def test_cli_skips_camchain_for_ucm_alpha_out_of_range(caplog, tmp_path):
    """D8: ucm with alpha >= 1 writes no camchain (warning), the MVIS yaml
    still is; an in-range ucm keeps its camchain path."""
    import logging
    from types import SimpleNamespace
    import gtsam
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.cli.main import _kalibr_camchain_path
    out = SimpleNamespace(kalibr_yaml=None, path=str(tmp_path / "run.yaml"))
    good, bad = [400., 401., 320., 240., 0.6], [400., 401., 320., 240., 1.02]
    assert _kalibr_camchain_path(out, ["ucm", "radtan"],
                                 [good, MODELS["radtan"]]) == str(tmp_path / "run_camchain-imucam.yaml")
    with caplog.at_level(logging.WARNING, logger="mvis"):
        assert _kalibr_camchain_path(out, ["radtan", "ucm"], [MODELS["radtan"], bad]) is None
    assert "cam1" in caplog.text and "alpha=1.020" in caplog.text
    assert _kalibr_camchain_path(out, ["ucm"]) is not None   # no values: by model only
    # the writer itself refuses before opening the camchain file
    cal = CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.array(bad), model="ucm")
    chain = tmp_path / "chain.yaml"
    with pytest.raises(ValueError, match="alpha"):
        co.to_kalibr_camchain_imucam(SimpleNamespace(cameras=[{"calib": cal}]))
    assert not chain.exists()
