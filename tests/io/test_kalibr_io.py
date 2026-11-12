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
"""P5 (code-cleanup): one io/kalibr.py for Kalibr interop. The writer existed but there was NO
reader for a full camchain-imucam (T_cam_imu); read_camchain_imucam closes the loop."""
import json
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest
import yaml

from mvis.io.kalibr import (
    _cam_entry, camera_imu, camera_only, centre_in_imu_mm, norm_topic, parse_kalibr_camchain, parse_results,
    read_camchain, read_camchain_imucam, write_camchain_imucam,
)


def test_read_camchain_imucam_extracts_intrinsics_and_extrinsics():
    doc = {"cam0": {
        "camera_model": "pinhole", "distortion_model": "radtan",
        "intrinsics": [400, 401, 320, 240], "distortion_coeffs": [0.1, -0.2, 0.001, 0.002],
        "resolution": [640, 480], "rostopic": "/cam0/image_raw",
        "T_cam_imu": [[1, 0, 0, 0.1], [0, 1, 0, 0.02], [0, 0, 1, 0.03], [0, 0, 0, 1]],
    }}
    cams = read_camchain_imucam(doc)
    assert cams[0]["intrinsics"] == [400, 401, 320, 240]
    assert cams[0]["distortion_model"] == "radtan"
    assert cams[0]["T_cam_imu"][0][3] == 0.1          # the extrinsic the old reader dropped


def test_camchain_imucam_round_trips_T_cam_imu():
    T_CtoI = gtsam.Pose3(gtsam.Rot3.Rx(0.1), gtsam.Point3(0.1, 0.02, 0.03))
    cal = SimpleNamespace(intr=np.array([400., 401, 320, 240, 0.1, -0.2, 0.001, 0.002]),
                          model="radtan", tr=0.0, T=T_CtoI)
    result = SimpleNamespace(cameras=[{"calib": cal, "image_size": [640, 480]}])
    doc = write_camchain_imucam(result)
    cams = read_camchain_imucam(doc)
    assert np.allclose(cams[0]["T_cam_imu"], np.linalg.inv(T_CtoI.matrix()))
    assert np.allclose(cams[0]["intrinsics"], [400, 401, 320, 240])


def test_read_camchain_is_the_camera_only_reader():
    cams = read_camchain({"cam0": {"camera_model": "pinhole", "distortion_model": "equidistant",
                                   "intrinsics": [1, 2, 3, 4], "distortion_coeffs": [0, 0, 0, 0],
                                   "resolution": [8, 6], "rostopic": "/c0"}})
    assert cams[0]["intrinsics"] == [1, 2, 3, 4] and "T_cam_imu" not in cams[0]


def test_kalibr_primitives_reachable_via_io_kalibr():
    """module-cleanup P4: the low-level Kalibr pack/model primitives are reachable from the one
    surface (io.kalibr), not just from camera_input/cam_imu_output internals."""
    from mvis.io.kalibr import kalibr_model_to_internal, pack_intrinsics_kalibr
    cam_m, dist_m, intr, coeffs = pack_intrinsics_kalibr("radtan", [400, 401, 320, 240, 0.1, -0.2, 0.0, 0.0])
    assert cam_m == "pinhole" and dist_m == "radtan"
    assert intr == [400.0, 401.0, 320.0, 240.0]
    assert kalibr_model_to_internal("pinhole equidistant") == "equidistant"


# ---------- reading finished Kalibr runs, moved from scripts/regression_report/kalibr_io.py -------
def _write_yaml(path, doc):
    path.write_text(yaml.safe_dump(doc))


_RT_IDENT = [[1.0, 0, 0, 0.1], [0, 1.0, 0, 0.0], [0, 0, 1.0, 0.0], [0, 0, 0, 1.0]]
_RESULTS_TXT = """\
Reprojection error (cam0) [px]:     mean 0.25, median 0.2
Gyroscope error (imu0) [rad/s]:     mean 0.01, median 0.005
Transformation (cam0):
-----------------------
T_ci:  (imu0 to cam0):
[[ 1.  0.  0.  0.1]
 [ 0.  1.  0.  0. ]
 [ 0.  0.  1.  0. ]
 [ 0.  0.  0.  1. ]]

timeshift cam0 to imu0: [s] (t_imu = t_cam + shift)
0.002

IMU0:
----------------------------
Gyroscope:
M:
[[ 2.  0.  0.]
 [ 0.  2.  0.]
 [ 0.  0.  2.]]
Accelerometer:
M:
[[ 4.  0.  0.]
 [ 0.  4.  0.]
 [ 0.  0.  4.]]
"""


def test_parse_results_reads_reproj_transform_and_shift(tmp_path):
    p = tmp_path / "r-results-imucam.txt"
    p.write_text(_RESULTS_TXT)
    res = parse_results(str(p))
    cam = res["cams"]["cam0"]
    assert (cam["reproj_mean"], cam["reproj_median"]) == (0.25, 0.2)
    assert cam["shift_ms"] == 2.0
    assert cam["T_ci"][0, 3] == 0.1
    assert res["resid"][("imu0", "Gyroscope")] == (0.01, 0.005)
    assert res["imus"][0]["idx"] == 0
    assert res["imus"][0]["M_g"][0, 0] == 2.0 and res["imus"][0]["M_a"][1, 1] == 4.0


def test_centre_in_imu_mm_is_minus_rt_t_in_mm():
    R = gtsam.Rot3.Rz(np.pi / 2).matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [0.1, 0.0, 0.0]
    assert np.allclose(centre_in_imu_mm(T), -R.T @ [100.0, 0.0, 0.0])


def test_norm_topic_matches_euroc_and_bag_names():
    assert norm_topic("/cam0/image_raw") == "/cam0"
    assert norm_topic("cam0") == "/cam0"
    assert norm_topic("/imu0") == "/imu0"
    assert norm_topic(None) == "" and norm_topic("") == ""


_DS = {"camera_model": "ds", "distortion_model": "none", "intrinsics": [0.5, 0.6, 400., 401., 320., 240.],
       "distortion_coeffs": [], "resolution": [640, 480], "rostopic": "/cam0/image_raw"}
_OMNI = {"camera_model": "omni", "distortion_model": "radtan", "intrinsics": [0.9, 400., 401., 320., 240.],
         "distortion_coeffs": [0.1, 0.2, 0.3, 0.4], "resolution": [640, 480]}


def test_cam_entry_keeps_raw_intrinsics_and_parse_kalibr_camchain_reorders():
    """The two readers are different on purpose: keep both behaviours explicit."""
    raw = _cam_entry(_DS)
    assert raw["intr"] == [0.5, 0.6, 400., 401., 320., 240.] and raw["dist"] == []
    assert raw["topic"] == "/cam0" and raw["rows"] == 480 and "tr_ms" not in raw
    assert _cam_entry({**_DS, "line_delay": 1e-5})["tr_ms"] == 1e-5 * 480 * 1000

    ds, omni = parse_kalibr_camchain({"cam0": _DS, "cam1": _OMNI, "other": {}})
    assert ds["camera_id"] == 0 and ds["intrinsics"] == [400., 401., 320., 240.] and ds["distortion"] == [0.5, 0.6]
    assert omni["camera_id"] == 1 and omni["intrinsics"] == [400., 401., 320., 240.]
    assert omni["distortion"] == [0.9, 0.1, 0.2, 0.3, 0.4]


def test_parse_kalibr_camchain_omni_none_is_mvis_ucm():
    """Kalibr omni-none [xi, gamma_u, gamma_v, cu, cv] -> MVIS ucm [fu, fv, cu, cv] + [alpha],
    with f = gamma / (1 + xi) and alpha = xi / (1 + xi); omni-radtan keeps its own branch."""
    xi, gu, gv = 0.6, 480., 481.6
    cam = {"camera_model": "omni", "distortion_model": "none", "intrinsics": [xi, gu, gv, 320., 240.],
           "distortion_coeffs": [], "resolution": [640, 480]}
    ucm, no_dist_key = parse_kalibr_camchain({"cam0": cam, "cam1": {k: v for k, v in cam.items()
                                                                     if k != "distortion_model"}})
    assert ucm["model"] == "ucm" and ucm["camera_id"] == 0 and ucm["resolution"] == [640, 480]
    assert ucm["intrinsics"] == pytest.approx([gu / (1 + xi), gv / (1 + xi), 320., 240.])
    assert ucm["distortion"] == pytest.approx([xi / (1 + xi)])
    assert no_dist_key["model"] == "ucm" and no_dist_key["distortion"] == pytest.approx([xi / (1 + xi)])
    assert parse_kalibr_camchain({"cam0": _OMNI})[0]["model"] == "omni"


def test_parse_kalibr_camchain_pinhole_is_passed_through():
    cam = {"camera_model": "pinhole", "distortion_model": "radtan", "intrinsics": [1., 2., 3., 4.],
           "distortion_coeffs": [0.1, 0.2, 0., 0.], "resolution": [8, 6], "T_cn_cnm1": _RT_IDENT}
    out = parse_kalibr_camchain({"cam0": cam})[0]
    assert out["intrinsics"] == [1., 2., 3., 4.] and out["distortion"] == [0.1, 0.2, 0., 0.]
    assert out["T_cn_cnm1"] is _RT_IDENT and out["resolution"] == [8, 6]


def test_camera_only_needs_ok_status_and_chains_baselines(tmp_path):
    assert camera_only(str(tmp_path)) is None                  # no status file
    (tmp_path / "kalibr_status.json").write_text('{"ok": false}')
    assert camera_only(str(tmp_path)) is None                  # failed run
    (tmp_path / "kalibr_status.json").write_text('{"ok": true}')
    assert camera_only(str(tmp_path)) is None                  # no camchain
    cam = {"distortion_model": "radtan", "intrinsics": [1., 2., 3., 4.], "distortion_coeffs": [0., 0., 0., 0.],
           "resolution": [8, 6], "rostopic": "/cam0/image_raw"}
    T1 = np.eye(4)
    T1[0, 3] = 0.1
    T2 = np.eye(4)
    T2[1, 3] = 0.2
    _write_yaml(tmp_path / "bag-camchain.yaml", {"cam0": cam, "cam1": {**cam, "T_cn_cnm1": T1.tolist()},
                                                 "cam2": {**cam, "T_cn_cnm1": T2.tolist()}})
    res = camera_only(str(tmp_path))
    assert "baseline_mm" not in res["cams"][0]
    assert np.isclose(res["cams"][1]["baseline_mm"], 100.0)
    assert np.isclose(res["cams"][2]["baseline_mm"], np.hypot(100.0, 200.0))


def test_camera_only_refuses_an_ambiguous_directory(tmp_path):
    (tmp_path / "kalibr_status.json").write_text('{"ok": true}')
    (tmp_path / "a-camchain.yaml").write_text("{}")
    (tmp_path / "b-camchain.yaml").write_text("{}")
    with pytest.raises(AssertionError, match="ambiguous"):
        camera_only(str(tmp_path))


def _imu_run(d, comparison=None):
    cam = {"distortion_model": "radtan", "intrinsics": [1., 2., 3., 4.], "distortion_coeffs": [0., 0., 0., 0.],
           "resolution": [8, 6], "rostopic": "/cam0/image_raw", "T_cam_imu": _RT_IDENT, "timeshift_cam_imu": 0.002}
    _write_yaml(d / "input-camchain-imucam.yaml", {"cam0": cam})
    M2, M4 = (2 * np.eye(3)).tolist(), (4 * np.eye(3)).tolist()
    T_ib = np.eye(4)
    T_ib[2, 3] = 0.05
    _write_yaml(d / "input-imu.yaml", {
        "imu0": {"model": "scale-misalignment", "rostopic": "/imu0",
                 "gyroscopes": {"M": M2, "A": np.eye(3).tolist(), "C_gyro_i": np.eye(3).tolist()},
                 "accelerometers": {"M": M4}},
        "imu1": {"model": "calibrated", "rostopic": "imu1", "gyroscopes": {"M": M2}, "accelerometers": {"M": M4},
                 "T_i_b": T_ib.tolist(), "time_offset": 0.001}})
    (d / "input-results-imucam.txt").write_text(_RESULTS_TXT)
    if comparison:
        (d / "comparison.json").write_text(json.dumps(comparison))


def test_camera_imu_converts_to_mvis_conventions(tmp_path):
    assert camera_imu(str(tmp_path)) is None                   # incomplete run
    _imu_run(tmp_path)
    res = camera_imu(str(tmp_path))
    cam = res["cams"][0]
    assert np.allclose(cam["p_mm"], [-100.0, 0, 0]) and np.isclose(cam["lever_mm"], 100.0)
    assert cam["td_ms"] == 2.0 and cam["reproj_median"] == 0.2
    imu0, imu1 = res["imus"]
    assert np.allclose(imu0["Dw"], 0.5 * np.eye(3)) and np.allclose(imu0["Da"], 0.25 * np.eye(3))
    assert np.allclose(imu0["Tg"], np.eye(3)) and "p_mm" not in imu0
    assert np.allclose(imu1["Tg"], np.zeros((3, 3)))           # A and C_gyro_i default
    assert imu1["topic"] == "/imu1" and np.allclose(imu1["p_mm"], [0, 0, -50.0]) and imu1["td_ms"] == 1.0


def test_camera_imu_adds_back_the_retime_shift(tmp_path):
    _imu_run(tmp_path, {"kalibr": {"timeshift_applied_ms": {"cam0": 5.0}}})
    assert camera_imu(str(tmp_path))["cams"][0]["td_ms"] == 7.0


# ---- camchain readers: number order, one conversion (F15) ----
def _pin(i):
    return {"camera_model": "pinhole", "distortion_model": "radtan", "intrinsics": [100. + i, 100., 3., 4.],
            "distortion_coeffs": [0., 0., 0., 0.], "resolution": [8, 6], "rostopic": f"/cam{i}/image_raw",
            "T_cam_imu": np.eye(4).tolist()}


def test_kalibr_cam_keys_sort_by_number():
    from mvis.io.camera_input import kalibr_cam_keys
    doc = {f"cam{i}": None for i in (10, 2, 0, 1, 11)}
    doc.update({"camx": None, "imu0": None, 3: None})
    assert kalibr_cam_keys(doc) == ["cam0", "cam1", "cam2", "cam10", "cam11", "camx"]
    # up to ten cameras the order is the old string sort
    ten = {f"cam{i}": None for i in range(10)}
    assert kalibr_cam_keys(ten) == sorted(ten)


def test_camchain_readers_keep_camera_order_past_cam9():
    doc = {f"cam{i}": _pin(i) for i in (11, 0, 10, 2, 1, 3, 4, 5, 6, 7, 8, 9)}
    want = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert [c["intrinsics"][0] - 100 for c in read_camchain(doc)] == want
    assert [c["rostopic"] for c in read_camchain_imucam(doc)] == [f"/cam{i}/image_raw" for i in want]
    assert [c["camera_id"] for c in parse_kalibr_camchain(doc)] == want
    assert [c["intrinsics"][0] - 100 for c in parse_kalibr_camchain(doc)] == want


@pytest.mark.parametrize("cam", [
    _DS, {**_DS, "camera_model": "eucm"}, _OMNI,
    {"camera_model": "omni", "distortion_model": "none", "intrinsics": [0.6, 480., 481.6, 320., 240.],
     "distortion_coeffs": [], "resolution": [640, 480]},
    {"camera_model": "pinhole", "distortion_model": "equidistant", "intrinsics": [1., 2., 3., 4.],
     "distortion_coeffs": [0.1, 0.2, 0.3, 0.4], "resolution": [8, 6]},
    {"camera_model": "pinhole", "distortion_model": "fov", "intrinsics": [1., 2., 3., 4.],
     "distortion_coeffs": [0.9], "resolution": [8, 6]}])
def test_parse_kalibr_camchain_is_read_camchain_with_kalibr_names(cam):
    """parse_kalibr_camchain converts through read_camchain (the cam-IMU input reader)."""
    out = parse_kalibr_camchain({"cam0": cam})[0]
    ref = read_camchain({"cam0": {**cam, "rostopic": ""}})[0]
    assert out["intrinsics"] == ref["intrinsics"] and out["distortion"] == ref["distortion_coeffs"]
    want = "ucm" if (cam["camera_model"], cam["distortion_model"]) == ("omni", "none") else cam["camera_model"]
    assert out["model"] == want and out["distortion_model"] == cam["distortion_model"]


def test_parse_kalibr_camchain_edge_cases_follow_read_camchain():
    """Pinned F15 behaviour (was: pass-through). pinhole-none is not an MVIS model and raises, as
    the cam-IMU camera-input reader does; a missing distortion_coeffs key gives the model's zero
    coefficients; string intrinsics are converted to float."""
    pin = {"camera_model": "pinhole", "intrinsics": ["400", "401", "320", "240"], "resolution": [640, 480]}
    with pytest.raises(ValueError, match="pinhole none"):
        parse_kalibr_camchain({"cam0": {**pin, "distortion_model": "none", "distortion_coeffs": []}})
    out = parse_kalibr_camchain({"cam0": {**pin, "distortion_model": "radtan"}})[0]
    assert out["intrinsics"] == [400., 401., 320., 240.] and out["distortion"] == [0., 0., 0., 0.]
