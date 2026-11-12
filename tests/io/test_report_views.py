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
"""analysis.views: camera geometry through the forward model, specs, markup."""
import json

import numpy as np
import pytest

from mvis.analysis.views.geometry import fov_deg, project, ray, rim
from mvis.analysis.views.html import assets_html, rig_html, scene_html
from mvis.analysis.views.spec import board_outline, without_frames
from mvis.calibration.factors.camera_projection import k_cal_vec

MODELS = {
    "equidistant": ([330.7, 330.6, 270.3, 320.2], [-0.0235, 0.0175, -0.0245, 0.0093], (544, 640)),
    "radtan": ([458.6, 457.3, 367.2, 248.4], [-0.283, 0.074, 0.0002, 0.00002], (752, 480)),
    "eucm": ([380.0, 380.0, 320.0, 240.0], [0.6, 1.1], (640, 480)),
    "double_sphere": ([350.0, 350.0, 320.0, 240.0], [-0.2, 0.6], (640, 480)),
}


@pytest.mark.parametrize("model", sorted(MODELS))
def test_ray_inverts_the_forward_model(model):
    intr, dist, size = MODELS[model]
    cal = k_cal_vec(model, intr, dist)
    for uv in [(10.0, 10.0), (size[0] - 5.0, size[1] / 2), (size[0] * 0.3, size[1] * 0.8)]:
        r = ray(model, cal, uv)
        assert r is not None and abs(np.linalg.norm(r) - 1) < 1e-9
        assert np.hypot(*(project(model, cal, r) - np.array(uv))) < 0.1


def test_rim_goes_round_the_image_and_fov_is_sane():
    intr, dist, size = MODELS["equidistant"]
    cal = k_cal_vec("equidistant", intr, dist)
    assert len(rim("equidistant", cal, size)) == 32
    f = fov_deg("equidistant", cal, size)
    assert 90 < f["h"] < 100 and 108 < f["v"] < 118          # the Insight 9 stereo lens


def test_pinhole_fov_matches_the_closed_form():
    cal = k_cal_vec("radtan", [400.0, 400.0, 320.0, 240.0], [0, 0, 0, 0])
    f = fov_deg("radtan", cal, (640, 480))
    assert f["h"] == pytest.approx(2 * np.degrees(np.arctan(320 / 400)), abs=0.1)


def test_board_outline_grows_the_corners_by_one_square():
    sq = 0.04
    pts = np.array([[i * sq, j * sq, 0] for j in range(6) for i in range(9)])
    o = board_outline(pts)
    assert np.allclose(o[0], [-sq, -sq, 0]) and np.allclose(o[2], [9 * sq, 6 * sq, 0])


def test_without_frames_strips_images_and_refuses_leftovers():
    spec = {"v": 2, "cams": [{"name": "cam0", "uvb": "AAA", "jpg": ["/9j/xx"], "jpg_idx": [0], "jpg_size": [10, 10]}]}
    out = without_frames(spec)
    assert out["cams"][0] == {"name": "cam0", "uvb": "AAA"} and "jpg" in spec["cams"][0]
    with pytest.raises(ValueError):
        without_frames({"v": 2, "cams": [{"name": "cam0", "note": "data:image/png;base64,iVBOR"}]})
    # packed corner pixels may contain the letters by chance; only a string's start counts
    assert without_frames({"v": 2, "cams": [{"uvb": "AAAA/9j/AAAA"}]})["cams"][0]["uvb"] == "AAAA/9j/AAAA"


def test_markup_keeps_specs_parseable_and_scripts_closed():
    spec = {"v": 2, "cams": [{"name": "</script><b>x"}]}
    h = scene_html(spec) + rig_html(spec)
    assert h.count("</script>") == 2                          # only the two JSON elements close
    body = h[h.index('json">') + 6:h.index("</script>")]
    assert json.loads(body)["cams"][0]["name"] == "</script><b>x"
    a = assets_html()
    assert a.count("</script>") == 1 and "<script src" not in a


def test_legacy_report_converts_to_the_same_scene():
    """An older Plotly report reads back as the v2 spec the new builder makes, without frames."""
    from mvis.analysis.cam_imu_report import _build_3d_section_plotly, _cam_views
    from mvis.analysis.views.legacy import scene_from_report
    from mvis.analysis.views.spec import scene_spec
    from mvis.detection.chessboard import Chessboard
    from tests.io.test_cam_imu_report import _result_with_geometry
    r, board = _result_with_geometry(), Chessboard(7, 7, 0.10)
    old = _build_3d_section_plotly(r, board, None)
    assert 'mvisProgress("' in old
    cams = [{"model": e["calib"].model, "intr": list(e["calib"].intr), "size": (640, 480)} for e in r.cameras]
    got = scene_from_report(old, cams)
    want = scene_spec(r, board=board, views=_cam_views(r, board, None)[1], state_idx=None, image_sizes={0: (640, 480)})
    assert got["v"] == want["v"] == 2 and np.allclose(got["xyz"], want["xyz"], atol=1e-4)
    assert np.allclose(got["board"]["pts"], want["board"]["pts"], atol=1e-4)
    assert got["cams"][0]["p"] == want["cams"][0]["p"] and "jpg" not in got["cams"][0]


def test_fisheye_past_90_degrees_and_a_folded_lens():
    """T265-like fisheyes see past 90 deg, where MVIS's kb projection stops: the rim comes from the lens polynomial.
    A polynomial that turns over before the image edge (VI-Rig bag07 cam0) has no field of view; its rim stops at the fold."""
    ok = k_cal_vec("equidistant", [286.38, 285.30, 413.92, 399.14], [-0.00249, 0.03444, -0.03411, 0.00556])
    f = fov_deg("equidistant", ok, (848, 800))   # side edges past 90 deg (past what MVIS projects): no value; top-bottom ~88.6 each
    assert f["h"] is None and f["v"] == pytest.approx(177.3, abs=0.2) and len(rim("equidistant", ok, (848, 800), per_side=12)) == 48
    wide = k_cal_vec("equidistant", [286.38, 285.30, 413.92, 399.14], [0.0, 0.0, 0.0, 0.0])   # theta_d = theta
    assert fov_deg("equidistant", wide, (848, 800))["h"] == pytest.approx(np.degrees(413.92 / 286.38 + 434.08 / 286.38), abs=0.2)
    folded = k_cal_vec("equidistant", [286.52, 285.64, 414.86, 405.04], [-0.00278, 0.03279, -0.03053, 0.00416])
    assert fov_deg("equidistant", folded, (848, 800)) == {"h": None, "v": None}
    r = np.array(rim("equidistant", folded, (848, 800), per_side=12))
    assert len(r) == 48 and np.degrees(np.arccos(r[:, 2])).max() < 110     # stops at the fold, about 100 deg


def test_reference_rig_from_levers_rotvecs_and_alignment():
    from mvis.analysis.views.spec import reference_rig
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])      # MVIS cam0 = 90 deg about z
    ref = {"label": "factory", "lever_mm": {0: [1.0, 2.0, 3.0], "1": [4.0, 0.0, 0.0]}, "rotvec": {"cam0": [0, 0, 0]}}
    out = reference_rig(ref, [Rz, Rz], 2)
    assert out["label"] == "factory" and out["cams"][0]["p"] == [1.0, 2.0, 3.0] and out["cams"][1]["R"] is None
    aligned = reference_rig(dict(ref, align="cam0_rotation"), [Rz, Rz], 2)   # rotate the reference so its cam0 matches
    assert np.allclose(aligned["cams"][0]["R"], Rz.reshape(9)) and np.allclose(aligned["cams"][1]["p"], [0.0, 4.0, 0.0])
    assert reference_rig({"label": "x"}, [Rz], 1) is None and reference_rig(None, [Rz], 1) is None


def test_reference_rig_from_a_kalibr_camchain(tmp_path):
    import yaml
    from mvis.analysis.views.spec import reference_rig
    T = np.eye(4); T[:3, 3] = [0.01, -0.02, 0.03]                    # T_cam_imu: imu -> cam
    cam = {"camera_model": "pinhole", "intrinsics": [300, 300, 320, 240], "distortion_model": "radtan",
           "distortion_coeffs": [0, 0, 0, 0], "resolution": [640, 480], "rostopic": "/cam0/image_raw", "T_cam_imu": T.tolist()}
    f = tmp_path / "camchain-imucam.yaml"
    f.write_text(yaml.safe_dump({"cam0": cam, "cam1": cam}))
    out = reference_rig({"label": "Kalibr", "camchain": str(f), "order": [1, 0]}, [np.eye(3)] * 2, 2)
    assert np.allclose(out["cams"][0]["p"], [-10.0, 20.0, -30.0]) and np.allclose(out["cams"][0]["R"], np.eye(3).reshape(9))


def test_report_carries_the_reference_ghost_and_its_deltas():
    from mvis.analysis.cam_imu_report import cam_imu_html_report, cam_imu_text_report
    from tests.io.test_cam_imu_report import _result_with_geometry, view_specs
    r = _result_with_geometry()
    p0 = np.asarray(r.cameras[0]["calib"].T.translation()) * 1e3
    ref = {"label": "factory", "lever_mm": {0: (p0 + [3.0, 4.0, 0.0]).tolist()}, "rotvec": {0: [0.0, 0.0, 0.0]}}
    rig = view_specs(cam_imu_html_report(r, reference=ref))["rig"]
    assert rig["ref"]["label"] == "factory" and np.allclose(rig["ref"]["cams"][0]["p"], p0 + [3.0, 4.0, 0.0])
    txt = cam_imu_text_report(r, ref)
    assert "factory" in txt and "5.0" in txt      # |lever delta| = 5 mm
