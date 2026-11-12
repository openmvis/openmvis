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
"""One `reference:` parser (mvis.analysis.reference) for the verdict, the report's
vs-reference section and the rig ghost: `camN`, `N` and `'N'` keys, scalar or vector
baselines. Before it, the report dropped `cam0` keys (the tumvi_raw release report had
an empty lever table) and the verdict dropped `0` keys and crashed on vector baselines."""
import json

import numpy as np
import pytest
import yaml

from mvis.analysis.reference import (baseline_norm_mm, by_camera, cam_index,
                                     camera_value, reference_deltas)

L0, L1 = [45.38, -72.68, -45.07], [-55.52, -71.10, -47.83]


@pytest.mark.parametrize("key,idx", [
    ("cam0", 0), ("cam1", 1), ("cam12", 12), (0, 0), (3, 3), ("0", 0), ("2", 2),
    (" 1 ", 1), ("left", None), ("cam", None), ("camx", None), ("cam-1", None),
    (-1, None), (True, None), (None, None), ("1a", None)])
def test_cam_index_key_forms(key, idx):
    assert cam_index(key) == idx


def test_by_camera_keeps_order_drops_unknown_and_first_duplicate_wins():
    with pytest.warns(UserWarning, match="two keys name camera 0"):
        assert list(by_camera({"cam1": "a", "left": "x", 0: "b", "0": "c"}).items()) == \
            [(1, "a"), (0, "b")]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")    # equal duplicates are silent
        assert by_camera({"cam0": [1, 2], 0: [1, 2]}) == {0: [1, 2]}
    assert by_camera(None) == {} and by_camera({}) == {}
    assert camera_value({"cam2": 5}, 2) == 5 and camera_value({"2": 5}, 2) == 5
    assert camera_value({2: 5}, 2) == 5 and camera_value({2: 5}, 1) is None
    assert camera_value(None, 0) is None


@pytest.mark.parametrize("value,norm", [
    (None, None), (134, 134.0), (100.95, 100.95), ("134.0", 134.0),
    ([3.0, 4.0, 0.0], 5.0), ((3, 4, 12), 13.0), (np.array([3.0, 4.0, 0.0]), 5.0),
    (np.float64(7.5), 7.5), (np.array(7.5), 7.5), ([100.95], 100.95)])
def test_baseline_shapes(value, norm):
    got = baseline_norm_mm(value)
    assert got == norm if norm is None else got == pytest.approx(norm, abs=0, rel=1e-15)
    assert got is None or type(got) is float


def test_reference_deltas_normalises_every_part():
    assert reference_deltas(None) is None and reference_deltas({}) is None
    out = reference_deltas({"lever_mm": {"cam0": L0, 1: L1}, "rotation_deg": {"1": 0.5},
                            "baseline_mm": [100.0, 0.0, 0.0]})
    assert out == {"label": "reference", "lever_mm": {0: L0, 1: L1},
                   "baseline_mm": 100.0}
    assert reference_deltas({"label": "TUM"})["label"] == "TUM"


# ---- the report's vs-reference section -------------------------------------
def _rows():
    return [dict(id=0, lever_mm=[46.45, -69.16, -46.06], rot_rad=[0.0, 0.0, float(np.radians(1.0))],
                 rot_angle_rad=float(np.radians(1.0))),
            dict(id=1, lever_mm=[-54.21, -66.57, -49.77], rot_rad=[0.0, 0.0, float(np.radians(2.0))],
                 rot_angle_rad=float(np.radians(2.0)), baseline_norm_mm=100.76)]


@pytest.mark.parametrize("keys", [("cam0", "cam1"), (0, 1), ("0", "1")])
@pytest.mark.parametrize("base", [100.95, [100.95, 0.0, 0.0], "100.95"])
def test_report_deltas_accept_every_key_form_and_baseline_shape(keys, base):
    from mvis.analysis.cam_imu_report import _ref_deltas
    ref = {"label": "TUM", "lever_mm": {keys[0]: L0, keys[1]: L1},
           "rotation_deg": {keys[1]: 2.5}, "baseline_mm": base}
    out = _ref_deltas(_rows(), ref)
    assert out["label"] == "TUM"
    assert [c["id"] for c in out["cams"]] == [0, 1]
    assert out["cams"][0]["lever_delta_mm"] == pytest.approx(
        float(np.linalg.norm(np.subtract(_rows()[0]["lever_mm"], L0))), rel=1e-15)
    assert "rotation_delta_rad" not in out["cams"][0]
    assert out["cams"][1]["rotation_delta_rad"] == pytest.approx(np.radians(0.5))
    assert out["baseline"]["ref_mm"] == pytest.approx(100.95)
    assert out["baseline"]["delta_mm"] == pytest.approx(0.19)


def test_report_text_and_html_show_the_lever_rows_for_cam_keys():
    """The tumvi_raw release config form (cam0/cam1 keys) yields lever rows."""
    from mvis.analysis.cam_imu_report import cam_imu_html_report, cam_imu_text_report
    from tests.helpers.cam_imu_report_fakes import _fake_result
    r = _fake_result()
    p0 = (np.asarray(r.cameras[0]["calib"].T.translation()) * 1e3 + [3.0, 4.0, 0.0]).tolist()
    for key in ("cam0", 0, "0"):
        ref = {"label": "factory", "lever_mm": {key: p0}}
        assert "cam0: lever Δ 5.00 mm" in cam_imu_text_report(r, ref)
        html = cam_imu_html_report(r, reference=ref)
        sec = html[html.index('id="sec-reference"'):]
        assert "<td>cam0</td><td>5.00</td>" in sec[:sec.index("</table>")].replace("\n", "")


def test_report_without_matching_keys_has_no_camera_rows():
    from mvis.analysis.cam_imu_report import _ref_deltas
    out = _ref_deltas(_rows(), {"lever_mm": {"left": L0}})
    assert out["cams"] == [] and out["baseline"] is None


# ---- the verdict ------------------------------------------------------------
def _summary(tmp_path):
    d = {"ok": True, "cams": [
        {"name": "cam0", "p_CinI_mm": [46.45, -69.16, -46.06]},
        {"name": "cam1", "p_CinI_mm": [-54.21, -66.57, -49.77],
         "p_CtoC0_mm": [100.74, 1.89, 0.39]}]}
    p = tmp_path / "cam_imu.json"
    p.write_text(json.dumps(d))
    return str(p)


def _ref_checks(v):
    return [(c.id, c.subject, c.value, c.status) for c in v.checks if c.id.startswith("ref.")]


def test_verdict_reference_checks_match_for_every_key_form_and_baseline_shape(tmp_path):
    from mvis.analysis.verdict import assess_cam_imu
    p = _summary(tmp_path)
    want = _ref_checks(assess_cam_imu(p, {"label": "TUM", "lever_mm": {"cam0": L0, "cam1": L1},
                                          "baseline_mm": 100.95}))
    assert [w[:2] for w in want] == [("ref.lever", "cam0"), ("ref.lever", "cam1"),
                                     ("ref.baseline", "cam1")]
    assert want[0][2] == round(float(np.linalg.norm(np.subtract([46.45, -69.16, -46.06], L0))), 2)
    for keys in [(0, 1), ("0", "1")]:
        for base in (100.95, "100.95", [100.95, 0.0, 0.0], (0.0, 100.95, 0.0)):
            ref = {"label": "TUM", "lever_mm": {keys[0]: L0, keys[1]: L1}, "baseline_mm": base}
            assert _ref_checks(assess_cam_imu(p, ref)) == want


def test_verdict_reference_from_a_yaml_config(tmp_path):
    """YAML gives int keys for `0:` and str keys for `'0':` and `cam0:`."""
    from mvis.analysis.verdict import assess_cam_imu
    p = _summary(tmp_path)
    for text in ("lever_mm: {0: %s}" % L0, "lever_mm: {'0': %s}" % L0, "lever_mm: {cam0: %s}" % L0):
        ref = yaml.safe_load(text)
        assert [c[:2] for c in _ref_checks(assess_cam_imu(p, ref))] == [("ref.lever", "cam0")]


def test_rig_ghost_reads_the_same_key_forms():
    from mvis.analysis.views.spec import reference_rig
    for key in ("cam0", 0, "0"):
        out = reference_rig({"lever_mm": {key: L0}}, [np.eye(3)], 1)
        assert np.allclose(out["cams"][0]["p"], L0)
