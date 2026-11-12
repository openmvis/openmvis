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
"""P1b (code-cleanup): shared cam-IMU result metrics used by roundtrip.py, repeatability.py,
and cam_imu_adapter.py — one definition instead of three hand-rolled copies."""
import numpy as np
import gtsam
from mvis.recalib.result_metrics import imu_rows, lever_mm, lever_rms_mm, mean_rot, rot_rms_deg, stereo_baseline


def test_stereo_baseline_cam1_minus_cam0():
    cams = [{"p_CinI_mm": [100, 0, 0]}, {"p_CinI_mm": [164, 0, 0]}]
    assert abs(stereo_baseline(cams) - 64.0) < 1e-9


def test_stereo_baseline_none_for_mono():
    assert stereo_baseline([{"p_CinI_mm": [1, 2, 3]}]) is None


def test_lever_mm_is_norm_of_a_vec3():
    assert abs(lever_mm([3, 4, 0]) - 5.0) < 1e-9


def test_imu_rows_base_then_aux_with_names_and_biases():
    res = {"Dw_diag": [1, 1, 1], "Da_diag": [1, 1, 1], "b_g": [0, 0, 0], "b_a": [0, 0, 0],
           "imus": [{"name": "t265", "Dw_diag": [1.01, 1, 1], "Da_diag": [1, 1, 1],
                     "b_g": [0.1, 0, 0], "b_a": [0, 0, 0]}]}
    rows = imu_rows(res)
    assert [r["name"] for r in rows] == ["imu0", "t265"]
    assert rows[1]["Da"] == [1, 1, 1]
    assert rows[1]["b_g"] == [0.1, 0, 0]


def test_imu_rows_missing_fields_are_none():
    rows = imu_rows({"Dw_diag": [1, 1, 1], "imus": [{"name": "aux"}]})  # no Da/biases
    assert rows[0]["b_g"] is None and rows[1]["Da"] is None


# ---------- across-run spread ---------------------------------------------------------------------
def test_mean_rot_of_identical_rotations_is_that_rotation():
    rv = [0.1, -0.2, 0.3]
    assert np.allclose(gtsam.Rot3.Logmap(mean_rot([rv, rv, rv])), rv)


def test_mean_rot_of_symmetric_rotations_about_a_centre():
    """+-d about the identity along one axis: the chordal mean is the identity."""
    assert np.allclose(gtsam.Rot3.Logmap(mean_rot([[0, 0, 0.02], [0, 0, -0.02]])), 0, atol=1e-12)


def test_mean_rot_is_a_proper_rotation():
    m = mean_rot([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]]).matrix()
    assert np.isclose(np.linalg.det(m), 1.0) and np.allclose(m.T @ m, np.eye(3))


def test_rot_rms_deg_of_symmetric_rotations():
    d = 0.01
    assert np.isclose(rot_rms_deg([[0, 0, d], [0, 0, -d]]), np.rad2deg(d))
    assert rot_rms_deg([[0.1, 0.2, 0.3]] * 3) < 1e-6


def test_lever_rms_mm_is_about_the_mean():
    assert lever_rms_mm([[1.0, 0, 0], [3.0, 0, 0]]) == 1.0
    assert lever_rms_mm([[5.0, 5, 5]] * 4) == 0.0


def _batch_result(k, two_cams=True):
    rng = np.random.default_rng(k)
    cam = lambda c: {"name": f"cam{c}", "p_CinI_mm": list(rng.normal([10 * c, 2, 3], 0.5)),
                     "rotvec": list(rng.normal(0, 1e-3, 3)), "td_ms": float(rng.normal(1, 0.1))}
    return {"reproj_rms_px": 0.3 + 0.01 * k, "cams": [cam(c) for c in range(2 if two_cams else 1)],
            "Dw_diag": list(1 + rng.normal(0, 1e-3, 3)), "Da_diag": list(1 + rng.normal(0, 1e-3, 3)),
            "imus": [{"name": "aux", "p_IsinI_mm": list(rng.normal([50, 0, 0], 0.3)),
                      "rotvec": list(rng.normal(0, 1e-3, 3)), "td_ms": float(rng.normal(0, 0.05)),
                      "Dw_diag": [1.0, 1.0, 1.0], "Da_diag": [1.0, 1.0, 1.0]}]}


def test_discover_batch_results_keeps_numbered_dirs_only(tmp_path):
    import json
    from mvis.recalib.result_metrics import discover_batch_results
    for name in ("bag02", "bag01", "mic07", "bag09_t265_2_only", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "cam_imu.json").write_text(json.dumps({"id": name}))
    (tmp_path / "bag03").mkdir()                                       # no result
    got = discover_batch_results(str(tmp_path))
    assert got == [("01", {"id": "bag01"}), ("02", {"id": "bag02"}), ("07", {"id": "mic07"})]


def test_repeatability_stats_are_the_spreads_of_the_runs():
    from mvis.recalib.result_metrics import repeatability_stats, rot_rms_deg
    data = [(f"{k:02d}", _batch_result(k)) for k in range(5)]
    st = repeatability_stats(data)
    assert st["runs"] == ["00", "01", "02", "03", "04"]
    assert np.array_equal(st["reproj"], [0.3 + 0.01 * k for k in range(5)])
    p = lambda c: np.array([r["cams"][c]["p_CinI_mm"] for _, r in data])
    assert np.array_equal(st["baseline"], np.linalg.norm(p(1) - p(0), axis=1))
    c1 = st["cams"][1]
    assert c1["name"] == "cam1" and np.array_equal(c1["lever_mean"], p(1).mean(0))
    assert c1["lever_rms"] == lever_rms_mm(p(1))
    assert c1["rot_rms"] == rot_rms_deg([r["cams"][1]["rotvec"] for _, r in data])
    assert c1["td_std"] == np.std([r["cams"][1]["td_ms"] for _, r in data])
    assert [m["name"] for m in st["imus"]] == ["imu0(base)", "aux"]
    assert np.array_equal(st["imus"][0]["W"], [r["Dw_diag"] for _, r in data])
    assert st["aux"][0]["name"] == "aux" and st["aux"][0]["td_std"] > 0


def test_repeatability_stats_mono_rig_has_no_baseline():
    from mvis.recalib.result_metrics import repeatability_stats
    st = repeatability_stats([(str(k), _batch_result(k, two_cams=False)) for k in range(3)])
    assert st["baseline"] is None and len(st["cams"]) == 1
