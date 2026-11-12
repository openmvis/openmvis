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
"""P1 (roundtrip-cli): roundtrip_report compares a forward calib to its recalibration on the
apply-calib-corrected bag — IMU intrinsics/biases must collapse to identity/0, geometry must
be preserved, td is a warn (observability-dependent)."""
from mvis.recalib.roundtrip import roundtrip_report


def _json(Dw, Da, levers, tds, b_g=None, b_a=None):
    cams = [{"name": f"cam{i}", "td_ms": tds[i], "reproj": 0.5,
             "p_CinI_mm": levers[i], "rotvec": [0, 0, 0]} for i in range(len(levers))]
    d = {"reproj_rms_px": 0.5, "imu_model": "calyx", "Dw_diag": Dw, "Da_diag": Da,
         "imus": [], "cams": cams}
    if b_g is not None:
        d["b_g"] = b_g
        d["b_a"] = b_a
    return d


def test_clean_roundtrip_passes():
    f = _json([1.0, 1.005, 1.008], [1.058, 1.06, 1.075], [[100, 0, 0], [164, 0, 0]], [-1.7, -1.7])
    r = _json([1.0, 0.999, 1.0], [1.001, 1.001, 1.0], [[100.5, 0, 0], [164.4, 0, 0]], [0.5, 0.5])
    assert roundtrip_report(f, r)["ok"] is True


def test_noncollapsing_da_fails():
    f = _json([1.0, 1.0, 1.0], [1.06, 1.06, 1.06], [[100, 0, 0], [164, 0, 0]], [0, 0])
    r = _json([1.0, 1.0, 1.0], [1.06, 1.06, 1.06], [[100, 0, 0], [164, 0, 0]], [0, 0])  # Da unchanged
    rep = roundtrip_report(f, r)
    assert rep["ok"] is False
    assert any("Da" in c["name"] and not c["pass"] for c in rep["checks"])


def test_baseline_drift_fails():
    f = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [0, 0])   # baseline 64
    r = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [170, 0, 0]], [0, 0])   # baseline 70 (drift 6mm)
    assert roundtrip_report(f, r)["ok"] is False


def test_td_residual_warns_not_fails():
    f = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [-1.7, -1.7])
    r = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [1.0, 1.0])   # 1 ms td residual
    rep = roundtrip_report(f, r)
    assert rep["ok"] is True
    assert any(c["kind"] == "td" and c["warn"] for c in rep["checks"])


def test_bias_collapse_checked_when_present():
    f = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [0, 0], b_g=[0.01, 0, 0], b_a=[0, 0, -0.25])
    r = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [0, 0], b_g=[0, 0, 0], b_a=[0, 0, -0.005])
    assert roundtrip_report(f, r)["ok"] is True
    bad = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [0, 0], b_g=[0, 0, 0], b_a=[0, 0, -0.25])
    assert roundtrip_report(f, bad)["ok"] is False        # accel bias not removed


def test_run_roundtrip_rejects_non_ros1():
    """P4: run_roundtrip raises a clear ValueError on a non-ros1 (euroc) config, before any
    calibration — apply-calib is ros1-bag-only."""
    import pytest
    from mvis.recalib.roundtrip import run_roundtrip
    with pytest.raises(ValueError, match="ros1|ROS1"):
        run_roundtrip("examples/regression/tum_vi_mic/cam_imu_mic05.yaml")


def test_with_biases_injects_base_AND_aux_biases(tmp_path):
    """P1a: _with_biases must source base+aux biases via load_rig_calib (today it injects only
    base, so an un-removed AUX accel bias is invisible to the report)."""
    import json, yaml
    from mvis.recalib.roundtrip import _with_biases
    dj = tmp_path / "cam_imu.json"
    json.dump({"Dw_diag": [1, 1, 1], "Da_diag": [1, 1, 1], "cams": [],
               "imus": [{"name": "t265", "Dw_diag": [1, 1, 1], "Da_diag": [1, 1, 1]}]},
              open(dj, "w"))
    cy = tmp_path / "cam_imu_calib.yaml"
    yaml.safe_dump({"imus": [
        {"name": "imu0", "role": "base", "model": "calyx", "imu_intr_xi": [0.0] * 24,
         "b_g": [0.01, 0, 0], "b_a": [0, 0, 0]},
        {"name": "t265", "role": "aux", "b_g": [0, 0, 0], "b_a": [0, 0, 0.3]}]}, open(cy, "w"))
    out = _with_biases(str(dj))
    assert list(out["b_a"]) == [0, 0, 0]              # base injected
    assert list(out["imus"][0]["b_a"]) == [0, 0, 0.3]  # AUX injected (currently MISSING)


def test_roundtrip_report_aligns_imus_by_name_not_order(tmp_path):
    """P1a hardening: forward/recal IMUs in different order must still pair correctly (was a
    positional zip)."""
    from mvis.recalib.roundtrip import roundtrip_report
    base = {"Dw_diag": [1, 1, 1], "Da_diag": [1, 1, 1], "cams": []}
    fwd = {**base, "imus": [{"name": "a", "Da_diag": [1.05, 1, 1]}, {"name": "b", "Da_diag": [1, 1, 1]}]}
    rec = {**base, "imus": [{"name": "b", "Da_diag": [1, 1, 1]}, {"name": "a", "Da_diag": [1.05, 1, 1]}]}
    rep = roundtrip_report(fwd, rec)
    # 'a' did not collapse (still 1.05) -> must FAIL under name alignment (positional zip would
    # mis-pair a<->b and call it ok)
    assert rep["ok"] is False
    assert any(c["name"].startswith("a ") and "Da" in c["name"] and not c["pass"] for c in rep["checks"])


def test_aux_t_ga_residual_is_checked_as_a_warning():
    f = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [0, 0])
    r = _json([1.0] * 3, [1.0] * 3, [[100, 0, 0], [164, 0, 0]], [0, 0])
    f["imus"] = [{"name": "t265", "t_ga_ms": -6.74}]
    r["imus"] = [{"name": "t265", "t_ga_ms": -6.74}]          # not corrected
    rep = roundtrip_report(f, r)
    row = [c for c in rep["checks"] if c["name"].startswith("t265 t_ga")]
    assert len(row) == 1 and row[0]["warn"] and not row[0]["pass"]
    assert rep["ok"] is True                                   # a warning, like the base t_ga
    r["imus"] = [{"name": "t265", "t_ga_ms": 0.1}]
    assert [c for c in roundtrip_report(f, r)["checks"] if c["name"].startswith("t265 t_ga")][0]["pass"]
