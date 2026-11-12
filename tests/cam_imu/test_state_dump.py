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
"""Full-precision state dump + comparison (cam-imu-speedup S0).

The end-to-end gates of the speed-up steps compare `MVIS_DUMP_STATE` npz
files, so the dump must round-trip every Values key at full precision and
the comparison must (a) accept identical dumps bitwise, (b) reject a 1-ulp
change in exact mode, (c) apply the plan's per-quantity tolerances, sigma-
normalised where a sigma exists, and (d) fail on missing keys / NaN-pattern
changes. `perturb_values` is the noise-floor knob."""
import os
import subprocess
import sys

import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.state_dump import (
    Tolerances, build_state_dump, compare_states, load_state_dump, next_dump_path,
    perturb_values, summarize, symbol_name, values_to_arrays, write_state_dump,
    _PASS_COUNTER)

_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..")


def _values():
    v = gtsam.Values()
    T = gtsam.Pose3(gtsam.Rot3.Ypr(0.3, -0.2, 0.1), gtsam.Point3(1.0, -2.0, 0.5))
    v.insert(gtsam.symbol('x', 0), T)
    v.insert(gtsam.symbol('x', 1), T.retract(np.array([0.01, 0, 0, 0.1, 0, 0])))
    v.insert(gtsam.symbol('b', 0), gtsam.Pose3())
    v.insert(gtsam.symbol('v', 0), np.array([0.1, 0.2, 0.3, 1e-3, -2e-3, 3e-3, 0.05, -0.03, 0.02]))
    v.insert(gtsam.symbol('c', 0), np.r_[np.zeros(8), np.array([0.5, -0.3, 1.0, 2.0, 1e-3, 0, 0, 0])])
    v.insert(gtsam.symbol('I', 0), np.full(24, 1e-4))
    v.insert(gtsam.symbol('H', 0), np.linspace(-1e-3, 1e-3, 36))
    v.insert(gtsam.symbol('f', 7), np.array([2.5e-3]))
    v.insert(gtsam.symbol('o', 0), np.array([-1.2e-4]))
    return v


def _dump(v=None):
    v = v or _values()
    cal = CameraCalib(gtsam.Pose3(gtsam.Rot3.Rx(0.1), gtsam.Point3(0.05, 0, 0)),
                      1e-3, 0.0, np.array([430., 428., 320., 240., 0, 0, 0, 0]),
                      model="radtan")
    sig_c = np.full(16, 1e-3); sig_c[8:] = 0.2
    return build_state_dump(
        v, calibs={"c0": cal}, sigmas={"c0": sig_c, "I0": np.full(24, 1e-2), "A0": None},
        scalars={"err_init": 1234.5, "err_final": 12.25, "iterations": 7, "ok": True},
        arrays={"rms_pre": np.array([1.0, np.nan, 0.5]), "rms_post": np.array([0.3, np.nan, 0.2]),
                "fac_cam": np.array([0, 1, 1]), "corner_px": np.array([0.1, 0.2, np.nan]),
                "corner_cam": np.array([0, 0, 1]),
                "imu_cost_split": {"theta": 1.0, "p": 2.0}},
        timings={"solve_s": 1.5, "init_s": 0.2})


def test_symbol_names_and_values_roundtrip():
    v = _values()
    arr = values_to_arrays(v)
    assert symbol_name(gtsam.symbol('x', 12)) == "x12"
    assert set(arr) == {f"values/{k}" for k in
                        ("x0", "x1", "b0", "v0", "c0", "I0", "H0", "f7", "o0")}
    assert arr["values/x0"].shape == (4, 4)
    assert np.array_equal(arr["values/x0"], v.atPose3(gtsam.symbol('x', 0)).matrix())
    assert np.array_equal(arr["values/v0"], v.atVector(gtsam.symbol('v', 0)))
    assert arr["values/f7"].shape == (1,)


def test_dump_roundtrips_bitwise(tmp_path):
    d = _dump()
    p = str(tmp_path / "a.npz")
    write_state_dump(p, d)
    z = load_state_dump(p)
    assert set(z) == set(d)
    for k in d:
        assert np.array_equal(np.asarray(d[k]), z[k], equal_nan=True), k
    assert z["timings/solve_s"] == 1.5 and z["iterations"] == 7 and bool(z["ok"]) is True
    assert "sigma/A0" not in z                      # None sigmas are skipped
    assert z["imu_cost_split/p"] == 2.0 and z["calib/c0/T"].shape == (4, 4)


def test_next_dump_path_counts_passes():
    _PASS_COUNTER["n"] = 0
    assert next_dump_path("/x/state.npz") == "/x/state.npz"
    assert next_dump_path("/x/state.npz") == "/x/state_pass2.npz"
    assert next_dump_path("/x/state.npz") == "/x/state_pass3.npz"
    _PASS_COUNTER["n"] = 0


def test_exact_mode_accepts_identical_and_rejects_one_ulp():
    a = _dump()
    ok, rows = compare_states(a, a, exact=True)
    assert ok and all(r.ok for r in rows)
    b = dict(a)
    x = a["values/x1"].copy()
    x[0, 3] = np.nextafter(x[0, 3], np.inf)        # one ulp on a translation
    b["values/x1"] = x
    ok, rows = compare_states(a, b, exact=True)
    assert not ok
    bad = [r.key for r in rows if not r.ok]
    assert bad == ["values/x1"]
    # the same change passes the tolerance gate (far below 1e-8 m)
    ok, _ = compare_states(a, b, exact=False)
    assert ok


def test_nan_patterns_are_equal_not_different():
    a = _dump()
    ok, rows = compare_states(a, a, exact=False)
    assert ok
    b = dict(a)
    b["rms_pre"] = np.array([1.0, 0.7, 0.5])         # NaN -> number
    ok, rows = compare_states(a, b, exact=False)
    assert not ok
    assert [r.key for r in rows if not r.ok] == ["rms_pre"]


def test_missing_key_and_timings_ignored():
    a = _dump()
    b = dict(a)
    b["timings/solve_s"] = np.array(99.0)            # never gated
    ok, rows = compare_states(a, b, exact=True)
    assert ok
    row = next(r for r in rows if r.key == "timings/solve_s")
    assert row.max_abs == pytest.approx(97.5) and np.isnan(row.max_ratio)
    del b["values/H0"]
    ok, rows = compare_states(a, b, exact=True)
    assert not ok and any(r.key == "values/H0" and "missing" in r.note for r in rows)


def test_tolerance_rules_per_quantity():
    a = _dump()
    tol = Tolerances()

    def shifted(key, delta, idx=None):
        b = dict(a)
        x = a[key].copy()
        if idx is None:
            x = x + delta
        else:
            x[idx] += delta
        b[key] = x
        return b

    # Pose3 translation: 1e-8 m
    assert compare_states(a, shifted("values/x0", 0.9e-8, (0, 3)), tol=tol)[0]
    assert not compare_states(a, shifted("values/x0", 1.1e-8, (0, 3)), tol=tol)[0]
    # Pose3 rotation: 1e-8 rad (rotate about z by 2e-8 fails, 1e-10 passes and
    # is measured at full resolution -- not at the arccos(trace) floor ~1.5e-8)
    for ang, expect_ok in ((2e-8, False), (1e-10, True), (1e-13, True)):
        b = dict(a)
        R = gtsam.Rot3.Rz(ang).matrix()
        T = a["values/x0"].copy(); T[:3, :3] = R @ T[:3, :3]
        b["values/x0"] = T
        ok, rows = compare_states(a, b, tol=tol)
        assert ok == expect_ok, ang
        r = next(r for r in rows if r.key == "values/x0")
        assert r.max_abs == pytest.approx(ang, rel=1e-3, abs=2e-16), ang
    # per-frame dt: 1e-9 s
    assert compare_states(a, shifted("values/f7", 0.5e-9), tol=tol)[0]
    assert not compare_states(a, shifted("values/f7", 2e-9), tol=tol)[0]
    # heights: 1e-8 m
    assert not compare_states(a, shifted("values/H0", 2e-8, 3), tol=tol)[0]
    # reprojection norms: 1e-6 px
    assert compare_states(a, shifted("corner_px", 0.5e-6, 0), tol=tol)[0]
    assert not compare_states(a, shifted("corner_px", 2e-6, 0), tol=tol)[0]
    # cost: 1e-9 relative
    assert not compare_states(a, shifted("err_final", 12.25 * 2e-9), tol=tol)[0]
    # integer / bool entries are exact
    assert not compare_states(a, shifted("iterations", 1), tol=tol)[0]
    assert not compare_states(a, shifted("fac_cam", 1, 0), tol=tol)[0]
    # intrinsics: 1e-9 relative of the full value (calib/c0/intr), not of xi
    assert compare_states(a, shifted("calib/c0/intr", 430 * 0.5e-9, 0), tol=tol)[0]
    assert not compare_states(a, shifted("calib/c0/intr", 430 * 2e-9, 0), tol=tol)[0]
    # xi intrinsics entry: 1e-9 * |intr| = 4.3e-7 abs, OR 1e-4 sigma (0.2 -> 2e-5)
    assert compare_states(a, shifted("values/c0", 1.5e-5, 8), tol=tol)[0]
    assert not compare_states(a, shifted("values/c0", 3e-5, 8), tol=tol)[0]
    # xi rotation entry: 1e-8 rad OR 1e-4 * sigma (1e-3 -> 1e-7)
    assert compare_states(a, shifted("values/c0", 0.8e-7, 0), tol=tol)[0]
    assert not compare_states(a, shifted("values/c0", 1.2e-7, 0), tol=tol)[0]
    # scale multiplies everything; --tol overrides one group
    assert compare_states(a, shifted("values/f7", 2e-9), tol=Tolerances(scale=10))[0]
    assert compare_states(a, shifted("values/f7", 2e-9),
                          tol=Tolerances().with_overrides({"time": 1e-8}))[0]
    with pytest.raises(KeyError):
        Tolerances().with_overrides({"nope": 1.0})


def test_sigma_ratio_reported():
    a = _dump()
    b = dict(a)
    x = a["values/c0"].copy(); x[3] += 5e-7          # trans entry, sigma 1e-3
    b["values/c0"] = x
    ok, rows = compare_states(a, b)
    r = next(r for r in rows if r.key == "values/c0")
    assert r.max_sigma == pytest.approx(5e-4)
    assert not ok                                     # 5e-7 > max(1e-8, 1e-4 * 1e-3)
    g = {s["group"]: s for s in summarize(rows)}
    assert g["values/c"]["max_sigma"] == pytest.approx(5e-4) and not g["values/c"]["ok"]


def test_perturb_values_is_tiny_and_deterministic():
    v1, v2 = _values(), _values()
    ref = values_to_arrays(_values())
    n1 = perturb_values(v1, 1e-15, seed=0)
    n2 = perturb_values(v2, 1e-15, seed=0)
    assert n1 == n2 == 9
    a1, a2 = values_to_arrays(v1), values_to_arrays(v2)
    for k in ref:
        assert np.array_equal(a1[k], a2[k]), k                  # deterministic
        d = np.abs(a1[k] - ref[k])
        # rel * (|x| + 1) plus the rounding of x + d itself (pose entries go
        # through retract, so allow a few ulps there)
        bound = 1e-15 * (np.abs(ref[k]) + 1.0) + 4 * np.spacing(np.abs(ref[k]) + 1.0)
        assert np.all(d <= bound), (k, d, bound)
    changed = [k for k in ref if not np.array_equal(a1[k], ref[k])]
    assert len(changed) >= 7                                     # most keys move
    v3 = _values()
    perturb_values(v3, 1e-15, seed=1)
    assert not np.array_equal(values_to_arrays(v3)["values/v0"], a1["values/v0"])


_COMPARE_SCRIPT = os.path.join(_ROOT, "scripts", "compare_cam_imu_state.py")


@pytest.mark.skipif(not os.path.exists(_COMPARE_SCRIPT),
                    reason="maintainer script scripts/compare_cam_imu_state.py not in this tree")
def test_cli_script_exit_codes(tmp_path):
    a = _dump()
    pa, pb = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    write_state_dump(pa, a)
    b = dict(a)
    x = a["values/v0"].copy(); x[0] += 1e-12
    b["values/v0"] = x
    write_state_dump(pb, b)
    script = _COMPARE_SCRIPT
    env = dict(os.environ, PYTHONPATH=os.path.join(_ROOT, "src"))
    r = subprocess.run([sys.executable, script, pa, pb, "--exact"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 1 and "RESULT: FAIL" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([sys.executable, script, pa, pb], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "RESULT: PASS" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([sys.executable, script, pa, pa, "--exact"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "RESULT: PASS" in r.stdout, r.stdout + r.stderr
