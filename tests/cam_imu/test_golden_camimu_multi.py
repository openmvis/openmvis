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
"""Golden snapshot of `_run_cam_imu_calibration_multi_once` on a fixed sim
(cam-IMU speed-up, S0 golden).

Scenario (~5.6 s, 168 states at 30 Hz, 3 LM iterations, ~10-15 s wall):
  * cam0 radtan, global shutter, `vision_pose: interp`, tr pinned by `_TINY`;
  * cam1 equidistant, rolling shutter (20 ms readout, rolling_mode 0),
    `vision_pose: imu`, tr estimated from 0;
  * calyx IMU with a non-identity Dw, gyro calibrated (`imu_intrinsics dw_da`),
    per-frame `frame_dt` offsets on the IMU-propagated camera, estimated board
    heights, t_ga estimated (the config defaults).

Stored in `tests/data/golden_camimu_multi_once.npz` (numpy only): rms_pre /
rms_post / fac_cam / corner_px / corner_cam / err_init / err_final /
imu_interval_cost / imu_cost_split / per-camera stats and sigmas / v_init /
the final calib vectors / pose matrices / vb / xi_imu / heights / frame-dt /
t_ga, plus the measured NOISE FLOOR per array (`floor__<name>`) and the
metadata of the run that took it (`meta__*`: BLAS thread count, BLAS build,
numpy, CPU architecture, the thread counts used for the floor).

BLAS THREADS (S0 fix round 1). numpy's OpenBLAS reduces in a thread-dependent
order, so the solve is bitwise reproducible only at ONE thread count: this
golden is taken with GOLDEN_THREADS = 1 (the same single-thread setting S0b
requires for every bitwise bag 2 C gate). With the default 32 threads the same
sim ends 1.3e-5 away in err_final (26 of 32 arrays differ) - deterministic per
thread count, different between them. The test pins the thread count of this
process itself through `tests/_blas_threads.py` (ctypes into the loaded
OpenBLAS), so `OPENBLAS_NUM_THREADS` need not be set; it is honoured when it is.

Two modes:
  * `GOLDEN_EXACT=1` -> `np.array_equal` on every array, run with the golden's
    thread count pinned in-process (skipped, not failed, when it cannot be
    pinned and the environment does not provide it). Still machine-specific
    (SIMD sin/cos differ across CPUs): only the machine that took the golden
    can expect it; the BITWISE steps (D1, K1, K2, K8) use it locally.
  * default -> |Δ| <= max(ATOL + RTOL·|golden|, FLOOR_MULT · noise_floor) at
    whatever thread count the process has. The floor is measured when the
    golden is taken as the max over (a) two ~1e-15-relative
    perturbations of the initial Values at 1 thread and (b) unperturbed runs
    at the other thread counts (2, 4, 8, all cores), so this mode is
    thread-agnostic. Rounding steps (K3-K7, D2) must pass this mode, then
    `--regen-golden` once accepted (note it in the commit).

LINEARIZED SNAPSHOT. The exact factor's golden pins `vision_imu_linearized=False`, so it keeps
guarding that factor. `golden_camimu_multi_linearized_once.npz` is the same scenario with the
linearized factor, the default on real rows. It catches changes to that path; it does not say the
factor is right, which `test_linearized_fit_matches_the_exact_factor` checks on the fit.
Regenerate one at a time: `-k exact` or `-k linearized` with `--regen-golden`.

`pytest tests/cam_imu/test_golden_camimu_multi.py --regen-golden` rewrites the npz
(~8 sim runs): the golden at 1 thread, a second run for determinism (bitwise),
the two perturbed runs and the runs at the other thread counts for the floor.
"""
import os
import time

import numpy as np
import gtsam
import pytest

from mvis.simulation.from_result import GroundTruth, cal_from_vec
from mvis.simulation.camera_measurements import simulate_camera
from mvis.simulation.imu_measurements import simulate_imu
from mvis.calibration import pipeline_cam_imu as P
from mvis.calibration.pipeline_cam_imu import CamImuCalibConfig, X, _Vk, _Cc, H0, I0, O0
from mvis.simulation.synth import _spline, _imu_intr
from tests._blas_threads import (get_num_threads, blas_threads, env_num_threads,
                                 describe as describe_blas)

pytestmark = pytest.mark.slow

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden_camimu_multi_once.npz")
GOLDEN_LINEARIZED = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden_camimu_multi_linearized_once.npz")
ATOL = 1e-10            # S0 default-mode floor ...
RTOL = 1e-9             # ... and relative tolerance
FLOOR_MULT = 10.0       # amendment 3: tolerance = multiple of the measured noise floor
# The floor is measured on the machine that took the golden. Another CPU rounds differently
# (seen on CI: err_init differs by 1.5e-8 relative, rms_pre by 3e-7), so CI gets these instead.
OTHER_MACHINE_ATOL = 1e-6
OTHER_MACHINE_RTOL = 1e-7
GOLDEN_THREADS = 1      # BLAS threads the golden is taken with (the bitwise setting)
N_STATES = 168          # 5.6 s at 30 Hz
_FLOOR_PREFIX = "floor__"
_META_PREFIX = "meta__"


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------
def _gt_rig():
    dt = 1.0 / 30.0
    ts = 1.0 + np.arange(N_STATES) * dt
    T_states = []
    for t in ts:
        M = np.eye(4)
        M[:3, :3] = (gtsam.Rot3.Rz(0.5 * np.sin(1.3 * t))
                     .compose(gtsam.Rot3.Rx(0.3 * np.sin(0.7 * t)))).matrix()
        M[:3, 3] = [0.3 * np.sin(t), 0.3 * np.cos(0.8 * t), 0.1 * np.sin(0.5 * t)]
        T_states.append(M.tolist())
    Tb = np.eye(4); Tb[:3, 3] = [0.0, 0.0, 1.3]
    T1 = np.eye(4)
    T1[:3, :3] = gtsam.Rot3.Rodrigues(0.02, -0.35, 0.03).matrix()
    T1[:3, 3] = [0.1, 0.02, -0.01]
    xi = np.zeros(24); xi[0] = 0.01; xi[4] = -0.005; xi[8] = 0.002    # calyx Dw diagonal
    return GroundTruth(
        cams=[dict(intr8=[400., 405., 320., 240., -0.05, 0.01, 0.0, 0.0],
                   T_CtoI=np.eye(4).tolist(), td=0.003, is_fisheye=False,
                   model="radtan", image_size=[640, 480]),
              dict(intr8=[380., 382., 330., 245., 0.01, -0.002, 0.0005, 0.0],
                   T_CtoI=T1.tolist(), td=-0.002, is_fisheye=True,
                   model="equidistant", image_size=[640, 480])],
        imus=[dict(name="base", is_base=True, xi24=xi.tolist(), T_IstoI=None, td=0.0)],
        b_g=[0.001, 0.0, -0.001], b_a=[0.01, -0.01, 0.0],
        T_BinW=Tb.tolist(), t_states=ts.tolist(), T_states=T_states,
        imu_model="calyx", g_inW=[0.0, 0.0, -9.81])


def golden_scenario(seed=7, readout=0.02, linearized=False):
    """(cams, board_points, t_imu, w_m, a_m, cfg) of the golden sim."""
    gt = _gt_rig()
    pts_local = np.array([[0.05 * j, 0.05 * i, 0.0] for i in range(6) for j in range(6)])
    spline, _ = _spline(gt)
    Tb = np.asarray(gt.T_BinW, float)
    pts_world = (Tb[:3, :3] @ pts_local.T).T + Tb[:3, 3]
    ts = np.asarray(gt.t_states)
    frame_times = [ts[3:-3], ts[3:-3] + 0.5 / 30.0]
    cams = []
    for ci, cgt in enumerate(gt.cams):
        cal = cal_from_vec(cgt["intr8"], cgt["is_fisheye"], cgt["model"])
        cs = simulate_camera(spline, frame_times[ci], cal,
                             gtsam.Pose3(np.asarray(cgt["T_CtoI"])), pts_world,
                             tuple(cgt["image_size"]), td=cgt["td"], sigma_px=0.5,
                             readout=(readout if ci == 1 else 0.0), rolling_mode=0,
                             seed=seed + ci)
        cams.append(dict(t_cam=cs.t, ids_list=cs.ids, uv_list=cs.uv,
                         cal_vec=np.asarray(cgt["intr8"], float),
                         is_fisheye=cgt["is_fisheye"], model=cgt["model"],
                         image_size=tuple(cgt["image_size"]), pin_intrinsics=False,
                         rolling_mode=0, estimate_readout=(ci == 1),
                         vision_pose=("imu" if ci == 1 else "interp")))
    imu_t = np.arange(ts[1], ts[-1], 1.0 / 400.0)
    bs = simulate_imu(spline, imu_t, _imu_intr(gt.imus[0]["xi24"], gt.imu_model),
                      g_inG=np.asarray(gt.g_inW), b_g0=gt.b_g, b_a0=gt.b_a,
                      seed=seed + 100, sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001,
                      sigma_ba=0.002)
    cfg = CamImuCalibConfig(max_iterations=3, vision_imu_frame_dt_sigma=0.004,
                            estimate_board_heights=True, vision_imu_linearized=linearized)
    return cams, pts_local, bs.t, bs.w_m, bs.a_m, cfg


# ---------------------------------------------------------------------------
# run + flatten
# ---------------------------------------------------------------------------
def _perturb_values(values: gtsam.Values, rng, rel=1e-15):
    """In-place ~`rel` relative perturbation of every initial value (poses:
    retract by rel-scaled tangent; vectors: multiply by (1 ± rel))."""
    for k in values.keys():
        try:
            pose = values.atPose3(k)
        except RuntimeError:
            v = np.asarray(values.atVector(k), float)
            values.update(k, v * (1.0 + rel * rng.choice([-1.0, 1.0], size=v.shape)))
            continue
        scale = np.concatenate([np.ones(3), np.maximum(np.abs(pose.translation()), 1.0)])
        values.update(k, pose.retract(rel * scale * rng.choice([-1.0, 1.0], size=6)))


def run_golden_sim(perturb_seed=None, linearized=False):
    """Run `_run_cam_imu_calibration_multi_once` on the golden scenario and
    return its result flattened to a {name: ndarray} dict (numpy only).
    `perturb_seed` != None perturbs the initial Values by ~1e-15 relative
    (noise-floor measurement)."""
    cams, pts, t, w, a, cfg = golden_scenario(linearized=linearized)
    captured = {}
    orig = P.build_joint_graph

    def spy(*args, **kw):
        out = orig(*args, **kw)
        captured["v_init"] = np.asarray(args[3], float).copy()
        if perturb_seed is not None:
            _perturb_values(out[1], np.random.default_rng(perturb_seed))
        return out

    P.build_joint_graph = spy
    try:
        r = P._run_cam_imu_calibration_multi_once(cams, pts, t, w, a, config=cfg)
    finally:
        P.build_joint_graph = orig
    d = r.diagnostics
    res = r.values
    out = {
        "iterations": np.array(d["iterations"]),
        "err_init": np.array(d["err_init"]), "err_final": np.array(d["err_final"]),
        "rms_pre": np.asarray(d["reproj_rms_pre_px"], float),
        "rms_post": np.asarray(d["reproj_rms_post_px"], float),
        "fac_cam": np.asarray(d["reproj_frame_cam"], int),
        "reproj_stats": np.array([d["reproj_rms_px"], d["reproj_median_px"], d["reproj_p95_px"]]),
        "chi2": np.array([d["cam_chi2_dof"], d["imu_chi2_dof"]]),
        "imu_interval_cost": np.asarray(d["imu_interval_cost"], float),
        "imu_cost_split_keys": np.array(sorted(d["imu_cost_split"])),
        "imu_cost_split": np.array([d["imu_cost_split"][k] for k in sorted(d["imu_cost_split"])]),
        "v_init": captured["v_init"],
        "t_states": np.asarray(r.t_states, float),
        "xi_imu": np.asarray(res.atVector(I0), float),
        "imu_intr_sigma": (np.asarray(r.imu_intr_sigma, float) if r.imu_intr_sigma is not None
                           else np.full(24, np.nan)),
        "heights": np.asarray(res.atVector(H0), float),
        "t_ga": np.asarray(res.atVector(O0), float) if res.exists(O0) else np.zeros(0),
    }
    K = len(r.t_states)
    out["poses"] = np.stack([res.atPose3(X(k)).matrix() for k in range(K)])
    out["vb"] = np.stack([np.asarray(res.atVector(_Vk(k)), float) for k in range(K)])
    fdt = []
    n = 0
    while res.exists(gtsam.symbol('f', n)):
        fdt.append(float(res.atVector(gtsam.symbol('f', n))[0])); n += 1
    out["frame_dt"] = np.asarray(fdt, float)
    for c, e in enumerate(r.cameras):
        cal = e["calib"]
        out[f"cam{c}_xi"] = np.asarray(res.atVector(_Cc(c)), float)
        out[f"cam{c}_T"] = np.asarray(cal.T.matrix(), float)
        out[f"cam{c}_td_tr"] = np.array([cal.td, cal.tr])
        out[f"cam{c}_intr"] = np.asarray(cal.intr, float)
        out[f"cam{c}_sigma"] = np.asarray(e["calib_sigma"], float)
        out[f"cam{c}_stats"] = np.array([e["reproj_rms_px"], e["reproj_median_px"],
                                         e["reproj_p95_px"], e["n_frames"]])
    return out


# The pooled corner norms are not kept on the result: wrap the pipeline helper
# so the run records exactly what the diagnostics consumed (no second pass).
_corner_capture = {}
_orig_reproj_corner_norms = P._reproj_corner_norms


def _capturing_reproj_corner_norms(cam_factors, values, table=None):
    # `table` (D1): the fused corner-norm table the pipeline built at `values`;
    # forwarded so the capture records what the diagnostics consumed, no new pass
    px, cams = _orig_reproj_corner_norms(cam_factors, values, table=table)
    _corner_capture["corner_px"] = np.asarray(px, float).copy()
    _corner_capture["corner_cam"] = np.asarray(cams, int).copy()
    return px, cams


def run_golden(perturb_seed=None, linearized=False):
    P._reproj_corner_norms = _capturing_reproj_corner_norms
    try:
        out = run_golden_sim(perturb_seed, linearized)
    finally:
        P._reproj_corner_norms = _orig_reproj_corner_norms
    out["corner_px"] = _corner_capture["corner_px"]
    out["corner_cam"] = _corner_capture["corner_cam"]
    return out


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _data_keys(g):
    return [k for k in g if not (k.startswith(_FLOOR_PREFIX) or k.startswith(_META_PREFIX))]


def compare_to_golden(out, golden, exact):
    """Returns a list of (name, max |Δ|, allowed) violations; [] = pass."""
    bad = []
    for k in _data_keys(golden):
        A = np.asarray(out[k]); B = np.asarray(golden[k])
        if A.dtype.kind in "US":
            if not np.array_equal(A, B):
                bad.append((k, "label mismatch", ""))
            continue
        if A.shape != B.shape:
            bad.append((k, f"shape {A.shape} vs {B.shape}", ""))
            continue
        if k == "iterations" or A.dtype.kind in "iub":
            if not np.array_equal(A, B):
                bad.append((k, f"{A} vs {B}", "equal"))
            continue
        nanA, nanB = np.isnan(A), np.isnan(B)
        if not np.array_equal(nanA, nanB):
            bad.append((k, "NaN pattern", ""))
            continue
        m = ~nanB
        if not m.any():
            continue
        d = np.abs(A[m] - B[m])
        if exact:
            if d.max() != 0.0:
                bad.append((k, float(d.max()), 0.0))
            continue
        floor = np.asarray(golden[_FLOOR_PREFIX + k])[m] if _FLOOR_PREFIX + k in golden else 0.0
        allowed = np.maximum(ATOL + RTOL * np.abs(B[m]), FLOOR_MULT * floor)
        if os.environ.get("CI"):
            allowed = np.maximum(allowed, OTHER_MACHINE_ATOL + OTHER_MACHINE_RTOL * np.abs(B[m]))
        if np.any(d > allowed):
            i = int(np.argmax(d - allowed))
            bad.append((k, float(d[i]), float(allowed[i])))
    return bad


def _accumulate_floor(floors, ref, other, what):
    """floors[k] = elementwise max |ref - other| over the float arrays; the
    non-float arrays must be identical (an iteration-count flip is not noise)."""
    for k in _data_keys(ref):
        A = np.asarray(ref[k]); B = np.asarray(other[k])
        if A.dtype.kind not in "fc":
            assert np.array_equal(A, B), f"{k} changed under {what}"
            continue
        d = np.abs(A - B)
        d = np.where(np.isnan(d), 0.0, d)
        floors[k] = np.maximum(floors.get(k, 0.0), d)


def _worst(floors, n=8):
    return sorted(((float(np.max(v)) if np.size(v) else 0.0, k) for k, v in floors.items()),
                  reverse=True)[:n]


def _floor_thread_counts(initial):
    """Thread counts the floor is measured at besides GOLDEN_THREADS."""
    cands = {2, 4, 8, os.cpu_count() or 1, initial or 1}
    return sorted(c for c in cands if c != GOLDEN_THREADS)


def golden_meta(golden):
    """{'blas_threads': int, 'blas': str, 'floor_threads': [int], 'taken': str}."""
    def _s(k, default=""):
        return str(golden[_META_PREFIX + k]) if _META_PREFIX + k in golden else default
    return dict(blas_threads=int(golden[_META_PREFIX + "blas_threads"]),
                blas=_s("blas"), taken=_s("taken"),
                floor_threads=[int(x) for x in np.atleast_1d(golden[_META_PREFIX + "floor_threads"])]
                if _META_PREFIX + "floor_threads" in golden else [])


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------
def _regen(path, linearized, wall_budget=120.0):
    initial = get_num_threads()
    with blas_threads(GOLDEN_THREADS) as n_eff:
        assert n_eff == GOLDEN_THREADS or env_num_threads() == GOLDEN_THREADS, (
            f"--regen-golden needs {GOLDEN_THREADS} BLAS thread(s): no OpenBLAS to pin "
            f"in-process (threads {n_eff}); set OPENBLAS_NUM_THREADS={GOLDEN_THREADS}")
        t0 = time.time()
        out = run_golden(linearized=linearized)
        wall = time.time() - t0
        again = run_golden(linearized=linearized)
        nondet = compare_to_golden(again, out, exact=True)
        assert not nondet, f"the golden sim is not deterministic run-to-run: {nondet}"
        pert_floors = {}
        for s in (1, 2):
            _accumulate_floor(pert_floors, out, run_golden(perturb_seed=s, linearized=linearized),
                              f"a 1e-15 perturbation (seed {s})")
    thread_floors, used = {}, []
    for n in _floor_thread_counts(initial):
        with blas_threads(n) as n_eff:
            if n_eff != n:
                continue
            alt = run_golden(linearized=linearized)
        used.append(n)
        _accumulate_floor(thread_floors, out, alt, f"{n} BLAS threads")
    floors = dict(pert_floors)
    for k, v in thread_floors.items():
        floors[k] = np.maximum(floors.get(k, 0.0), v)
    store = dict(out)
    store.update({_FLOOR_PREFIX + k: v for k, v in floors.items()})
    store[_META_PREFIX + "blas_threads"] = np.array(GOLDEN_THREADS)
    store[_META_PREFIX + "blas"] = np.array(describe_blas())
    store[_META_PREFIX + "floor_threads"] = np.array(used, int)
    store[_META_PREFIX + "taken"] = np.array(time.strftime("%Y-%m-%d %H:%M"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **store)
    print(f"\n[regen-golden] wrote {path} ({wall:.1f} s per run, {GOLDEN_THREADS} BLAS "
          f"thread(s), {describe_blas()}); noise floor = max of the two 1e-15 "
          f"perturbations and the runs at {used} threads.\n"
          f"  worst 8 perturbation : {_worst(pert_floors)}\n"
          f"  worst 8 thread count : {_worst(thread_floors)}\n"
          f"  worst 8 combined     : {_worst(floors)}")
    assert wall < wall_budget, f"golden sim took {wall:.1f} s (budget {wall_budget:.0f} s)"


def test_other_machine_allowance_applies_only_on_ci(monkeypatch):
    golden = {"x": np.array([1e8]), _FLOOR_PREFIX + "x": np.array([0.0])}
    near, far = {"x": np.array([1e8 + 3.0])}, {"x": np.array([1e8 + 100.0])}
    monkeypatch.delenv("CI", raising=False)
    assert compare_to_golden(near, golden, False)           # this machine stays strict
    monkeypatch.setenv("CI", "true")
    assert not compare_to_golden(near, golden, False)       # rounding on another CPU
    assert compare_to_golden(far, golden, False)            # a real change still fails


@pytest.mark.parametrize("linearized", [False, True], ids=["exact", "linearized"])
def test_golden_camimu_multi_once(request, linearized):
    path = GOLDEN_LINEARIZED if linearized else GOLDEN
    if request.config.getoption("--regen-golden"):
        _regen(path, linearized)
        return
    assert os.path.exists(path), "run with --regen-golden first"
    golden = dict(np.load(path, allow_pickle=False))
    exact = os.environ.get("GOLDEN_EXACT", "0") == "1"
    meta = golden_meta(golden)
    t0 = time.time()
    if exact:
        want = meta["blas_threads"]
        with blas_threads(want) as n_eff:
            if n_eff != want and env_num_threads() != want:
                pytest.skip(f"GOLDEN_EXACT needs {want} BLAS thread(s) (the golden's setting); "
                            f"this process has {get_num_threads()} and cannot pin them "
                            f"(no OpenBLAS): set OPENBLAS_NUM_THREADS={want}")
            out = run_golden(linearized=linearized)
        mode = f"GOLDEN_EXACT ({want} BLAS thread(s); golden taken on {meta['blas']})"
    else:
        out = run_golden(linearized=linearized)
        mode = (f"tolerance (atol {ATOL}, rtol {RTOL}, {FLOOR_MULT}x floor measured at "
                f"{[meta['blas_threads']] + meta['floor_threads']} threads; this run "
                f"{get_num_threads()} threads)")
    wall = time.time() - t0
    missing = [k for k in _data_keys(golden) if k not in out]
    assert not missing, missing
    bad = compare_to_golden(out, golden, exact)
    assert not bad, f"golden mismatch in {mode} mode ({wall:.1f} s): {bad}"
    assert wall < 120.0, f"golden sim took {wall:.1f} s (budget 2 min)"


def test_linearized_fit_matches_the_exact_factor():
    """Full solve: the linearized factor is an approximation, so compare the fit, not the values.
    The levers are not compared: on this 5.6 s scenario they are nearly free."""
    exact, lin = run_golden(), run_golden(linearized=True)
    assert abs(lin["err_final"] - exact["err_final"]) < 0.01 * exact["err_final"]
    assert abs(lin["reproj_stats"][0] - exact["reproj_stats"][0]) < 0.02
    assert abs(lin["t_ga"][0] - exact["t_ga"][0]) < 1e-4
    assert int(lin["iterations"]) == int(exact["iterations"])
