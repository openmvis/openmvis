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
"""mvis.simulation.twin without a solve: fixture round trip, windows, simulation, the dense
trajectory and capture."""
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

from mvis.simulation.camera_measurements import project_points_rolling, simulate_camera
from mvis.simulation.from_result import cal_from_vec, fit_bspline_from_states
from mvis.calibration import pipeline_cam_imu as P
from mvis.calibration.factors.lie_groups import log_so3
from mvis.imu import CALYX
from mvis.simulation import twin as twin_module
from mvis.simulation.twin import (PERTURB, PERTURB_SEED, Twin, capture, config_for, dense_trajectory,
                                  noise_from_solve, imu_noise_scale, perturb, perturb_rng_seed, simulate, simulate_aux)
from tests.helpers.shaking_rig import _imu, _truth, _vb

DATA = Path(__file__).parents[1] / "data" / "twins"


@pytest.fixture(params=["twin_gs", "twin_rs"])
def twin(request):
    return Twin.load(str(DATA / f"{request.param}.npz"))


def test_fixture_loads_and_is_small(twin):
    assert (DATA / f"{twin.name}.npz").stat().st_size < 2e6
    assert twin.T_traj.shape[1:] == (4, 4) and np.allclose(twin.T_traj[:, 3], [0, 0, 0, 1])
    assert len(twin.truth["heights_m"]) == len(twin.board_points)
    for cam in twin.cams:
        assert len(cam["t_cam"]) == len(cam["ids_list"]) > 100


def test_save_load_round_trip(twin, tmp_path):
    p = tmp_path / "t.npz"
    twin.save(str(p))
    back = Twin.load(str(p))
    assert back.name == twin.name and back.truth == twin.truth
    assert back.solve_config == twin.solve_config and back.imu_noise == twin.imu_noise
    assert np.array_equal(back.T_traj, twin.T_traj) and np.array_equal(back.t_imu, twin.t_imu)
    for a, b in zip(back.cams, twin.cams):
        assert np.array_equal(a["t_cam"], b["t_cam"])
        assert all(np.array_equal(x, y) for x, y in zip(a["ids_list"], b["ids_list"]))
        assert a["image_size"] == b["image_size"] and a["model"] == b["model"]


def test_window_cuts_every_stream(twin):
    t0 = float(twin.t_traj[0]) + 2.0
    w = twin.window(t0, 3.0)
    assert w.t_traj.min() >= t0 and w.t_traj.max() <= t0 + 3.0
    assert w.t_imu.min() >= t0 and w.t_imu.max() <= t0 + 3.0
    for full, cam in zip(twin.cams, w.cams):
        assert 0 < len(cam["t_cam"]) < len(full["t_cam"])
        assert len(cam["ids_list"]) == len(cam["t_cam"])
    with pytest.raises(AssertionError):
        twin.window(float(twin.t_traj[-1]) + 100.0, 1.0)


def test_config_keeps_the_recording_settings_and_takes_overrides(twin):
    assert config_for(twin).corner_subsample == twin.solve_config["corner_subsample"]
    assert config_for(twin, vision_pose="interp").vision_pose == "interp"


def test_simulation_is_reproducible_and_noise_is_seeded(twin):
    short = twin.window(float(twin.t_traj[0]), 2.0)
    a = simulate(short, seed=1)
    b = simulate(short, seed=1)
    c = simulate(short, seed=2)
    assert np.array_equal(a[3], b[3]) and np.array_equal(a[0][0]["uv_list"][0], b[0][0]["uv_list"][0])
    assert not np.array_equal(a[3], c[3])
    clean = simulate(short, noise=False)
    assert not np.array_equal(clean[3], a[3])
    # the solver must not be handed real detections
    assert all(cam["frames"] is None for cam in a[0])


def test_true_t_ga_only_moves_the_accel(twin):
    short = twin.window(float(twin.t_traj[0]), 2.0)
    base = simulate(short, noise=False)
    moved = simulate(short, noise=False, true_t_ga=0.0)
    assert np.array_equal(base[2], moved[2])              # gyro unchanged
    assert not np.array_equal(base[3], moved[3])
    assert np.array_equal(base[0][0]["uv_list"][0], moved[0][0]["uv_list"][0])


def _scene():
    t = np.linspace(0.0, 2.0, 400)
    Ts = []
    for ti in t:
        T = np.eye(4)
        a = 0.8 * np.sin(2.0 * ti)
        T[:3, :3] = gtsam.Rot3.Rz(a).matrix()
        T[:3, 3] = [0.05 * ti, 0.0, -0.5]
        Ts.append(T)
    spline, _ = fit_bspline_from_states(t, Ts, 0.01)
    pts = np.array([[x, y, 0.0] for x in np.linspace(-0.1, 0.1, 5) for y in np.linspace(-0.1, 0.1, 5)])
    cal = cal_from_vec([400.0, 400.0, 320.0, 240.0, 0, 0, 0, 0], False, "radtan")
    return spline, pts, cal, gtsam.Pose3(), (640, 480)


@pytest.mark.parametrize("readout", [0.02, -0.02])
def test_simulate_camera_applies_rolling_shutter_for_either_sign(readout):
    """Negative readout used to fall through to a global shutter."""
    spline, pts, cal, T, size = _scene()
    times = np.array([0.7])
    rs = simulate_camera(spline, times, cal, T, pts, size, readout=readout)
    gs = simulate_camera(spline, times, cal, T, pts, size, readout=0.0)
    ids, uv, _ = project_points_rolling(spline, 0.7, cal, T, pts, size, readout, 0)
    assert np.array_equal(rs.ids[0], ids) and np.allclose(rs.uv[0], uv)
    assert np.abs(rs.uv[0] - gs.uv[0]).max() > 0.1


def _solved_states(t_states):
    """What dense_trajectory reads from a result, with every state at the truth."""
    values = gtsam.Values()
    for k, t in enumerate(t_states):
        values.insert(gtsam.symbol_shorthand.X(k), _truth(t))
        values.insert(P._Vk(k), _vb(t))
    return SimpleNamespace(values=values, t_states=t_states, imu_intr_xi=np.zeros(24))


def test_dense_trajectory_fills_states_with_imu_stamps_on_the_truth():
    t_imu, w_m, a_m = _imu(T=0.5)
    t_states = np.array([0.0, 0.041, 0.082, 0.123])
    t, T = dense_trajectory(_solved_states(t_states), t_imu, w_m, a_m, CALYX)

    inner = t_imu[(t_imu > t_states[0]) & (t_imu < t_states[-1])]
    inner = inner[np.min(np.abs(inner[:, None] - t_states), axis=1) > 1e-6]
    np.testing.assert_array_equal(np.sort(t), np.sort(np.concatenate([t_states, inner])))
    assert np.all(np.diff(t) > 0) and T.shape == (len(t), 4, 4)
    # the states themselves come through untouched, the IMU-filled poses stay on the truth
    for tk in t_states:
        np.testing.assert_array_equal(T[t == tk][0], _truth(tk).matrix())
    for ti, Ti in zip(t, T):
        Tt = _truth(ti).matrix()
        # measured 4.4e-7 m and 3.0e-7 rad; a chord over the 41 ms interval would be far off
        assert np.linalg.norm(Ti[:3, 3] - Tt[:3, 3]) < 2e-6
        assert np.linalg.norm(gtsam.Rot3.Logmap(gtsam.Rot3(Tt[:3, :3].T @ Ti[:3, :3]))) < 2e-6


def test_dense_trajectory_leaves_long_gaps_unfilled():
    t_imu, w_m, a_m = _imu(T=0.6)
    t_states = np.array([0.0, 0.041, 0.5])              # 0.459 s gap: geodesic only
    t, _ = dense_trajectory(_solved_states(t_states), t_imu, w_m, a_m, CALYX)
    assert t[-2] < 0.041 + 1e-9 and t[-1] == 0.5
    assert np.sum(t < 0.041) > 2


def test_capture_takes_the_solve_inside_the_cli_as_the_truth(monkeypatch, tmp_path):
    import mvis.cli.main as cli
    from mvis.core.config import CamImuRunConfig
    monkeypatch.setattr(P, "run_cam_imu_calibration_multi", lambda *a, **k: "solved")
    inner = P.run_cam_imu_calibration_multi
    config = tmp_path / "c.yaml"
    config.write_text("x: 1")
    cfg = SimpleNamespace(output=SimpleNamespace(path=None),
                          input=SimpleNamespace(cam_topics=None, path="bag", format="ros1"))
    monkeypatch.setattr(CamImuRunConfig, "from_yaml", staticmethod(lambda path: cfg))

    def fake_cli(c):                                    # the real CLI looks the solver up here too
        P.run_cam_imu_calibration_multi("cams", "pts", "t", "w", "a", config="cfg")
    monkeypatch.setattr(cli, "calibrate_cam_imu", fake_cli)
    got = {}
    monkeypatch.setattr(twin_module, "from_solve",
                        lambda *a, **k: got.update(args=a, kw=k) or "twin")

    monkeypatch.setattr(twin_module, "perturb", lambda t, seed: (t, seed))
    out = capture(str(config), "n", sigma_px=0.3, imu_noise=dict(sigma_w=1.0),
                  work_dir=str(tmp_path / "w"))
    assert out == ("twin", PERTURB_SEED)                # the real solve is perturbed by default
    assert got["args"] == ("n", "solved", "cams", "pts", "t", "w", "a", "cfg")
    assert got["kw"] == dict(sigma_px=0.3, imu_noise=dict(sigma_w=1.0), aux_imus=None)
    assert cfg.output.path == str(tmp_path / "w" / "cam_imu.json") and cfg._source_yaml == "x: 1"
    assert (tmp_path / "w").is_dir()
    assert P.run_cam_imu_calibration_multi is inner     # the spy is gone
    assert capture(str(config), "n", work_dir=str(tmp_path / "w"), perturb_seed=None) == "twin"


def _fake_cli_solve(monkeypatch, fail=False):
    import mvis.cli.main as cli
    monkeypatch.setattr(P, "run_cam_imu_calibration_multi", lambda *a, **k: "solved")

    def fake_cli(c):
        P.run_cam_imu_calibration_multi("cams", "pts", "t", "w", "a", aux_imus="aux", config="cfg")
        if fail:
            raise RuntimeError("boom")
        return dict(reproj_rms_px=0.5)
    monkeypatch.setattr(cli, "calibrate_cam_imu", fake_cli)
    return P.run_cam_imu_calibration_multi


def test_capture_solve_returns_the_inner_solve_and_its_inputs(monkeypatch):
    inner = _fake_cli_solve(monkeypatch)
    cfg = SimpleNamespace(input=SimpleNamespace(cam_topics=("/c0", "/c1"), path="x.bag", format="ros1"))
    grab = twin_module.capture_solve(cfg)
    assert grab == dict(r="solved", cams="cams", board_points="pts", t_imu="t", w_m="w", a_m="a",
                        config="cfg", aux_imus="aux", summary=dict(reproj_rms_px=0.5),
                        cam_topics=["/c0", "/c1"], src_path="x.bag", src_format="ros1")
    assert P.run_cam_imu_calibration_multi is inner


def test_capture_solve_restores_the_solver_when_the_run_fails(monkeypatch):
    inner = _fake_cli_solve(monkeypatch, fail=True)
    cfg = SimpleNamespace(input=SimpleNamespace(cam_topics=None, path="m", format="euroc"))
    with pytest.raises(RuntimeError):
        twin_module.capture_solve(cfg)
    assert P.run_cam_imu_calibration_multi is inner


def _with_aux(twin):
    """The twin plus one aux IMU 100 mm from the base, stamped 5 ms late, accel t_ga 3 ms."""
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 0.0]
    aux = dict(name="aux", xi24=twin.truth["xi24"], T_IstoI=T.tolist(), td=0.005, t_ga=0.003,
               b_g=[0.001, 0.0, 0.0], b_a=[0.0, 0.02, 0.0], gyro_only=False)
    return dataclasses.replace(twin, truth={**twin.truth, "aux": [aux]}, aux_t_imu=[twin.t_imu[::2].copy()])


def test_aux_imu_and_camera_settings_survive_save_and_window(twin, tmp_path):
    twin = _with_aux(twin)
    twin.cams[0]["factor_hz"] = 4.0
    twin.sigma_px = [0.2] * len(twin.cams)
    p = tmp_path / "t.npz"
    twin.save(str(p))
    back = Twin.load(str(p))
    assert back.truth["aux"] == twin.truth["aux"] and np.array_equal(back.aux_t_imu[0], twin.aux_t_imu[0])
    assert back.cams[0]["factor_hz"] == 4.0 and back.sigma_px == twin.sigma_px
    t0 = float(twin.t_traj[0]) + 2.0
    cut = twin.window(t0, 3.0).aux_t_imu[0]
    assert cut.min() >= t0 and cut.max() <= t0 + 3.0 and len(cut) < len(twin.aux_t_imu[0])
    assert simulate(twin, noise=False)[0][0]["factor_hz"] == 4.0


def test_shipped_fixtures_are_perturbed_with_the_fixed_seed(twin):
    assert twin.truth["perturbation"] == dict(seed=PERTURB_SEED, name=twin.name,
                                              rng_seed=perturb_rng_seed(PERTURB_SEED, twin.name), **PERTURB)


def _unperturbed(twin):
    return dataclasses.replace(twin, truth={k: v for k, v in twin.truth.items() if k != "perturbation"})


def test_perturbation_is_seeded_bounded_and_moves_the_truth_off_the_source(twin):
    src = _with_aux(_unperturbed(twin))
    a, b = perturb(src), perturb(src)
    assert a.truth == b.truth                             # seeded: the same twin twice
    assert perturb(src, seed=PERTURB_SEED + 1).truth != a.truth
    # each twin name gets its own draws; the hash is stable across processes (crc32)
    other = perturb(dataclasses.replace(src, name=src.name + "_x"))
    step = lambda p: np.subtract(p.truth["cams"][0]["intr8"], src.truth["cams"][0]["intr8"])
    assert not np.allclose(step(other), step(a))
    assert perturb_rng_seed(3, "twin_gs") == 3 * 2 ** 32 + 0x58e73758
    assert PERTURB_SEED == 3 and PERTURB == dict(focal_rel=0.01, center_px=4.0, dist_rel=0.05, rot_rad=0.01,
                                                 lever_m=0.005, td_s=0.001, t_ga_s=0.0005)
    s = PERTURB
    for c0, c1, g0, g1 in zip(src.cams, a.cams, src.truth["cams"], a.truth["cams"]):
        i0, i1 = np.asarray(g0["intr8"]), np.asarray(g1["intr8"])
        step = i1 - i0
        assert np.all(step != 0)                          # truth != the source solve, every coefficient
        assert np.all(np.abs(step[:2]) <= s["focal_rel"] * np.abs(i0[:2]))
        assert np.all(np.abs(step[2:4]) <= s["center_px"])
        assert np.all(np.abs(step[4:]) <= s["dist_rel"] * np.abs(i0[4:]))
        assert np.allclose(c1["cal_vec"] - c0["cal_vec"], step)     # the prior moves with the truth
        T0, T1 = np.asarray(g0["T_CtoI"]), np.asarray(g1["T_CtoI"])
        rot = log_so3(T0[:3, :3].T @ T1[:3, :3])
        assert 0 < np.abs(rot).max() <= s["rot_rad"] + 1e-12
        assert 0 < np.abs(T1[:3, 3] - T0[:3, 3]).max() <= s["lever_m"]
        assert 0 < abs(g1["td"] - g0["td"]) <= s["td_s"] and g1["tr"] == g0["tr"]
    assert 0 < abs(a.truth["t_ga"] - src.truth["t_ga"]) <= s["t_ga_s"]
    x0, x1 = src.truth["aux"][0], a.truth["aux"][0]
    assert 0 < abs(x1["td"] - x0["td"]) <= s["td_s"] and 0 < abs(x1["t_ga"] - x0["t_ga"]) <= s["t_ga_s"]
    p0, p1 = np.asarray(x0["T_IstoI"])[:3, 3], np.asarray(x1["T_IstoI"])[:3, 3]
    assert 0 < np.abs(p1 - p0).max() <= s["lever_m"]
    for k in ("xi24", "b_g", "b_a", "g_inW", "T_BinW", "heights_m"):
        assert a.truth[k] == src.truth[k]
    assert np.array_equal(a.T_traj, src.T_traj) and src.truth == _with_aux(_unperturbed(twin)).truth
    with pytest.raises(ValueError, match="already perturbed"):
        perturb(a)


def test_twin_without_aux_stays_as_it_was(twin):
    assert twin.aux_t_imu == [] and simulate_aux(twin) == []


def test_aux_samples_follow_the_model(twin):
    """The aux gyro is the base gyro at t + td, and its accel carries t_ga and the lever arm."""
    from mvis.imu import parse_imu_model
    from mvis.imu.imu_model import ImuModel
    from mvis.simulation.imu_measurements import simulate_imu
    from mvis.simulation.twin import _truth_spline_and_window
    t2 = _with_aux(twin)
    aux = simulate_aux(t2, noise=False)[0]
    at = t2.truth["aux"][0]
    spline, lo, hi = _truth_spline_and_window(t2)
    assert aux["t_imu"].min() + at["td"] > lo and aux["t_imu"].max() + at["td"] < hi
    assert aux["w_m"].shape == aux["a_m"].shape == (len(aux["t_imu"]), 3)
    kw = dict(g_inG=np.asarray(t2.truth["g_inW"], float), R_IstoI=np.eye(3), p_IsinI=np.array([0.1, 0.0, 0.0]),
              b_g0=at["b_g"], b_a0=at["b_a"])
    intr = ImuModel.from_tangent(np.asarray(at["xi24"], float), parse_imu_model(t2.truth["imu_model"]))
    want_g = simulate_imu(spline, aux["t_imu"], intr, td=0.005, **kw)
    want_a = simulate_imu(spline, aux["t_imu"], intr, td=0.002, **kw)
    assert np.array_equal(aux["w_m"], want_g.w_m) and np.array_equal(aux["a_m"], want_a.a_m)
    # the 3 ms offset matters: the accel is not the one at the gyro's time
    assert np.abs(aux["a_m"] - want_g.a_m).max() > 1e-4


def test_noise_is_measured_from_the_real_residuals():
    result = SimpleNamespace(diagnostics=dict(imu_chi2_dof=99.0),
                             cameras=[dict(reproj_rms_px=0.3), dict(reproj_rms_px=0.5)])
    cfg = SimpleNamespace(sigma_g=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002)
    px, imu = noise_from_solve(result, cfg)
    assert px == pytest.approx([0.3 / 2 ** 0.5, 0.5 / 2 ** 0.5])     # RMS of a 2D error, per axis
    # IMU noise starts at the solver's sigmas; the chi2 does not set it (it can't: free states absorb it)
    assert imu == pytest.approx(dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002))


def test_imu_noise_scale_matches_the_real_chi2():
    # chi2 grows with the noise squared: a pilot at 4x the real chi2 needs half the noise
    assert imu_noise_scale(0.04, 0.16) == pytest.approx(0.5)
    assert imu_noise_scale(0.25, 0.25) == pytest.approx(1.0)


def test_true_board_heights_replace_the_initial_ones_and_are_not_added(twin):
    """A row with a heights file starts from heights; the simulated board must sit at the truth."""
    start = 0.002 * np.ones(len(twin.board_points))
    truth = 0.003 * np.ones(len(twin.board_points))
    bumped = twin.board_points.copy()
    bumped[:, 2] = start
    t = dataclasses.replace(twin, board_points=bumped, truth={**twin.truth, "heights_m": truth.tolist()})
    uv_t = simulate(t, noise=False)[0][0]["uv_list"][10]
    flat = dataclasses.replace(twin, truth={**twin.truth, "heights_m": truth.tolist()})
    uv_flat = simulate(flat, noise=False)[0][0]["uv_list"][10]
    assert np.allclose(uv_t, uv_flat, atol=1e-9)


def test_knot_spacing_is_saved_and_makes_the_truth_smoother(twin, tmp_path):
    wide = dataclasses.replace(twin, knot_dt=0.02)
    p = tmp_path / "t.npz"
    wide.save(str(p))
    assert Twin.load(str(p)).knot_dt == 0.02
    jump = lambda t: np.abs(np.diff(simulate(t, noise=False)[3], axis=0)).mean()
    assert jump(wide) < jump(twin)
