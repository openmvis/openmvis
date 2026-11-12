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
"""Pipeline smoke for solve.vision_pose = "imu" (elp-stereo-imu-investigation
E8): the IMU-propagated vision factor builds through the multi-camera joint
pipeline, converges on the seeded synthetic scenario of
test_cam_imu_sim_recovery, and the default "interp" path is byte-identical in
its factor list. Marked slow (real GTSAM solves)."""
import re
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from mvis.calibration.pipeline_cam_imu import CamImuCalibConfig
from mvis.calibration.factors.camera_imu_factor import CameraCalibImuFactor
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.simulation.synth import build_synth, recover, zscore_table
from tests.helpers.sim_scenarios import _polled_scenario, _small_gt, _STRICT


def _scenario(seed=7):
    gt = _small_gt()
    gt.board_points_local = np.array(
        [[0.05 * j, 0.05 * i, 0.0] for i in range(6) for j in range(6)])
    ts = np.asarray(gt.t_states)
    imu_t = np.arange(ts[2], ts[-2], 1.0 / 400.0)
    cams, t_imu, w_m, a_m, aux = build_synth(
        gt, [ts[5:-5:3]], [imu_t], sigma_px=1.0,
        imu_sigmas=dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002),
        seed=seed)
    return gt, cams, t_imu, w_m, a_m, aux


@pytest.mark.parametrize("gyro,anchor", [("calibrated", "lower"),
                                         ("calibrated", "nearest"),
                                         ("raw", "lower")])
def test_vision_imu_converges_on_sim(gyro, anchor):
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    cfg = CamImuCalibConfig(vision_pose="imu", vision_imu_gyro=gyro,
                            vision_imu_anchor=anchor)
    res = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, cfg)
    assert res.ok
    assert res.diagnostics["reproj_rms_px"] < 2.0       # ~ sigma_px noise floor
    rows = zscore_table(gt, res)
    strict = [r for r in rows if _STRICT.match(r["param"])]
    assert strict
    bad = [r for r in strict if r["sigma"] == r["sigma"] and abs(r["z"]) >= 3.5]
    assert not bad, \
        f"camera params outside 3.5 sigma: {[(r['param'], round(r['z'], 2)) for r in bad]}"


def test_imu_is_the_default_and_each_setting_builds_its_factor():
    """Default config builds only interp factors (as before); vision_pose=imu
    builds only IMU-propagated factors with the velocity (and I0) keys."""
    import gtsam
    from mvis.calibration import pipeline_cam_imu as P
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    captured = {}
    orig = P.build_joint_graph

    def spy(*a, **k):
        out = orig(*a, **k)
        captured.setdefault("runs", []).append(out[2])     # cam_factors
        return out
    P.build_joint_graph = spy
    try:
        recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux,
                CamImuCalibConfig(max_iterations=1, vision_pose="interp"))
        recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux,
                CamImuCalibConfig(max_iterations=1))
    finally:
        P.build_joint_graph = orig
    interp, imu = captured["runs"]
    assert all(isinstance(f, CameraCalibInterpFactor) for _c, _j, f in interp)
    assert all(isinstance(f, CameraCalibImuFactor) for _c, _j, f in imu)
    assert len(imu) == len(interp)
    for (_c, j, f) in imu:
        assert f.keys[0] == P.X(j - 1) and f.keys[1] == P._Vk(j - 1)   # lower anchor
        assert f.keys[4] == P.I0


def test_frame_dt_pipeline_converges_and_reports():
    from mvis.calibration.pipeline_cam_imu import result_to_dict
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    cfg = CamImuCalibConfig(vision_pose="imu", vision_imu_frame_dt_sigma=0.004)
    res = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, cfg)
    assert res.ok and res.diagnostics["reproj_rms_px"] < 2.0
    fd = result_to_dict(res)["diagnostics"]["vision_imu"]["frame_dt_ms"]
    assert fd["n"] > 50
    assert abs(fd["mean"]) < 1.0 and fd["std"] < 2.0      # no jitter in this sim


def test_frame_dt_can_be_limited_to_some_cameras():
    """A camera dict with frame_dt False gets no per-frame offsets even when
    vision_imu_frame_dt_sigma > 0 (used by the sensor-clock pass 1)."""
    from mvis.calibration import pipeline_cam_imu as P
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    got = {}
    orig = P.build_joint_graph

    def spy(*a, **k):
        out = orig(*a, **k)
        got.setdefault("f", []).append([f.frame_dt for _c, _j, f in out[2]])
        return out
    P.build_joint_graph = spy
    try:
        for flag in (False, True):
            cams[0]["frame_dt"] = flag
            recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux,
                    CamImuCalibConfig(max_iterations=1, vision_pose="imu",
                                      vision_imu_frame_dt_sigma=0.01))
    finally:
        P.build_joint_graph = orig
        cams[0].pop("frame_dt", None)
    off, on = got["f"]
    assert off and not any(off)
    assert on and all(on)


def test_sensor_clock_retiming_two_pass():
    """Polled camera: without retiming the sawtooth offsets inflate the
    residuals; with retime_sensor_clock the sensor clock is recovered, the
    frames are re-stamped and the camera parameters come back within 3.5 sigma."""
    gt, cams, t_imu, w_m, a_m, aux, T_s = _polled_scenario()
    cfg = CamImuCalibConfig(vision_pose="imu")
    plain = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, cfg)
    cams_rt = [dict(cams[0], retime_sensor_clock=True)]
    res = recover(cams_rt, gt.board_points_local, t_imu, w_m, a_m, aux, cfg)
    sc = res.diagnostics["sensor_clock"][0]
    assert sc["retimed"], sc
    assert abs(sc["rate_hz"] - 1.0 / T_s) < 0.01
    assert sc["n_retimed"] >= 0.9 * len(cams[0]["t_cam"])
    assert res.ok
    floor = 0.5 * np.sqrt(2.0)                 # sigma_px 0.5 per axis -> 0.707 px rms
    assert res.diagnostics["reproj_rms_px"] < 1.05 * floor
    assert plain.diagnostics["reproj_rms_px"] > 1.15 * res.diagnostics["reproj_rms_px"]
    rows = zscore_table(gt, res)
    strict = [r for r in rows if _STRICT.match(r["param"])]
    bad = [r for r in strict if r["sigma"] == r["sigma"] and abs(r["z"]) >= 3.5]
    assert not bad, [(r["param"], round(r["z"], 2)) for r in bad]


def test_sensor_clock_t_ga_off_in_pass1_on_in_pass2():
    """Pass 1 carries per-frame offsets and no t_ga term; pass 2 the reverse. A few iterations are
    enough: this is about which factors are built, and pass 2 runs whether or not the lattice fits."""
    from mvis.calibration import pipeline_cam_imu as P
    gt, cams, t_imu, w_m, a_m, aux, _T_s = _polled_scenario()
    passes = []
    orig = P.build_joint_graph

    def spy(*a, **k):
        out = orig(*a, **k)
        passes.append({"tga": {f.tga for _c, _j, f in out[2]}, "frame_dt": {f.frame_dt for _c, _j, f in out[2]},
                       "O0_in_factor": {P.O0 in f.keys for _c, _j, f in out[2]}, "O0_state": out[1].exists(P.O0)})
        return out
    P.build_joint_graph = spy
    try:
        res = recover([dict(cams[0], retime_sensor_clock=True)], gt.board_points_local, t_imu, w_m, a_m, aux,
                      CamImuCalibConfig(vision_pose="imu", estimate_t_ga=True, max_iterations=3))
    finally:
        P.build_joint_graph = orig
    assert passes == [
        {"tga": {False}, "frame_dt": {True}, "O0_in_factor": {False}, "O0_state": True},   # the IMU factor keeps O0
        {"tga": {True}, "frame_dt": {False}, "O0_in_factor": {True}, "O0_state": True}]
    assert "t_ga" in res.diagnostics


def test_sensor_clock_requires_imu_vision_pose():
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    with pytest.raises(ValueError):
        recover([dict(cams[0], retime_sensor_clock=True)], gt.board_points_local,
                t_imu, w_m, a_m, aux, CamImuCalibConfig(vision_pose="interp"))


@pytest.mark.parametrize("kw,expect", [
    (dict(estimate_t_ga=True), True),
    (dict(estimate_t_ga=True, vision_imu_t_ga=False), False),
    (dict(estimate_t_ga=False), False),
    (dict(estimate_t_ga=True, vision_imu_gyro="raw"), False),
    (dict(estimate_t_ga=True, vision_imu_frame_dt_sigma=0.02), False),
])
def test_t_ga_key_only_when_estimated_and_on(kw, expect):
    """The factor takes O0 only with estimate_t_ga, the switch on and the calibrated gyro."""
    from mvis.calibration import pipeline_cam_imu as P
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    captured = {}
    orig = P.build_joint_graph

    def spy(*a, **k):
        out = orig(*a, **k)
        captured["facs"] = out[2]
        return out
    P.build_joint_graph = spy
    try:
        recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux,
                CamImuCalibConfig(max_iterations=1, vision_pose="imu", **kw))
    finally:
        P.build_joint_graph = orig
    assert captured["facs"]
    for _c, _j, f in captured["facs"]:
        assert f.tga is expect
        assert (P.O0 in f.keys) is expect


def test_t_ga_term_solves_end_to_end():
    """Term on and off both converge through the pipeline and report t_ga. Accuracy is
    covered at factor level (test_vision_imu_tga.py): this scene's motion is too smooth
    to observe t_ga."""
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    for on in (True, False):
        res = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux,
                      CamImuCalibConfig(vision_pose="imu", estimate_t_ga=True, vision_imu_t_ga=on))
        assert res.ok and res.diagnostics["iterations"] > 0
        assert res.diagnostics["reproj_rms_px"] < 2.0
        assert res.diagnostics["t_ga"] == res.diagnostics["t_ga"]


def test_config_switch_parses_from_solve_yaml(tmp_path):
    from mvis.core.config import CamImuRunConfig
    p = tmp_path / "c.yaml"
    p.write_text("input: {format: euroc, path: x}\n"
                 "board: {board_type: apriltag, rows: 6, cols: 6, tag_size: 0.03, tag_spacing: 0.3}\n"
                 "camera_input: {path: y}\nsolve: {vision_imu_t_ga: false}\noutput: {path: z.json}\n")
    assert CamImuRunConfig.from_yaml(str(p)).solve.vision_imu_t_ga is False
    assert CamImuCalibConfig().vision_imu_t_ga is True


def _run_with_singular_extra_keys(monkeypatch, cfg):
    """The real pipeline with a tiny-sigma row beside other keys added to its graph: Cholesky fails at
    every lambda, as it did with O0 beside the delta_f keys on a real bag before `imu_tga` went off in
    per-frame offset mode (that sim does not fail by itself: on `_scenario` it takes 5 iterations)."""
    import gtsam
    from mvis.calibration import pipeline_cam_imu as P
    a, b = gtsam.symbol("z", 0), gtsam.symbol("z", 1)

    def err(_f, v, H):
        if H is not None:
            H[0], H[1] = np.eye(1), np.eye(1)
        return v.atVector(a) + v.atVector(b) - 1.0
    orig = P.build_joint_graph

    def bad(*args, **kw):
        graph, values, *rest = orig(*args, **kw)
        graph.add(gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(1, 1e-12), gtsam.KeyVector([a, b]), err))
        for k in (a, b):
            graph.add(gtsam.PriorFactorVector(k, np.zeros(1), gtsam.noiseModel.Isotropic.Sigma(1, 1.0)))
            values.insert(k, np.zeros(1))
        return (graph, values, *rest)
    monkeypatch.setattr(P, "build_joint_graph", bad)
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    return recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, cfg)


def test_silent_lm_stop_is_reported_and_not_ok(monkeypatch, caplog):
    from mvis.calibration.pipeline_cam_imu import result_to_dict
    with caplog.at_level("WARNING", logger="mvis"):
        res = _run_with_singular_extra_keys(monkeypatch, CamImuCalibConfig(vision_pose="imu"))
    d = res.diagnostics
    assert d["iterations"] == 0 and d["err_final"] == d["err_init"]
    assert d["lm_no_step"]["failed"] is True and "Indeterminant" in d["lm_no_step"]["reason"]
    assert res.ok is False
    assert result_to_dict(res)["diagnostics"]["lm_no_step"]["failed"] is True
    assert any("LM made no step" in r.getMessage() for r in caplog.records)


def test_solve_that_iterates_has_no_no_step_key():
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    res = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, CamImuCalibConfig(max_iterations=2))
    assert res.diagnostics["iterations"] > 0 and "lm_no_step" not in res.diagnostics and res.ok


def test_failed_pass1_marks_the_retimed_solve_not_ok(monkeypatch):
    """A dead pass 1 has all-zero offsets; pass 2 may iterate, but its retiming means nothing."""
    import dataclasses
    from mvis.calibration import pipeline_cam_imu as P
    gt, cams, t_imu, w_m, a_m, aux = _scenario()
    orig = P._run_cam_imu_calibration_multi_once
    calls = []

    def once(*a, **k):
        calls.append(k["config"].vision_imu_frame_dt_sigma)
        r = orig(*a, **dict(k, config=dataclasses.replace(k["config"], max_iterations=1)))
        if len(calls) == 1:
            r.diagnostics["lm_no_step"] = {"failed": True, "reason": "forced", "err": 1.0}
        return r
    monkeypatch.setattr(P, "_run_cam_imu_calibration_multi_once", once)
    res = recover([dict(cams[0], retime_sensor_clock=True)], gt.board_points_local, t_imu, w_m, a_m, aux,
                  CamImuCalibConfig(vision_pose="imu"))
    assert len(calls) == 2 and res.ok is False
    assert res.diagnostics["sensor_clock_pass1_lm_no_step"]["reason"] == "forced"
    assert "lm_no_step" not in res.diagnostics
