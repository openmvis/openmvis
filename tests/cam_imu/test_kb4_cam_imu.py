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
"""kb4 through the cam-IMU pipelines (plan kb4-kb8, G2): board mode with intrinsics pinned and
free, and the multi-board path, all with the model named, from a scenario simulated through the
model. The simulator gates on z > 0, so this stays in front; past 90 deg is `test_kb4_sim.py`."""
import os
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest
import yaml

from mvis.calibration.pipeline_cam_imu import (
    CamImuCalibConfig, format_report, run_cam_imu_calibration_multi)
from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG
from mvis.io.cam_imu_output import write_cam_imu_calibration
from mvis.io.camera_input import load_camera_intrinsics
from mvis.simulation.from_result import ModelCal
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario
from mvis.simulation.trajectory_io import load_trajectory

KB4 = np.array([300.0, 301.0, 321.0, 239.0, 0.02, -0.01, 3e-3, -5e-4])
T_CtoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])), gtsam.Point3(0.05, -0.02, 0.01))


def _scenario():
    board = Chessboard(7, 7, 0.10)
    traj = load_trajectory(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                                        "sim_trajectories", "calib_3D_v0.txt"), dt=0.05)
    t0 = traj.start_time + 1.0
    sc = make_scenario(traj, t0=t0, depth=3.0, board=board, cal=ModelCal("kb4", KB4),
                       T_CtoI=T_CtoI, imu_intr=ImuModel(model=RPNG),
                       b_g=np.array([0.01, -0.02, 0.005]), b_a=np.array([0.05, -0.03, 0.02]),
                       image_size=(640, 480), td_cam=0.01, sigma_px=0.0)
    return sc, board, t0


def _run(start, pin):
    sc, board, t0 = _scenario()
    t_cam = np.arange(t0, t0 + 15.0, 1.0 / 20.0)
    cam = sc.camera(t_cam, seed=7)
    s = sc.base_imu(imu_times(t0 - 0.3, t0 + 15.3, 400.0), seed=7)
    return run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv), cal_vec=start, model="kb4",
              image_size=(640, 480), pin_intrinsics=pin)],
        np.asarray(board.all_points_3d(), float), s.t, s.w_m, s.a_m,
        config=CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0))


@pytest.mark.slow
def test_board_mode_pinned_e2e_and_output(tmp_path):
    res = _run(KB4, True)
    assert res.ok and res.calib.model == "kb4" and res.calib.n_intr == 8 and res.calib.DIM == 16
    np.testing.assert_allclose(res.calib.intr, KB4, atol=1e-6)
    d = T_CtoI.localCoordinates(res.calib.T)
    assert np.linalg.norm(d[:3]) < 1e-2 and abs(res.calib.td - 0.01) < 3e-3
    assert res.diagnostics["reproj_rms_px"] < 0.01
    txt = format_report(res)
    assert "dist (k1 k2 k3 k4)" in txt and "cam0 kb4 valid domain" in txt
    a = res.cameras[0]["corner_accounting"]
    assert a["model"] == "kb4" and a["used"] > 0 and a["past_fold"] == a["other_invalid"] == 0
    out, chain = tmp_path / "r_calib.yaml", tmp_path / "chain.yaml"
    write_cam_imu_calibration(res, str(out), kalibr_path=str(chain))
    c0 = yaml.safe_load(chain.read_text())["cam0"]
    assert (c0["camera_model"], c0["distortion_model"]) == ("pinhole", "equidistant")
    np.testing.assert_allclose(c0["intrinsics"] + c0["distortion_coeffs"], res.calib.intr, atol=1e-12)
    blk = yaml.safe_load(out.read_text())["cameras"][0]
    assert blk["model"] == "pinhole_kb4" and blk["corner_accounting"]["used"] == a["used"]
    lc = load_camera_intrinsics(str(out))[0]
    assert lc["distortion_model"] == "kb4"
    np.testing.assert_allclose(lc["intrinsics"] + lc["distortion_coeffs"], res.calib.intr, atol=1e-9)


@pytest.mark.slow
def test_board_mode_free_intrinsics_recovered():
    start = KB4 + np.r_[0.8, -0.6, 0.5, -0.4, 2e-3, -1e-3, 3e-4, -5e-5]
    res = _run(start, False)
    assert res.ok and res.calib.model == "kb4"
    # 15 s leaves focal weakly observed: 0.2 px
    assert np.abs(res.calib.intr[:4] - KB4[:4]).max() < 0.2
    assert res.diagnostics["reproj_rms_px"] < 0.01


@pytest.mark.slow
def test_multi_board_path_keeps_the_model(graph_factors):
    from mvis.calibration.multi_board_pipeline import run_multi_board_calibration
    from mvis.simulation.multi_board import MultiBoardScenario, forward_T_CtoI, place_boards_ring
    from mvis.simulation.trajectories import load_sine3d
    traj = load_sine3d(target_span=1.0)
    board = Chessboard(11, 11, 0.10)
    poses = place_boards_ring(board, radius=1.5, directions=4)
    cal = np.array([288.7, 288.9, 416.1, 406.0, -0.0078, 0.0442, -0.0412, 0.0073])
    sc = MultiBoardScenario(traj=traj, boards=[board] * 4, T_BinW_list=poses, cal=ModelCal("kb4", cal),
                            is_fisheye=False, T_CtoI=forward_T_CtoI(), image_size=(848, 800),
                            imu_intr=ImuModel(model=RPNG), b_g=np.zeros(3), b_a=np.zeros(3),
                            sigma_px=0.5)
    t = imu_times(traj.start_time, traj.start_time + 30.0, 10.0)
    cs = sc.camera(t, seed=1)
    imu = sc.base_imu(imu_times(traj.start_time, traj.start_time + 30.0, 200.0), seed=1)
    res = run_multi_board_calibration(
        dict(t_cam=cs.t, ids_list=cs.ids, uv_list=cs.uv, cal_vec=cal, model="kb4",
             image_size=(848, 800)),
        {b: sc.board_points(b) for b in range(4)}, imu.t, imu.w_m, imu.a_m)
    assert res.calib.model == "kb4" and res.calib.n_intr == 8
    a = res.diagnostics["corner_accounting"]
    assert a["used"] > 0
    _check(a, graph_factors[0])


def _check(a, factor_corners):
    """Against the factors the graph was built from, not a remainder."""
    assert a["n_corners"] == factor_corners
    assert a["not_in_solve"] >= 0
    assert a["detected"] == a["n_corners"] + a["no_pose_corners"] + a["outliers"] + a["not_in_solve"]


@pytest.fixture
def graph_factors(monkeypatch):
    """Corners per camera of the factors of the last graph built."""
    import mvis.calibration.multi_board_pipeline as mb
    import mvis.calibration.pipeline_cam_imu as pci
    seen = {}
    real = pci.build_joint_graph

    def spy(*a, **k):
        out = real(*a, **k)
        seen.clear()
        for c, _j, f in out[2]:
            seen[c] = seen.get(c, 0) + f.n
        return out
    monkeypatch.setattr(pci, "build_joint_graph", spy)
    monkeypatch.setattr(mb, "build_joint_graph", spy)
    return seen


@pytest.mark.slow
def test_lost_frames_count_as_no_pose(graph_factors):
    """A frame PnP refuses and one under 6 corners reach no factor and must count, with their corners.
    The review found no_pose_frames held only bracket failures, and no_pose_corners stayed 0."""
    sc, board, t0 = _scenario()
    t_cam = np.arange(t0, t0 + 15.0, 1.0 / 20.0)
    cam = sc.camera(t_cam, seed=7)
    ids, uv = [np.asarray(x) for x in cam.ids], [np.asarray(x) for x in cam.uv]
    assert all(len(i) >= 6 for i in ids[100:102])
    ids[100], uv[100] = ids[100][:0], uv[100][:0]      # no corners: PnP refuses it
    ids[101], uv[101] = ids[101][:5], uv[101][:5]      # 5 corners: under PnP's 6
    s = sc.base_imu(imu_times(t0 - 0.3, t0 + 15.3, 400.0), seed=7)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=ids, uv_list=uv, cal_vec=KB4, model="kb4",
              image_size=(640, 480), pin_intrinsics=True)],
        np.asarray(board.all_points_3d(), float), s.t, s.w_m, s.a_m,
        config=CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0))
    a = res.cameras[0]["corner_accounting"]
    assert a["no_pose_frames"] >= 2
    assert a["no_pose_corners"] >= 5
    assert res.cameras[0]["n_dropped"] < a["no_pose_frames"]
    assert a["detected"] == sum(len(x) for x in ids)
    _check(a, graph_factors[0])


@pytest.mark.slow
@pytest.mark.parametrize("path", ["single", "multi"])
def test_feature_mode_has_the_count(path, graph_factors):
    from mvis.calibration.pipeline_cam_imu import (
        run_cam_imu_calibration_feature, run_cam_imu_calibration_feature_multi)
    sc, _board, t0 = _scenario()
    t_cam = np.arange(t0, t0 + 20.0, 1.0 / 20.0)
    cam = sc.camera(t_cam, seed=3)
    s = sc.base_imu(imu_times(t0 - 0.3, t0 + 20.3, 400.0), seed=3)
    cfg = CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0)
    if path == "multi":
        res = run_cam_imu_calibration_feature_multi(
            [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv), cal_vec=KB4, model="kb4",
                  image_size=(640, 480))], s.t, s.w_m, s.a_m, sfm_mode="planar", config=cfg)
    else:
        res = run_cam_imu_calibration_feature(
            t_cam, list(cam.ids), list(cam.uv), KB4, s.t, s.w_m, s.a_m, image_size=(640, 480),
            model="kb4", sfm_mode="planar", config=cfg)
    a = res.cameras[0]["corner_accounting"]
    assert a["model"] == "kb4" and a["used"] > 0 and a["past_fold"] == 0
    assert a["n_corners"] == a["used"] + a["other_invalid"]
    _check(a, graph_factors[0])


def test_feature_wrapper_routes_to_multi(monkeypatch):
    """G4: the single-camera feature entry hands feature_multi one camera with its vision
    factors at the state rate and td_pin, and passes the solve options through."""
    import mvis.calibration.pipeline_cam_imu as pci
    seen = {}
    monkeypatch.setattr(pci, "run_cam_imu_calibration_feature_multi",
                        lambda cams, *a, **k: seen.update(cams=cams, a=a, k=k) or "res")
    cfg = CamImuCalibConfig(state_hz=7.0)
    out = pci.run_cam_imu_calibration_feature(
        [0.0], [[1]], [[[1.0, 2.0]]], KB4, "t", "w", "a", (640, 480), model="kb4",
        sfm_mode="planar", config=cfg, pin_intrinsics=False, td_pin=0.004)
    (cam,) = seen["cams"]
    assert out == "res" and seen["a"] == ("t", "w", "a")
    assert cam["factor_hz"] == 7.0 and cam["td_pin"] == 0.004 and cam["model"] == "kb4"
    assert seen["k"] == dict(config=cfg, pin_intrinsics=False, sfm_mode="planar", sfm_kwargs=None)


@pytest.mark.slow
def test_feature_wrapper_holds_td_pin():
    """G4: the single-camera feature entry is feature_multi with the frames at the states and
    td started (and, under a tight td prior, held) at td_pin instead of the seed's."""
    from mvis.calibration.pipeline_cam_imu import run_cam_imu_calibration_feature
    sc, _board, t0 = _scenario()
    t_cam = np.arange(t0, t0 + 20.0, 1.0 / 20.0)
    cam = sc.camera(t_cam, seed=3)
    s = sc.base_imu(imu_times(t0 - 0.3, t0 + 20.3, 400.0), seed=3)
    cfg = CamImuCalibConfig(state_hz=0.0, sigma_px=0.5, huber_px=0.0, sigma_td_prior=1e-6)
    res = run_cam_imu_calibration_feature(
        t_cam, list(cam.ids), list(cam.uv), KB4, s.t, s.w_m, s.a_m, image_size=(640, 480),
        model="kb4", sfm_mode="planar", config=cfg, td_pin=0.0123)
    assert res.ok and res.diagnostics["sfm_mode"] == "planar"
    assert abs(res.calib.td - 0.0123) < 1e-5 and abs(res.seed.td - 0.0123) > 1e-4


@pytest.mark.slow
def test_failed_count_leaves_the_board_mode_result(monkeypatch, caplog):
    import mvis.calibration.pipeline_cam_imu as pci
    start = KB4 + np.r_[0.8, -0.6, 0.5, -0.4, 2e-3, -1e-3, 3e-4, -5e-5]
    good = _run(start, False)

    def boom(*a, **k):
        raise ValueError("kb4: expected 4 distortion coefficients")
    monkeypatch.setattr(pci, "_corner_accounting", boom)
    with caplog.at_level("WARNING"):
        res = _run(start, False)
    assert res.ok and res.cameras[0]["corner_accounting"] is None
    assert "corner_accounting failed" in caplog.text
    np.testing.assert_allclose(res.calib.intr, good.calib.intr, atol=1e-6)
