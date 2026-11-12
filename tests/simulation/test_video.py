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
"""Video render module (mvis/video-render subtask)."""
import os

import numpy as np
import gtsam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mvis.detection.chessboard import Chessboard


def test_animate_writes_nonempty(tmp_path):
    from mvis.simulation.video import _animate
    fig, ax = plt.subplots()

    def frame_fn(i):
        ax.clear()
        ax.plot([0, i], [0, 1])

    written = _animate(fig, frame_fn, n_frames=3, fps=5, out=str(tmp_path / "a.gif"))
    plt.close(fig)
    assert os.path.exists(written) and os.path.getsize(written) > 0


def test_animate_mp4_request_falls_back(tmp_path):
    """An .mp4 request must still write something when ffmpeg is unavailable."""
    from mvis.simulation.video import _animate
    fig, ax = plt.subplots()
    written = _animate(fig, lambda i: ax.plot([0, i], [0, 1]), n_frames=2, fps=5,
                       out=str(tmp_path / "a.mp4"))
    plt.close(fig)
    assert os.path.exists(written) and os.path.getsize(written) > 0


def test_scene_axes_draws(tmp_path):
    from mvis.simulation.video import _scene_axes
    board = Chessboard(7, 7, 0.10)
    pose = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0.0, 0.0, 2.0))
    cam_path = np.random.RandomState(0).randn(20, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _scene_axes(ax, [board], [pose], cam_path)
    assert len(ax.collections) >= 1            # board face/corners were drawn
    plt.close(fig)


def test_render_sim_video_single_board(tmp_path):
    from mvis.simulation.video import render_sim_video
    from mvis.simulation.multi_board import SimConfig
    out = render_sim_video(SimConfig.realistic_single_board(),
                           str(tmp_path / "s.mp4"), fps=4, max_seconds=3.0)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_render_sim_video_multiboard(tmp_path):
    from mvis.simulation.video import render_sim_video, _BOARD_COLORS
    from mvis.simulation.multi_board import SimConfig
    out = render_sim_video(SimConfig.synthetic_multiboard_fisheye(),
                           str(tmp_path / "m.gif"), fps=3, max_seconds=2.0)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def _calib_scene(cfg):
    """Build a CalibScene + detections from a SimConfig (stands in for a real
    calibration result + its detection cache)."""
    from mvis.simulation.video import CalibScene
    scn, cam, _imu, bpts, gt, _aux = cfg.simulate()
    scene = CalibScene(
        traj=scn.traj, board_points=bpts,
        board_poses={i: gt['board_poses'][i] for i in range(len(gt['board_poses']))},
        cal_vec=gt['cam_intrinsics'], is_fisheye=cfg.is_fisheye,
        image_size=tuple(cfg.image_size), T_CtoI=gt['T_CtoI'])
    return scene, cam


def test_render_calib_video_plays_detections(tmp_path):
    from mvis.simulation.video import render_calib_video
    from mvis.simulation.multi_board import SimConfig
    scene, cam = _calib_scene(SimConfig.realistic_single_board())
    out = render_calib_video(scene, cam, str(tmp_path / "calib.mp4"), fps=4, max_frames=6)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_render_calib_video_show_trajectory(tmp_path):
    from mvis.simulation.video import render_calib_video
    from mvis.simulation.multi_board import SimConfig
    scene, cam = _calib_scene(SimConfig.realistic_single_board())
    out = render_calib_video(scene, cam, str(tmp_path / "tr.mp4"), fps=4, max_frames=5,
                             show_trajectory=True)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_reproject_matches_detections_noise_free():
    """GT-derived detections must reproject ~onto themselves (overlay correctness)."""
    from mvis.simulation.video import _reproject_frame
    from mvis.simulation.multi_board import SimConfig
    scene, cam = _calib_scene(SimConfig.realistic_single_board(sigma_px=0.0))
    k = next(i for i, ids in enumerate(cam['ids_list']) if len(ids))
    det = np.asarray(cam['uv_list'][k], float).reshape(-1, 2)
    rep = _reproject_frame(scene, float(cam['t_cam'][k]), np.asarray(cam['ids_list'][k]))
    assert np.nanmax(np.linalg.norm(det - rep, axis=1)) < 1.0


def test_calib_scene_from_result_renders(tmp_path):
    """The real-result flow: recover, build a scene from the RESULT, render the
    detections vs the RECOVERED reprojection (should be close for a good calib)."""
    from mvis.simulation.multi_board import SimConfig
    from mvis.calibration.multi_board_pipeline import run_multi_board_calibration
    from mvis.simulation.video import (
        calib_scene_from_result, render_calib_video, _reproject_frame)

    cfg = SimConfig.realistic_single_board(duration=22.0)
    scn, cam, imu, bpts, gt, aux = cfg.simulate()
    res = run_multi_board_calibration(cam, bpts, imu.t, imu.w_m, imu.a_m,
                                      config=cfg.estimator_config(), pin_intrinsics=False)
    scene = calib_scene_from_result(res, bpts, image_size=cfg.image_size)
    k = next(i for i, ids in enumerate(cam['ids_list']) if len(ids))
    det = np.asarray(cam['uv_list'][k], float).reshape(-1, 2)
    rep = _reproject_frame(scene, float(cam['t_cam'][k]), np.asarray(cam['ids_list'][k]))
    assert np.nanmedian(np.linalg.norm(det - rep, axis=1)) < 5.0   # recovered calib reprojects near detections
    out = render_calib_video(scene, cam, str(tmp_path / "r.mp4"), fps=4, max_frames=5)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_render_calib_video_from_path(tmp_path):
    """Render straight from files on disk: a GroundTruth json + a DetectionCache json
    (detection-only playback; board_points omitted)."""
    import json
    from mvis.simulation.from_result import gt_from_json
    from mvis.simulation.video import render_calib_video_from_path

    from mvis.simulation.multi_board import DEFAULT_GT_FILE
    gt_path = DEFAULT_GT_FILE                      # packaged captured GT (any cwd)
    ts = [float(t) for t in gt_from_json(gt_path).t_states[20:25]]
    cache = {"cameras": {"0": {"frames": [
        {"camera_id": 0, "timestamp": t, "image_size": [640, 480],
         "observations": [{"point_id": i, "pixel": [300 + 5 * i, 200 + 3 * i], "row": 0}
                          for i in range(12)]}
        for t in ts]}}}
    cpath = tmp_path / "detections.json"
    with open(cpath, "w") as f:
        json.dump(cache, f)
    out = render_calib_video_from_path(gt_path, str(tmp_path / "p.mp4"),
                                       detections_path=str(cpath), fps=4)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_render_detections_video(tmp_path):
    from mvis.simulation.video import render_detections_video
    rng = np.random.RandomState(0)
    det = dict(t_cam=np.arange(6) * 0.1,
               uv_list=[rng.uniform(0, 640, (20, 2)) for _ in range(6)],
               ids_list=[np.arange(20) for _ in range(6)])
    out = render_detections_video(det, (640, 480), str(tmp_path / "d.mp4"),
                                  fps=4, max_frames=5)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_board_points_from_object_and_yaml(tmp_path):
    from mvis.simulation.video import board_points_from
    bp = board_points_from(Chessboard(7, 7, 0.10))
    assert set(bp) == {0} and bp[0].shape == (49, 3)
    assert np.allclose(bp[0][5], Chessboard(7, 7, 0.10).point_3d(5))   # indexed by point_id

    yml = tmp_path / "target.yaml"
    yml.write_text("target_type: 'aprilgrid'\ntagCols: 6\ntagRows: 6\n"
                   "tagSize: 0.088\ntagSpacing: 0.3\n")
    bp2 = board_points_from(str(yml))
    assert bp2[0].shape == (6 * 6 * 4, 3)


def test_render_calib_from_path_with_board_spec(tmp_path):
    """board_spec (a YAML) → reprojection overlay in the path renderer."""
    import json
    from mvis.simulation.from_result import gt_from_json
    from mvis.simulation.video import render_calib_video_from_path

    from mvis.simulation.multi_board import DEFAULT_GT_FILE
    gt_path = DEFAULT_GT_FILE                      # packaged captured GT (any cwd)
    ts = [float(t) for t in gt_from_json(gt_path).t_states[20:24]]
    cache = {"cameras": {"0": {"frames": [
        {"camera_id": 0, "timestamp": t, "image_size": [640, 480],
         "observations": [{"point_id": i, "pixel": [300 + 5 * i, 200 + 3 * i], "row": 0}
                          for i in range(20)]} for t in ts]}}}
    cp = tmp_path / "d.json"
    with open(cp, "w") as f:
        json.dump(cache, f)
    yml = tmp_path / "t.yaml"
    yml.write_text("target_type: 'aprilgrid'\ntagCols: 6\ntagRows: 6\n"
                   "tagSize: 0.088\ntagSpacing: 0.3\n")
    out = render_calib_video_from_path(gt_path, str(tmp_path / "b.mp4"),
                                       detections_path=str(cp), board_spec=str(yml), fps=4)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_rotation_curve_bounded_through_gimbal_lock():
    """Oscillating motion around pitch=90 deg (Euler singularity) must yield a
    smooth, bounded curve — rpy+unwrap ratcheted this to thousands of degrees."""
    from mvis.simulation.video import rotation_curve_deg
    base = gtsam.Rot3.Pitch(np.pi / 2)                 # rest pose AT the singularity
    t = np.linspace(0.0, 10.0, 400)
    Rs = [(base * gtsam.Rot3.Expmap(0.6 * np.sin(2 * np.pi * 0.5 * tk) * np.array([1.0, 0.7, 0.3]))).matrix()
          for tk in t]
    th = rotation_curve_deg(Rs)
    assert th.shape == (len(t), 3)
    assert np.all(th[0] == 0.0)
    assert np.abs(th).max() < 90.0                     # bounded like the true motion (~40 deg)
    assert np.abs(np.diff(th, axis=0)).max() < 10.0    # no 180-deg flips


def test_rotation_log_rad_is_logmap_from_start():
    """log(R) curve = Log(R_0ᵀ·R_i) in rad: zero at start, exact for a known rotation,
    smooth through the pitch=90 deg Euler singularity."""
    from mvis.simulation.video import rotation_log_rad
    base = gtsam.Rot3.Pitch(np.pi / 2)
    w = np.array([0.3, -0.2, 0.5])
    Rs = [(base * gtsam.Rot3.Expmap(a * w)).matrix() for a in np.linspace(0.0, 1.0, 50)]
    th = rotation_log_rad(Rs)
    assert th.shape == (50, 3)
    assert np.allclose(th[0], 0.0)
    assert np.allclose(th[-1], w, atol=1e-9)
    assert np.abs(np.diff(th, axis=0)).max() < 0.05


def _fake_capture(n_states=3, n_frames=5):
    """A `twin.capture_solve`-shaped capture: two cameras, one aux IMU."""
    from types import SimpleNamespace
    from mvis.calibration.pipeline_cam_imu import X
    vals = gtsam.Values()
    for i in range(n_states):
        vals.insert(X(i), gtsam.Pose3(gtsam.Rot3.Rz(0.1 * i), gtsam.Point3(i, 0, 0)))
    cal = lambda k: SimpleNamespace(intr=[400.0 + k, 401.0, 320.0, 240.0, 0.1, 0.0, 0.0, 0.0],
                                    is_fisheye=bool(k), model=None if k else "radtan",
                                    T=gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.01 * k, 0, 0)),
                                    td=0.002 * k, tr=0.0)
    r = SimpleNamespace(t_states=np.arange(n_states) * 0.1, values=vals,
                        cameras=[dict(calib=cal(k), reproj_rms_px=0.3 + k) for k in range(2)],
                        T_BinW=gtsam.Pose3(), imus=[dict(name="aux", T_IstoI=gtsam.Pose3()), dict(name="x")])
    cams = [dict(t_cam=np.arange(n_frames) * 0.05, image_size=(640, 480),
                 uv_list=[np.full((2, 2), f, float) for f in range(n_frames)],
                 ids_list=[np.array([0, 1]) for _ in range(n_frames)]) for _ in range(2)]
    return dict(r=r, cams=cams, board_points=[[0, 0, 0], [0.1, 0, 0]], summary=dict(reproj_rms_px=0.4),
                cam_topics=["/cam0"], src_path="/d/x.bag", src_format="ros1")


def test_world_from_capture_keeps_every_camera_and_strides_detections():
    import json
    from mvis.simulation.video import world_from_capture
    W = world_from_capture(_fake_capture(), stride=2)
    json.dumps(W)                                            # JSON-ready
    assert len(W["T_states"]) == 3 and W["t_states"] == [0.0, 0.1, 0.2]
    assert W["board_points"] == [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]] and W["reproj_rms_px"] == 0.4
    c0, c1 = W["cameras"]
    assert c0["source"] == "/cam0" and c1["source"] == "cam1"
    assert c0["model"] == "radtan" and c1["model"] == "equidistant" and c1["is_fisheye"]
    assert c0["det"]["t_cam"] == [0.0, 0.1, 0.2] and c0["det"]["uv_list"][1] == [[2.0, 2.0], [2.0, 2.0]]
    assert c1["td"] == 0.002 and c1["reproj_rms_px"] == 1.3
    assert [m["name"] for m in W["imus"]] == ["imu0", "aux"]          # aux without a pose is skipped
