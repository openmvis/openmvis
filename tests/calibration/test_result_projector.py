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
"""ResultProjector (G2): camera-only residual walkers score corners against the board the solve
fitted — `board_points_refined` when set, the nominal board otherwise, then bit for bit as before."""

import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_projection import (
    find_bracketing_poses, k_cal_vec, project_observation, synced_time,
)
from mvis.calibration.outlier_rejection import (
    board_pose_timeline, compute_per_observation_residuals,
)
from mvis.calibration.result_projector import ResultProjector
from mvis.core.types import (
    CalibrationResult, CameraParams, ExtrinsicTransform, FrameObservations, Observation,
)
from mvis.detection.chessboard import Chessboard

IMG = (640, 480)


def _board():
    return Chessboard(rows=7, cols=5, square_size=0.03)


def _heights(board, amp=0.004):
    pts = board.all_points_3d()
    return amp * np.sin(17.0 * pts[:, 0]) * np.cos(23.0 * pts[:, 1])


def _setup(refined: bool, dist_model="radtan"):
    """Two cameras (cam 1 with extrinsic, td and RS readout), 6 ref poses, frames projected from
    the refined board (heights of a few mm) plus small noise."""
    board = _board()
    nominal = board.all_points_3d()
    pts_true = nominal.copy()
    pts_true[:, 2] += _heights(board)
    dist = [0.05, -0.02, 0.001, -0.0005] if dist_model == "radtan" else [0.01, -0.003, 0.001, 0.0]
    cams = [CameraParams(camera_id=c, intrinsics=np.array([460.0 + c, 458.0, 320.0, 240.0]),
                         distortion=np.array(dist), distortion_model=dist_model, image_size=IMG,
                         rolling_shutter_readout=0.0 if c == 0 else 0.02) for c in (0, 1)]
    T_ext = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.01, -0.02, 0.005), gtsam.Point3(0.1, 0.002, -0.001))
    ext = [ExtrinsicTransform(from_camera=1, to_camera=0, T_to_from=T_ext.matrix(),
                              so3=np.zeros(3), translation=np.zeros(3))]
    times = [0.1 * k for k in range(6)]
    poses = []
    for k, t in enumerate(times):
        # T_board_cam: camera ~0.5 m in front of the board, looking at it
        R = gtsam.Rot3.Rodrigues(0.05 * k, 0.03 * k, 0.02)
        poses.append(gtsam.Pose3(R, gtsam.Point3(0.06 + 0.005 * k, 0.04, -0.5)).matrix())
    poses.insert(3, np.eye(4))                      # a sanitized pose the walkers must skip
    times.insert(3, 0.25)
    result = CalibrationResult(
        cameras=cams, extrinsics=ext, reference_camera=0, time_offsets={1: 0.004},
        board_poses=poses, board_pose_times=times,
        board_points_refined=pts_true if refined else None)
    rng = np.random.default_rng(0)
    frames = {0: [], 1: []}
    idx, tl, gp = board_pose_timeline(result)
    for cid in (0, 1):
        cal_vec = k_cal_vec(dist_model, cams[cid].intrinsics, cams[cid].distortion)
        for t in [0.0, 0.1, 0.13, 0.2, 0.37, 0.5, 0.9]:     # 0.9 is off the timeline
            obs = []
            for pid in range(board.num_points()):
                tt = synced_time(t, tl, 0.0)
                b = find_bracketing_poses(tt, tl, idx)
                if b is None:
                    uv = np.array([100.0, 100.0])
                else:
                    uv = project_observation(
                        board_point=pts_true[pid], bp0=gp[b[0]], bp1=gp[b[1]], time_meas=tt,
                        time_0=b[2], time_1=b[3], cal_vec=cal_vec, distortion_model=dist_model,
                        T_ext=None if cid == 0 else T_ext, td=0.0 if cid == 0 else 0.004,
                        tr=cams[cid].rolling_shutter_readout, pixel_row=240.0, image_height=IMG[1])
                uv = uv + rng.normal(0.0, 0.05, 2)
                obs.append(Observation(point_id=pid, pixel=uv, row=240.0))
            frames[cid].append(FrameObservations(camera_id=cid, timestamp=t, observations=obs,
                                                 image_size=IMG))
    return result, frames, board


def _old_residuals(result, all_frames, board, image_sizes):
    """The pre-G2 walker, verbatim in substance: nominal `board.point_3d`, `project_observation`."""
    ext_map = {e.from_camera: gtsam.Pose3(e.T_to_from) for e in result.extrinsics}
    cams = {c.camera_id: c for c in result.cameras}
    idx, tl, gp = board_pose_timeline(result)
    out = {}
    for cid, frames in all_frames.items():
        cam = cams[cid]
        cal_vec = k_cal_vec(cam.distortion_model, cam.intrinsics, cam.distortion)
        is_ref = cid == result.reference_camera
        per = []
        for frame in frames:
            b = None
            if len(tl) >= 2:
                t = synced_time(frame.timestamp, tl, result.snap_sync_tolerance_s)
                b = find_bracketing_poses(t, tl, idx)
            if b is None:
                per.append([None] * len(frame.observations))
                continue
            row = []
            for obs in frame.observations:
                p = project_observation(
                    board_point=board.point_3d(obs.point_id), bp0=gp[b[0]], bp1=gp[b[1]],
                    time_meas=t, time_0=b[2], time_1=b[3], cal_vec=cal_vec,
                    distortion_model=cam.distortion_model,
                    T_ext=None if is_ref else ext_map.get(cid),
                    td=0.0 if is_ref else float(result.time_offsets.get(cid, 0.0)),
                    tr=float(cam.rolling_shutter_readout), pixel_row=obs.row,
                    image_height=image_sizes.get(cid, (640, 480))[1])
                row.append(None if p is None else float(np.linalg.norm(p - obs.pixel)))
            per.append(row)
        out[cid] = per
    return out


@pytest.mark.parametrize("model", ["radtan", "equidistant", "kb4"])
def test_without_refined_board_residuals_are_bitwise_unchanged(model):
    result, frames, board = _setup(refined=False, dist_model=model)
    sizes = {0: IMG, 1: IMG}
    new = compute_per_observation_residuals(result, frames, board, sizes)
    old = _old_residuals(result, frames, board, sizes)
    assert new == old                                    # exact float equality, None in place
    assert any(r is None for fr in new[0] for r in fr)   # the off-timeline frame
    assert not ResultProjector(result, board, sizes).uses_refined_board


def test_board_point_is_refined_when_set_else_nominal():
    result, _, board = _setup(refined=True)
    p = ResultProjector(result, board)
    assert p.uses_refined_board
    for pid in (0, 7, 34):
        np.testing.assert_array_equal(p.board_point(pid), result.board_points_refined[pid])
    result.board_points_refined = None
    p = ResultProjector(result, board)
    for pid in (0, 7, 34):
        np.testing.assert_array_equal(p.board_point(pid), board.point_3d(pid))


def test_refined_board_scores_corners_against_the_fitted_board():
    """Frames seen on a board with mm heights: against the refined board the residuals are the
    noise; against the flat board (the pre-G2 behavior) they carry the height parallax."""
    result, frames, board = _setup(refined=True)
    sizes = {0: IMG, 1: IMG}
    new = compute_per_observation_residuals(result, frames, board, sizes)
    flat = _old_residuals(result, frames, board, sizes)
    r_new = np.array([r for c in new.values() for fr in c for r in fr if r is not None])
    r_flat = np.array([r for c in flat.values() for fr in c for r in fr if r is not None])
    assert len(r_new) == len(r_flat) > 0
    assert np.sqrt(np.mean(r_new ** 2)) < 0.12           # ~ the 0.05 px/axis noise
    assert np.sqrt(np.mean(r_flat ** 2)) > 3 * np.sqrt(np.mean(r_new ** 2))


def test_report_and_corner_accounting_use_the_refined_board(monkeypatch, tmp_path):
    """The HTML report's residuals and the corner accounting walk the same projection."""
    import mvis.analysis.generate_report as gr
    from mvis.calibration.corner_accounting import _PoseTrack
    result, frames, board = _setup(refined=True)
    sizes = {0: IMG, 1: IMG}
    seen = {}
    monkeypatch.setattr(gr, "generate_html_report", lambda **kw: seen.update(kw))
    gr.generate_calibration_report(result, frames, board, {}, sizes,
                                   output_path=str(tmp_path / "r.html"))
    res = compute_per_observation_residuals(result, frames, board, sizes)
    proj = ResultProjector(result, board, sizes)
    for cid in (0, 1):
        # the report leaves out frames the solve had no pose for (ref frames off a node; 0.9 is
        # off the timeline for both)
        solved = [proj.solved_bracket(proj.camera(cid), f.timestamp) is not None
                  for f in frames[cid]]
        assert solved.count(False) == (3 if cid == 0 else 1)
        want = [r for fr, ok in zip(res[cid], solved) if ok for r in fr if r is not None]
        # the report takes hypot(ex, ey), the filter norm(proj - obs): equal to rounding
        np.testing.assert_allclose(seen["error_data"][cid]["errors"], want, rtol=1e-14, atol=0)
    track = _PoseTrack(result, board, sizes)
    b = track.bracket(0.13)
    obs = frames[1][2].observations[5]
    p_cam, ok = track.corner(1, b, obs, track.projector.camera(1).cal_vec, "radtan")
    assert ok
    T = track.projector.observation_pose(
        track.projector.camera(1), obs, (b[0], track.poses[b[1]], track.poses[b[2]], b[3], b[4]))
    q = result.board_points_refined[5]
    np.testing.assert_array_equal(
        p_cam, np.asarray(T.transformTo(gtsam.Point3(*map(float, q))), float).ravel()[:3])


def test_regenerate_rebuilds_the_refined_board_from_stored_heights(tmp_path):
    import yaml
    from mvis.analysis.regenerate_html import _apply_stored_board_shape
    board = _board()
    h = _heights(board)
    y = tmp_path / "run.yaml"
    y.write_text(yaml.safe_dump({"board_shape": {"heights_m": [float(v) for v in h]}}))
    result, _, _ = _setup(refined=False)
    assert _apply_stored_board_shape(result, y, board)
    want = board.all_points_3d()
    want[:, 2] += h
    np.testing.assert_array_equal(result.board_points_refined, want)
    y.write_text(yaml.safe_dump({"cameras": []}))
    result, _, _ = _setup(refined=False)
    assert not _apply_stored_board_shape(result, y, board)
    assert result.board_points_refined is None


def test_solved_bracket_skips_frames_off_every_pose_node():
    """A frame the solve dropped (no pose node of its own) is not scored: the ref camera always,
    a non-ref camera when the solve snapped (no td). Interpolating it between ref poses scores a
    pose the solve never fitted (mvis_t265 stereo: gated frames pushed the report RMS to ~6 px)."""
    result, _, board = _setup(refined=False)
    proj = ResultProjector(result, board, {0: IMG, 1: IMG})
    ref, other = proj.camera(0), proj.camera(1)
    assert proj.solved_bracket(ref, 0.1) is not None          # on its own pose node
    assert proj.solved_bracket(ref, 0.5) is not None          # the last node
    assert proj.bracket(0.13) is not None
    assert proj.solved_bracket(ref, 0.13) is None              # between nodes: dropped
    assert proj.solved_bracket(other, 0.13) is not None        # td solve interpolates non-ref
    result.time_offsets, result.snap_sync_tolerance_s = {}, 0.008
    proj = ResultProjector(result, board, {0: IMG, 1: IMG})
    other = proj.camera(1)
    assert proj.solved_bracket(other, 0.103) is not None       # snaps to 0.1
    assert proj.solved_bracket(other, 0.13) is None            # no ref pose within tolerance


def test_regenerate_restores_snap_sync_from_time_offsets():
    from mvis.analysis.regenerate_html import _restore_snap_sync
    from mvis.calibration.pipeline_cam import PipelineConfig
    result, _, _ = _setup(refined=False)
    _restore_snap_sync(result)
    assert result.snap_sync_tolerance_s == 0.0                 # td solve: interpolates
    result.time_offsets = {}
    _restore_snap_sync(result)
    assert result.snap_sync_tolerance_s == PipelineConfig.pair_sync_tolerance_s
