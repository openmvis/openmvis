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
"""`corner_accounting` describes the result as returned (plan kb4-kb8, review of PR A): after the
outlier rerun, which removes every corner that failed to project, and after final pose cleanup."""
import math
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest
import yaml

from mvis.calibration.factors.custom_projections import _kb_fold
from mvis.calibration.corner_accounting import count_corners
from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
from mvis.core.config import CameraConfig
from mvis.core.types import CalibrationResult, CameraParams, FrameObservations, Observation
from mvis.detection.chessboard import Chessboard
from tests._reference import ref_kb4
from tests.camera.test_kb4_sim import GT, _IMG, _frames

FOLDING = np.array([250.0, 251.0, 501.0, 498.0, 0.02, -0.005, 5e-4, -1.67e-3])   # folds near 97 deg
T_01 = gtsam.Pose3(gtsam.Rot3.Rz(0.01), gtsam.Point3(0.06, 0.0, 0.0))            # cam1 in cam0


def _stereo(board, seed=0, n=30):
    """Frames of two synced cameras from the frozen formula. cam0's board poses, cam1 by T_01."""
    frames0, th_max, _nb, poses = _frames(board, seed, n=n)
    pts = board.all_points_3d().astype(float)
    frames1 = []
    rng = np.random.RandomState(seed + 100)
    for f, T in zip(frames0, poses):
        obs = []
        for i, p in enumerate(pts):
            P = np.asarray(T_01.transformTo(T.transformFrom(gtsam.Point3(*p))))
            uv = np.array(ref_kb4.project(P, GT))
            if 5 <= uv[0] < _IMG[0] - 5 and 5 <= uv[1] < _IMG[1] - 5:
                obs.append(Observation(point_id=i, pixel=uv + rng.standard_normal(2) * 0.05,
                                       timestamp=f.timestamp))
        frames1.append(FrameObservations(camera_id=1, timestamp=f.timestamp, observations=obs,
                                         image_size=_IMG))
    return {0: frames0, 1: frames1}, th_max


def _run(board, frames, rerun):
    seeds = [CameraConfig(camera_id=c, distortion_model="kb4", initial_intrinsics=FOLDING[:4].tolist(),
                          initial_distortion=FOLDING[4:].tolist()) for c in (0, 1)]
    pc = PipelineConfig(camera_configs=seeds, max_iterations=8, max_iterations_single=8, pixel_sigma=1.0,
                        huber_threshold=1.345, corner_subsample="all", frozen_intrinsics_cameras={0, 1},
                        optimize_time_offset=False, max_init_frames=100,
                        init_frame_selection="uniform", reference_camera=0,
                        rerun_after_outlier_rejection=rerun,
                        distortion_models={0: "kb4", 1: "kb4"}, verbose=False)
    pipe = CalibrationPipeline(board, pc)
    return pipe, pipe.run(frames, {0: _IMG, 1: _IMG})


@pytest.fixture
def folding_after_stage2(monkeypatch):
    """Clean data makes a free kb4 fit recover its monotonic truth, so hand Stage 4 and 5 a folding
    polynomial, frozen. The joint solves, the residuals and the filter stay the real ones."""
    orig = CalibrationPipeline._stage2_single_cam_optimize

    def patched(self, all_frames, image_sizes, active, state):
        orig(self, all_frames, image_sizes, active, state)
        for cid in active:
            state.dist_coeffs[cid] = FOLDING[4:].copy()
    monkeypatch.setattr(CalibrationPipeline, "_stage2_single_cam_optimize", patched)


@pytest.fixture
def factor_count(monkeypatch):
    """Vision factors the last solve really built, per camera: a count from the graph side."""
    import mvis.calibration.optimizer as opt
    made = {}
    make, solve = opt._make_unified_factor, opt.Optimizer.solve

    def counted(*a, **k):
        made[k["cal_key"]] = made.get(k["cal_key"], 0) + 1
        return make(*a, **k)

    def fresh(self, *a, **k):
        made.clear()
        return solve(self, *a, **k)
    monkeypatch.setattr(opt, "_make_unified_factor", counted)
    monkeypatch.setattr(opt.Optimizer, "solve", fresh)
    return lambda cid: made.get(opt._sym("K", cid), 0)


def _check_counts(a, n_factors):
    """Against the graph: the corners counted as having a factor, not a remainder."""
    assert a["n_corners"] - a["removed_invalid"] + a["discarded_pose_corners"] == n_factors
    assert a["not_in_solve"] >= 0
    assert a["detected"] == (a["used"] + a["past_fold"] + a["other_invalid"] + a["outliers"]
                             + a["no_pose_corners"] + a["not_in_solve"])


@pytest.mark.slow
def test_alarm_survives_the_outlier_rerun(caplog, folding_after_stage2, factor_count):
    """The filter removes every corner past pass 1's fold before the final solve, so a count of
    that solve alone reads past_fold 0. A real stereo run with the rerun on, the default."""
    board = Chessboard(7, 9, 0.04)
    frames, th_max = _stereo(board)
    fold = _kb_fold(tuple(FOLDING[4:]))
    assert math.pi / 2 < fold < th_max
    with caplog.at_level("WARNING"):
        pipe, r = _run(board, frames, rerun=True)
    assert r is not pipe.first_pass_result, "the rerun must have been kept"
    assert pipe.first_pass_result.corner_accounting[0]["past_fold"] > 0
    for cid in (0, 1):
        a, st = r.corner_accounting[cid], pipe.outlier_stats[cid]
        assert a["past_fold"] > 0 and a["past_fold_pass1"] > 0 and a["fold_level"] == "alarm"
        assert a["max_theta_deg"] > math.degrees(fold)
        assert a["filter_removed"] == st.n_observations_before - st.n_observations_after
        assert a["outliers"] + a["removed_invalid"] <= a["filter_removed"]
        _check_counts(a, factor_count(cid))
    assert "ALARM" in caplog.text
    # one figure per fact: the log names the report's two counts, it does not add them
    a0 = r.corner_accounting[0]
    assert f"ALARM: {a0['past_fold']} observed corners" in caplog.text


@pytest.mark.slow
def test_clean_kb4_stereo_run_has_no_alarm(caplog, factor_count):
    board = Chessboard(7, 9, 0.04)
    frames, _ = _stereo(board, n=20)
    seeds = {0: "kb4", 1: "kb4"}
    pc = PipelineConfig(max_iterations=10, max_iterations_single=10, pixel_sigma=1.0,
                        huber_threshold=1.345, corner_subsample="all", optimize_time_offset=False,
                        max_init_frames=100, init_frame_selection="uniform", reference_camera=0,
                        rerun_after_outlier_rejection=True, distortion_models=seeds, verbose=False)
    pipe = CalibrationPipeline(board, pc)
    with caplog.at_level("WARNING"):
        r = pipe.run(frames, {0: _IMG, 1: _IMG})
    assert r is not pipe.first_pass_result
    for cid in (0, 1):
        a = r.corner_accounting[cid]
        assert a["past_fold"] == a["past_fold_pass1"] == 0 and a["fold_level"] != "alarm"
        assert a["max_theta_deg"] > 95.0
        _check_counts(a, factor_count(cid))
    assert "ALARM" not in caplog.text


@pytest.mark.slow
def test_stage5_gate_drop_is_not_a_filter_removal(monkeypatch, factor_count):
    """The Stage 5 gates change the filter's dict in place. A real run with a frame dropped there: it
    stays out of filter_removed, which keeps matching the filter's own stats, and of outliers."""
    import mvis.detection.quality_gate as qg
    board = Chessboard(7, 9, 0.04)
    frames, _ = _stereo(board, n=20)
    pc = PipelineConfig(max_iterations=10, max_iterations_single=10, pixel_sigma=1.0,
                        huber_threshold=1.345, corner_subsample="all", optimize_time_offset=False,
                        max_init_frames=100, init_frame_selection="uniform", reference_camera=0,
                        rerun_after_outlier_rejection=True, distortion_models={0: "kb4", 1: "kb4"},
                        verbose=False)
    pipe = CalibrationPipeline(board, pc)
    real, dropped = qg.apply_quality_gates, []

    def gate(all_frames, *a, **k):
        if hasattr(pipe, "filter_kept_frames") and not dropped:     # Stage 5's gate
            dropped.append(len(all_frames[1][7].observations))
            all_frames[1] = all_frames[1][:7] + all_frames[1][8:]
        return real(all_frames, *a, **k)
    monkeypatch.setattr(qg, "apply_quality_gates", gate)
    r = pipe.run(frames, {0: _IMG, 1: _IMG})
    assert r is not pipe.first_pass_result and dropped and dropped[0] > 0
    for cid in (0, 1):
        a, st = r.corner_accounting[cid], pipe.outlier_stats[cid]
        assert a["filter_removed"] == st.n_observations_before - st.n_observations_after
        _check_counts(a, factor_count(cid))
    assert r.corner_accounting[1]["not_in_solve"] >= dropped[0]


@pytest.mark.slow
@pytest.mark.parametrize("how", ["pathological", "solve_failed"])
def test_reverted_rerun_counts_the_first_pass_frames(monkeypatch, folding_after_stage2, how):
    """After a revert the result is pass 1's, so the count is over pass 1's frames, and nothing was
    removed from them. The rerun's frames must not leak in."""
    import mvis.calibration.pipeline_cam as pc
    board = Chessboard(7, 9, 0.04)
    frames, _ = _stereo(board, n=20)
    if how == "pathological":
        monkeypatch.setattr(pc, "_pass2_is_pathological", lambda pass1, pass2: True)
    else:
        solve = CalibrationPipeline._stage4_joint_optimize

        def fails_on_rerun(self, *a, **k):
            if k.get("max_iterations") is not None:
                raise ValueError("too few frames")
            return solve(self, *a, **k)
        monkeypatch.setattr(CalibrationPipeline, "_stage4_joint_optimize", fails_on_rerun)
    pipe, r = _run(board, frames, rerun=True)
    assert r is pipe.first_pass_result
    for cid in (0, 1):
        a = r.corner_accounting[cid]
        assert a["outliers"] == a["filter_removed"] == a["removed_invalid"] == 0
        assert a["past_fold_pass1"] == 0 and a["past_fold"] > 0 and a["fold_level"] == "alarm"
        n_first = sum(len(f.observations) for f in pipe.first_pass_frames[cid])
        assert a["n_corners"] + a["no_pose_corners"] == n_first
        assert a["not_in_solve"] >= 0


# ── the frames a solve skips, on the real optimizer ──

def test_frames_with_no_factor_are_not_counted_as_used(factor_count):
    """A frame with its own pose None or too few points gets no factor, even inside the timeline."""
    from mvis.calibration.optimizer import Optimizer, OptimizerOptions
    board = Chessboard(7, 9, 0.04)
    frames, _th, _nb, poses = _frames(board, 0, n=14)
    frames = list(frames)
    poses = [T.inverse() for T in poses]
    poses[5] = None                                              # no pose of its own
    few = board.min_points_for_pose() - 1
    frames[8] = FrameObservations(camera_id=0, timestamp=frames[8].timestamp, image_size=_IMG,
                                  observations=frames[8].observations[:few])
    opts = OptimizerOptions(max_iterations=2, pixel_sigma=1.0, huber_threshold=1.345, verbose=False)
    r = Optimizer(board, opts).solve_single_camera(frames, GT[:4].copy(), GT[4:].copy(), poses, "kb4",
                                                   _IMG, camera_id=0)
    a = r.corner_accounting[0]
    assert a["no_pose_frames"] == 2
    assert a["no_pose_corners"] == len(frames[5].observations) + few
    assert a["n_corners"] == factor_count(0)
    assert a["used"] < r.reprojection_errors[0].num_observations      # the stats keep frame 8


# ── the rules, on a stub result: point ids are off-axis angles in degrees ──

K = np.array([250.0, 251.0, 501.0, 498.0, -0.125, 0.0, 0.0, 0.0])      # folds at 93.6 deg
MONOTONIC = np.r_[K[:4], 0.0, 0.0, 0.0, 0.0]
SIZES = {0: (1000, 1000)}


class _Board:
    def point_3d(self, pid):
        t = math.radians(pid)
        return np.array([math.sin(t), 0.0, math.cos(t)])


def _ok():
    T = np.eye(4)
    T[0, 3] = 1e-4                    # not identity, which the pose sanitizer would discard
    return T


def _result(poses, times, cal=K, factored=None):
    return CalibrationResult(cameras=[CameraParams(0, cal[:4], cal[4:], "kb4", (1000, 1000))],
                             extrinsics=[], reference_camera=0, board_poses=poses,
                             board_pose_times=times,
                             factor_frames={0: set(times if factored is None else factored)})


def _frame(t, ids):
    return FrameObservations(camera_id=0, timestamp=t, image_size=(1000, 1000),
                             observations=[Observation(point_id=i, pixel=np.zeros(2), timestamp=t)
                                           for i in ids])


def _count(res, frames, **k):
    for name in ("before_filter", "after_filter", "pass1_frames"):
        if name in k:
            k[name] = {0: k[name]}
    return count_corners(res, {0: frames}, _Board(), SIZES, **k)[0]


def test_discarded_pose_counts_nothing_and_raises_no_alarm():
    frames = [_frame(0.0, [10, 20]), _frame(1.0, [10, 110]), _frame(2.0, [10, 20])]
    live = _count(_result([_ok(), _ok(), _ok()], [0.0, 1.0, 2.0]), frames)
    assert live["past_fold"] == 1 and live["fold_level"] == "alarm"
    assert live["max_theta_deg"] == pytest.approx(110.0, abs=0.1)
    a = _count(_result([_ok(), None, _ok()], [0.0, 1.0, 2.0]), frames)
    assert a["past_fold"] == 0 and a["fold_level"] != "alarm"
    assert a["no_pose_corners"] >= 2 and a["discarded_pose_corners"] == a["no_pose_corners"]
    assert a["max_theta_deg"] is None or a["max_theta_deg"] < 93.0


def _filtered():
    return dict(
        before=[_frame(0.0, [10, 20, 110]), _frame(1.0, [10, 20, 30]), _frame(2.0, [10, 40])],
        after=[_frame(0.0, [10, 20]), _frame(1.0, [10, 20])])


def test_filter_removed_corners_are_judged_at_the_final_fit():
    """Corner 110 removed and past the fold; corner 30 removed but it projects."""
    d = _filtered()
    res = _result([_ok(), _ok()], [0.0, 1.0])
    first = _result([_ok(), _ok(), _ok()], [0.0, 1.0, 2.0])
    a = _count(res, d["after"], before_filter=d["before"], after_filter=d["after"],
               pass1=first, pass1_frames=d["before"])
    assert (a["used"], a["past_fold"], a["other_invalid"]) == (4, 1, 0)
    assert a["removed_invalid"] == 1 and a["outliers"] == 3       # 30, and frame 2 at its pass 1 pose
    assert a["filter_removed"] == 8 - 4 and a["past_fold_pass1"] == 1
    assert a["fold_level"] == "alarm" and a["max_theta_deg"] == pytest.approx(110.0, abs=0.1)


def test_whole_frame_drop_is_judged_at_its_pass1_pose():
    """The filter drops a frame whole, the one furthest past the fold: it has no final pose, its
    pass 1 pose is the only estimate, and the fit is the final one."""
    frames = [_frame(0.0, [10, 20]), _frame(1.0, [110, 120]), _frame(2.0, [10, 20])]
    after = [frames[0], frames[2]]
    res = _result([_ok(), _ok()], [0.0, 2.0])
    first = _result([_ok(), _ok(), _ok()], [0.0, 1.0, 2.0])
    a = _count(res, after, before_filter=frames, after_filter=after, pass1=first, pass1_frames=frames)
    assert a["past_fold"] == 2 and a["removed_invalid"] == 2 and a["outliers"] == 0
    assert a["max_theta_deg"] == pytest.approx(120.0, abs=0.1) and a["no_pose_frames"] == 0


def test_frame_with_no_bracket_in_pass1_is_no_pose_not_outliers():
    frames = [_frame(0.0, [10, 20]), _frame(1.0, [10, 20]), _frame(5.0, [10, 20])]
    after = frames[:2]
    res = _result([_ok(), _ok()], [0.0, 1.0])
    a = _count(res, after, before_filter=frames, after_filter=after, pass1=res, pass1_frames=after)
    assert a["outliers"] == 0 and a["no_pose_frames"] == 1 and a["no_pose_corners"] == 2
    assert a["filter_removed"] == 2


def test_gate_drop_is_left_to_not_in_solve():
    """The filter kept this frame and a gate after it dropped it: no category claims it."""
    frames = [_frame(0.0, [10, 20]), _frame(1.0, [10, 20]), _frame(2.0, [10, 20])]
    solve = [frames[0], frames[2]]
    res = _result([_ok(), _ok()], [0.0, 2.0])
    a = _count(res, solve, before_filter=frames, after_filter=frames, pass1=res, pass1_frames=frames)
    assert a["used"] == 4 and a["outliers"] == a["filter_removed"] == a["no_pose_corners"] == 0


def test_bad_pass1_pose_on_a_clean_final_fit_raises_no_alarm():
    """A pose final cleanup would discard from pass 1 can show corners past the fold that were never
    seen there. It must not leave a lasting alarm, or raise max_theta."""
    frames = [_frame(0.0, [10, 20]), _frame(1.0, [110, 120]), _frame(2.0, [10, 20])]
    after = [frames[0], frames[2]]
    res = _result([_ok(), _ok()], [0.0, 2.0], cal=MONOTONIC)
    first = _result([_ok(), _ok(), _ok()], [0.0, 1.0, 2.0])
    a = _count(res, after, before_filter=frames, after_filter=after, pass1=first, pass1_frames=frames,
               pass1_bad_poses=[False, True, False])
    assert a["past_fold_pass1"] == 0 and a["past_fold"] == 0 and a["fold_level"] != "alarm"
    assert a["max_theta_deg"] < 93.0 and a["no_pose_corners"] == 2


def test_pass1_alarm_stands_when_the_final_fold_moved_past_the_corner():
    frames = [_frame(0.0, [10, 20, 110]), _frame(1.0, [10, 20])]
    after = [_frame(0.0, [10, 20]), _frame(1.0, [10, 20])]
    final = _result([_ok(), _ok()], [0.0, 1.0], cal=MONOTONIC)
    first = _result([_ok(), _ok()], [0.0, 1.0])
    a = _count(final, after, before_filter=frames, after_filter=after, pass1=first, pass1_frames=frames)
    assert a["past_fold"] == 0 and a["outliers"] == 1
    assert a["past_fold_pass1"] == 1 and a["fold_level"] == "alarm"


def test_pass1_is_not_walked_without_a_kb_camera(monkeypatch):
    import mvis.calibration.corner_accounting as ca
    frames = [_frame(0.0, [10, 20]), _frame(1.0, [10, 20])]
    res = _result([_ok(), _ok()], [0.0, 1.0], cal=np.array([250.0, 251.0, 501.0, 498.0, 0.6, 1.0]))
    res.cameras[0].distortion_model = "eucm"
    calls = []
    real = ca.count_corners
    monkeypatch.setattr(ca, "count_corners", lambda *x, **k: calls.append(1) or real(*x, **k))
    a = real(res, {0: frames}, _Board(), SIZES, before_filter={0: frames}, after_filter={0: frames},
             pass1=res, pass1_frames={0: frames})[0]
    assert calls == []
    assert a["model"] == "eucm" and a["fold_deg"] is None and a["fold_level"] == "none"
    assert a["past_fold"] == a["past_fold_pass1"] == 0 and a["used"] == 4


# ── regenerated reports keep it ──

def test_yaml_round_trip_restores_the_in_memory_shape(tmp_path):
    from mvis.analysis.html_report import _corner_accounting_rows
    from mvis.analysis.regenerate_html import _result_from_yaml
    from mvis.io.output_writer import OutputWriter
    r = _result([np.eye(4)], [1.0])
    r.factor_frames = None
    r.corner_accounting = {0: {"camera": 0, "model": "kb4", "n_corners": 12, "used": 10, "past_fold": 2,
                               "past_fold_pass1": 1, "other_invalid": 0, "no_pose_frames": 1,
                               "no_pose_corners": 40, "outliers": 3, "max_theta_deg": 101.0,
                               "max_theta_used_deg": 96.0, "fold_deg": 100.0, "fold_level": "alarm"}}
    p = tmp_path / "m_calib.yaml"
    OutputWriter.write_calibration_yaml(r, str(p))
    back = _result_from_yaml(p)
    assert back.corner_accounting == r.corner_accounting
    assert "ALARM" in "".join(_corner_accounting_rows(back))
    doc = yaml.safe_load(p.read_text())
    doc.pop("corner_accounting")
    p.write_text(yaml.safe_dump(doc))
    assert _result_from_yaml(p).corner_accounting is None         # an older file


def _stereo_kb4_run(board, frames):
    pc = PipelineConfig(max_iterations=10, max_iterations_single=10, pixel_sigma=1.0,
                        huber_threshold=1.345, corner_subsample="all", optimize_time_offset=False,
                        max_init_frames=100, init_frame_selection="uniform", reference_camera=0,
                        rerun_after_outlier_rejection=True, distortion_models={0: "kb4", 1: "kb4"},
                        verbose=False)
    pipe = CalibrationPipeline(board, pc)
    return pipe, pipe.run(frames, {0: _IMG, 1: _IMG})


def test_failed_count_cannot_change_a_calibration(monkeypatch, caplog):
    """A ValueError from the count inside the Stage 5 solve would read as "Stage 5 rerun failed" and
    silently keep pass 1."""
    import mvis.calibration.corner_accounting as ca
    board = Chessboard(7, 9, 0.04)
    frames, _ = _stereo(board, n=20)
    _, good = _stereo_kb4_run(board, frames)

    def boom(*a, **k):
        raise ValueError("kb4: expected 4 distortion coefficients")
    monkeypatch.setattr(ca, "count_corners", boom)
    with caplog.at_level("WARNING"):
        pipe, r = _stereo_kb4_run(board, frames)

    assert r is not pipe.first_pass_result
    assert "Stage 5 rerun failed" not in caplog.text
    assert "corner_accounting failed" in caplog.text
    assert r.corner_accounting is None
    for c, g in zip(r.cameras, good.cameras):
        np.testing.assert_allclose(np.r_[c.intrinsics, c.distortion],
                                   np.r_[g.intrinsics, g.distortion], atol=1e-5)
    assert r.overall_rms == pytest.approx(good.overall_rms, abs=1e-5)


def test_negative_not_in_solve_warns(caplog):
    from mvis.calibration.corner_accounting import not_in_solve
    a = dict(camera=1, detected=10, n_corners=8, no_pose_corners=3, outliers=1)
    with caplog.at_level("WARNING"):
        assert not_in_solve(a) == -2
    assert "not_in_solve is -2" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert not_in_solve(dict(a, detected=12)) == 0
    assert caplog.text == ""
