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
"""Task 1: aux_imu_factor_hz config field + _aux_knot_pairs helper + full-aux decimation.

Tests:
  1. Helper unit tests (TDD, written before implementation).
  2. Graph-count test: decimated run has fewer factors/variables than default-off.
"""

import numpy as np
import gtsam
import pytest

from mvis.calibration.pipeline_cam_imu import _aux_knot_pairs
from tests.helpers.sim_scenarios import (_AUX_WIN as _WIN, _make_aux_scenario, _TRAJ,
                                         _IMG, _RADTAN, _BG, _BA, _T_CtoI, _TD, _CAM_HZ,
                                         _IMU_HZ)

# ---------------------------------------------------------------------------
# Part 1 — helper unit tests (written BEFORE the helper exists, TDD RED->GREEN)
# ---------------------------------------------------------------------------

def test_aux_knot_pairs_default_off_is_stride1():
    t = np.arange(0.0, 1.0 + 1e-9, 1/30)          # 31 knots, 30 Hz
    knots, pairs = _aux_knot_pairs(t, 0.0)         # off (aux_imu_stride defaults to 0)
    assert knots == list(range(len(t)))            # every knot
    assert pairs == [(k, k+1) for k in range(len(t)-1)]   # every interval


def test_aux_knot_pairs_stride_param_wins():
    """aux_imu_stride>0 must win over aux_imu_factor_hz (even if factor_hz=0)."""
    t = np.arange(0.0, 1.0 + 1e-9, 1/30)          # 31 knots, 30 Hz
    knots, pairs = _aux_knot_pairs(t, 0.0, aux_imu_stride=2)
    assert all(kb - ka == 2 for ka, kb in pairs[:-1])


def test_aux_knot_pairs_decimates_to_rate():
    t = np.arange(0.0, 1.0 + 1e-9, 1/30)           # 30 Hz, K=30 intervals
    knots, pairs = _aux_knot_pairs(t, 10.0)        # 10 Hz -> stride 3
    assert knots[0] == 0 and knots[-1] == len(t)-1 # first + last always present
    assert all(kb - ka == 3 for ka, kb in pairs[:-1])     # stride-3 spans
    assert len(pairs) <= (len(t)-1)//3 + 1         # ~3x fewer factors


def test_aux_knot_pairs_remainder_last_interval_shorter():
    t = np.arange(0.0, 1.0 + 1e-9, 1/29)           # 30 knots, K=29 (not div by 3)
    knots, pairs = _aux_knot_pairs(t, 10.0)
    assert knots[-1] == len(t)-1                    # final knot included
    assert pairs[-1][1] == len(t)-1                 # last factor reaches the end
    assert pairs[-1][0] == knots[-2]                # ...from the previous knot


# ---------------------------------------------------------------------------
# Part 2 — graph-count test (default-off identity + decimation drops factors)
# ---------------------------------------------------------------------------


def _build_graph(scenario_data, aux_imu_factor_hz, aux_imu_stride=None):
    """Run seed+init then call build_joint_graph directly; return (graph, values).

    aux_imu_stride=None means "use the config field default" (currently 2).
    Pass aux_imu_stride=1 explicitly to build the no-decimation reference.
    """
    from mvis.detection.chessboard import Chessboard
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, build_joint_graph,
        _decimate, _assign_brackets, _densify_velocities, _cam_model,
        _cam_pin_tr, _cam_pin_extr, _cam_factor_hz, _corner_mask,
    )
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.calibration.cam_imu_init import (
        initialize_cam_imu, initialize_imu_imu, pnp_pose_track,
    )
    import gtsam, numpy as np

    t_cam = scenario_data['t_cam']
    cam   = scenario_data['cam']
    s     = scenario_data['s']
    sa    = scenario_data['sa']
    board_pts = scenario_data['board_pts']

    cfg_kwargs = dict(state_hz=0.0, sigma_px=0.5, huber_px=2.0,
                      imu_intrinsics="pinned",
                      aux_imu_factor_hz=aux_imu_factor_hz)
    if aux_imu_stride is not None:
        cfg_kwargs['aux_imu_stride'] = aux_imu_stride
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              **cfg_kwargs)
    cal_vec = _RADTAN
    model = None  # default

    t_kept, T_CtoB_kept, _ = pnp_pose_track(
        t_cam, list(cam.ids), list(cam.uv), board_pts, cal_vec)
    seed = initialize_cam_imu(t_kept, T_CtoB_kept, s.t, s.w_m, s.a_m,
                              **cfg.init_kwargs)

    keep = _decimate(t_kept, cfg.state_hz)
    t_states = np.asarray(t_kept, float)[keep]
    T_CtoB_states = [T for T, kp in zip(T_CtoB_kept, keep) if kp]

    g_inW = np.array([0.0, 0.0, -9.80665])
    T_CtoI_inv = seed.T_CtoI.inverse()
    T_init = [seed.T_BinW.compose(T_CtoB).compose(T_CtoI_inv)
              for T_CtoB in T_CtoB_states]
    v_init = _densify_velocities(t_states, T_init, seed, s.t, s.w_m, s.a_m,
                                 cfg, g_inW)

    calib_bases = [CameraCalib(seed.T_CtoI, seed.td, 0.0, cal_vec)]
    image_sizes = [_IMG]

    keep_f = _decimate(np.asarray(t_kept, float), _cam_factor_hz({'cal_vec': cal_vec}, cfg))
    t_frames = np.asarray(t_kept, float)[keep_f]
    j, dt_frame, ok_f = _assign_brackets(t_frames, seed.td, t_states, cfg.max_interp_dt)
    board_pts_all = board_pts
    idx_by_time = {float(t): i for i, t in enumerate(np.asarray(t_cam, float))}
    frames = []
    for m, t in enumerate(t_frames):
        if not ok_f[m]:
            continue
        i = idx_by_time.get(float(t))
        if i is None:
            continue
        ids = np.asarray(cam.ids[i], int)
        uv  = np.asarray(cam.uv[i], float).reshape(-1, 2)
        cm = _corner_mask(ids, cfg.corner_subsample)
        if np.sum(cm) < 6:
            continue
        frames.append((uv[cm], board_pts_all[ids[cm]], int(j[m]), float(dt_frame[m])))
    frames_per_cam = [frames]

    # aux seed
    aux_sd = initialize_imu_imu(s.t, s.w_m, sa.t, sa.w_m, b_g_base=seed.b_g)
    aux_data = [dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux', seed=aux_sd)]

    graph, values, _, _, _ = build_joint_graph(
        t_states, frames_per_cam, T_init, v_init,
        seed, calib_bases, image_sizes,
        s.t, s.w_m, s.a_m, cfg,
        aux_data=aux_data)
    return graph, values


@pytest.fixture(scope="module")
def _aux_scenario():
    return _make_aux_scenario()


def test_decimation_reduces_aux_factors_and_states(_aux_scenario):
    """Stride-2-decimated run must have strictly fewer factors AND variables than
    the stride-1 (no-decimation) reference. The reference pins aux_imu_stride=1
    because the config default is now 2; the decimated run uses config default
    (stride=2, which wins over aux_imu_factor_hz in the resolution order)."""
    g_full, v_full = _build_graph(_aux_scenario, aux_imu_factor_hz=0.0,
                                  aux_imu_stride=1)
    g_dec,  v_dec  = _build_graph(_aux_scenario, aux_imu_factor_hz=10.0)
    # No aux_imu_stride kwarg → config default aux_imu_stride=2 takes effect.

    assert g_dec.size() < g_full.size(),   \
        f"Expected fewer factors under decimation: {g_dec.size()} vs {g_full.size()}"
    assert v_dec.size() < v_full.size(),   \
        f"Expected fewer variables under decimation: {v_dec.size()} vs {v_full.size()}"


def test_default_aux_imu_stride_decimates(_aux_scenario):
    """The config-default aux_imu_stride=2 must produce strictly fewer graph
    factors AND variables than an explicit stride-1 (full-rate) reference build.
    This validates that the new default actually changes behavior."""
    g_ref, v_ref = _build_graph(_aux_scenario, aux_imu_factor_hz=0.0,
                                aux_imu_stride=1)
    g_def, v_def = _build_graph(_aux_scenario, aux_imu_factor_hz=0.0)  # default stride 2
    assert g_def.size() < g_ref.size() and v_def.size() < v_ref.size(), (
        f"Expected default stride-2 to decimate: "
        f"graph {g_def.size()} vs ref {g_ref.size()}, "
        f"values {v_def.size()} vs ref {v_ref.size()}"
    )


def test_decimated_full_aux_solve_completes(_aux_scenario):
    """End-to-end solve with a FULL aux IMU at aux_imu_factor_hz=10 must run AND
    extract results without crashing. Exercises solve + result-assembly, which
    requests _VBs(s, ka)/_VBs(s, kb) at the DECIMATED knots — a regression guard
    against the result loop assuming consecutive (k, k+1) states."""
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, run_cam_imu_calibration_multi)

    sd = _aux_scenario
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=30.0, sigma_px=0.5, huber_px=2.0,
                            imu_intrinsics="pinned", aux_imu_factor_hz=10.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=sd['t_cam'], ids_list=list(sd['cam'].ids),
              uv_list=list(sd['cam'].uv), cal_vec=_RADTAN, image_size=_IMG)],
        sd['board_pts'], sd['s'].t, sd['s'].w_m, sd['s'].a_m,
        aux_imus=[dict(t_imu=sd['sa'].t, w_m=sd['sa'].w_m, a_m=sd['sa'].a_m,
                       name='aux')],
        config=cfg)

    assert res.ok
    assert len(res.imus) == 1
    m = res.imus[0]
    assert np.isfinite(m['td'])
    assert np.all(np.isfinite(np.asarray(m['T_IstoI'].matrix(), float)))
    assert np.isfinite(m['chi2_dof'])


# ---------------------------------------------------------------------------
# Part 3 — gyro-only aux decimation (Task 2)
# ---------------------------------------------------------------------------

def _make_gyro_aux_scenario():
    """Return scenario kwargs for a 1-cam + base-IMU + 1-gyro-only-aux-IMU setup."""
    from mvis.detection.chessboard import Chessboard
    from mvis.imu import ImuModel, RPNG
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
    from mvis.simulation.trajectory_io import load_trajectory
    from mvis.simulation.imu_measurements import imu_times
    from mvis.simulation.scenario import make_scenario, AuxImu

    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    bg_s = np.array([-0.008, 0.015, -0.01])
    ba_s = np.array([0.02, -0.04, 0.03])
    td_s = 0.004

    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=0.0,
        aux=AuxImu(T_IstoI=T_IstoI, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_s))

    t0 = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + _WIN, 1.0 / _CAM_HZ)
    cam = sc.camera(t_cam, seed=21)
    t_imu = imu_times(t0 - 0.3, t0 + _WIN + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=21)
    t_aux = imu_times(t0 - 0.3, t0 + _WIN + 0.3, 250.0)
    sa = sc.aux_imu(t_aux, seed=22)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    return dict(
        t_cam=t_cam, cam=cam, s=s, sa=sa,
        board_pts=board_pts,
    )


def _build_gyro_graph(scenario_data, aux_imu_factor_hz, aux_imu_stride=None):
    """Same as _build_graph but with gyro_only=True in aux_data.

    aux_imu_stride=None means "use the config field default" (currently 2).
    Pass aux_imu_stride=1 explicitly to build the no-decimation reference.
    """
    from mvis.detection.chessboard import Chessboard
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, build_joint_graph,
        _decimate, _assign_brackets, _densify_velocities, _cam_model,
        _cam_pin_tr, _cam_pin_extr, _cam_factor_hz, _corner_mask,
    )
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.calibration.cam_imu_init import (
        initialize_cam_imu, initialize_imu_imu, pnp_pose_track,
    )
    import gtsam, numpy as np

    t_cam = scenario_data['t_cam']
    cam   = scenario_data['cam']
    s     = scenario_data['s']
    sa    = scenario_data['sa']
    board_pts = scenario_data['board_pts']

    cfg_kwargs = dict(state_hz=0.0, sigma_px=0.5, huber_px=2.0,
                      imu_intrinsics="pinned",
                      aux_imu_factor_hz=aux_imu_factor_hz)
    if aux_imu_stride is not None:
        cfg_kwargs['aux_imu_stride'] = aux_imu_stride
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              **cfg_kwargs)
    cal_vec = _RADTAN

    t_kept, T_CtoB_kept, _ = pnp_pose_track(
        t_cam, list(cam.ids), list(cam.uv), board_pts, cal_vec)
    seed = initialize_cam_imu(t_kept, T_CtoB_kept, s.t, s.w_m, s.a_m,
                              **cfg.init_kwargs)

    keep = _decimate(t_kept, cfg.state_hz)
    t_states = np.asarray(t_kept, float)[keep]
    T_CtoB_states = [T for T, kp in zip(T_CtoB_kept, keep) if kp]

    g_inW = np.array([0.0, 0.0, -9.80665])
    T_CtoI_inv = seed.T_CtoI.inverse()
    T_init = [seed.T_BinW.compose(T_CtoB).compose(T_CtoI_inv)
              for T_CtoB in T_CtoB_states]
    v_init = _densify_velocities(t_states, T_init, seed, s.t, s.w_m, s.a_m,
                                 cfg, g_inW)

    calib_bases = [CameraCalib(seed.T_CtoI, seed.td, 0.0, cal_vec)]
    image_sizes = [_IMG]

    keep_f = _decimate(np.asarray(t_kept, float), _cam_factor_hz({'cal_vec': cal_vec}, cfg))
    t_frames = np.asarray(t_kept, float)[keep_f]
    j, dt_frame, ok_f = _assign_brackets(t_frames, seed.td, t_states, cfg.max_interp_dt)
    board_pts_all = board_pts
    idx_by_time = {float(t): i for i, t in enumerate(np.asarray(t_cam, float))}
    frames = []
    for m, t in enumerate(t_frames):
        if not ok_f[m]:
            continue
        i = idx_by_time.get(float(t))
        if i is None:
            continue
        ids = np.asarray(cam.ids[i], int)
        uv  = np.asarray(cam.uv[i], float).reshape(-1, 2)
        cm = _corner_mask(ids, cfg.corner_subsample)
        if np.sum(cm) < 6:
            continue
        frames.append((uv[cm], board_pts_all[ids[cm]], int(j[m]), float(dt_frame[m])))
    frames_per_cam = [frames]

    # aux seed (gyro-only)
    aux_sd = initialize_imu_imu(s.t, s.w_m, sa.t, sa.w_m, b_g_base=seed.b_g)
    aux_data = [dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux',
                     gyro_only=True, seed=aux_sd)]

    graph, values, _, _, _ = build_joint_graph(
        t_states, frames_per_cam, T_init, v_init,
        seed, calib_bases, image_sizes,
        s.t, s.w_m, s.a_m, cfg,
        aux_data=aux_data)
    return graph, values


@pytest.fixture(scope="module")
def _gyro_aux_scenario():
    return _make_gyro_aux_scenario()


def test_gyro_only_aux_decimates(_gyro_aux_scenario):
    """Stride-2-decimated gyro-only run must have strictly fewer factors AND
    fewer _BGs variables than the stride-1 (no-decimation) reference.
    The reference pins aux_imu_stride=1; the decimated run uses config default
    (stride=2, which wins over aux_imu_factor_hz in the resolution order)."""
    g_full, v_full = _build_gyro_graph(_gyro_aux_scenario, aux_imu_factor_hz=0.0,
                                       aux_imu_stride=1)
    g_dec,  v_dec  = _build_gyro_graph(_gyro_aux_scenario, aux_imu_factor_hz=10.0)
    # No aux_imu_stride kwarg → config default aux_imu_stride=2 takes effect.
    assert g_dec.size() < g_full.size(), \
        f"Expected fewer factors under decimation: {g_dec.size()} vs {g_full.size()}"
    assert v_dec.size() < v_full.size(), \
        f"Expected fewer variables under decimation: {v_dec.size()} vs {v_full.size()}"


def test_decimated_gyro_only_solve_completes(_gyro_aux_scenario):
    """End-to-end solve with a GYRO-ONLY aux at aux_imu_factor_hz=10 must complete
    and extract results without crashing. Exercises result-assembly (which would
    crash without the (ka,kb,f) triple fix in the gyro-only result loop)."""
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, run_cam_imu_calibration_multi)

    sd = _gyro_aux_scenario
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=30.0, sigma_px=0.5, huber_px=2.0,
                            imu_intrinsics="pinned", aux_imu_factor_hz=10.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=sd['t_cam'], ids_list=list(sd['cam'].ids),
              uv_list=list(sd['cam'].uv), cal_vec=_RADTAN, image_size=_IMG)],
        sd['board_pts'], sd['s'].t, sd['s'].w_m, sd['s'].a_m,
        aux_imus=[dict(t_imu=sd['sa'].t, w_m=sd['sa'].w_m, a_m=sd['sa'].a_m,
                       name='aux', gyro_only=True)],
        config=cfg)

    assert res.ok
    assert len(res.imus) == 1
    m = res.imus[0]
    assert np.isfinite(m['td'])
    assert np.all(np.isfinite(np.asarray(m['T_IstoI'].matrix(), float)))
    assert np.isfinite(m['chi2_dof'])


# ---------------------------------------------------------------------------
# Part 4 — accuracy recovery under decimation (Task 3)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_decimated_aux_recovers_within_tolerance():
    """End-to-end accuracy test: decimating aux-IMU factors (aux_imu_factor_hz=
    10, stride 2 at the ~20 Hz state rate) must still recover the known aux
    extrinsic (translation), td, and Dw/Da within the SAME tolerances that the
    full-rate aux-recovery reference (test_aux_imu_joint_recovery in
    test_pipeline_cam_imu.py) uses.

    Window note: this runs on a 30 s window (NOT the module's 5 s _WIN, which is
    sized for the fast graph-count / solve-completes tests). A 30 s window — the
    same length the reference full-rate aux-recovery test uses — is required for
    the aux extrinsic/td/intrinsics to be observable to the full-rate
    tolerances. Empirically the short 5 s window is information-limited at BOTH
    full rate and 10 Hz (td ~2 ms, Dw ~0.027), so it cannot distinguish a
    decimation cost from a window-excitation cost; at 30 s both full-rate and
    decimated solves land inside the tolerances and decimation costs ~nothing
    (10 Hz: td 0.13 ms, lever 0.003 m, Dw 0.006, Da 0.003 — as good as full
    rate), which is the property this test asserts.

    Tolerances (copied verbatim from that test):
      lever arm translation:  atol = 2e-2 m
      td:                     atol = 1e-3 s
      Dw (identity):          atol = 0.02  (2 %)
      Da (identity):          atol = 0.02  (2 %)
    """
    from mvis.detection.chessboard import Chessboard
    from mvis.imu import ImuModel, RPNG
    from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
    from mvis.calibration.pipeline_cam_imu import (
        CamImuCalibConfig, run_cam_imu_calibration_multi)
    from mvis.simulation.trajectory_io import load_trajectory
    from mvis.simulation.imu_measurements import imu_times
    from mvis.simulation.scenario import make_scenario, AuxImu

    # ----- known ground-truth aux calibration -----
    T_IstoI_true = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                               gtsam.Point3(0.08, -0.04, 0.03))
    td_true    = 0.004
    lever_true = np.array([0.08, -0.04, 0.03])   # == T_IstoI_true.translation()

    # Aux IMU uses identity intrinsics → truth Dw = I, Da = I
    # Pipeline default imu_intrinsics_model is 'calyx'; use same model to decode
    # intr_xi24 so that Dw()/Da() are correctly reconstructed.
    from mvis.imu import CALYX
    aux_model = CALYX
    Dw_true   = np.eye(3)
    Da_true   = np.eye(3)

    bg_s = np.array([-0.008,  0.015, -0.01])
    ba_s = np.array([ 0.02,  -0.04,   0.03])

    # ----- well-excited calib trajectory (same as existing aux tests) -----
    # 30 s window (NOT the module's short _WIN=5.0) so the aux calib is
    # observable to the full-rate tolerances — matches the reference test.
    win = 30.0
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=0.5,
        aux=AuxImu(T_IstoI=T_IstoI_true, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_true))

    t0    = sc.traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + win, 1.0 / _CAM_HZ)
    cam   = sc.camera(t_cam, seed=21)
    t_imu = imu_times(t0 - 0.3, t0 + win + 0.3, _IMU_HZ)
    s     = sc.base_imu(t_imu, seed=21)
    t_aux = imu_times(t0 - 0.3, t0 + win + 0.3, 250.0)
    sa    = sc.aux_imu(t_aux, seed=22)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # ----- run with decimated aux factors (10 Hz / stride 2 at ~20 Hz states) -----
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              
        state_hz=30.0, sigma_px=0.5, huber_px=2.0,
        imu_intrinsics="dw_da",
        aux_imu_factor_hz=10.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=_RADTAN, image_size=_IMG)],
        board_pts, s.t, s.w_m, s.a_m,
        aux_imus=[dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux')],
        config=cfg)

    assert res.ok, f"solve failed: {res.diagnostics}"
    assert len(res.imus) == 1
    m = res.imus[0]

    # ---- tolerances from test_aux_imu_joint_recovery (test_pipeline_cam_imu.py) ----
    LEVER_TOL = 2e-2   # m          (lever arm translation)
    TD_TOL    = 1e-3   # s          (inter-IMU time offset)
    DADW_TOL  = 0.02   # (2 %)     (Dw/Da matrix entries vs identity)

    lever_rec = np.asarray(m['T_IstoI'].translation(), float)
    td_rec    = float(m['td'])
    xi24      = np.asarray(m['intr_xi24'], float)
    ii        = ImuModel.from_tangent(xi24, aux_model)
    Dw_rec    = ii.Dw()
    Da_rec    = ii.Da()

    print(f"\n[decimated-aux-recovery]"
          f"\n  lever  rec={lever_rec}  true={lever_true}"
          f"\n  td     rec={td_rec:.6f}  true={td_true:.6f}"
          f"\n  Dw_rec=\n{Dw_rec}"
          f"\n  Da_rec=\n{Da_rec}")

    np.testing.assert_allclose(
        lever_rec, lever_true, atol=LEVER_TOL,
        err_msg=f"aux lever arm: rec={lever_rec}  true={lever_true}")
    assert abs(td_rec - td_true) < TD_TOL, (
        f"aux td: rec={td_rec:+.6f}  true={td_true:+.6f}"
        f"  diff={abs(td_rec - td_true):.6f} >= {TD_TOL}")
    np.testing.assert_allclose(
        Dw_rec, Dw_true, atol=DADW_TOL,
        err_msg=f"aux Dw: rec=\n{Dw_rec}\ntrue=\n{Dw_true}")
    np.testing.assert_allclose(
        Da_rec, Da_true, atol=DADW_TOL,
        err_msg=f"aux Da: rec=\n{Da_rec}\ntrue=\n{Da_true}")
