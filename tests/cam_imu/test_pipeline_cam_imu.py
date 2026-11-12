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
"""End-to-end joint cam-IMU calibration pipeline test (C1).

Unlike test_cam_imu_recovery (which starts LM from perturbed GROUND TRUTH),
this drives the production path: raw simulated detections + raw IMU ->
cam_imu_init seed (no GT touched) -> joint graph -> LM -> scored against the
simulator truth only at the end. IMU intrinsics identity + pinned (phase A).
"""


import numpy as np
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.imu import ImuModel, RPNG, CALYX
from mvis.calibration.factors.camera_intrinsics import _cal_from_vector
from mvis.calibration.factors.lie_groups import log_so3
from mvis.calibration.pipeline_cam_imu import (
    CamImuCalibConfig, GaugeFactorPose3, run_cam_imu_calibration,
    run_cam_imu_calibration_multi, format_report, result_to_dict)
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.imu_measurements import imu_times
from mvis.simulation.scenario import make_scenario, CalibScenario
from tests.helpers.sim_scenarios import (_TD, scenario_for_pipeline as _scenario,
                                         data_for_pipeline as _data, _TRAJ, _IMG,
                                         _RADTAN, _BG, _BA, _T_CtoI, _CAM_HZ, _IMU_HZ)


def test_gauge_factor_jacobian():
    """FD check of the 4-DOF gauge Jacobian near the gauge point."""
    T0 = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -0.2, 0.4])),
                     gtsam.Point3(1.0, -2.0, 0.5))
    g = GaugeFactorPose3(T0)
    pose = T0.retract(np.array([0.003, -0.002, 0.004, 0.01, -0.02, 0.015]))
    e, J = g.error_and_jacobian(pose)
    eps = 1e-6
    J_fd = np.zeros((4, 6))
    for i in range(6):
        d = np.zeros(6); d[i] = eps
        J_fd[:, i] = (g.error(pose.retract(d)) - g.error(pose.retract(-d))) \
            / (2 * eps)
    assert np.allclose(J, J_fd, atol=1e-3), f"\n{J}\nvs FD\n{J_fd}"


@pytest.mark.parametrize("sigma_px", [0.0, 0.5])
def test_pipeline_recovers_calibration(sigma_px):
    """Raw detections + raw IMU -> seed -> joint LM. Recovers T_CtoI, td, b_g,
    b_a within tolerance; reprojection at the noise floor; chi2/dof sane."""
    sc = _scenario(sigma_px=sigma_px)
    t_cam, cam, s = _data(sc, seed=3)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)
    ids_list = list(cam.ids)
    uv_list = list(cam.uv)

    # clean: pure quadratic at FULL state rate (interp chord error scales
    # with the state interval — 20 Hz states keep it sub-mm); noisy: the
    # PRODUCTION config — Huber is required with pose-pair vision factors
    # (unbounded quadratic influence of large PnP-noise residuals can break
    # the LM linearization at wild iterates: adjacent-state relative
    # rotations approach pi where the interpolation Jacobian is singular)
    # phase A: IMU intrinsics pinned at identity (the sim injects none) — set
    # explicitly now that the config default is dw_da/calyx (unified-estimate).
    cfg = (CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=0.0, sigma_px=0.5, huber_px=0.0,
                             imu_intrinsics="pinned")
           if sigma_px == 0 else
           CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=10.0, sigma_px=sigma_px, huber_px=2.0,
                             imu_intrinsics="pinned"))
    res = run_cam_imu_calibration(t_cam, ids_list, uv_list, board_pts,
                                  _RADTAN, s.t, s.w_m, s.a_m, _IMG, config=cfg)
    print("\n" + format_report(res))
    assert res.ok

    # tolerances reflect the window's information content (the marginals on
    # this trajectory report sigma_trans ~1-3 cm — extrinsic translation and
    # b_a are the weakly-excited directions; rot/td/intrinsics are sharp)
    tol_rot = 2e-3 if sigma_px == 0 else 5e-3            # rad
    tol_trans = 1.5e-2 if sigma_px == 0 else 6e-2        # m (sigma ~3 cm here)
    tol_td = 5e-4 if sigma_px == 0 else 2e-3             # s
    d_ext = _T_CtoI.localCoordinates(res.calib.T)
    assert np.linalg.norm(d_ext[0:3]) < tol_rot, f"rot {d_ext[0:3]}"
    assert np.linalg.norm(d_ext[3:6]) < tol_trans, f"trans {d_ext[3:6]}"
    assert abs(res.calib.td - _TD) < tol_td, f"td {res.calib.td:+.5f}"
    assert abs(res.calib.tr) < 1e-9                       # pinned
    assert np.linalg.norm(res.b_g - _BG) < (1e-3 if sigma_px == 0 else 4e-3)
    assert np.linalg.norm(res.b_a - _BA) < (4e-2 if sigma_px == 0 else 6e-2)

    # fit quality: reprojection at the data floor, chi2/dof sane under noise
    floor = 0.05 if sigma_px == 0 else 1.5 * sigma_px
    assert res.diagnostics['reproj_rms_px'] < floor, \
        f"reproj {res.diagnostics['reproj_rms_px']:.3f}"
    if sigma_px > 0:
        assert res.diagnostics['cam_chi2_dof'] < 3.0
        # noisy case: solved calib consistent with its own marginals (3 sigma)
        sig = res.calib_sigma
        if np.all(np.isfinite(sig)):
            assert np.all(np.abs(d_ext[0:3]) < 4 * sig[0:3] + 1e-4)
            assert np.all(np.abs(d_ext[3:6]) < 4 * sig[3:6] + 1e-4)

    # report serializes
    d = result_to_dict(res)
    assert d['camera']['td_s'] == pytest.approx(res.calib.td)


def test_phase_b_recovers_imu_intrinsics():
    """C3: sim data generated with NON-identity IMU intrinsics; phase-B
    ('full') unlock must pull xi_in toward the true deviation while keeping
    the spatial-temporal calibration intact. Phase A on the same data shows
    the misfit (the b_a-drift signature seen on the real bags)."""
    sigma_px = 0.5
    true_intr = ImuModel(
        x_dw=np.array([1.01, 0.02, 0.99, -0.01, 0.015, 1.02]),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.zeros(9),
        R=gtsam.Rot3.Expmap(np.array([0.02, -0.01, 0.03])).matrix(),
        model=RPNG)
    xi_true = ImuModel(model=RPNG).local_coordinates(true_intr)

    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=true_intr, b_g=_BG, b_a=_BA, image_size=_IMG,
        td_cam=_TD, sigma_px=sigma_px)
    t_cam, cam, s = _data(sc, seed=7)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # this test characterizes the RPNG parameterization (test_phase_b_recovers_
    # calyx_intrinsics covers calyx); pin the model now that the default is calyx.
    cfg_b = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=10.0, sigma_px=sigma_px, huber_px=0.0,
                              imu_intrinsics="full", imu_intrinsics_model="rpng")
    res = run_cam_imu_calibration(t_cam, list(cam.ids), list(cam.uv),
                                  board_pts, _RADTAN, s.t, s.w_m, s.a_m, _IMG,
                                  config=cfg_b)
    print("\n" + format_report(res))
    assert res.ok
    assert res.imu_intr_sigma is not None

    # spatial-temporal still recovered with I free
    d_ext = _T_CtoI.localCoordinates(res.calib.T)
    # 12.3 mrad imu and 12.4 mrad interp at worst over seeds; this seed 6.7 and 5.4 mrad
    assert np.linalg.norm(d_ext[0:3]) < 13e-3
    assert abs(res.calib.td - _TD) < 2e-3
    # xi_in moved toward truth: Dw/Da blocks (the strongly-excited ones)
    xi = res.imu_intr_xi
    err_b = np.abs(xi[0:12] - xi_true[0:12])
    err_a = np.abs(xi_true[0:12])              # phase A = staying at zero
    assert np.sum(err_b) < 0.6 * np.sum(err_a), \
        f"Dw/Da not recovered: |err| {np.sum(err_b):.4f} vs pinned " \
        f"{np.sum(err_a):.4f}\nxi    {xi[0:12]}\nxi_gt {xi_true[0:12]}"
    # ... and each freed component is consistent with its marginal
    sig = res.imu_intr_sigma
    assert np.all(np.abs(xi[0:12] - xi_true[0:12]) < 4 * sig[0:12] + 5e-3), \
        f"d {xi[0:12] - xi_true[0:12]}\nsig {sig[0:12]}"


def test_phase_b_recovers_calyx_intrinsics():
    """K2: same as the RPNG recovery test but with the CALYX model — sim data
    generated with a FULL 3x3 gyro Dw + upper-tri Da; the 'full' unlock with
    imu_intrinsics_model='calyx' must recover the absolute Dw/Da (15 freed
    params, no rotation DOF) while keeping the spatial-temporal calib intact,
    and the result must carry the model + a valid R_wtoI·U_w decomposition."""
    sigma_px = 0.5
    Dw_true = np.array([[1.01, 0.02, -0.01],
                        [0.005, 0.99, 0.015],
                        [-0.008, 0.01, 1.02]])
    true_intr = ImuModel(
        x_dw=Dw_true.flatten(),
        x_da=np.array([0.98, -0.02, 1.01, 0.01, -0.01, 0.99]),
        x_tg=np.zeros(9), model=CALYX)
    xi_true = ImuModel(model=CALYX).local_coordinates(true_intr)

    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=true_intr, b_g=_BG, b_a=_BA, image_size=_IMG,
        td_cam=_TD, sigma_px=sigma_px)
    t_cam, cam, s = _data(sc, seed=7)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=10.0, sigma_px=sigma_px, huber_px=0.0,
                            imu_intrinsics="full", imu_intrinsics_model="calyx")
    res = run_cam_imu_calibration(t_cam, list(cam.ids), list(cam.uv),
                                  board_pts, _RADTAN, s.t, s.w_m, s.a_m, _IMG,
                                  config=cfg)
    assert res.ok and res.imu_model == "calyx"
    # spatial-temporal intact with the full Dw free
    d_ext = _T_CtoI.localCoordinates(res.calib.T)
    assert np.linalg.norm(d_ext[0:3]) < 6e-3
    assert abs(res.calib.td - _TD) < 2e-3
    # Dw(9)+Da(6) deviation pulled toward truth vs staying pinned at zero
    xi = res.imu_intr_xi
    err_b = np.abs(xi[0:15] - xi_true[0:15])
    err_pinned = np.abs(xi_true[0:15])
    assert np.sum(err_b) < 0.6 * np.sum(err_pinned), \
        f"Calyx Dw/Da not recovered: |err| {np.sum(err_b):.4f} vs pinned " \
        f"{np.sum(err_pinned):.4f}"
    # the reported decomposition reconstructs the recovered full Dw
    ii = ImuModel.from_tangent(xi, "calyx")
    R_wtoI, U_w = ii.decompose()
    assert np.allclose(R_wtoI @ U_w, ii.Dw(), atol=1e-12)
    assert np.allclose(np.tril(U_w, -1), 0.0, atol=1e-12)


def test_two_camera_rig(sigma_px=0.5):
    """C5a payoff: base camera + extra camera (different T_CtoI/td, stamps
    offset by half a period -> interior lambda), SAME board and IMU. The
    multi-cam graph must recover BOTH bundles and the composed rig extrinsic
    T_C1toC0 — with the extra camera initialized purely against the base IMU
    (no FOV-overlap machinery used)."""
    import dataclasses

    T_CtoI_2 = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.03, 0.17, -0.02])),
                           gtsam.Point3(-0.12, 0.03, 0.02))
    td2 = -0.005
    sc = _scenario(sigma_px=sigma_px)
    sc2 = dataclasses.replace(sc, T_CtoI=T_CtoI_2, td_cam=td2)
    assert np.allclose(sc2.T_BinW.matrix(), sc.T_BinW.matrix())  # shared board

    t0 = sc.traj.start_time + 1.0
    win = 30.0
    t_cam1 = np.arange(t0, t0 + win, 1.0 / _CAM_HZ)
    t_cam2 = t_cam1 + 0.5 / _CAM_HZ          # half-period offset
    cam1 = sc.camera(t_cam1, seed=11)
    cam2 = sc2.camera(t_cam2, seed=12)
    t_imu = imu_times(t0 - 0.3, t0 + win + 0.3, _IMU_HZ)
    s = sc.base_imu(t_imu, seed=11)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # FULL state rate: the interpolation chord error is the dominant model
    # error on this aggressive trajectory and scales ~x4 per state-rate
    # halving (it pushes the weakly-observable common-mode td direction;
    # the sigma_td_prior seed anchor pins the remainder)
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=0.0, sigma_px=sigma_px, huber_px=2.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam1, ids_list=list(cam1.ids), uv_list=list(cam1.uv),
              cal_vec=_RADTAN, image_size=_IMG),
         dict(t_cam=t_cam2, ids_list=list(cam2.ids), uv_list=list(cam2.uv),
              cal_vec=_RADTAN, image_size=_IMG)],
        board_pts, s.t, s.w_m, s.a_m, config=cfg)
    print("\n" + format_report(res))
    assert res.ok
    assert len(res.cameras) == 2

    # base bundle. Absolute td carries the weakly-observable COMMON-MODE
    # slide pushed by the interpolation chord error (~2 ms at full state
    # rate on this aggressive trajectory; sub-ms on smooth real motion) —
    # tolerance reflects it. The RELATIVE td between cameras is
    # vision-determined and tight.
    d0 = _T_CtoI.localCoordinates(res.calib.T)
    assert np.linalg.norm(d0[0:3]) < 6e-3, f"base rot {d0[0:3]}"
    assert abs(res.calib.td - _TD) < 3e-3, f"base td {res.calib.td:+.5f}"
    # extra bundle: initialized against the base IMU only, recovered jointly
    c1 = res.cameras[1]
    d1 = T_CtoI_2.localCoordinates(c1['calib'].T)
    assert np.linalg.norm(d1[0:3]) < 7e-3, f"extra rot {d1[0:3]}"   # imu 6.1 mrad at this noise; passes at 0.02 px
    assert abs(c1['calib'].td - td2) < 3e-3, f"extra td {c1['calib'].td:+.5f}"
    # relative td: the common mode cancels
    d_td_rel = (res.calib.td - c1['calib'].td) - (_TD - td2)
    assert abs(d_td_rel) < 1e-3, f"relative td {d_td_rel:+.5f}"
    assert c1['reproj_rms_px'] < 1.5 * sigma_px
    # extra-cam seed consistency gate: gravity direction in the SHARED board
    # frame must agree between the two independent inits (yaw-invariant)
    assert c1['seed_gates']['g_dir_deg'] < 2.0, c1['seed_gates']
    assert abs(c1['seed_gates']['g_norm_ls'] - 9.81) < 0.15

    # composed rig extrinsic: common-mode trajectory error cancels
    T_rel_gt = _T_CtoI.inverse().compose(T_CtoI_2)
    d_rel = T_rel_gt.localCoordinates(c1['T_CtoC0'])
    assert np.linalg.norm(d_rel[0:3]) < 4e-3, f"rig rot {d_rel[0:3]}"
    assert np.linalg.norm(d_rel[3:6]) < 2e-2, f"rig trans {d_rel[3:6]}"
    if 'T_CtoC0_sigma' in c1 and np.all(np.isfinite(c1['T_CtoC0_sigma'])):
        sig = c1['T_CtoC0_sigma']
        assert np.all(np.abs(d_rel) < 5 * sig + 1e-3), \
            f"rig extrinsic vs sigma: {d_rel} vs {sig}"


def test_aux_imu_joint_recovery(sigma_px=0.5):
    """C6b: base camera + base IMU + one AUX IMU. The aux is initialized by
    I6 (gyro-gyro: td + rotation; lever arm starts at ZERO) and the joint
    graph must recover the full T_IstoI — including the lever arm — plus
    td_s and the aux biases."""
    from mvis.simulation.scenario import AuxImu

    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    td_s, bg_s = 0.004, np.array([-0.008, 0.015, -0.01])
    ba_s = np.array([0.02, -0.04, 0.03])
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=sigma_px,
        aux=AuxImu(T_IstoI=T_IstoI, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_s))
    t_cam, cam, s = _data(sc, win=30.0, seed=21)
    t0 = sc.traj.start_time + 1.0
    t_aux = imu_times(t0 - 0.3, t0 + 30.3, 250.0)     # different rate
    sa = sc.aux_imu(t_aux, seed=22)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # aux extrinsic/td recovery with IMU intrinsics pinned at identity (sim
    # injects none) — explicit now that the config default is dw_da/calyx.
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=0.0, sigma_px=sigma_px, huber_px=2.0,
                            imu_intrinsics="pinned")
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=_RADTAN, image_size=_IMG)],
        board_pts, s.t, s.w_m, s.a_m,
        aux_imus=[dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux')],
        config=cfg)
    assert res.ok
    assert len(res.imus) == 1
    m = res.imus[0]
    print("\naux:", {k: m[k] for k in ('td', 'chi2_dof')},
          "\nT_IstoI:", m['T_IstoI'], "\nseed:", m['seed_report'])

    assert m['seed_report']['ok']
    d = T_IstoI.localCoordinates(m['T_IstoI'])
    assert np.linalg.norm(d[0:3]) < 5e-3, f"aux rot {d[0:3]}"
    # the lever arm started at ZERO and must be recovered by the joint solve
    assert np.linalg.norm(d[3:6]) < 2e-2, f"aux lever arm {d[3:6]}"
    assert abs(m['td'] - td_s) < 1e-3, f"aux td {m['td']:+.5f}"
    assert np.linalg.norm(m['b_g_s'] - bg_s) < 4e-3
    assert np.linalg.norm(m['b_a_s'] - ba_s) < 4e-2
    assert m['chi2_dof'] < 3.0


def test_aux_imu_gyro_only_recovery(sigma_px=0.5):
    """C6b gyro-only (`gyro_only=True`): the aux is calibrated from its GYRO
    alone via `ImuFactorACI_Multi_In_SO3`. The solve must recover the
    extrinsic ROTATION + td (a gyro can't see the lever arm or accel intr),
    leaving the lever arm and accel bias pinned at ZERO."""
    from mvis.simulation.scenario import AuxImu

    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    td_s, bg_s = 0.004, np.array([-0.008, 0.015, -0.01])
    ba_s = np.array([0.02, -0.04, 0.03])
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=sigma_px,
        aux=AuxImu(T_IstoI=T_IstoI, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_s))
    t_cam, cam, s = _data(sc, win=30.0, seed=21)
    t0 = sc.traj.start_time + 1.0
    t_aux = imu_times(t0 - 0.3, t0 + 30.3, 250.0)
    sa = sc.aux_imu(t_aux, seed=22)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # factor_hz=0.0 -> dense vision factors (one per accepted frame, == state_hz
    # fallback). The cam-IMU default is now 10.0; this gyro-scale accuracy test
    # relies on dense base-camera factors, so it pins the prior dense behavior.
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=0.0, sigma_px=sigma_px, huber_px=2.0,
                            imu_intrinsics="dw_da", factor_hz=0.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=_RADTAN, image_size=_IMG)],
        board_pts, s.t, s.w_m, s.a_m,
        aux_imus=[dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux',
                       gyro_only=True)],
        config=cfg)
    assert res.ok
    m = res.imus[0]
    print("\ngyro-only aux:", {k: m[k] for k in ('td', 'chi2_dof')},
          "\nT_IstoI:", m['T_IstoI'])
    assert m['seed_report']['ok']
    d = T_IstoI.localCoordinates(m['T_IstoI'])
    # rotation is gyro-only's PURPOSE and recovers tightly
    assert np.linalg.norm(d[0:3]) < 6e-3, f"aux rot {d[0:3]}"   # imu 5.04 mrad at this noise; passes at 0.02 px
    assert np.linalg.norm(m['b_g_s'] - bg_s) < 4e-3
    # gyro-only: lever arm and accel bias are NOT estimated -> stay at zero
    assert np.allclose(m['b_a_s'], 0.0)
    assert np.allclose(np.asarray(m['T_IstoI'].translation()), 0.0)
    # td is WEAKLY observable from the gyro alone (no accel/velocity coupling
    # to stiffen the time gauge): marginal sigma ~3.5 ms here, so the estimate
    # sits within ~2 sigma of truth rather than pinned to <1 ms like the full
    # factor. Assert it stays in that band and that the softness is reported.
    assert abs(m['td'] - td_s) < 3 * m['calib_sigma'][6], (
        f"aux td {m['td']:+.5f} vs {td_s:+.5f}, sigma {m['calib_sigma'][6]:.5f}")
    assert m['calib_sigma'][6] > 1e-3, "td expected weakly observable (gyro-only)"
    assert m['chi2_dof'] < 3.0


def test_aux_imu_large_td_window_shift():
    """Robustness: a LARGE aux-base time offset (50 ms) is recovered because
    the aux preint WINDOW is shifted by the init td_s, leaving only the small
    residual for the factor's first-order Exp(w*dt) propagation. Without the
    window shift, the 50 ms `v*td` propagation would carry ~cm-level error."""
    from mvis.simulation.scenario import AuxImu

    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    td_s = 0.050                                   # LARGE offset
    bg_s = np.array([-0.008, 0.015, -0.01])
    ba_s = np.array([0.02, -0.04, 0.03])
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA, image_size=_IMG,
        td_cam=_TD, sigma_px=0.5,
        aux=AuxImu(T_IstoI=T_IstoI, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_s))
    t_cam, cam, s = _data(sc, win=30.0, seed=21)
    t0 = traj.start_time + 1.0
    t_aux = imu_times(t0 - 0.3, t0 + 30.3, 250.0)
    sa = sc.aux_imu(t_aux, seed=22)
    pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)
    # factor_hz=0.0 -> dense vision factors (prior default). The cam-IMU default
    # is now 10.0; this large-td-shift test relies on dense base-camera factors.
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=0.0, sigma_px=0.5, huber_px=2.0, factor_hz=0.0)
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=_RADTAN, image_size=_IMG)],
        pts, s.t, s.w_m, s.a_m,
        aux_imus=[dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux')],
        config=cfg)
    m = res.imus[0]
    d = T_IstoI.localCoordinates(m['T_IstoI'])
    assert abs(m['td'] - td_s) < 2e-3, f"large aux td {m['td']:+.5f}"
    assert np.linalg.norm(d[0:3]) < 6e-3, f"rot {d[0:3]}"
    assert np.linalg.norm(d[3:6]) < 3e-2, f"lever {d[3:6]}"
    assert m['chi2_dof'] < 3.0


def test_aux_imu_does_not_bias_base_gyro_scale():
    """C7a arbiter: adding an aux IMU must NOT introduce a common-mode bias
    in the BASE gyro scale (the real-data C3->C6 +0.5% shift was suspected to
    come from the 2nd IMU). Ground-truth sim with KNOWN non-identity Dw on
    both IMUs: the base Dw recovery error is ~identical with and without the
    aux. (It also documents the genuine finding: per-axis gyro scale carries
    an excitation-limited bias up to ~0.5% even with one IMU.)"""
    from mvis.simulation.scenario import AuxImu

    base_intr = ImuModel(
        x_dw=np.array([1.010, 0.0, 1.005, 0.0, 0.0, 0.995]),
        x_da=np.array([1.002, 0.0, 0.998, 0.0, 0.0, 1.001]),
        x_tg=np.zeros(9), model=RPNG)
    aux_intr = ImuModel(
        x_dw=np.array([1.008, 0.0, 1.012, 0.0, 0.0, 0.997]),
        x_da=np.array([0.999, 0.0, 1.003, 0.0, 0.0, 1.002]),
        x_tg=np.zeros(9), model=RPNG)
    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=base_intr, b_g=_BG, b_a=_BA, image_size=_IMG, td_cam=0.0,
        sigma_px=0.5, aux=AuxImu(T_IstoI=T_IstoI, intr=aux_intr,
                                 b_g=_BG * 0.5, b_a=_BA * 0.5, td=0.003))
    t0 = traj.start_time + 1.0
    t_cam = np.arange(t0, t0 + 40.0, 1 / 20.)
    cam = sc.camera(t_cam, seed=5)
    t_b = imu_times(t0 - 0.3, t0 + 40.3, 400.); sb = sc.base_imu(t_b, seed=5)
    t_a = imu_times(t0 - 0.3, t0 + 40.3, 250.); sa = sc.aux_imu(t_a, seed=6)
    pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)
    dw = lambda xi: np.array([1 + xi[0], 1 + xi[2], 1 + xi[5]])
    gt_base = dw(ImuModel(model=RPNG).local_coordinates(base_intr))
    cam_in = [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
                   cal_vec=_RADTAN, image_size=_IMG)]
    # gyro-scale (Dw) cross-talk tolerance was tuned for the RPNG model — pin it
    # now that the default is calyx (different Dw parameterization).
    # factor_hz=0.0 -> dense vision factors (prior default). The cam-IMU default
    # is now 10.0; this gyro-scale cross-talk test relies on dense base factors.
    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=0.0, sigma_px=0.5, huber_px=0.0,
                            imu_intrinsics="dw_da", imu_intrinsics_model="rpng",
                            factor_hz=0.0)

    res1 = run_cam_imu_calibration_multi(cam_in, pts, sb.t, sb.w_m, sb.a_m,
                                         config=cfg)
    res2 = run_cam_imu_calibration_multi(
        cam_in, pts, sb.t, sb.w_m, sb.a_m,
        aux_imus=[dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux')],
        config=cfg)
    err1 = dw(res1.imu_intr_xi) - gt_base
    err2 = dw(res2.imu_intr_xi) - gt_base
    # the aux must not shift the base gyro scale (the common-mode hypothesis)
    assert np.max(np.abs(err2 - err1)) < 1.5e-3, \
        f"aux shifted base Dw: 1-IMU {err1} vs 2-IMU {err2}"


def test_intrinsics_refinement_improves_or_holds():
    """Start from slightly-off camera intrinsics (the realistic input: the
    camera-only calibration is good but not exact); the joint solve must not
    diverge and the reprojection must still reach the floor."""
    sigma_px = 0.5
    sc = _scenario(sigma_px=sigma_px)
    t_cam, cam, s = _data(sc, seed=5)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    # offset must exceed the intrinsics' marginal sigma (~2.5 px on this
    # window) or refinement has no signal — ~3 sigma per component here
    cal_off = _RADTAN.copy()
    cal_off[0] += 8.0; cal_off[1] -= 6.0    # ~2% focal error
    cal_off[2] += 5.0; cal_off[3] -= 4.0

    cfg = CamImuCalibConfig(estimate_t_ga=False,  # sim data has zero t_ga; recovery asserted in isolation
                              state_hz=10.0, sigma_px=sigma_px, huber_px=0.0)
    res = run_cam_imu_calibration(t_cam, list(cam.ids), list(cam.uv),
                                  board_pts, cal_off, s.t, s.w_m, s.a_m, _IMG,
                                  config=cfg)
    assert res.ok
    assert res.diagnostics['reproj_rms_px'] < 1.5 * sigma_px
    # the solved intrinsics moved back toward truth (free refinement works)
    err_in = np.abs(cal_off[:4] - _RADTAN[:4])
    err_out = np.abs(res.calib.intr[:4] - _RADTAN[:4])
    assert np.sum(err_out) < np.sum(err_in), \
        f"intrinsics did not improve: {err_out} vs {err_in}"


def test_aux_imu_recovery_with_tga_estimated(sigma_px=0.5):
    """aux-tga A3 coupling guard: same scenario as test_aux_imu_joint_recovery
    but with estimate_t_ga=True (the shipping default). Sim injects NO t_ga, so
    base + aux t_ga must stay ~0 and the aux extrinsic/td recovery must hold —
    guarding the known t_ga<->td trade on this well-excited trajectory."""
    from mvis.simulation.scenario import AuxImu

    T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.3, -1.2, 0.5])),
                          gtsam.Point3(0.08, -0.04, 0.03))
    td_s, bg_s = 0.004, np.array([-0.008, 0.015, -0.01])
    ba_s = np.array([0.02, -0.04, 0.03])
    traj = load_trajectory(_TRAJ, dt=0.05)
    sc = make_scenario(
        traj, t0=traj.start_time + 1.0, depth=3.0, board=Chessboard(7, 7, 0.10),
        cal=_cal_from_vector(_RADTAN, False), T_CtoI=_T_CtoI,
        imu_intr=ImuModel(model=RPNG), b_g=_BG, b_a=_BA,
        image_size=_IMG, td_cam=_TD, sigma_px=sigma_px,
        aux=AuxImu(T_IstoI=T_IstoI, intr=ImuModel(model=RPNG),
                   b_g=bg_s, b_a=ba_s, td=td_s))
    t_cam, cam, s = _data(sc, win=30.0, seed=21)
    t0 = sc.traj.start_time + 1.0
    t_aux = imu_times(t0 - 0.3, t0 + 30.3, 250.0)
    sa = sc.aux_imu(t_aux, seed=22)
    board_pts = np.asarray(Chessboard(7, 7, 0.10).all_points_3d(), float)

    cfg = CamImuCalibConfig(estimate_t_ga=True,
                            state_hz=0.0, sigma_px=sigma_px, huber_px=2.0,
                            imu_intrinsics="pinned")
    res = run_cam_imu_calibration_multi(
        [dict(t_cam=t_cam, ids_list=list(cam.ids), uv_list=list(cam.uv),
              cal_vec=_RADTAN, image_size=_IMG)],
        board_pts, s.t, s.w_m, s.a_m,
        aux_imus=[dict(t_imu=sa.t, w_m=sa.w_m, a_m=sa.a_m, name='aux')],
        config=cfg)
    assert res.ok
    m = res.imus[0]
    # aux t_ga present, ~zero (sim injects none)
    assert 't_ga' in m, "aux t_ga variable missing from result"
    assert abs(m['t_ga']) < 1.0e-3, f"aux t_ga {m['t_ga']:+.5f}"
    # base t_ga wanders ~1.6 ms on this noisy scenario (the documented
    # 2026-08-09 base-side coupling; NOT introduced by the aux column) — bound
    # it loosely; the tight zero-case lives in test_aci_tga closed-loop tests.
    base_tga = res.diagnostics.get('t_ga')
    assert base_tga is not None and abs(base_tga) < 2.5e-3, f"base t_ga {base_tga}"
    # geometry/temporal recovery must hold with the extra variables in
    d = T_IstoI.localCoordinates(m['T_IstoI'])
    assert np.linalg.norm(d[0:3]) < 5e-3, f"aux rot {d[0:3]}"
    assert np.linalg.norm(d[3:6]) < 2e-2, f"aux lever arm {d[3:6]}"
    # timing family: with base+aux t_ga free, td trades against t_ga by
    # ~1.4 ms on this noisy 30 s sim (documented coupling; geometry above is
    # unaffected). Tight td recovery is asserted with t_ga pinned in
    # test_aux_imu_joint_recovery.
    assert abs(m['td'] - td_s) < 2.5e-3, f"aux td {m['td']:+.5f}"
    assert m['chi2_dof'] < 3.0
