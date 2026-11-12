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
"""Dense pose knots (solve.knot_hz) — the H1 chord-error fix
(elp-stereo-imu-investigation E7).

The vision factor interpolates the pose between its two bracketing states with a
constant twist. With states only at the base-camera frames (41 ms for the ELP)
that model is wrong by ~ angular acceleration x dt^2 at the corner capture
times. knot_hz subdivides the state intervals so the same factor sees a short
chord, and each rolling-shutter row is bracketed by its own capture time.
"""
import numpy as np
import gtsam

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_interp_factor import (
    CameraCalibInterpFactor, _interp_pose)
from mvis.core.shutter import row_offset
from mvis.calibration.factors.lie_groups import log_so3
from mvis.calibration.pipeline_cam_imu import _subdivide_states
from tests.helpers.shaking_rig import (_CV, _W, _TR, _truth, _board, _rs_measure, _H)


def _frame_error(t_states, t_cap, split_rows):
    """Max reprojection error (px) of one RS frame captured at t_cap, evaluated
    with every state at TRUTH — any residual is pure interpolation-model error."""
    calib = CameraCalib(gtsam.Pose3(), 0.0, _TR, _CV, model="radtan")
    T_BinW, p_B = _board()
    uv = _rs_measure(t_cap, T_BinW, p_B, calib)
    xi = np.zeros(calib.DIM)
    rho = np.array([row_offset(v, _H, 0) for v in uv[:, 1]])
    if split_rows:
        jj = np.clip(np.searchsorted(t_states, t_cap + rho * _TR), 1,
                     len(t_states) - 1)
    else:
        jj = np.full(len(uv), np.clip(np.searchsorted(t_states, t_cap), 1,
                                      len(t_states) - 1))
    errs = []
    for j in np.unique(jj):
        g = jj == j
        f = CameraCalibInterpFactor(uv[g], p_B[g], t_cap - t_states[j - 1],
                                    t_states[j] - t_states[j - 1], calib,
                                    (_W, _H), rolling_mode=0)
        e = f.error(_truth(t_states[j - 1]), _truth(t_states[j]), T_BinW, xi)
        errs.append(np.linalg.norm(e.reshape(-1, 2), axis=1))
    return float(np.max(np.concatenate(errs)))


# ---------------------------------------------------------------------------
def test_subdivide_off_is_identity():
    t = np.array([0.0, 0.041, 0.082])
    T = [_truth(x) for x in t]
    t2, T2 = _subdivide_states(t, T, 0.0, 0.3)
    np.testing.assert_array_equal(t2, t)
    assert len(T2) == 3


def test_subdivide_keeps_stamps_and_caps_interval():
    t = np.array([0.0, 0.041, 0.083, 0.6, 0.641])     # 0.517 s gap > max_gap
    T = [_truth(x) for x in t]
    t2, T2 = _subdivide_states(t, T, 100.0, 0.3)
    assert len(t2) == len(T2)
    assert np.all(np.diff(t2) > 0)
    for x in t:                                        # original knots kept
        assert np.min(np.abs(t2 - x)) < 1e-12
    d = np.diff(t2)
    long = d > 0.3
    assert long.sum() == 1                             # the gap is left alone
    assert np.all(d[~long] <= 1 / 100.0 + 1e-12)
    # inserted knots lie on the constant-twist interpolant of their neighbours
    k = int(np.argmin(np.abs(t2 - 0.0205)))
    lam = (t2[k] - 0.0) / 0.041
    ref = _interp_pose(T[0], T[1], lam)[0]
    assert np.allclose(T2[k].matrix(), ref.matrix())


def test_chord_error_shrinks_with_dense_knots():
    """At truth: base-rate knots (41 ms) leave a pixel-level model error on an
    RS frame captured mid-interval; 125 Hz knots + per-row brackets remove it."""
    t_base = np.arange(0.0, 0.25, 0.041)
    t_cap = 0.041 * 2 + 0.012                     # capture inside an interval
    sparse = _frame_error(t_base, t_cap, split_rows=False)
    t_dense, _ = _subdivide_states(t_base, [_truth(x) for x in t_base],
                                   125.0, 0.3)
    dense = _frame_error(t_dense, t_cap, split_rows=True)
    dense_one_bracket = _frame_error(t_dense, t_cap, split_rows=False)
    assert sparse > 0.5                           # pixel-level at 41 ms
    assert dense < 0.1 * sparse
    assert dense < dense_one_bracket              # per-row brackets help


def test_chord_error_scales_with_dt_squared():
    t_cap = 0.1234
    e = []
    for dt in (0.04, 0.02):
        t = np.arange(0.0, 0.3, dt)
        e.append(_frame_error(t, t_cap, split_rows=True))
    assert 2.5 < e[0] / e[1] < 6.0                # ~4x for half the spacing
