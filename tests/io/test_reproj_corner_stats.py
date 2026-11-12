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
"""Kalibr-convention reprojection statistics.

A single reprojection error is the per-corner Euclidean norm in raw pixels;
the reported median/p95/RMS pool ALL corners of ALL frames (per camera and
globally) — the same population Kalibr's printErrorStatistics uses. Replaces
the previous per-frame per-axis RMS convention (which read ~1/sqrt(2) low and
weighted a 4-corner frame the same as a 200-corner one). Corners skipped by
the cheirality/depth guards are excluded (NaN), not counted as zeros.
"""
import numpy as np
import gtsam
import pytest

from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.factors.camera_interp_factor import CameraCalibInterpFactor
from mvis.calibration.factors.camera_feature_factor import CameraFeatureInterpFactor
from mvis.calibration.pipeline_cam_imu import _reproj_corner_norms, X, B0, _Cc

_PK = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.05), gtsam.Point3(0, 0, 0))
_PK1 = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.09), gtsam.Point3(0.01, 0, 0))
_TB = gtsam.Pose3(gtsam.Rot3.Rodrigues(1.3, 0.1, 0), gtsam.Point3(0, 0, 1.5))
_CV = [350., 350, 512, 512, 0.0, 0.0, 0.0, 0.0]


def _board_corners(n=12, seed=1):
    rng = np.random.default_rng(seed)
    p_B = np.zeros((n, 3))
    p_B[:, 0] = rng.uniform(-0.15, 0.15, n)
    p_B[:, 1] = rng.uniform(-0.15, 0.15, n)
    return p_B


def _interp_factor_with_offsets(offsets, seed=1, pk=_PK, pk1=_PK1):
    """Board factor whose residual at xi=0 (evaluated at pk, pk1) equals
    -offsets exactly."""
    p_B = _board_corners(len(offsets), seed)
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.array(_CV), model="radtan")
    f0 = CameraCalibInterpFactor(np.zeros((len(p_B), 2)), p_B, 0.01, 0.05,
                                 base, (1024, 1024))
    uv = f0.error(pk, pk1, _TB, np.zeros(base.DIM)).reshape(-1, 2)
    return CameraCalibInterpFactor(uv + offsets, p_B, 0.01, 0.05,
                                   base, (1024, 1024)), base


def test_interp_factor_corner_norms_match_error_rows():
    offsets = np.tile([[0.3, -0.4]], (12, 1))          # norm 0.5 each
    f, base = _interp_factor_with_offsets(offsets)
    xi = np.zeros(base.DIM)
    norms = f.corner_norms(_PK, _PK1, _TB, xi)
    ref = np.linalg.norm(f.error(_PK, _PK1, _TB, xi).reshape(-1, 2), axis=1)
    assert norms.shape == (12,)
    np.testing.assert_allclose(norms, ref)
    np.testing.assert_allclose(norms, 0.5, atol=1e-9)


def test_feature_factor_corner_norms_nan_for_guarded():
    base = CameraCalib(gtsam.Pose3(), 0.0, 0.0, np.array(_CV), model="radtan")
    P = np.array([[0.0, 0.0, 2.0],       # valid, in front
                  [0.1, 0.0, 2.0],       # valid
                  [0.0, 0.0, -1.0]])     # behind the camera: guard skips it
    f = CameraFeatureInterpFactor(np.full((3, 2), 512.0), [0, 1, 2],
                                  0.01, 0.05, base, (1024, 1024))
    norms = f.corner_norms(_PK, _PK1, np.zeros(base.DIM), P)
    assert norms.shape == (3,)
    assert np.isfinite(norms[:2]).all()
    ref = np.linalg.norm(
        f.error(_PK, _PK1, np.zeros(base.DIM), P).reshape(-1, 2), axis=1)
    np.testing.assert_allclose(norms[:2], ref[:2])
    assert np.isnan(norms[2])            # excluded, not counted as zero


def test_reproj_corner_norms_pools_corners_across_frames():
    """Stats population = concatenated per-corner norms, NOT per-frame RMS."""
    _PK2 = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.12), gtsam.Point3(0.02, 0, 0))
    fa, base = _interp_factor_with_offsets(np.tile([[0.3, -0.4]], (12, 1)), seed=1)
    fb, _ = _interp_factor_with_offsets(np.tile([[0.6, -0.8]], (4, 1)), seed=2,
                                        pk=_PK1, pk1=_PK2)
    cam_factors = [(0, 1, fa), (0, 2, fb)]
    values = gtsam.Values()
    for k, pose in [(0, _PK), (1, _PK1), (2, _PK2)]:
        values.insert(X(k), pose)
    values.insert(B0, _TB)
    values.insert(_Cc(0), np.zeros(base.DIM))

    norms, cams = _reproj_corner_norms(cam_factors, values)
    assert norms.shape == cams.shape == (16,)          # 12 + 4 corners pooled
    assert (cams == 0).all()
    np.testing.assert_allclose(np.sort(np.unique(np.round(norms, 6))), [0.5, 1.0])
    # Kalibr convention: median over the 16 corner norms (12 of 0.5, 4 of 1.0)
    assert np.median(norms) == pytest.approx(0.5)
    # NOT the old convention: median of per-frame per-axis RMS would be ~0.53
    rms_per_axis = [np.sqrt(np.mean(f.error(
        values.atPose3(X(j - 1)), values.atPose3(X(j)), _TB,
        np.zeros(base.DIM)) ** 2)) for _, j, f in cam_factors]
    assert np.median(norms) != pytest.approx(np.median(rms_per_axis))


def test_shared_corner_stats_kalibr_convention():
    """One shared helper for BOTH pipelines: stats over pooled finite corner
    norms; NaN (guard-skipped) corners excluded from the population."""
    from mvis.calibration.reproj_stats import corner_stats
    norms = np.array([0.5] * 12 + [1.0] * 4 + [np.nan])
    s = corner_stats(norms)
    pool = np.array([0.5] * 12 + [1.0] * 4)
    assert s["n_corners"] == 16
    assert s["median"] == pytest.approx(np.median(pool))
    assert s["p90"] == pytest.approx(np.percentile(pool, 90))
    assert s["p95"] == pytest.approx(np.percentile(pool, 95))
    assert s["rms"] == pytest.approx(np.sqrt(np.mean(pool ** 2)))
    assert s["mean"] == pytest.approx(np.mean(pool))
    assert s["max"] == pytest.approx(1.0)
    empty = corner_stats([np.nan])
    assert empty["n_corners"] == 0 and np.isnan(empty["median"])


def test_cam_only_stats_match_shared_helper():
    """Camera-only pipeline stats must equal the shared helper's output —
    same convention, same code path."""
    from mvis.calibration.optimizer import Optimizer
    from mvis.calibration.reproj_stats import corner_stats
    errors = [0.5] * 12 + [1.0] * 4
    st = Optimizer._errors_to_stats(None, errors, 3)
    s = corner_stats(errors)
    assert st.camera_id == 3
    assert (st.rms, st.mean, st.median, st.p90, st.p95, st.max) == \
        (s["rms"], s["mean"], s["median"], s["p90"], s["p95"], s["max"])
    assert st.num_observations == s["n_corners"]


def test_reproj_grading_rescaled_for_corner_norm_convention():
    """Grades must mean the same fit quality as before the sqrt(2) convention
    switch: thresholds scaled from per-axis (1.0/2.0 median, 2.0/4.0 p95) to
    corner-norm units. A 1.2 px corner-norm median (= 0.85 px per-axis, 'good'
    before) must still grade good."""
    from mvis.analysis.cam_imu_report import (_grade, _MED_GRADE, _P95_GRADE,
                                              _RMS_GRADE)
    assert _MED_GRADE == pytest.approx((np.sqrt(2), 2 * np.sqrt(2)))
    assert _P95_GRADE == pytest.approx((2 * np.sqrt(2), 4 * np.sqrt(2)))
    assert _RMS_GRADE == _MED_GRADE
    assert _grade(1.2, *_MED_GRADE) == "good"          # was "marginal" pre-scale
    assert _grade(2.5, *_P95_GRADE) == "good"
    assert _grade(3.0, *_MED_GRADE) == "poor"
    assert _grade(6.0, *_P95_GRADE) == "poor"


def test_report_plot_guide_lines_use_corner_stats():
    """The per-frame plot's median/p95 guide lines must match the headline
    (corner-pooled) stats when the diagnostics carry them."""
    from mvis.analysis.cam_imu_report import _plot_reproj_frames
    n = 40
    d = dict(reproj_per_frame=np.linspace(0.2, 1.0, n),
             reproj_frame_cam=np.zeros(n, int),
             reproj_median_px=0.35, reproj_p95_px=0.88)
    fig = _plot_reproj_frames(d)
    ax1 = fig.axes[0]
    hline_ys = sorted(ln.get_ydata()[0] for ln in ax1.lines
                      if len(set(ln.get_ydata())) == 1 and len(ln.get_xdata()) == 2)
    assert hline_ys == [0.35, 0.88]
