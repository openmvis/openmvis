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
"""Unit tests for _pass2_is_pathological.

The guardrail runs in CalibrationPipeline._rerun_after_outlier_rejection.
It reverts pass-2 to pass-1 when pass-2 produced a pathological
calibration (e.g. MIC02/03 cam1 fx flipping sign).
"""

import numpy as np

from mvis.calibration.pipeline_cam import _pass2_is_pathological
from mvis.core.types import (
    CalibrationResult, CameraParams, ReprojectionErrorStats,
)


def _cam(cid: int, fx: float = 420.0, fy: float | None = None,
         dist: tuple = (0.19, 0.04, -0.23, 0.09)) -> CameraParams:
    fy = fx if fy is None else fy
    return CameraParams(
        camera_id=cid,
        intrinsics=np.array([fx, fy, 480.0, 480.0]),
        distortion=np.array(dist),
        distortion_model="equidistant",
        image_size=(960, 960),
    )


def _err(cid: int, p95: float) -> ReprojectionErrorStats:
    return ReprojectionErrorStats(
        camera_id=cid, rms=p95 / 2, mean=p95 / 3,
        median=p95 / 4, p90=p95 * 0.9, p95=p95, max=p95 * 5,
        num_observations=10000,
    )


def _result(cams, errs) -> CalibrationResult:
    return CalibrationResult(
        cameras=cams,
        reprojection_errors=errs,
        reference_camera=0,
    )


def test_healthy_pass2_accepted():
    p1 = _result([_cam(0), _cam(1)], [_err(0, 0.3), _err(1, 0.35)])
    p2 = _result([_cam(0, 420.5), _cam(1, 421.1)],
                 [_err(0, 0.28), _err(1, 0.33)])
    assert _pass2_is_pathological(p1, p2) is False


def test_negative_fx_flagged():
    p1 = _result([_cam(0), _cam(1)], [_err(0, 0.3), _err(1, 0.35)])
    p2 = _result([_cam(0, 420.5), _cam(1, fx=-185.0)],
                 [_err(0, 0.28), _err(1, 2.0)])
    assert _pass2_is_pathological(p1, p2) is True


def test_fx_fy_asymmetry_flagged():
    # |fx - fy|/max = 200/420 = 48% > 30%
    p1 = _result([_cam(0)], [_err(0, 0.3)])
    p2 = _result([_cam(0, fx=420.0, fy=220.0)], [_err(0, 0.3)])
    assert _pass2_is_pathological(p1, p2) is True


def test_distortion_blown_flagged():
    p1 = _result([_cam(0)], [_err(0, 0.3)])
    # k1 = -6.32 exceeds |2.0| — the MIC02 cam1 pass-2 distortion signature.
    p2 = _result(
        [_cam(0, dist=(-6.32, 1.0, 1.62, -0.37))],
        [_err(0, 0.3)],
    )
    assert _pass2_is_pathological(p1, p2) is True


def test_p95_regression_flagged():
    p1 = _result([_cam(0), _cam(1)], [_err(0, 2.0), _err(1, 2.0)])
    # cam1 P95 went from 2.0 → 4.0 (2× regression)
    p2 = _result([_cam(0, 421.0), _cam(1, 421.0)],
                 [_err(0, 2.1), _err(1, 4.0)])
    assert _pass2_is_pathological(p1, p2) is True


def test_small_p95_regression_ignored():
    # 0.3 → 0.4 is 1.33× — below 1.5× threshold, should NOT trigger.
    p1 = _result([_cam(0)], [_err(0, 0.3)])
    p2 = _result([_cam(0, 420.5)], [_err(0, 0.4)])
    assert _pass2_is_pathological(p1, p2) is False


def test_mic02_pattern_reproduced():
    """Exact fingerprint of the MIC02 cam1 pathology we discovered:
    fx: +418 → -185, P95: 10 → 672, distortion k1: 0.20 → -6.32.
    All three triggers should fire, any one sufficient to revert."""
    p1 = _result(
        [_cam(0, 418.0, dist=(0.19, 0.07, -0.25, 0.10)),
         _cam(1, 418.7, dist=(0.20, 0.03, -0.20, 0.08))],
        [_err(0, 2.8), _err(1, 10.4)],
    )
    p2 = _result(
        [_cam(0, 419.1, dist=(0.19, 0.04, -0.23, 0.09)),
         _cam(1, fx=-185.0, fy=532.0,
              dist=(-6.32, 1.00, 1.62, -0.37))],
        [_err(0, 2.6), _err(1, 672.0)],
    )
    assert _pass2_is_pathological(p1, p2) is True
