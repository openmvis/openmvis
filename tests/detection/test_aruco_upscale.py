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
"""ArUco / feature-detector upscale (cleanup plan PR F, F6): corners found in a resized
image map back through `board.scaled_to_original` (pixel centres, not x / scale), the native
detection is kept unless the upscale finds more markers, and the detectors share the lazy
`_get_detector` scaffolding. No regression row uses these detectors."""
import copy

import cv2
import numpy as np
import pytest

from mvis.detection.apriltag_board import scaled_to_original as old_path
from mvis.detection.aruco_board import ArucoBoard
from mvis.detection.aruco_features import ArucoFeatureDetector
from mvis.detection.board import scaled_to_original

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
SS = 8                       # supersampling of the rendered canvas


def _scene(marker_px, n=3, gap_px=None):
    """A 3x3 ArUco grid rendered SS x larger and area-downsampled, so each marker spans
    `marker_px` image pixels with anti-aliased edges. Returns (image, {id: 4x2 truth})."""
    gap_px = marker_px if gap_px is None else gap_px
    m, g, pad = marker_px * SS, gap_px * SS, 2 * marker_px * SS
    size = 2 * pad + n * m + (n - 1) * g
    canvas = np.full((size, size), 255, np.uint8)
    truth = {}
    for r in range(n):
        for c in range(n):
            mid = r * n + c
            y0, x0 = pad + r * (m + g), pad + c * (m + g)
            canvas[y0:y0 + m, x0:x0 + m] = cv2.aruco.generateImageMarker(DICT, mid, m)
            big = np.array([[x0, y0], [x0 + m, y0], [x0 + m, y0 + m], [x0, y0 + m]], float) - 0.5
            truth[mid] = (big + 0.5) / SS - 0.5       # edge of the big image in small pixels
    small = cv2.resize(canvas, None, fx=1 / SS, fy=1 / SS, interpolation=cv2.INTER_AREA)
    return small, truth


def _errors(obs, truth):
    return np.array([o.pixel - truth[o.point_id // 4][o.point_id % 4] for o in obs])


def test_scaled_to_original_has_one_home():
    assert old_path is scaled_to_original
    np.testing.assert_allclose(scaled_to_original([[1.5, 1.5]], 2.0), [[0.5, 0.5]])


def _forced_2x(det, small):
    """Detect with the native pass blanked, so the 2x upscale decides."""
    real = det._detect_raw
    calls = []

    def raw(gray):
        calls.append(gray.shape)
        if len(calls) == 1:
            return (), None
        return real(gray)
    det._detect_raw = raw
    obs = det.detect(small)
    assert calls[1] == (small.shape[0] * 2, small.shape[1] * 2)
    return obs


@pytest.mark.parametrize("make", [
    lambda: ArucoBoard(3, 3, 0.04, 0.04, min_tag_px=1.0),
    lambda: ArucoFeatureDetector(min_tag_px=1.0, max_scale=2.0, refine_subpixel=False),
])
def test_2x_upscale_corner_bias_below_0p05_px(make):
    """Unrefined cv2 corners scatter by up to 0.75 px on their own; their mean is the
    mapping's bias: 0.014 px now, +0.25 px with the old x / scale."""
    small, truth = _scene(24)
    obs = _forced_2x(make(), small)
    assert len(obs) == 36
    e = _errors(obs, truth)
    assert np.abs(e.mean(axis=0)).max() < 0.05
    assert np.abs((e + 0.25).mean(axis=0)).min() > 0.2      # what the old mapping gave


def test_2x_upscale_refined_corners_within_0p05_px():
    """With subpixel refinement every marker centre lands within 0.05 px (0.006 px here;
    0.256 px with the old mapping) and every corner within 0.1 px."""
    small, truth = _scene(24)
    obs = _forced_2x(ArucoFeatureDetector(min_tag_px=1.0, max_scale=2.0), small)
    assert len(obs) == 36
    e = _errors(obs, truth)
    mid = np.array([o.point_id // 4 for o in obs])
    centre = np.array([e[mid == m].mean(axis=0) for m in range(9)])
    assert np.abs(centre).max() < 0.05
    assert np.abs(centre + 0.25).min() > 0.2
    assert np.abs(e).max() < 0.1


def test_native_detection_kept_unless_upscale_finds_more():
    small, truth = _scene(14)                    # all 9 found natively, below min_tag_px
    for det in (ArucoBoard(3, 3, 0.04, 0.04, min_tag_px=40.0),
                ArucoFeatureDetector(min_tag_px=40.0, refine_subpixel=False)):
        native = det._detect_raw(small)
        assert len(native[1]) == 9
        obs = det.detect(small)
        got = {o.point_id: o.pixel for o in obs}
        for c, i in zip(native[0], native[1].flatten()):
            for k in range(4):
                assert got[int(i) * 4 + k].tobytes() == c.reshape(4, 2)[k].astype(float).tobytes()


def test_upscale_still_used_when_it_finds_more():
    small, truth = _scene(24)
    det = ArucoBoard(3, 3, 0.04, 0.04, min_tag_px=1.0)
    real = det._detect_raw
    first = []

    def raw(gray):
        c, i = real(gray)
        if not first:                            # native pass: drop all but one marker
            first.append(1)
            return c[:1], i[:1]
        return c, i
    det._detect_raw = raw
    assert len(det.detect(small)) == 36


def test_detector_is_built_once_and_dropped_on_copy():
    small, _ = _scene(24)
    for det in (ArucoBoard(3, 3, 0.04, 0.04), ArucoFeatureDetector()):
        det.detect(small)
        d = det._get_detector()
        assert det._get_detector() is d
        twin = copy.deepcopy(det)                # the pipeline copies one board per camera
        assert twin._detector is None
        assert len(twin.detect(small)) == len(det.detect(small))


def test_fingerprint_changes_so_old_caches_redetect():
    from mvis.detection.apriltag_features import AprilTagFeatureDetector
    from mvis.io.capture_key import board_fingerprint
    for det in (ArucoBoard(3, 3, 0.04, 0.04), ArucoFeatureDetector(), AprilTagFeatureDetector()):
        assert board_fingerprint(det)["_corner_map"] == "pixel_centre"


class _Tag:
    def __init__(self, tag_id, corners):
        self.tag_id, self.corners = tag_id, np.asarray(corners, float)


def _square(x0, y0, side):
    return np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]], float)


def _apriltag_with(native, upscaled):
    """An AprilTagFeatureDetector whose raw detector returns `native` on the input image and
    `upscaled(scale)` on the resized one (aprilgrid itself returns whole-pixel corners
    here, so a cv2-rendered scene cannot measure a quarter-pixel mapping bias)."""
    from mvis.detection.apriltag_features import AprilTagFeatureDetector
    det = AprilTagFeatureDetector(min_tag_px=30.0, max_scale=2.0, refine_subpixel=False)
    shapes = []

    def raw(gray):
        shapes.append(gray.shape)
        return native if len(shapes) == 1 else upscaled(gray.shape[0] / 100)
    det._detect_raw = raw
    return det, shapes


def test_apriltag_features_forced_2x_corner_error_below_0p05_px():
    """Corners found at 2x map back through pixel centres: the true corner (x, y) sits at
    2 (x + 0.5) - 0.5 in the upscaled image. The old x / scale put it 0.25 px off."""
    truth = {i: _square(10.3 + 25 * i, 40.7, 12.0) for i in range(3)}
    det, shapes = _apriltag_with(
        [], lambda s: [_Tag(i, (c + 0.5) * s - 0.5) for i, c in truth.items()])
    obs = det.detect(np.zeros((100, 100), np.uint8))
    assert shapes == [(100, 100), (200, 200)]
    e = np.array([o.pixel - truth[o.point_id // 4][o.point_id % 4] for o in obs])
    assert len(obs) == 12 and np.abs(e).max() < 0.05
    old = np.array([((truth[o.point_id // 4][o.point_id % 4] + 0.5) * 2 - 0.5) / 2 for o in obs])
    assert np.allclose(np.abs(old - [o.pixel for o in obs]), 0.25)   # what x / scale gave


def test_apriltag_features_native_wins_a_tie_and_upscale_wins_with_more():
    small = [_Tag(0, _square(10.0, 10.0, 12.0))]     # 12 px < min_tag_px: try 2x
    tie, shapes = _apriltag_with(small, lambda s: [_Tag(0, _square(21.0, 21.0, 24.0))])
    got = [o.pixel for o in tie.detect(np.zeros((100, 100), np.uint8))]
    assert len(shapes) == 2
    np.testing.assert_array_equal(np.array(got), small[0].corners)   # native kept
    more, _ = _apriltag_with(small, lambda s: [_Tag(0, _square(21.0, 21.0, 24.0)),
                                              _Tag(1, _square(61.0, 21.0, 24.0))])
    obs = more.detect(np.zeros((100, 100), np.uint8))
    assert len(obs) == 8
    np.testing.assert_allclose(obs[0].pixel, scaled_to_original([[21.0, 21.0]], 2.0)[0])
