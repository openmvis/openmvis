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
"""Multi-scale AprilTag detection: corners from an upscaled pass must map back
to the source image with cv2.resize's pixel-centre convention, and native
(scale 1.0) detections win ties against upscaled copies of the same tag.

Regression for the -0.375 px (4x) / -0.25 px (2x) bias of `corners / scale`
found on 180-deg wide-FOV data.
"""
import cv2
import numpy as np
import pytest

from mvis.detection.apriltag_board import AprilTagBoard, scaled_to_original


@pytest.mark.parametrize("scale", [2.0, 3.0, 4.0])
def test_scaled_to_original_inverts_cv2_resize(scale):
    img = np.zeros((60, 80), np.float32)
    x0, y0 = 40.3, 25.7                     # sub-pixel blob centre
    yy, xx = np.mgrid[0:60, 0:80]
    img += np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 2.0 ** 2))
    up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    w = np.clip(up, 0, None)
    YY, XX = np.mgrid[0:up.shape[0], 0:up.shape[1]]
    c = np.array([[(w * XX).sum() / w.sum(), (w * YY).sum() / w.sum()]])
    assert np.allclose(scaled_to_original(c, scale), [[x0, y0]], atol=0.02)


def test_offset_is_added_after_unscaling():
    c = np.array([[9.5, 19.5]])
    assert np.allclose(scaled_to_original(c, 2.0, (100.0, 200.0)), [[104.5, 209.5]])


def _render_aprilgrid(tag_px=40, gap_px=12, rows=8, cols=8, margin=60, k=8):
    """Kalibr-style AprilGrid, rendered at k x supersampling and INTER_AREA-
    downsampled so edges are exact. Kalibr tags carry a 2-module black border
    (OpenCV's marker has 1, so pad one more module); black squares of the gap
    size sit at every grid intersection. Returns the image and the ground-truth
    outer tag corners in final pixel coords (pixel centres at integers)."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    T, G, M = tag_px * k, gap_px * k, margin * k
    mod = T // 10                                    # 6 data + 2x2 border modules
    W = 2 * M + cols * T + (cols - 1) * G
    H = 2 * M + rows * T + (rows - 1) * G
    hi = np.full((H, W), 255, np.uint8)
    gt = []
    for r in range(rows):
        for c in range(cols):
            x, y = M + c * (T + G), M + r * (T + G)
            tag = np.zeros((T, T), np.uint8)
            tag[mod:T - mod, mod:T - mod] = cv2.aruco.generateImageMarker(d, r * cols + c, T - 2 * mod)
            hi[y:y + T, x:x + T] = tag
            gt += [(xx / k - 0.5, yy / k - 0.5) for xx in (x, x + T) for yy in (y, y + T)]
    for r in range(rows + 1):
        for c in range(cols + 1):
            cx, cy = M + c * (T + G) - G, M + r * (T + G) - G
            hi[cy:cy + G, cx:cx + G] = 0
    img = cv2.resize(hi, (W // k, H // k), interpolation=cv2.INTER_AREA)
    return img, np.array(gt)


def test_small_tag_board_corners_unbiased():
    """Tags of 40 px (5 px modules) trigger the upscale pass; the detected
    corners must be unbiased against the rendered ground truth."""
    img, gt = _render_aprilgrid()
    board = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3)
    obs = board.detect(img)
    assert len(obs) >= 0.9 * 4 * 64
    px = np.array([o.pixel for o in obs])
    j = np.argmin(np.linalg.norm(px[:, None, :] - gt[None], axis=2), axis=1)
    err = px - gt[j]
    assert np.linalg.norm(err, axis=1).max() < 1.0          # matched to the right corner
    assert np.all(np.abs(err.mean(axis=0)) < 0.1), f"corner bias {err.mean(axis=0)}"


@pytest.mark.parametrize("tag_px, want_upscaled", [(24, True), (48, False)])
def test_tie_rule_depends_on_module_size(monkeypatch, tag_px, want_upscaled):
    """Frames whose tags have modules under _NATIVE_MIN_MODULE_PX keep the
    upscaled copy (native corners are the noisy ones there: TUM-VI 512, 2.4 px
    modules); frames of larger tags keep the native copy on a near-tie. Upscaled corners are
    tagged with a +0.3 px shift so the winner is visible in the output."""
    real = AprilTagBoard._detect_at_scale
    native = []

    def tagged(self, gray, scale=1.0):
        dets, s = real(self, gray, scale)
        if scale == 1.0:
            native.append(len(dets))
            return dets, s
        for d in dets:
            d.corners = [np.asarray(c) + 0.3 * scale for c in d.corners]
        return dets, s

    monkeypatch.setattr(AprilTagBoard, "_detect_at_scale", tagged)
    img, gt = _render_aprilgrid(tag_px=tag_px, gap_px=tag_px * 3 // 10)
    obs = AprilTagBoard(rows=8, cols=8, tag_size=0.07, tag_spacing=0.3).detect(img)
    assert native and native[0] >= 0.5 * 64, "native pass must see the tags"
    px = np.array([o.pixel for o in obs])
    j = np.argmin(np.linalg.norm(px[:, None, :] - gt[None], axis=2), axis=1)
    bias = (px - gt[j]).mean(axis=0)
    assert np.allclose(bias, 0.3 if want_upscaled else 0.0, atol=0.05), bias
