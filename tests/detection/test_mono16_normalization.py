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
"""Tests for mono16 → mono8 auto-detection in the ROS bag loader.

Two real-world conventions exist for mono16 images and the loader must
distinguish them:

1. **10/12-bit IR padded into u16** (e.g. TUM RGB-D IR sensors):
   raw values clamped to [0, 1023] or [0, 4095]. Per-frame min-max
   normalization stretches to full 0-255 range.

2. **u8 left-shifted into the upper byte** (e.g. TUM-VI 1024_16
   calibration bags): raw values span the full u16 range
   (typically ~800 to ~65500). Min-max compresses everything to
   near-black; correct conversion is `>> 8` to recover the original u8.

Bug discovered 2026-05-01: TUM-VI 1024 detection found 0/436 corners
because the loader applied min-max to a left-shifted u8 image, producing
mean=0.0 (essentially all black). Fix auto-detects via `max > 4095` →
left-shifted, else 10/12-bit IR.

These tests directly exercise the conversion logic without the heavier
ROS bag plumbing.
"""
from __future__ import annotations

import numpy as np
import pytest


def _convert_mono16(img16: np.ndarray) -> np.ndarray:
    """Replica of the conversion logic, in `io/image_decode.py:mono16_to_mono8` since G6
    (in `data_loader.py:_stream_ros1bag` before).
    Kept here so the test contract is explicit and breaks loudly if the
    loader is ever refactored.
    """
    img_max = int(img16.max())
    if img_max > 4095:
        return (img16 >> 8).astype(np.uint8)
    lo, hi = int(img16.min()), img_max
    if hi > lo:
        # Cast to int32 to avoid uint16 overflow: (1023-50)*255 = 248k.
        scaled = (img16.astype(np.int32) - lo) * 255
        return (scaled // (hi - lo)).astype(np.uint8)
    return np.zeros_like(img16, dtype=np.uint8)


# ----- 10-bit IR (TUM RGB-D-style) -----

def test_10bit_ir_uses_minmax_normalization():
    """10-bit IR data (max in [0, 1023]) must use min-max normalization.
    Pre-fix behavior — must remain unchanged."""
    rng = np.random.default_rng(0)
    img16 = rng.integers(50, 1023, size=(64, 64), dtype=np.uint16)
    out = _convert_mono16(img16)
    assert out.dtype == np.uint8
    # Min-max normalization should always span 0-255 across the image
    assert out.min() == 0
    assert out.max() == 255
    # Mean should be around the middle of [0, 255]
    assert 30 < out.mean() < 225, f"mean={out.mean()} suggests broken normalization"


def test_12bit_ir_uses_minmax_normalization():
    """12-bit padded data (max in [0, 4095]) also uses min-max."""
    rng = np.random.default_rng(1)
    img16 = rng.integers(100, 4095, size=(64, 64), dtype=np.uint16)
    out = _convert_mono16(img16)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_constant_ir_image_returns_zeros():
    """An IR image with no contrast (all same value) must not crash."""
    img16 = np.full((32, 32), 500, dtype=np.uint16)
    out = _convert_mono16(img16)
    assert out.dtype == np.uint8
    assert out.shape == (32, 32)
    assert np.all(out == 0)


# ----- TUM-VI 1024 (u8 left-shifted into upper byte) -----

def test_left_shifted_u8_uses_right_shift():
    """TUM-VI 1024 mono16: raw u8 left-shifted by 8 (≈ ×256).
    Must use `>> 8` recovery, NOT min-max (which produces near-black)."""
    # Realistic TUM-VI grayscale: u8 image left-shifted by 8 bits
    rng = np.random.default_rng(42)
    original_u8 = rng.integers(20, 240, size=(64, 64), dtype=np.uint8)
    img16 = (original_u8.astype(np.uint16) << 8) + rng.integers(0, 255, (64, 64))
    out = _convert_mono16(img16)
    assert out.dtype == np.uint8
    # Should recover the original u8 to within ±1 (lower byte is noise)
    np.testing.assert_allclose(out, original_u8, atol=1,
                                err_msg="Right-shift recovery failed")
    # Mean should be in a meaningful range, NOT near zero
    assert out.mean() > 30, (
        f"mean={out.mean():.2f} too low — left-shifted u8 was likely "
        "incorrectly min-max-normalized to near-black")


def test_real_tumvi_value_range_does_not_collapse_to_black():
    """Reproduce the actual failure: TUM-VI bags had u16 range
    [832, 65520] which min-max normalized to mean=0.0 (image too dark
    for AprilTag detection). The fix must NOT do that."""
    # Synthetic u16 image with TUM-VI-like distribution: most pixels in
    # the middle-to-high range with a wide spread.
    rng = np.random.default_rng(7)
    # Roughly model the histogram seen in the real bag
    pix = rng.integers(800, 65520, size=(128, 128), dtype=np.uint16)
    out = _convert_mono16(pix)
    assert out.dtype == np.uint8
    # Critical regression guard: image must NOT be near-black
    assert out.mean() > 50, (
        f"mean={out.mean():.2f} — looks like the buggy min-max path "
        "compressed the image to black again")
    # And there must be reasonable contrast
    assert (out.max() - out.min()) > 100, (
        "insufficient contrast — won't be detectable by AprilTag")


def test_threshold_4095_is_inclusive_of_12bit():
    """Boundary case: max == 4095 should still use min-max (12-bit IR)."""
    img16 = np.array([[0, 1000, 2000, 4095]] * 4, dtype=np.uint16)
    out = _convert_mono16(img16)
    assert out.dtype == np.uint8
    # Min-max: 0 → 0, 4095 → 255
    assert out[0, 0] == 0
    assert out[0, 3] == 255


def test_threshold_4096_switches_to_shift():
    """Boundary case: max == 4096 should use right-shift (left-shifted u8)."""
    img16 = np.array([[256, 4096]] * 2, dtype=np.uint16)
    out = _convert_mono16(img16)
    assert out.dtype == np.uint8
    # 256 >> 8 = 1, 4096 >> 8 = 16 (verifies shift path is used)
    assert out[0, 0] == 1
    assert out[0, 1] == 16


# ----- Loader-level integration smoke (synthesizes mono16 messages) -----

def test_shared_decoder_matches_helper():
    """The one decoder every reader uses since G6 (`io.image_decode.mono16_to_mono8`, reached
    from a mono16 message through `ros_image_to_gray`) must give the documented conversion
    above on both conventions."""
    from types import SimpleNamespace

    from mvis.io.image_decode import mono16_to_mono8, ros_image_to_gray

    rng = np.random.default_rng(3)
    cases = [rng.integers(50, 1023, (16, 24), dtype=np.uint16),          # 10-bit IR
             rng.integers(100, 4095, (16, 24), dtype=np.uint16),         # 12-bit
             (rng.integers(3, 255, (16, 24), dtype=np.uint16) << 8),     # u8 in the upper byte
             np.full((16, 24), 500, np.uint16)]                          # constant
    for img16 in cases:
        want = _convert_mono16(img16)
        np.testing.assert_array_equal(mono16_to_mono8(img16), want)
        msg = SimpleNamespace(height=16, width=24, encoding="mono16", is_bigendian=0, step=48,
                              data=np.frombuffer(img16.astype("<u2").tobytes(), np.uint8))
        np.testing.assert_array_equal(ros_image_to_gray(msg), want)
