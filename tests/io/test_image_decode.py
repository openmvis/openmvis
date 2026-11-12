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
"""Golden pixels for the one image decoder (io.image_decode), per encoding.

The expected values are worked out by hand from the rule in the module docstring, not taken
from the decoder, so a change of rule fails here first."""
import array
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mvis.io.image_decode import (MONO16_SHIFT_ABOVE, array_to_gray, image_file_to_gray,
                                  mono16_to_mono8, ros_image_to_gray)


def msg(pixels, encoding, step=None, bigendian=False, pad=0xAB, data_type=np.ndarray):
    """A raw sensor_msgs/Image with `pixels` (h, w[, c]) laid out row by row with `step`
    bytes per row; padding bytes are filled with `pad` so a decoder that reads them shows."""
    pixels = np.asarray(pixels)
    h, w = pixels.shape[:2]
    dt = pixels.dtype.newbyteorder(">" if bigendian else "<")
    rows = pixels.astype(dt).reshape(h, -1).view(np.uint8).reshape(h, -1)
    step = rows.shape[1] if step is None else step
    buf = np.full((h, step), pad, np.uint8)
    buf[:, :rows.shape[1]] = rows
    data = buf.reshape(-1)
    if data_type is bytes:
        data = data.tobytes()
    elif data_type is array.array:
        data = array.array("B", data.tobytes())
    return SimpleNamespace(height=h, width=w, encoding=encoding, step=step,
                           is_bigendian=int(bigendian), data=data)


# ---- mono8 ------------------------------------------------------------------------------

@pytest.mark.parametrize("enc", ["mono8", "8UC1"])
def test_mono8_is_the_bytes(enc):
    px = np.array([[0, 1, 2], [128, 254, 255]], np.uint8)
    got = ros_image_to_gray(msg(px, enc))
    assert got.dtype == np.uint8 and got.shape == (2, 3)
    np.testing.assert_array_equal(got, px)


def test_mono8_odd_step_skips_row_padding():
    px = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], np.uint8)
    np.testing.assert_array_equal(ros_image_to_gray(msg(px, "mono8", step=5)), px)


@pytest.mark.parametrize("data_type", [np.ndarray, bytes, array.array])
def test_data_as_rosbags_array_bytes_or_rclpy_array(data_type):
    px = np.array([[3, 4], [5, 6]], np.uint8)
    np.testing.assert_array_equal(ros_image_to_gray(msg(px, "mono8", data_type=data_type)), px)


# ---- mono16: both conventions -----------------------------------------------------------

@pytest.mark.parametrize("enc", ["mono16", "16UC1"])
def test_mono16_upper_byte_convention_shifts(enc):
    # max 0xFFF0 > 4095: the data sit in the upper byte (TUM-VI 12-bit << 4), so >> 8
    px = np.array([[0x0000, 0x0140, 0x12F0], [0x8000, 0xFEFF, 0xFFF0]], np.uint16)
    want = np.array([[0x00, 0x01, 0x12], [0x80, 0xFE, 0xFF]], np.uint8)
    np.testing.assert_array_equal(ros_image_to_gray(msg(px, enc)), want)


def test_mono16_10bit_convention_stretches_min_max_floored():
    # max 1023 <= 4095: 10-bit IR, (v - lo) * 255 // (hi - lo) with lo 0, hi 1023
    px = np.array([[0, 1, 4], [512, 1022, 1023]], np.uint16)
    want = np.array([[0, 0, 0], [127, 254, 255]], np.uint8)   # 512*255//1023 = 127
    np.testing.assert_array_equal(ros_image_to_gray(msg(px, "mono16")), want)


def test_mono16_12bit_convention_with_an_offset_floor():
    px = np.array([[100, 2000, 4095]], np.uint16)
    want = np.array([[0, 121, 255]], np.uint8)                 # 1900*255//3995 = 121
    np.testing.assert_array_equal(ros_image_to_gray(msg(px, "mono16")), want)


def test_mono16_threshold_is_4095():
    assert MONO16_SHIFT_ABOVE == 4095
    np.testing.assert_array_equal(mono16_to_mono8(np.array([[0, 4095]], np.uint16)), [[0, 255]])
    np.testing.assert_array_equal(mono16_to_mono8(np.array([[0, 4096]], np.uint16)), [[0, 16]])


def test_mono16_constant_frame_is_black():
    np.testing.assert_array_equal(mono16_to_mono8(np.full((2, 2), 700, np.uint16)), 0)
    np.testing.assert_array_equal(mono16_to_mono8(np.full((2, 2), 0, np.uint16)), 0)


def test_mono16_big_endian_flag_swaps_bytes():
    px = np.array([[0x1234, 0xABCD], [0x0100, 0xFF00]], np.uint16)
    little = ros_image_to_gray(msg(px, "mono16"))
    big = ros_image_to_gray(msg(px, "mono16", bigendian=True))
    np.testing.assert_array_equal(little, [[0x12, 0xAB], [0x01, 0xFF]])
    np.testing.assert_array_equal(big, little)


def test_mono16_odd_step_and_big_endian_together():
    px = np.array([[100, 600, 1100], [1600, 2100, 2600]], np.uint16)   # 12-bit stretch
    want = ((px.astype(int) - 100) * 255 // 2500).astype(np.uint8)
    np.testing.assert_array_equal(ros_image_to_gray(msg(px, "mono16", step=9, bigendian=True)), want)
    np.testing.assert_array_equal(want, [[0, 51, 102], [153, 204, 255]])


# ---- colour -----------------------------------------------------------------------------

RGB = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[255, 255, 255], [0, 0, 0], [200, 90, 40]]], np.uint8)
# OpenCV luma, 0.299 R + 0.587 G + 0.114 B, rounded: 76.2, 149.7, 29.1, 255, 0, 117.2
LUMA = np.array([[76, 150, 29], [255, 0, 117]], np.uint8)


def test_rgb8_luma():
    np.testing.assert_array_equal(ros_image_to_gray(msg(RGB, "rgb8")), LUMA)


@pytest.mark.parametrize("enc", ["bgr8", "8UC3"])
def test_bgr8_and_8uc3_are_bgr_order(enc):
    np.testing.assert_array_equal(ros_image_to_gray(msg(RGB[..., ::-1], enc)), LUMA)


def test_rgba8_and_bgra8_ignore_alpha():
    alpha = np.full(RGB.shape[:2] + (1,), 7, np.uint8)
    np.testing.assert_array_equal(ros_image_to_gray(msg(np.dstack([RGB, alpha]), "rgba8")), LUMA)
    bgra = np.dstack([RGB[..., ::-1], alpha])
    for enc in ("bgra8", "8UC4"):
        np.testing.assert_array_equal(ros_image_to_gray(msg(bgra, enc)), LUMA)


def test_rgb8_odd_step():
    np.testing.assert_array_equal(ros_image_to_gray(msg(RGB, "rgb8", step=11)), LUMA)


def test_rgb16_is_luma_then_the_mono16_rule():
    px16 = RGB.astype(np.uint16) << 8                       # max > 4095: upper byte
    got = ros_image_to_gray(msg(px16, "rgb16"))
    want = mono16_to_mono8(cv2.cvtColor(px16, cv2.COLOR_RGB2GRAY))
    np.testing.assert_array_equal(got, want)
    assert np.abs(got.astype(int) - LUMA).max() <= 1


# ---- bayer ------------------------------------------------------------------------------

def _mosaic(pattern, rgb, n=8, dtype=np.uint8):
    """An n x n Bayer mosaic of one flat colour, laid out as `pattern` (ROS naming)."""
    val = dict(zip("rgb", rgb))
    m = np.zeros((n, n), dtype)
    for k, ch in enumerate(pattern):
        m[k // 2::2, k % 2::2] = val[ch]
    return m


@pytest.mark.parametrize("pattern", ["rggb", "bggr", "gbrg", "grbg"])
def test_bayer8_flat_colour_gives_its_luma_in_the_interior(pattern):
    got = ros_image_to_gray(msg(_mosaic(pattern, (200, 90, 40)), f"bayer_{pattern}8"))
    assert got.shape == (8, 8) and got.dtype == np.uint8
    np.testing.assert_array_equal(got[2:-2, 2:-2], 117)     # the luma of (200, 90, 40)


def test_bayer_pattern_name_matters():
    m = _mosaic("rggb", (200, 90, 40))
    right = ros_image_to_gray(msg(m, "bayer_rggb8"))[2:-2, 2:-2]
    wrong = ros_image_to_gray(msg(m, "bayer_bggr8"))[2:-2, 2:-2]
    assert (right == 117).all()
    assert (wrong == 88).all()      # R and B swapped: 0.299*40 + 0.587*90 + 0.114*200 = 87.6


def test_bayer16_upper_byte():
    m = _mosaic("rggb", (200, 90, 40), dtype=np.uint16) << 8
    got = ros_image_to_gray(msg(m, "bayer_rggb16"))
    # luma of (51200, 23040, 10240) = 30000.7 -> 30001, >> 8 = 117
    np.testing.assert_array_equal(got[2:-2, 2:-2], 117)


# ---- errors -----------------------------------------------------------------------------

@pytest.mark.parametrize("enc", ["yuv422", "32FC1", ""])
def test_unsupported_encoding_raises(enc):
    with pytest.raises(ValueError, match="unsupported image encoding"):
        ros_image_to_gray(msg(np.zeros((2, 2), np.uint8), enc), "/cam0")


def test_short_data_raises():
    m = msg(np.zeros((4, 4), np.uint8), "mono8")
    m.data = m.data[:10]
    with pytest.raises(ValueError, match="too short"):
        ros_image_to_gray(m)


# ---- compressed + files: same pixels as the raw frame -----------------------------------

def _compressed(img, ext=".png", fmt="png"):
    ok, buf = cv2.imencode(ext, img)
    assert ok
    stamp = SimpleNamespace(sec=1, nanosec=0)
    return SimpleNamespace(format=fmt, data=buf.reshape(-1), header=SimpleNamespace(stamp=stamp))


@pytest.mark.parametrize("img,enc", [
    (np.array([[0, 9, 255], [3, 4, 5]], np.uint8), "mono8"),
    (np.array([[0x0100, 0x8000, 0xFFF0], [0, 1, 2]], np.uint16), "mono16"),
    (np.array([[0, 512, 1023], [1, 2, 3]], np.uint16), "mono16"),
    (RGB[..., ::-1].copy(), "bgr8"),
])
def test_png_reads_like_the_raw_frame_and_the_file(img, enc, tmp_path):
    raw = ros_image_to_gray(msg(img, enc))
    np.testing.assert_array_equal(ros_image_to_gray(_compressed(img), "/cam0", set()), raw)
    cv2.imwrite(str(tmp_path / "f.png"), img)
    np.testing.assert_array_equal(image_file_to_gray(tmp_path / "f.png"), raw)


def test_image_file_bgra_png_ignores_alpha(tmp_path):
    bgra = np.dstack([RGB[..., ::-1], np.full(RGB.shape[:2], 9, np.uint8)])
    cv2.imwrite(str(tmp_path / "a.png"), bgra)
    np.testing.assert_array_equal(image_file_to_gray(tmp_path / "a.png"), LUMA)


def test_image_file_unreadable_is_none(tmp_path):
    assert image_file_to_gray(tmp_path / "missing.png") is None
    (tmp_path / "junk.png").write_bytes(b"not an image")
    assert image_file_to_gray(tmp_path / "junk.png") is None


def test_array_to_gray_rejects_float():
    with pytest.raises(ValueError, match="depth"):
        array_to_gray(np.zeros((2, 2), np.float32))


def test_image_file_ignores_exif_rotation(tmp_path):
    """Results stay in the sensor's native orientation: a portrait EXIF tag
    (Orientation=6, rotate 90°) must not swap the image size."""
    from PIL import Image
    from mvis.io.image_decode import image_file_to_gray
    p = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(np.zeros((40, 60), np.uint8)).save(p, exif=exif)
    assert image_file_to_gray(p).shape == (40, 60)
