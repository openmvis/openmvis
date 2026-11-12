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
# tests/recalib/test_undistort.py
import numpy as np
from types import SimpleNamespace
from mvis.recalib.calib_spec import MVIS
from mvis.recalib.undistort import undistort_image_msg

def test_zero_distortion_is_near_identity():
    h, w = 8, 8
    K = np.array([4.0, 4.0, 3.5, 3.5])          # fx,fy,cx,cy
    cam = MVIS("cam0", 0.0, 0.0, "radtan", K, np.zeros(4))
    data = np.arange(h * w, dtype=np.uint8)
    msg = SimpleNamespace(height=h, width=w, encoding="mono8", step=w,
                          data=data.tobytes())
    out = undistort_image_msg(msg, cam)
    arr = np.frombuffer(out.data, np.uint8).reshape(h, w)
    # zero distortion + matching K ⇒ image essentially unchanged (interior)
    assert np.abs(arr[2:6, 2:6].astype(int) -
                  data.reshape(h, w)[2:6, 2:6].astype(int)).mean() < 2.0


def test_unknown_model_passthrough():
    h, w = 8, 8
    K = np.array([4.0, 4.0, 3.5, 3.5])
    cam = MVIS("cam0", 0.0, 0.0, "omni", K, np.zeros(4))  # model not in cv2 fast-paths
    data = np.arange(h * w, dtype=np.uint8)
    msg = SimpleNamespace(height=h, width=w, encoding="mono8", step=w, data=data.tobytes())
    out = undistort_image_msg(msg, cam)
    assert np.frombuffer(out.data, np.uint8).tolist() == data.tolist()  # unchanged


def test_non_mono8_encoding_passthrough():
    # bgr8 is 3 bytes/pixel; the mono8 reshape would crash, so we pass through.
    h, w = 8, 8
    K = np.array([4.0, 4.0, 3.5, 3.5])
    cam = MVIS("cam0", 0.0, 0.0, "radtan", K, np.array([-0.1, 0.02, 0.0, 0.0]))
    data = np.arange(h * w * 3, dtype=np.uint8)  # 3 channels
    msg = SimpleNamespace(height=h, width=w, encoding="bgr8", step=3 * w,
                          data=data.tobytes())
    out = undistort_image_msg(msg, cam)
    assert out is msg                                          # same object, untouched
    assert np.frombuffer(out.data, np.uint8).tolist() == data.tolist()  # unchanged
    assert out.encoding == "bgr8" and out.step == 3 * w


def test_undistort_rejects_models_without_cv2_path():
    import pytest
    from mvis.recalib.undistort import check_undistortable
    K = np.array([4.0, 4.0, 3.5, 3.5])
    ok = SimpleNamespace(kind="cam", cam=MVIS("cam0", 0.0, 0.0, "equidistant", K, np.zeros(4)))
    imu = SimpleNamespace(kind="imu", cam=None)
    check_undistortable({"/a": ok, "/i": imu})                 # supported: no error
    kb4 = SimpleNamespace(kind="cam", cam=MVIS("cam1", 0.0, 0.0, "kb4", K, np.zeros(4)))
    with pytest.raises(ValueError, match="kb4"):
        check_undistortable({"/a": ok, "/b": kb4})
    ucm = SimpleNamespace(kind="cam", cam=MVIS("cam2", 0.0, 0.0, "ucm", K, np.array([0.6])))
    with pytest.raises(ValueError, match="ucm"):
        check_undistortable({"/a": ok, "/c": ucm})
