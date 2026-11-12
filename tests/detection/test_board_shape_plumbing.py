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
"""Board shape plumbing (imu-scale-followup M2–M4): gauge factor Jacobians,
result serialisation, config wiring, and heights_file consumption."""
import json

import gtsam
import numpy as np
import pytest
import yaml

from mvis.calibration.optimizer import _board_height_gauge_factor, _sym
from mvis.core.types import CalibrationResult, CameraParams, ReprojectionErrorStats
from mvis.io.output_writer import OutputWriter


def _grid(n=4, pitch=0.05):
    xs, ys = np.meshgrid(np.arange(n) * pitch, np.arange(n) * pitch)
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])


def test_gauge_factor_zero_for_plane_and_jacobian():
    pts = _grid()
    keys = [_sym("L", i) for i in range(len(pts))]
    fac = _board_height_gauge_factor(keys, pts)
    # heights that are exactly a plane -> only the plane is removed; residual must
    # vanish for the mean / tilt part and not for a twist
    vals = gtsam.Values()
    plane = 0.002 + 0.01 * (pts[:, 0] - pts[:, 0].mean()) - 0.02 * (pts[:, 1] - pts[:, 1].mean())
    for k, p, h in zip(keys, pts, plane):
        vals.insert(k, gtsam.Point3(p[0], p[1], h))
    e_plane = np.asarray(fac.unwhitenedError(vals))
    assert abs(e_plane[0] - 0.002) < 1e-12                       # row 0 = mean height
    # twist: zero mean, zero first moments -> residual 0
    vals2 = gtsam.Values()
    x = pts[:, 0] - pts[:, 0].mean(); y = pts[:, 1] - pts[:, 1].mean()
    twist = 0.01 * x * y
    twist -= twist.mean()
    for k, p, h in zip(keys, pts, twist):
        vals2.insert(k, gtsam.Point3(p[0], p[1], h))
    assert np.allclose(fac.unwhitenedError(vals2), 0.0, atol=1e-9)
    # Jacobian vs finite differences on a random configuration
    rng = np.random.default_rng(0)
    vals3 = gtsam.Values()
    h3 = rng.normal(0, 0.003, len(pts))
    for k, p, h in zip(keys, pts, h3):
        vals3.insert(k, gtsam.Point3(p[0], p[1], h))
    lin = fac.linearize(vals3)
    A, _b = lin.jacobian()                                       # whitened (sigma 1e-6)
    sig = 1e-6
    def values_with(i, ax, step):
        v = gtsam.Values()
        for j, (k, p) in enumerate(zip(keys, pts)):
            q = np.array([p[0], p[1], h3[j]])
            if j == i:
                q[ax] += step
            v.insert(k, gtsam.Point3(*q))
        return v
    for i in rng.choice(len(pts), 4, replace=False):
        for ax in range(3):
            v = values_with(i, ax, 1e-7)
            fd = (np.asarray(fac.unwhitenedError(v)) - np.asarray(fac.unwhitenedError(vals3))) / 1e-7
            assert np.allclose(A[:, 3 * i + ax] * sig, fd, atol=1e-6)


def _result(n=8):
    cam = CameraParams(camera_id=0, intrinsics=np.array([500., 500., 320., 240.]),
                       distortion=np.zeros(4), distortion_model="radtan", image_size=(640, 480))
    h = np.linspace(-0.002, 0.002, n)
    bias = np.column_stack([np.full(n, 0.1), np.zeros(n), np.full(n, 50.0)])
    return CalibrationResult(cameras=[cam], extrinsics=[], reference_camera=0,
                             reprojection_errors=[ReprojectionErrorStats(0, 0.3, 0.3, 0.3)],
                             overall_rms=0.3, board_heights=h, board_point_bias=bias)


def test_calibration_dict_carries_board_shape_and_bias():
    d = OutputWriter.to_calibration_dict(_result())
    assert d["board_shape"]["model"] == "pose + per-corner height"
    assert len(d["board_shape"]["heights_m"]) == 8
    assert abs(d["board_shape"]["peak_to_peak_mm"] - 4.0) < 1e-9
    assert abs(d["board_point_bias"]["median_px"] - 0.1) < 1e-12
    assert d["board_point_bias"]["n_points_seen"] == 8
    json.dumps(d)                                                 # serialisable


def test_calibration_dict_without_shape_is_unchanged():
    r = _result(); r.board_heights = None; r.board_point_bias = None
    d = OutputWriter.to_calibration_dict(r)
    assert "board_shape" not in d and "board_point_bias" not in d


def test_estimate_board_points_reaches_optimizer(tmp_path):
    from mvis.core.config import CamRunConfig
    cfg = tmp_path / "cam.yaml"
    cfg.write_text("""
input:
  cameras:
  - {camera_id: 0, name: c, source_type: euroc, source_path: /nonexistent, distortion_model: radtan}
board: {board_type: apriltag, rows: 2, cols: 2, tag_size: 0.05, tag_spacing: 0.3}
estimate: {board_points: true}
output: {path: /tmp/x.yaml}
""")
    c = CamRunConfig.from_yaml(str(cfg))
    assert c.optimizer.optimize_board_points is True


def test_heights_file_yaml_and_npz(tmp_path):
    from mvis.cli.main import apply_board_heights, load_board_heights
    from types import SimpleNamespace
    pts = _grid()
    h = np.linspace(-0.003, 0.003, len(pts))
    y = tmp_path / "calibration.yaml"
    y.write_text(yaml.safe_dump({"board_shape": {"heights_m": h.tolist()}}))
    out = apply_board_heights(pts, SimpleNamespace(heights_file=str(y)))
    assert np.allclose(out[:, 2], h) and np.allclose(out[:, :2], pts[:, :2])
    z = tmp_path / "warp.npz"
    dp = np.zeros((len(pts), 3)); dp[:, 2] = h; dp[0, 2] = np.nan
    np.savez(z, dp=dp)
    hz = load_board_heights(str(z), len(pts))
    assert hz[0] == 0.0 and np.allclose(hz[1:], h[1:])
    assert apply_board_heights(pts, SimpleNamespace(heights_file=None)) is pts
    with pytest.raises(ValueError):
        load_board_heights(str(y), len(pts) + 1)


@pytest.mark.parametrize("value", [True, False])
def test_board_points_is_honoured_without_a_warning(tmp_path, caplog, value):
    """estimate.board_points reaches both solvers; nothing may call it unsupported or ignored."""
    from mvis.core.config import CamImuRunConfig, CamRunConfig
    board = "board: {board_type: apriltag, rows: 6, cols: 6, tag_size: 0.03, tag_spacing: 0.3}\n"
    est = f"estimate: {{board_points: {str(value).lower()}}}\n"
    cam = tmp_path / "cam.yaml"
    cam.write_text("input: {cameras: []}\n" + board + "solve: {}\n" + est)
    imu = tmp_path / "imu.yaml"
    imu.write_text("input: {format: euroc, path: x}\n" + board
                   + "camera_input: {path: y}\nsolve: {}\noutput: {path: z.json}\n" + est)
    with caplog.at_level("WARNING", logger="mvis"):
        assert CamRunConfig.from_yaml(str(cam)).optimizer.optimize_board_points is value
        assert CamImuRunConfig.from_yaml(str(imu)).solve.estimate_board_heights is value
    assert not [r for r in caplog.records if "board_points" in r.getMessage()]
