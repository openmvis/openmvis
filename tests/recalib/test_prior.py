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
"""mvis.recalib.prior: the pinned camera prior from several camera-only solves."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mvis.recalib.prior import _chordal_mean, median_prior, per_camera_prior


def _cam(fx, k1=0.1, cid=0, readout=None):
    c = {"id": cid, "model": "pinhole", "intrinsics": {"fx": fx, "fy": fx + 1, "cx": 320.0, "cy": 240.0},
         "distortion": {"model": "radtan", "coeffs": [k1, 0.0]}, "image_size": (640, 480)}
    if readout is not None:
        c["rolling_shutter_readout"] = readout
    return c


def _result(fx, rz, tx, dt, k1=0.1):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0, 0, rz]).as_matrix()
    T[0, 3] = tx
    return {"cameras": [_cam(fx, k1), _cam(fx + 10, k1, cid=1)],
            "extrinsics": {"reference_camera": 0, "transforms": [{"from": 1, "to": 0, "T_to_from": T.tolist()}]},
            "time_offsets": {"reference_camera": 0, "offsets": [{"camera": 1, "dt": dt}]},
            "reprojection_errors": [0.3]}


def test_chordal_mean_of_symmetric_rotations():
    Rs = [Rotation.from_rotvec([0, 0, s * 0.1]).as_matrix() for s in (-1, 1)]
    assert np.allclose(_chordal_mean(Rs), np.eye(3))


def test_chordal_mean_is_a_proper_rotation():
    Rs = [Rotation.from_rotvec(v).as_matrix() for v in ([0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3])]
    m = _chordal_mean(Rs)
    assert np.isclose(np.linalg.det(m), 1.0) and np.allclose(m.T @ m, np.eye(3))


def test_median_prior_takes_medians_and_drops_reprojection():
    out = median_prior([_result(400, -0.01, 0.1, 1e-3, k1=0.1), _result(402, 0.0, 0.2, 2e-3, k1=0.2),
                        _result(410, 0.01, 0.9, 9e-3, k1=0.3)], "src")
    assert out["source"] == "src" and "reprojection_errors" not in out
    assert out["cameras"][0]["intrinsics"]["fx"] == 402.0
    assert out["cameras"][1]["intrinsics"]["fx"] == 412.0
    assert out["cameras"][0]["distortion"] == {"model": "radtan", "coeffs": [0.2, 0.0]}
    assert out["cameras"][0]["rolling_shutter_readout"] == 0.0
    assert out["cameras"][0]["image_size"] == [640, 480]
    t = out["extrinsics"]["transforms"][0]
    assert t["translation"][0] == 0.2 and np.allclose(np.array(t["T_to_from"])[:3, :3], np.eye(3))
    assert np.allclose(t["so3"], 0.0)
    assert out["time_offsets"]["offsets"] == [{"camera": 1, "dt": 2e-3}]


def test_median_prior_refuses_different_camera_sets():
    a, b = _result(400, 0, 0, 0), _result(400, 0, 0, 0)
    b["cameras"][0]["model"] = "omni"
    with pytest.raises(AssertionError, match="camera sets differ"):
        median_prior([a, b], "")


def test_per_camera_prior_replaces_only_the_named_camera():
    seed = _result(400, 0.0, 0.1, 5e-3)
    solves = [{"cameras": [_cam(500 + i, cid=9)]} for i in (0, 2, 4)]
    out = per_camera_prior(seed, {1: solves}, "src")
    assert out["source"] == "src" and "reprojection_errors" not in out
    assert out["cameras"][0] == seed["cameras"][0]
    assert out["cameras"][1]["intrinsics"]["fx"] == 502.0
    assert out["cameras"][1]["id"] == 1                        # the seed's id, not the solve's
    assert out["extrinsics"] == seed["extrinsics"] and out["time_offsets"] == seed["time_offsets"]


def test_per_camera_prior_refuses_a_model_change():
    solve = {"cameras": [{**_cam(500), "model": "omni"}]}
    with pytest.raises(AssertionError, match="model differs"):
        per_camera_prior(_result(400, 0, 0, 0), {0: [solve]}, "")
