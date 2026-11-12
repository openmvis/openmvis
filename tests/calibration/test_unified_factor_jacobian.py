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
"""The camera-only unified factor's analytic Jacobian (every key, every model)
against central finite differences of its own error (G3: the pose block used to
be a finite difference for the custom models)."""

import gtsam
import numpy as np
import pytest

from mvis.calibration.optimizer import _make_unified_factor, _sym
from mvis.core.camera_models import n_dist

CALS = {
    "radtan": [400.0, 401.0, 320.0, 240.0, -0.2, 0.05, 1e-3, -1e-3],
    "equidistant": [400.0, 401.0, 320.0, 240.0, 0.01, -0.005, 1e-3, 0.0],
    "fov": [400.0, 401.0, 320.0, 240.0, 0.9, 0.0, 0.0, 0.0],
    "double_sphere": [400.0, 401.0, 320.0, 240.0, -0.2, 0.6, 0.0, 0.0],
    "eucm": [400.0, 401.0, 320.0, 240.0, 0.6, 1.0, 0.0, 0.0],
    "ucm": [400.0, 401.0, 320.0, 240.0, 0.6, 0.0, 0.0, 0.0],
    "omni_radtan": [400.0, 401.0, 320.0, 240.0, 1.0, -0.1, 0.02, 1e-3, -1e-3],
    "kb4": [400.0, 401.0, 320.0, 240.0, 0.02, -0.01, 3e-3, -5e-4],
}
KEYS = [("B", 0, 6), ("B", 1, 6), ("L", 0, 3), ("K", 0, None), ("E", 0, 6), ("D", 0, 1),
        ("R", 0, 1)]


def _scene(rng, cal):
    """{key letter: value} of one observation: board poses B0/B1, landmark L,
    intrinsics K, extrinsic E, time offset D, readout R."""
    rot = lambda s: gtsam.Rot3.RzRyRx(*rng.uniform(-s, s, 3))
    b0 = gtsam.Pose3(rot(0.5), gtsam.Point3(*rng.uniform(-0.1, 0.1, 2), -rng.uniform(0.5, 0.8)))
    return {"B0": b0,
            "B1": b0.compose(gtsam.Pose3(rot(0.05), gtsam.Point3(*rng.uniform(-0.02, 0.02, 3)))),
            "L0": np.array([rng.uniform(0, 0.2), rng.uniform(0, 0.2), 0.0]),
            "K0": np.asarray(cal, float),
            "E0": gtsam.Pose3(rot(0.05), gtsam.Point3(*rng.uniform(-0.03, 0.03, 3))),
            "D0": np.array([0.004]), "R0": np.array([0.02])}


def _values(scene, key=None, j=0, eps=0.0):
    """gtsam Values of `scene`, with tangent coordinate `j` of `key` moved by `eps`."""
    v = gtsam.Values()
    for name, x in scene.items():
        if name == key:
            d = np.zeros(6 if isinstance(x, gtsam.Pose3) else len(x))
            d[j] = eps
            x = x.retract(d) if isinstance(x, gtsam.Pose3) else x + d
        k = _sym(name[0], int(name[1:]))
        if isinstance(x, gtsam.Pose3):
            v.insert(k, x)
        elif name[0] == "L":
            v.insert(k, gtsam.Point3(*x))
        else:
            v.insert(k, np.asarray(x, float))
    return v


@pytest.mark.parametrize("model", sorted(CALS))
def test_unified_factor_jacobian_matches_fd(model):
    rng = np.random.default_rng(3)
    cal = CALS[model]
    n = 0
    for _ in range(12):
        scene = _scene(rng, cal)
        f, _ = _make_unified_factor(
            np.array([320.0, 240.0]) + rng.normal(0, 30.0, 2), np.zeros(3), 300.0, 480.0,
            0.03, 0.0, 0.1, gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
            _sym("B", 0), _sym("B", 1), _sym("L", 0), _sym("K", 0),
            ext_key=_sym("E", 0), td_key=_sym("D", 0), tr_key=_sym("R", 0),
            distortion_model=model)
        A, _ = f.linearize(_values(scene)).jacobian()
        if not np.abs(A).any():                     # corner outside the model
            continue
        col = 0
        for key in KEYS:
            dim = key[2] or len(cal)
            for j in range(dim):
                if key[0] == "K" and j >= 4 + n_dist(model):
                    assert not A[:, col].any()      # K zero padding: never read, no gradient
                    col += 1
                    continue
                eps = 1e-6 * (max(1.0, abs(cal[j])) if key[0] == "K" else 1.0)
                name = f"{key[0]}{key[1]}"
                fd = (f.unwhitenedError(_values(scene, name, j, eps))
                      - f.unwhitenedError(_values(scene, name, j, -eps))) / (2 * eps)
                np.testing.assert_allclose(A[:, col], fd, rtol=1e-5, atol=2e-3,
                                           err_msg=f"{model} {key[0]}[{j}]")
                col += 1
        assert col == A.shape[1]
        n += 1
    assert n >= 6
