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
"""Frozen copy of the projection seam (S0, never edited):
`camera_model._gtsam_project` / `_kernel_project`, the per-model `project`
bodies of `pinhole_radtan` / `pinhole_equidistant` / the custom-kernel models,
and `projection._Projector` / `make_projector`.

`_RefCameraModel` stands in for `build_camera_model(model, cal_vec)`: it holds
the gtsam Cal3 (radtan / equidistant) or the custom kernel and reproduces each
CameraModel's `project` verbatim. The custom kernels themselves
(`custom_projections.project_*`) are imported live: the plan leaves them
untouched ("batched numpy kernels for kb8, ds, eucm, fov and omni ... later").
"""
from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.custom_projections import (   # unchanged by the plan
    project_fov, project_double_sphere, project_eucm, project_omni_radtan,
    project_kb)

# gtsam Cal3DS2/Cal3Fisheye expose a 9-col cal Jacobian [fx,fy,skew,cx,cy,d1..d4];
# our cal_vec drops skew -> 8.
_GTSAM_TO_OURS = [0, 1, 3, 4, 5, 6, 7, 8]

_N_DIST = {"radtan": 4, "equidistant": 4, "fov": 1, "double_sphere": 2,
           "eucm": 2, "omni_radtan": 5, "kb4": 4, "kb8": 8}
_KERNELS = {"fov": project_fov, "double_sphere": project_double_sphere,
            "eucm": project_eucm, "omni_radtan": project_omni_radtan,
            "kb4": project_kb, "kb8": project_kb}


def _gtsam_project(cal, p_cam, jac):
    """Camera-frame project via a gtsam Cal3 at IDENTITY pose. jac=True ->
    (uv, J_point[2x3 wrt p_cam], J_intr[2x8 our order]); invalid -> (nan, None,
    None) (jac) or nan (no jac)."""
    cam_cls = (gtsam.PinholeCameraCal3Fisheye if isinstance(cal, gtsam.Cal3Fisheye)
               else gtsam.PinholeCameraCal3DS2)
    pt = gtsam.Point3(*p_cam)
    if not jac:
        try:
            return np.asarray(cam_cls(gtsam.Pose3(), cal).project(pt), float)
        except RuntimeError:
            return np.array([np.nan, np.nan])
    Jpt = np.zeros((2, 3), order="F")
    J_pose = np.zeros((2, 6), order="F")
    Jc9 = np.zeros((2, 9), order="F")
    try:
        uv = cam_cls(gtsam.Pose3(), cal).project(pt, J_pose, Jpt, Jc9)
    except RuntimeError:
        return np.array([np.nan, np.nan]), None, None
    return np.asarray(uv, float), Jpt, Jc9[:, _GTSAM_TO_OURS]


def _kernel_project(kernel, p_cam, cal_vec, n_intr, jac):
    """Camera-frame project via a custom_projections kernel. Slices the kernel's
    J_intr (2 x max(8, len(cal_vec))) to n_intr; invalid -> (nan, None, None)
    (jac) or nan."""
    if not jac:
        uv, _, _ = kernel(p_cam, cal_vec, False)
        return np.array([np.nan, np.nan]) if uv is None else np.asarray(uv, float)
    uv, J_pc, J_intr = kernel(p_cam, cal_vec, True)
    if uv is None:
        return np.array([np.nan, np.nan]), None, None
    J_intr = np.asarray(J_intr)
    assert J_intr.shape[1] >= n_intr, (
        f"kernel J_intr is {J_intr.shape[1]} wide, model needs {n_intr}")
    return np.asarray(uv, float), J_pc, J_intr[:, :n_intr]


class _RefCameraModel:
    """`build_camera_model(model, cal_vec)` + `CameraModel.project`, frozen.
    radtan -> Cal3DS2 / equidistant -> Cal3Fisheye (`PinholeRadtan.project` /
    `PinholeEquidistant.project` bodies); the rest -> `_kernel_project` over the
    model's kernel with `cal_vec = [fx, fy, cx, cy, dist...]`, `n_intr = len`."""
    __slots__ = ("model", "cal_vec", "n_intr", "_cal", "_kernel")

    def __init__(self, model: str, cal_vec: np.ndarray):
        cal_vec = np.asarray(cal_vec, dtype=np.float64)
        assert model in _N_DIST, f"unknown camera model {model!r}"
        assert len(cal_vec) == 4 + _N_DIST[model], (
            f"{model}: cal_vec must be canonical (4 + {_N_DIST[model]}), got {len(cal_vec)}")
        self.model = model
        self.cal_vec = cal_vec
        self.n_intr = len(cal_vec)
        self._cal = None
        self._kernel = None
        fx, fy, cx, cy = cal_vec[:4]
        d = cal_vec[4:]
        if model == "radtan":
            self._cal = gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, d[0], d[1], d[2], d[3])
        elif model == "equidistant":
            self._cal = gtsam.Cal3Fisheye(fx, fy, 0.0, cx, cy, d[0], d[1], d[2], d[3])
        else:
            self._kernel = _KERNELS[model]

    def project(self, point_3d: np.ndarray, jac: bool = False):
        point_3d = np.asarray(point_3d, dtype=np.float64)
        if self._cal is not None:
            if jac:
                return _gtsam_project(self._cal, point_3d, True)
            try:
                p2d = self._cal.uncalibrate(point_3d[:2] / point_3d[2])
                return np.array([p2d[0], p2d[1]])
            except RuntimeError:
                return np.array([np.nan, np.nan])
        return _kernel_project(self._kernel, point_3d, self.cal_vec, self.n_intr, jac)


class _Projector:
    """Holds a CameraModel (built once per factor evaluation) and chains its
    camera-frame projection to world-point pose/point Jacobians."""
    __slots__ = ("model", "cal_vec", "_cm")

    def __init__(self, model: str, cal_vec: np.ndarray):
        self.model = model
        self.cal_vec = np.asarray(cal_vec, float)
        self._cm = _RefCameraModel(model, self.cal_vec)   # S0 adapter (was build_camera_model)

    def project(self, T_CtoW: gtsam.Pose3, p_W, jac: bool = True):
        """(uv[2], J_pose[2x6], J_point[2x3], J_intr[2 x n_intr]); jac=False ->
        (uv, None, None, None). Raises RuntimeError when the point is out of the
        model's range / behind the camera (the caller zeroes those rows)."""
        pt = gtsam.Point3(*np.asarray(p_W, float))
        if not jac:
            uv = self._cm.project(np.asarray(T_CtoW.transformTo(pt), float))
            if not np.isfinite(uv).all():
                raise RuntimeError(f"{self.model} projection out of model range")
            return uv, None, None, None
        D_pose = np.zeros((3, 6), order="F")
        D_point = np.zeros((3, 3), order="F")
        p_cam = T_CtoW.transformTo(pt, D_pose, D_point)
        uv, J_pc, J_intr = self._cm.project(np.asarray(p_cam, float), jac=True)
        if J_pc is None or not np.isfinite(uv).all():
            raise RuntimeError(f"{self.model} projection out of model range")
        return uv, J_pc @ D_pose, J_pc @ D_point, J_intr


def make_projector(model: str, cal_vec: np.ndarray) -> _Projector:
    """Build a projector for `model` (call once per factor eval, reuse per corner)."""
    return _Projector(model, cal_vec)
