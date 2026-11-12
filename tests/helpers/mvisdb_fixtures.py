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
"""MVISDatabase fixtures: a 2-camera, one-board db, and a stub cam-IMU result."""
from __future__ import annotations

import numpy as np

from mvis.mvisdb.mvisdatabase import (
    Board, BoardPt, BoardType, CalibChoice, CameraInfo, CameraModel, CameraStream,
    Corner, ImuInfo, ImuStream, MVISDatabase, Rig, Run, TagDetection,
)


def make_board(board_id: int = 0) -> Board:
    pts = {}
    for tag in range(4):
        for c in range(4):
            pid = tag * 4 + c
            pts[pid] = BoardPt(pid, tag, Corner(c),
                               np.array([0.1 * tag, 0.1 * c, 0.0]))
    return Board(board_id=board_id, board_type=BoardType.APRILGRID,
                 rows=2, cols=2, spacing=0.03, size=0.1, pts=pts)


def det(pid: int, u: float, v: float, order: int = 0) -> TagDetection:
    return TagDetection(uv=np.array([u, v]), point_id=pid,
                        detector_emission_order=order)


def make_db(calib=CalibChoice.CAM_ONLY) -> MVISDatabase:
    db = MVISDatabase(name="fixture/bag01",
                      rig=Rig(base_camera_id=0, base_imu_id=0),
                      run=Run(calib))
    db.boards[0] = make_board()
    db.streams = [
        CameraStream(0, CameraInfo("cam0", camera_model=CameraModel.RADTAN)),
        CameraStream(1, CameraInfo("cam1", camera_model=CameraModel.EQUIDISTANT)),
    ]
    db.imus = [ImuStream(0, ImuInfo("imu0"))]
    # cam0: 3 frames — 4 dets / 2 dets / 0 groups
    f0 = db.add_frame(0, 100.0)
    db.add_board_detection(f0, 0, [det(0, 10.0, 20.9), det(1, 30.0, 21.0),
                                   det(4, 50.0, 22.5), det(5, 70.0, 23.0)])
    f1 = db.add_frame(0, 100.1)
    db.add_board_detection(f1, 0, [det(2, 11.0, 40.0), det(3, 31.0, 41.0)])
    db.add_frame(0, 100.2)  # empty frame — no groups at all
    # cam1: 1 frame, 1 det
    f3 = db.add_frame(1, 100.05)
    db.add_board_detection(f3, 0, [det(8, 5.0, 6.0)])
    return db


def _rot(axis, th):
    c, s = np.cos(th), np.sin(th)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


class _Pose:
    def __init__(self, M): self._M = np.asarray(M, float)
    def matrix(self): return self._M


class _Calib:
    def __init__(self, M, td, tr, intr, model):
        self.T = _Pose(M); self.td = td; self.tr = tr
        self.intr = np.asarray(intr, float); self.model = model


def _imu_result():
    from types import SimpleNamespace
    T0 = np.eye(4); T0[:3, 3] = [0.01, 0.02, 0.03]
    T1 = np.eye(4); T1[:3, :3] = _rot('z', 0.2); T1[:3, 3] = [0.1, 0, 0]
    base = _Calib(T0, 0.004, 0.03, [400, 401, 320, 240, -0.1, 0.01, 0, 0], "radtan")
    extra = _Calib(T1, 0.005, 0.0, [402, 403, 321, 241, 0.2, 0.1, 0.05, 0.01],
                   "equidistant")
    return SimpleNamespace(
        ok=True, calib=base, imu_model="rpng",
        imu_intr_xi=np.arange(24.0), imu_intr_sigma=np.full(24, 0.5),
        cameras=[{"calib": base, "calib_sigma": None},
                 {"calib": extra, "calib_sigma": np.ones(16)}],
        imus=[{"T_IstoI": _Pose(T1), "td": 0.001}],
    )
