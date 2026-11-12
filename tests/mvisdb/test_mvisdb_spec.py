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
"""CaptureSpec identity tests (T0a): same spec -> same hash; EVERY field flips it."""
from __future__ import annotations

import dataclasses

from mvis.mvisdb.mvisdatabase import BoardType, ClockKind, DetectionMode, SourceType, TagFamily
from mvis.mvisdb.spec import (
    BoardDetectorConfig, BoardSpec, CameraAddress, CaptureSpec,
    FreeFeatureConfig, ImuAddress, SourceConfig, TagDetectorConfig,
)


def make_spec() -> CaptureSpec:
    return CaptureSpec(
        source=SourceConfig(
            type=SourceType.ROS1BAG, path="/data/bag01.bag", clock=ClockKind.HEADER,
            cameras=(CameraAddress(0, "/cam0/image_raw", "cam0"),
                     CameraAddress(1, "/cam1/image_raw", "cam1")),
            imus=(ImuAddress(0, "/imu/data", "base"),),
            source_mtime_ns=1000, source_size=2000),
        detection={
            DetectionMode.BOARD: BoardDetectorConfig(
                boards=(BoardSpec(TagFamily.T36H11_KALIBR, BoardType.APRILGRID,
                                  6, 6, 0.0264, 0.088, 0),),
                refine_subpixel=True, min_tag_px=12),
            DetectionMode.TAG: TagDetectorConfig(TagFamily.ARUCO_6X6_250, (1, 53), 12),
        })


def test_same_spec_same_hash():
    assert make_spec().identity_hash() == make_spec().identity_hash()


def test_capture_mode_is_derived():
    assert make_spec().capture_mode == {DetectionMode.BOARD, DetectionMode.TAG}


def test_detector_config_type_enforced():
    import pytest
    with pytest.raises(TypeError):
        CaptureSpec(source=make_spec().source,
                    detection={DetectionMode.BOARD: {"rows": 6}})


def _flips(base: CaptureSpec, other: CaptureSpec) -> bool:
    return base.identity_hash() != other.identity_hash()


def test_every_source_field_flips_the_hash():
    base = make_spec()
    src = base.source
    variants = [
        dataclasses.replace(src, type=SourceType.ROS2BAG),
        dataclasses.replace(src, path="/data/bag02.bag"),
        dataclasses.replace(src, clock=ClockKind.BAG),
        dataclasses.replace(src, cameras=src.cameras[:1]),
        dataclasses.replace(src, cameras=(dataclasses.replace(src.cameras[0], topic="/x"),
                                          src.cameras[1])),
        dataclasses.replace(src, cameras=(dataclasses.replace(src.cameras[0], camera_id=7),
                                          src.cameras[1])),
        dataclasses.replace(src, imus=()),
        dataclasses.replace(src, source_mtime_ns=1001),
        dataclasses.replace(src, source_size=2001),
    ]
    for v in variants:
        assert _flips(base, CaptureSpec(v, dict(base.detection))), v


def test_every_detection_field_flips_the_hash():
    base = make_spec()
    bd: BoardDetectorConfig = base.detection[DetectionMode.BOARD]
    td: TagDetectorConfig = base.detection[DetectionMode.TAG]
    bs = bd.boards[0]
    det_variants = [
        # board detector params
        {**base.detection, DetectionMode.BOARD: dataclasses.replace(bd, refine_subpixel=False)},
        {**base.detection, DetectionMode.BOARD: dataclasses.replace(bd, min_tag_px=8)},
        # every BoardSpec field
        *[{**base.detection,
           DetectionMode.BOARD: dataclasses.replace(bd, boards=(dataclasses.replace(bs, **{k: v}),))}
          for k, v in [("family", TagFamily.T36H11), ("board_type", BoardType.ARUCO_GRID),
                       ("rows", 7), ("cols", 7), ("spacing", 0.03), ("size", 0.1),
                       ("start_id", 100)]],
        # tag detector fields
        {**base.detection, DetectionMode.TAG: dataclasses.replace(td, family=TagFamily.ARUCO_4X4_50)},
        {**base.detection, DetectionMode.TAG: dataclasses.replace(td, id_range=(1, 54))},
        {**base.detection, DetectionMode.TAG: dataclasses.replace(td, min_tag_px=8)},
        # mode set membership itself
        {DetectionMode.BOARD: bd},
        {**base.detection, DetectionMode.FREE: FreeFeatureConfig()},
    ]
    for d in det_variants:
        assert _flips(base, CaptureSpec(base.source, d)), d
    # free-feature fields flip too
    with_free = CaptureSpec(base.source, {**base.detection, DetectionMode.FREE: FreeFeatureConfig()})
    for k, v in [("max_features", 301), ("quality", 0.02), ("klt_window", 25), ("min_distance", 10)]:
        d = {**base.detection,
             DetectionMode.FREE: dataclasses.replace(FreeFeatureConfig(), **{k: v})}
        assert _flips(with_free, CaptureSpec(base.source, d)), k
