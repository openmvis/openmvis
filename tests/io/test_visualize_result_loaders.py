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
"""scripts/visualize_result.py loaders: the safe numpy-aware YAML read and the legacy
detections.json fallback (the mvisdb path is the regenerate_html loader, tested there)."""
import json

import numpy as np
import pytest
import yaml

from scripts import visualize_result as vr


def _write_run(tmp_path, *, numpy_tags=True):
    fx = np.float64(400.0) if numpy_tags else 400.0
    doc = {
        "cameras": [{"id": 0, "intrinsics": {"fx": fx, "fy": 401.0, "cx": 320.0, "cy": 240.0},
                     "distortion": {"model": "radtan", "coeffs": [0.1, 0.0, 0.0, 0.0]},
                     "image_size": [640, 480]}],
        "board_poses": [{"frame_index": 0, "timestamp": 1.5, "T_board_cam": np.eye(4).tolist()}],
        "board": {"points_3d": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]},
    }
    (tmp_path / "run_full.yaml").write_text(yaml.dump(doc))       # python tags when numpy
    return doc


def test_load_result_reads_numpy_tags_safely(tmp_path):
    _write_run(tmp_path)
    assert "!!python/object/apply:numpy" in (tmp_path / "run_full.yaml").read_text()
    result, board_pts, _ = vr.load_result(tmp_path)
    assert result.cameras[0].intrinsics[0] == 400.0
    assert result.board_pose_times == [1.5]
    assert board_pts.shape == (2, 3)


def test_load_result_refuses_code_tags(tmp_path):
    (tmp_path / "run_full.yaml").write_text('cameras: !!python/object/apply:os.system ["true"]\n')
    with pytest.raises(yaml.YAMLError):
        vr.load_result(tmp_path)


def test_frames_fall_back_to_legacy_json(tmp_path):
    cache = {"cameras": {"0": {"frames": [{"camera_id": 0, "timestamp": 2.0,
                                          "image_size": [640, 480],
                                          "observations": [{"point_id": 3, "pixel": [1.0, 2.0]}]}]}}}
    (tmp_path / "detections.json").write_text(json.dumps(cache))
    frames, sizes, images = vr.load_frames_and_images(tmp_path)
    assert [f.timestamp for f in frames[0]] == [2.0]
    assert sizes == {0: (640, 480)}
    assert images == {}
