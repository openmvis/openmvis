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
"""I5 export view vs the writer's calib YAML for kb4 / omni_radtan rigs.

kb4's calib YAML carries a `valid_domain` block whose corner_radius_px /
covers_image are computed from image_size. The export view rebuilds the
result from the db estimate, which does not hold image_size, so the view
must restore it from the capture's stream resolutions; otherwise the I5
shadow in scripts/run_calibration.py fails every kb4 run (rc=3) and
regenerated kb4 reports lose the valid-domain columns.
"""
from __future__ import annotations

import numpy as np
import yaml

from mvis.core.types import CalibrationResult, CameraParams, ExtrinsicTransform
from mvis.io.output_writer import OutputWriter
from mvis.mvisdb.estimates import estimates_from_calibration_result
from mvis.mvisdb.exports import export_calib_yaml, normalise_calib_yaml_pair
from mvis.mvisdb.manager import DatabaseManager
from mvis.mvisdb.mvisdatabase import (
    CalibChoice, CameraInfo, CameraModel, CameraStream, Run,
)
from tests.helpers.mvisdb_fixtures import make_db

KB4 = np.array([250.0, 251.0, 501.0, 498.0, 0.02, -0.01, 3e-3, -5e-4])
OMNI = np.array([700.0, 701.0, 510.0, 495.0, 1.6, -0.2, 0.05, 1e-4, -2e-4])
SIZES = {0: (1000, 1000), 1: (1024, 960), 2: (640, 480)}


def _rig_result():
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 0.0]
    cams = [
        CameraParams(0, KB4[:4], KB4[4:], "kb4", SIZES[0]),
        CameraParams(1, OMNI[:4], OMNI[4:], "omni_radtan", SIZES[1]),
        CameraParams(2, [500.0, 501, 320, 240], [-0.1, 0.01, 0, 0], "radtan",
                     SIZES[2]),
    ]
    ext = [ExtrinsicTransform(c, 0, T, np.zeros(3), T[:3, 3]) for c in (1, 2)]
    return CalibrationResult(cameras=cams, extrinsics=ext, reference_camera=0,
                             board_poses=[np.eye(4)], board_pose_times=[1.0])


def _attach(tmp_path, result, *, with_resolution: bool):
    db = make_db(CalibChoice.CAM_ONLY)
    models = [CameraModel.KB4, CameraModel.OMNI_RADTAN, CameraModel.RADTAN]
    db.streams = [CameraStream(c, CameraInfo(
        f"cam{c}", camera_model=models[c],
        resolution=SIZES[c] if with_resolution else None)) for c in range(3)]
    db.run = Run(CalibChoice.CAM_ONLY, description="t", status="ok")
    db.estimate_states = estimates_from_calibration_result(result)
    path = tmp_path / "detections.mvisdb"
    mgr = DatabaseManager()
    rid = mgr.save(db, path)
    return mgr, path, rid


def _shadow(tmp_path, *, with_resolution: bool, result=None):
    """The I5 shadow exactly as run_calibration.py runs it."""
    result = result or _rig_result()
    writer = tmp_path / "run_calib.yaml"
    OutputWriter.write_calibration_yaml(result, str(writer))
    mgr, path, rid = _attach(tmp_path, result, with_resolution=with_resolution)
    view = tmp_path / "view.yaml"
    export_calib_yaml(mgr, path, rid, view)
    a = yaml.safe_load(writer.read_text())
    b = yaml.safe_load(view.read_text())
    return a, b


def test_i5_export_view_restores_image_size_kb4_omni(tmp_path):
    a, b = _shadow(tmp_path, with_resolution=True)
    # the view itself is complete: image_size and kb4 valid_domain match
    for cw, cv in zip(a["cameras"], b["cameras"]):
        assert cv["image_size"] == cw["image_size"]
    vd = b["cameras"][0]["valid_domain"]
    assert vd["corner_radius_px"] is not None
    assert vd["covers_image"] is not None
    assert vd == a["cameras"][0]["valid_domain"]
    normalise_calib_yaml_pair(a, b)
    assert a == b
    # strict: the image-derived kb4 fields stay in the compared dicts
    assert "corner_radius_px" in a["cameras"][0]["valid_domain"]


def test_i5_normalisation_without_stored_resolution(tmp_path):
    """A capture with no stored resolution: only the two image-derived kb4
    fields are dropped (the view cannot compute them); the rest of
    valid_domain is still compared."""
    a, b = _shadow(tmp_path, with_resolution=False)
    assert b["cameras"][0]["valid_domain"]["corner_radius_px"] is None
    normalise_calib_yaml_pair(a, b)
    assert a == b
    assert set(a["cameras"][0]["valid_domain"]) == {
        "fold_theta_rad", "max_theta_rad", "max_radius_px"}


def test_i5_normalisation_drops_writer_only_diagnostics(tmp_path):
    """The writer adds the formal readout sigma, the board-shape estimate and
    the per-corner bias; the view is built from estimate states and has none
    of them, so the I5 shadow must ignore them (they are diagnostics, not
    parameters)."""
    a, b = _shadow(tmp_path, with_resolution=True)
    a["cameras"][0]["rolling_shutter_readout_sigma"] = 1.6e-4
    a["board_shape"] = {"rms_mm": 0.12, "p2p_mm": 0.5}
    a["board_point_bias"] = {"median_px": 0.27, "max_px": 0.98, "n_points_seen": 108}
    normalise_calib_yaml_pair(a, b)
    assert a == b
    assert "rolling_shutter_readout_sigma" not in a["cameras"][0]
    assert "board_shape" not in a and "board_point_bias" not in a


def test_i5_shadow_of_a_result_with_board_shape_and_readout_sigma(tmp_path):
    """The same through the real writer, attach and export view."""
    r = _rig_result()
    r.cameras[2].rolling_shutter_readout = 0.03
    r.cameras[2].rolling_shutter_readout_sigma = 1.6e-4
    r.board_heights = np.linspace(-1e-3, 1e-3, 8)
    r.board_point_bias = np.column_stack([np.full(8, 0.1), np.zeros(8), np.full(8, 5.0)])
    a, b = _shadow(tmp_path, with_resolution=True, result=r)
    assert "board_shape" in a and "rolling_shutter_readout_sigma" in a["cameras"][2]
    normalise_calib_yaml_pair(a, b)
    assert a == b


def test_i5_normalisation_still_catches_a_real_mismatch(tmp_path):
    a, b = _shadow(tmp_path, with_resolution=True)
    b["cameras"][0]["valid_domain"]["covers_image"] = (
        not a["cameras"][0]["valid_domain"]["covers_image"])
    normalise_calib_yaml_pair(a, b)
    assert a != b


def test_regenerate_from_db_keeps_kb4_valid_domain(tmp_path):
    from mvis.analysis.html_report import _kb_domain_rows
    from mvis.analysis.regenerate_html import _result_from_mvisdb
    result = _rig_result()
    _attach(tmp_path, result, with_resolution=True)
    back = _result_from_mvisdb(tmp_path / "detections.mvisdb")
    assert [c.image_size for c in back.cameras] == [SIZES[0], SIZES[1], SIZES[2]]
    assert "".join(_kb_domain_rows(back.cameras)) == \
        "".join(_kb_domain_rows(result.cameras))
    assert '<td class="num"></td><td></td>' not in "".join(
        _kb_domain_rows(back.cameras))


def test_export_view_ucm_round_trip(tmp_path):
    """A ucm camera (one distortion coefficient) saves to a fresh db (enum +
    schema CHECK) and its export view matches the writer's calib YAML."""
    ucm = np.array([380.0, 380.5, 510.0, 514.0, 0.643])
    result = CalibrationResult(
        cameras=[CameraParams(0, ucm[:4], ucm[4:], "ucm", (1024, 1024))],
        extrinsics=[], reference_camera=0,
        board_poses=[np.eye(4)], board_pose_times=[1.0])
    writer = tmp_path / "run_calib.yaml"
    OutputWriter.write_calibration_yaml(result, str(writer))
    db = make_db(CalibChoice.CAM_ONLY)
    db.streams = [CameraStream(0, CameraInfo("cam0", camera_model=CameraModel.UCM,
                                             resolution=(1024, 1024)))]
    db.run = Run(CalibChoice.CAM_ONLY, description="t", status="ok")
    db.estimate_states = estimates_from_calibration_result(result)
    path = tmp_path / "detections.mvisdb"
    mgr = DatabaseManager()
    rid = mgr.save(db, path)
    view = tmp_path / "view.yaml"
    export_calib_yaml(mgr, path, rid, view)
    a = yaml.safe_load(writer.read_text())
    b = yaml.safe_load(view.read_text())
    assert b["cameras"][0]["distortion"] == {"model": "ucm", "coeffs": [0.643]}
    normalise_calib_yaml_pair(a, b)
    assert a == b
    from mvis.io.camera_input import load_camera_intrinsics
    lc = load_camera_intrinsics(str(writer))[0]
    assert lc["distortion_model"] == "ucm"
    np.testing.assert_allclose(lc["intrinsics"] + lc["distortion_coeffs"], ucm,
                               atol=1e-12, rtol=0)


def test_kalibr_camchain_view_ucm(tmp_path):
    """U2 accept: a cam-IMU ucm result writes an omni-none camchain; the db
    export view of the same run matches it (the CLI I5 check), and both
    reload to the solved cal_vec to 1e-12."""
    from types import SimpleNamespace
    import gtsam
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.io.cam_imu_output import to_kalibr_camchain_imucam
    from mvis.io.camera_input import load_camera_intrinsics
    from mvis.mvisdb.estimates import estimates_from_cam_imu_result
    from mvis.mvisdb.exports import export_kalibr_camchain
    cal = np.array([380.0, 380.5, 510.0, 514.0, 0.643])
    T = gtsam.Pose3(gtsam.Rot3.Rz(0.02), gtsam.Point3(0.01, 0.02, 0.03))
    base = CameraCalib(T, 0.004, 0.0, cal, model="ucm")
    r = SimpleNamespace(ok=True, calib=base, imu_model="rpng",
                        imu_intr_xi=np.arange(24.0), imu_intr_sigma=np.full(24, 0.5),
                        cameras=[{"calib": base, "calib_sigma": None,
                                  "image_size": (512, 512)}], imus=[])
    chain = tmp_path / "run_camchain-imucam.yaml"
    chain.write_text(yaml.safe_dump(to_kalibr_camchain_imucam(r), sort_keys=False))
    db = make_db(CalibChoice.CAM_IMU)
    db.streams[0] = CameraStream(0, CameraInfo("cam0", camera_model=CameraModel.UCM,
                                               resolution=(512, 512)))
    db.estimate_states = estimates_from_cam_imu_result(r, state_hz=20.0)
    path = tmp_path / "b.mvisdb"
    mgr = DatabaseManager()
    rid = mgr.save(db, path)
    view = tmp_path / "view.yaml"
    export_kalibr_camchain(mgr, path, rid, view)
    a = yaml.safe_load(chain.read_text())["cam0"]
    b = yaml.safe_load(view.read_text())["cam0"]
    assert (a["camera_model"], a["distortion_model"]) == ("omni", "none")
    for k in ("camera_model", "distortion_model", "intrinsics", "distortion_coeffs"):
        assert a[k] == b[k], k
    for src in (chain, view):
        lc = load_camera_intrinsics(str(src))[0]
        assert lc["distortion_model"] == "ucm"
        np.testing.assert_allclose(lc["intrinsics"] + lc["distortion_coeffs"], cal,
                                   atol=1e-12, rtol=0)


def test_regenerate_frames_are_the_solve_frames_not_every_detection(tmp_path):
    """The db keeps every detected frame; the run's overlay marks the ones the detection stage thinned
    away (`detection:*`). The live report scores only the frames the solve started from, so
    regenerate_html must too: scoring the thinned frames interpolates them between ref poses
    1/target_hz apart (T265 stereo: 11 px RMS for a 0.3 px solve)."""
    from mvis.analysis.regenerate_html import _frames_from_mvisdb
    db = make_db(CalibChoice.CAM_ONLY)
    f = db.camera(0).frames
    f[1].is_selected_to_use, f[1].exclude_reason = False, "detection:hz_subsample"
    f[2].is_selected_to_use, f[2].exclude_reason = False, "stage5_gate:quality_gate"
    db.run = Run(CalibChoice.CAM_ONLY, description="t", status="ok")
    path = tmp_path / "detections.mvisdb"
    DatabaseManager().save(db, path)
    frames = _frames_from_mvisdb(path)
    # the detection-thinned frame is out; a frame a later stage dropped stays, as in the live report
    assert [fr.timestamp for fr in frames[0]] == [100.0, 100.2]
    assert [fr.timestamp for fr in frames[1]] == [100.05]


def test_regenerate_frames_without_a_run_keep_every_frame(tmp_path):
    """A capture with no solved run has no overlay: every frame comes back, as before."""
    from mvis.analysis.regenerate_html import _frames_from_mvisdb
    path = tmp_path / "detections.mvisdb"
    DatabaseManager().save(make_db(CalibChoice.CAM_ONLY), path)
    assert [fr.timestamp for fr in _frames_from_mvisdb(path)[0]] == [100.0, 100.1, 100.2]
