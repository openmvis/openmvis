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
"""The kb8 model is gone: old configs and results naming it are refused with what to use
instead, never read as another model. A capture db whose camera row names it stays usable,
because its detections do not depend on the model."""
import sqlite3

import numpy as np
import pytest
import yaml

from mvis.core import camera_models as cm

SPELLINGS = ["kb8", "KB8", " Kannala_Brandt8 ", "kannala_brandt8", "kannala-brandt8", "kb_8",
             "kannala_brandt_8", "kannala-brandt-8"]
EMBEDDED = ["pinhole_kb8", "PINHOLE_KB8", "pinhole kannala_brandt8", "kannala_brandt-8",
            "kannala_brandt 8", "kannala-brandt_8", "kb-8", "kb 8", "omni_kb8"]
MESSAGE = "was removed from MVIS; use 'kb4' past 90 degrees, 'equidistant' below"
COEFFS = [0.02, -0.01, 3e-3, -5e-4, 1e-4, -1e-5, 1e-6, -1e-7]


def _refused():
    return pytest.raises(ValueError, match=MESSAGE)


@pytest.mark.parametrize("name", SPELLINGS)
def test_config_with_kb8_is_refused_at_load(name):
    from mvis.core.config import CameraConfig
    with _refused():
        cm.resolve_model_alias(name)
    with _refused():
        CameraConfig(distortion_model=name)


def test_yaml_config_with_kb8_is_refused(tmp_path):
    from mvis.core.config import CalibrationConfig
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"cameras": [{"camera_id": 0, "source_type": "images",
                                              "source_path": "x", "distortion_model": "kb8"}]}))
    with _refused():
        CalibrationConfig.from_yaml(str(p))


def test_cam_run_config_with_kb8_is_refused(tmp_path):
    from mvis.core.config import CamRunConfig
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"input": {"cameras": [
        {"camera_id": 0, "source_type": "rosbag_topic", "source_path": "x.bag", "topic": "/c",
         "distortion_model": "kb8"}]}}))
    with _refused():
        CamRunConfig.from_yaml(str(p))


@pytest.mark.parametrize("name", SPELLINGS + EMBEDDED)
def test_input_name_is_refused_not_read_as_equidistant(name):
    from mvis.io.camera_input import _norm_model
    with _refused():
        _norm_model(name)


@pytest.mark.parametrize("fn", [cm.check_model, cm.n_dist, cm.is_custom, cm.default_dist])
def test_registry_refuses(fn):
    with _refused():
        fn("kb8")


def test_registry_has_only_kb4():
    assert "kb8" not in cm.MODELS and "kb8" not in cm.N_DIST and "kb8" not in cm.DIST_NAMES
    assert cm.KB_THETA_MODELS == {"kb4"}
    assert not cm.is_removed_model("kb4")
    assert not cm.is_removed_model("equidistant")
    assert not cm.is_removed_model(None)


def test_camera_factories_refuse():
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.camera import create_camera_model
    from mvis.camera.camera_model import build_camera_model
    import gtsam
    cal = np.r_[300.0, 300.0, 320.0, 240.0, COEFFS]
    with _refused():
        create_camera_model("kb8", 300, 300, 320, 240)
    with _refused():
        build_camera_model("kb8", cal)
    with _refused():
        CameraCalib(gtsam.Pose3(), 0.0, 0.0, cal, model="kb8")


# ── stored results ──

def _result_yaml(tmp_path, **cam):
    p = tmp_path / "old_calib.yaml"
    p.write_text(yaml.safe_dump({"cameras": [{
        "id": 0, "model": "pinhole_kb8", "image_size": [640, 480],
        "intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 320.0, "cy": 240.0},
        "distortion": {"model": "kb8", "coeffs": COEFFS}, **cam}]}))
    return p


def test_stored_mvis_yaml_is_refused(tmp_path):
    from mvis.io.camera_input import load_camera_intrinsics
    with _refused():
        load_camera_intrinsics(str(_result_yaml(tmp_path)))


def test_stored_kalibr_camchain_is_refused(tmp_path):
    from mvis.io.camera_input import load_camera_intrinsics
    p = tmp_path / "old_camchain.yaml"
    p.write_text(yaml.safe_dump({"cam0": {
        "camera_model": "pinhole", "distortion_model": "kb8", "rostopic": "/c",
        "intrinsics": [300.0, 300.0, 320.0, 240.0], "distortion_coeffs": COEFFS,
        "resolution": [640, 480]}}))
    with _refused():
        load_camera_intrinsics(str(p))


def test_regenerate_html_refuses_a_stored_result(tmp_path):
    from mvis.analysis.regenerate_html import _result_from_yaml
    with _refused():
        _result_from_yaml(_result_yaml(tmp_path))


def test_calib_spec_refuses_a_stored_result():
    from mvis.recalib.calib_spec import _cam_from_block
    with _refused():
        _cam_from_block({"intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                         "distortion": {"model": "kb8", "coeffs": COEFFS}}, 0)


def test_regression_health_refuses_a_stored_result():
    with _refused():
        cm.dist_sanity_issue("kb8", COEFFS)


def test_stats_mark_one_old_kb8_row_instead_of_failing_the_batch(tmp_path):
    from types import SimpleNamespace
    from mvis.regression.cam_stats import collect_stats
    (tmp_path / "d_r_full.yaml").write_text(yaml.safe_dump({
        "cameras": [{"id": 0, "intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 320.0, "cy": 240.0},
                     "distortion": {"model": "kb8", "coeffs": COEFFS}}],
        "reprojection_errors": {"per_camera": [{"camera": 0, "rms": 0.2}]}}))
    stats = collect_stats(SimpleNamespace(output_dir=tmp_path, dataset="d", slug="r"))
    cam = stats["per_camera"][0]
    assert cam["health"] == "removed_model" and MESSAGE in cam["health_msg"]


# ── mvisdb ──

def _db_with_kb4(tmp_path):
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.mvisdatabase import CalibChoice, CameraInfo, CameraModel, CameraStream, Run
    from mvis.mvisdb.estimates import estimates_from_calibration_result
    from mvis.core.types import CalibrationResult, CameraParams
    from tests.helpers.mvisdb_fixtures import make_db
    db = make_db(CalibChoice.CAM_ONLY)
    db.streams = [CameraStream(0, CameraInfo("cam0", camera_model=CameraModel.KB4,
                                             resolution=(640, 480)))]
    db.run = Run(CalibChoice.CAM_ONLY, description="t", status="ok")
    db.estimate_states = estimates_from_calibration_result(CalibrationResult(cameras=[
        CameraParams(0, np.array([300.0, 300.0, 320.0, 240.0]), np.zeros(4), "kb4", (640, 480))]))
    path = tmp_path / "old.mvisdb"
    rid = DatabaseManager().save(db, path)
    return path, rid


def _as_old_file(path, table):
    """Put 'kb8' into a stored row, as a file written before the removal holds it. An old
    file's CHECK allows it; this test's fresh file does not, so drop the check for the edit."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA writable_schema = ON")
    con.execute("UPDATE sqlite_master SET sql = replace(sql, \"'omni_radtan','kb4'\", "
                "\"'omni_radtan','kb4','kb8'\") WHERE name = 'camera'")
    con.commit()
    con.close()
    con = sqlite3.connect(path)
    con.execute(f"UPDATE {table} SET camera_model = 'kb8'")
    con.commit()
    con.close()


def test_enum_refuses_with_the_message():
    from mvis.mvisdb.mvisdatabase import CameraModel
    assert len(CameraModel) == 8
    with _refused():
        CameraModel("kb8")
    with pytest.raises(ValueError, match="not a valid CameraModel"):
        CameraModel("kb9")


def test_new_db_schema_refuses_kb8(tmp_path):
    path, _ = _db_with_kb4(tmp_path)
    con = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE camera SET camera_model='kb8'")
    con.close()


def test_old_capture_camera_row_stays_usable_without_its_model(tmp_path, caplog):
    from mvis.mvisdb.manager import DatabaseManager
    path, _ = _db_with_kb4(tmp_path)
    _as_old_file(path, "camera")
    with caplog.at_level("WARNING", logger="mvis.mvisdb.manager"):
        db = DatabaseManager().load(path)
    assert db.streams[0].info.camera_model is None      # never another model
    assert MESSAGE in caplog.text and "ignoring the stored model" in caplog.text


def test_old_capture_takes_a_new_kb4_solve(tmp_path):
    """What the changelog promises: load a capture whose camera row names kb8, solve with kb4,
    save into the same file, then read the new result and export it."""
    from mvis.core.types import CalibrationResult, CameraParams
    from mvis.mvisdb.estimates import estimates_from_calibration_result
    from mvis.mvisdb.exports import export_calib_yaml
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.mvisdatabase import CalibChoice, CameraModel, Run
    path, _ = _db_with_kb4(tmp_path)
    _as_old_file(path, "camera")
    mgr = DatabaseManager()
    db = mgr.load(path)
    assert db.streams[0].info.camera_model is None

    intr = np.array([310.0, 311.0, 321.0, 241.0])
    dist = np.array([0.02, -0.01, 3e-3, -5e-4])
    db.streams[0].info.camera_model = CameraModel.KB4
    db.run = Run(CalibChoice.CAM_ONLY, description="new solve", status="ok")
    db.estimate_states = estimates_from_calibration_result(CalibrationResult(cameras=[
        CameraParams(0, intr, dist, "kb4", (640, 480))]))
    rid = mgr.save(db, path)

    mgr.state(path, rid)
    out = tmp_path / "view.yaml"
    export_calib_yaml(mgr, path, rid, out)
    cam = yaml.safe_load(out.read_text())["cameras"][0]
    assert cam["distortion"]["model"] == "kb4"
    np.testing.assert_allclose(cam["distortion"]["coeffs"], dist)
    assert cam["intrinsics"]["fx"] == pytest.approx(310.0)


def test_old_file_with_a_kb8_result_still_takes_a_new_kb4_solve(tmp_path):
    """The old result stays refused, by its own run id only."""
    from mvis.core.types import CalibrationResult, CameraParams
    from mvis.mvisdb.estimates import estimates_from_calibration_result
    from mvis.mvisdb.manager import DatabaseManager
    from mvis.mvisdb.mvisdatabase import CalibChoice, CameraModel, Run
    path, old = _db_with_kb4(tmp_path)
    _as_old_file(path, "camera")
    _as_old_file(path, "result_camera")
    mgr = DatabaseManager()
    db = mgr.load(path)
    db.streams[0].info.camera_model = CameraModel.KB4
    db.run = Run(CalibChoice.CAM_ONLY, description="new solve", status="ok")
    db.estimate_states = estimates_from_calibration_result(CalibrationResult(cameras=[
        CameraParams(0, np.array([310.0, 311.0, 321.0, 241.0]),
                     np.array([0.02, -0.01, 3e-3, -5e-4]), "kb4", (640, 480))]))
    rid = mgr.save(db, path)
    mgr.state(path, rid)
    with _refused():
        mgr.state(path, old)


def test_old_stored_result_is_refused(tmp_path):
    from mvis.mvisdb.manager import DatabaseManager
    path, rid = _db_with_kb4(tmp_path)
    _as_old_file(path, "result_camera")
    with _refused():
        DatabaseManager().state(path, rid)


# ── kb4 is written to Kalibr as pinhole-equi with its four coefficients, never as kb8 ──

def test_kalibr_pack_writes_kb4_as_pinhole_equi():
    from mvis.io.cam_imu_output import _kalibr_pack, kalibr_export_issue
    cal = [300.0, 300.0, 320.0, 240.0, 0.02, -0.01, 3e-3, -5e-4]
    assert kalibr_export_issue("kb4", cal) is None
    assert _kalibr_pack("kb4", cal) == ("pinhole", "equidistant", cal[:4], cal[4:])
    assert cm.KALIBR_EXPORTABLE == set(cm.MODELS)
