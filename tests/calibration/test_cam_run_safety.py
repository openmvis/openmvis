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
"""Safety of the camera-only driver (release cleanup W4):

* ``run_cam -o <dir>`` clears only a run folder MVIS created (marker, detection
  cache, or nothing but MVIS outputs), never the working/home directory, and
  refuses any other existing folder instead of ``rmtree``-ing it (F44);
* ``--no-rs`` really disables rolling-shutter estimation (F43).
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from mvis.calibration import cam_run
from mvis.core.config import CameraConfig


def _mvis_run_folder(tmp_path: Path) -> Path:
    out = tmp_path / "run"
    out.mkdir()
    (out / "tum_cam1.html").write_text("report")
    (out / "tum_cam1_calib.yaml").write_text("cameras: []")
    (out / "tum_cam1_full.yaml").write_text("cameras: []")
    (out / "tum_cam1.log").write_text("log")
    (out / "thumbnails").mkdir()
    (out / "thumbnails" / "f0.jpg").write_bytes(b"jpg")
    return out


def test_missing_folder_is_left_to_mkdir(tmp_path):
    cam_run.clean_output_dir(tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_marked_folder_is_cleared_but_caches_logs_and_marker_stay(tmp_path):
    out = _mvis_run_folder(tmp_path)
    (out / cam_run.OUTPUT_MARKER).write_text("marker")
    (out / "detections.mvisdb").write_bytes(b"db")
    (out / "detections.mvisdb.lock").write_bytes(b"")
    (out / "detections.json").write_text("{}")
    (out / "notes.txt").write_text("stale")       # a stray file in an owned folder goes too
    cam_run.clean_output_dir(out)
    assert sorted(p.name for p in out.iterdir()) == sorted([
        cam_run.OUTPUT_MARKER, "detections.mvisdb", "detections.mvisdb.lock",
        "detections.json", "thumbnails", "tum_cam1.log"])
    assert (out / "thumbnails" / "f0.jpg").exists()


def test_cache_alone_marks_the_folder_as_ours(tmp_path):
    """Run folders from before the marker existed keep working through their cache."""
    out = _mvis_run_folder(tmp_path)
    (out / "detections.json").write_text("{}")
    cam_run.clean_output_dir(out)
    assert not (out / "tum_cam1.html").exists()
    assert (out / "detections.json").exists()


def test_folder_of_only_this_runs_outputs_is_cleared(tmp_path):
    out = _mvis_run_folder(tmp_path)
    (out / "tum_cam1_pass1.html").write_text("")
    (out / "detections.mvisdb.export_check_3.yaml").write_text("")
    assert cam_run.clean_output_dir(out, "tum_cam1") is True
    assert sorted(p.name for p in out.iterdir()) == ["thumbnails", "tum_cam1.log"]


def test_html_and_logs_of_another_name_do_not_make_it_ours(tmp_path):
    """Generic *.html / *.log are not evidence: a reports folder stays untouched."""
    out = tmp_path / "reports"
    out.mkdir()
    (out / "paper.html").write_text("mine")
    (out / "build.log").write_text("mine")
    with pytest.raises(cam_run.OutputDirRefused):
        cam_run.clean_output_dir(out, "tum_cam1")
    with pytest.raises(cam_run.OutputDirRefused):
        cam_run.clean_output_dir(out)                  # no run name: nothing matches
    assert (out / "paper.html").read_text() == "mine"


def test_empty_folder_is_ours(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    assert cam_run.clean_output_dir(out) is True
    assert out.is_dir()


def test_missing_folder_is_ours():
    assert cam_run.clean_output_dir("/nonexistent/mvis/run") is True


def test_unrelated_folder_is_refused_and_untouched(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "thesis.tex").write_text("precious")
    (out / "config.yaml").write_text("cameras: []")     # a plain .yaml is not a run output
    (out / "data").mkdir()
    with pytest.raises(cam_run.OutputDirRefused, match="not created by MVIS"):
        cam_run.clean_output_dir(out, "tum_cam1")
    assert (out / "thesis.tex").read_text() == "precious"
    assert (out / "config.yaml").exists() and (out / "data").is_dir()
    with pytest.raises(FileExistsError):                # it is a FileExistsError too
        cam_run.clean_output_dir(str(out))      # str paths are what the CLI passes


def test_a_file_at_the_output_path_is_refused(tmp_path):
    f = tmp_path / "out"
    f.write_text("x")
    with pytest.raises(cam_run.OutputDirRefused, match="not a directory"):
        cam_run.clean_output_dir(f)
    assert f.read_text() == "x"


@pytest.mark.parametrize("relative", [".", "..", "sub/.."])
def test_cwd_and_its_parents_are_never_cleared(tmp_path, monkeypatch, caplog, relative):
    work = tmp_path / "parent" / "work"
    work.mkdir(parents=True)
    (work / "sub").mkdir()
    (tmp_path / "parent" / "keep.html").write_text("")
    (work / "keep.html").write_text("")
    (work / cam_run.OUTPUT_MARKER).write_text("")      # even a marked folder
    monkeypatch.chdir(work)
    with caplog.at_level("WARNING", logger="ablation"):
        assert cam_run.clean_output_dir(relative) is False
    assert (work / "keep.html").exists() and (tmp_path / "parent" / "keep.html").exists()
    assert "Not clearing" in caplog.text


def test_home_is_never_cleared(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "keep.html").write_text("")
    monkeypatch.setenv("HOME", str(home))
    assert Path.home() == home
    assert cam_run.clean_output_dir(home) is False
    assert cam_run.clean_output_dir(tmp_path) is False          # a parent of home
    assert (home / "keep.html").exists()


def test_run_cam_does_not_mark_the_working_directory(tmp_path, monkeypatch):
    """`-o .` must not plant the marker: a later run from elsewhere would then own cwd."""
    cams = [CameraConfig(camera_id=0)]
    monkeypatch.setattr(cam_run, "parse_config", _fake_parse_config(tmp_path, cams))
    monkeypatch.setattr(cam_run, "create_board", lambda spec: None)
    monkeypatch.setattr("mvis.io.data_loader.DataLoader", _stop)    # first step after the clean
    work = tmp_path / "work"
    work.mkdir()
    (work / "mine.txt").write_text("")
    monkeypatch.chdir(work)
    with pytest.raises(_Stop):
        cam_run.run_cam("cfg.yaml", ".")
    assert sorted(p.name for p in work.iterdir()) == ["mine.txt"]


def test_run_cam_marks_and_later_clears_its_own_folder(tmp_path, monkeypatch):
    cams = [CameraConfig(camera_id=0)]
    monkeypatch.setattr(cam_run, "parse_config", _fake_parse_config(tmp_path, cams))
    monkeypatch.setattr(cam_run, "create_board", lambda spec: None)
    monkeypatch.setattr("mvis.io.data_loader.DataLoader", _stop)
    out = tmp_path / "run"
    with pytest.raises(_Stop):
        cam_run.run_cam("cfg.yaml", str(out))
    assert (out / cam_run.OUTPUT_MARKER).exists()
    (out / "stale_cam.html").write_text("")
    with pytest.raises(_Stop):
        cam_run.run_cam("cfg.yaml", str(out))
    assert sorted(p.name for p in out.iterdir()) == [cam_run.OUTPUT_MARKER]


class _Stop(Exception):
    pass


def _stop(*args, **kwargs):
    raise _Stop()


def _fake_parse_config(tmp_path, cameras):
    config = types.SimpleNamespace(cameras=cameras, output_file=str(tmp_path / "calib.yaml"),
                                   board=None, reference_camera=0)
    return lambda path: (config, None, {}, {c.camera_id: c.distortion_model for c in cameras},
                         {c.camera_id: f"cam{c.camera_id}" for c in cameras},
                         [c.camera_id for c in cameras])


def test_run_cam_no_rs_marks_every_camera_global_shutter(tmp_path, monkeypatch):
    cams = [CameraConfig(camera_id=0, shutter_type="rolling"),
            CameraConfig(camera_id=1, shutter_type="auto")]
    monkeypatch.setattr(cam_run, "parse_config", _fake_parse_config(tmp_path, cams))
    monkeypatch.setattr(cam_run, "create_board", _stop)
    with pytest.raises(_Stop):
        cam_run.run_cam("cfg.yaml", str(tmp_path / "out"), no_rs=True)
    assert [c.shutter_type for c in cams] == ["global", "global"]


def test_run_cam_without_no_rs_keeps_the_shutter_types(tmp_path, monkeypatch):
    cams = [CameraConfig(camera_id=0, shutter_type="rolling")]
    monkeypatch.setattr(cam_run, "parse_config", _fake_parse_config(tmp_path, cams))
    monkeypatch.setattr(cam_run, "create_board", _stop)
    with pytest.raises(_Stop):
        cam_run.run_cam("cfg.yaml", str(tmp_path / "out"))
    assert cams[0].shutter_type == "rolling"


def test_run_calibration_with_ref_enable_rs_false_reaches_the_pipeline(monkeypatch):
    """The function behind scripts/run_calibration.py honours enable_rs on both the
    camera configs and the CalibrationConfig the PipelineConfig is built from."""
    cams = [CameraConfig(camera_id=0, shutter_type="rolling")]
    calib_config = types.SimpleNamespace(cameras=cams)
    seen = {}

    def fake_from_calibration_config(cfg, **kw):
        seen["shutters"] = [c.shutter_type for c in cfg.cameras]
        raise _Stop()

    monkeypatch.setattr(cam_run.PipelineConfig, "from_calibration_config",
                        staticmethod(fake_from_calibration_config))
    monkeypatch.setattr("mvis.io.frame_selector.subsample_frames_by_hz", lambda fr, target_hz: fr)
    with pytest.raises(_Stop):
        cam_run.run_calibration_with_ref({0: []}, {}, {}, board=None, ref_cam=0,
                                         cam_names={0: "cam0"}, dist_models={0: "radtan"},
                                         enable_rs=False, camera_configs=cams,
                                         calib_config=calib_config)
    assert seen["shutters"] == ["global"]


def test_no_rs_flag_is_still_the_cli_name():
    args = cam_run.build_parser().parse_args(["--no-rs"])
    assert args.no_rs is True
