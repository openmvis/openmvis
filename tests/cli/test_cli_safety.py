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
"""`mvis --help` lists only working subcommands: the `analyze` stub is gone (F46),
and `mvis cam -o <unrelated dir>` exits with a message instead of a traceback (F44)."""
from __future__ import annotations

import pytest

from mvis.cli import main as cli


def _main(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["mvis", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code


def test_help_does_not_list_analyze(monkeypatch, capsys):
    assert _main(monkeypatch, ["--help"]) == 0
    out = capsys.readouterr().out
    assert "analyze" not in out
    for cmd in ("cam", "cam-imu", "run", "apply-calib", "roundtrip", "datasets"):
        assert cmd in out


def test_analyze_is_an_unknown_subcommand(monkeypatch, capsys):
    assert _main(monkeypatch, ["analyze", "--results", "x.yaml"]) == 2
    assert "invalid choice: 'analyze'" in capsys.readouterr().err


def test_cam_help_says_only_mvis_folders_are_cleared(monkeypatch, capsys):
    assert _main(monkeypatch, ["cam", "--help"]) == 0
    assert "only when MVIS created it" in " ".join(capsys.readouterr().out.split())


def test_calibrate_cam_turns_a_refused_run_folder_into_an_exit_message(monkeypatch, tmp_path):
    cfg = tmp_path / "cam.yaml"
    cfg.write_text("mode: cam\ninput:\n  cameras:\n    - {camera_id: 0}\n"
                   f"output:\n  file: {tmp_path / 'calib.yaml'}\n")

    from mvis.calibration.cam_run import OutputDirRefused

    def refuse(path, output, result_file=None):
        raise OutputDirRefused(f"output directory {output} exists and was not created by MVIS")

    monkeypatch.setattr("mvis.calibration.cam_run.run_cam", refuse)
    monkeypatch.setattr("mvis.calibration.cam_run.configure_process", lambda: None)
    with pytest.raises(SystemExit) as e:
        cli.calibrate_cam(str(cfg), str(tmp_path / "results"))
    assert "not created by MVIS" in str(e.value.code)
    assert "Traceback" not in str(e.value.code)


def test_other_file_exists_errors_keep_their_traceback(monkeypatch, tmp_path):
    """Only the refused run folder is relabelled; an unrelated FileExistsError from
    deep inside the run (e.g. result_file's parent being a file) stays a traceback."""
    cfg = tmp_path / "cam.yaml"
    cfg.write_text("mode: cam\ninput:\n  cameras:\n    - {camera_id: 0}\n"
                   f"output:\n  file: {tmp_path / 'calib.yaml'}\n")

    def other(path, output, result_file=None):
        raise FileExistsError("[Errno 17] File exists: 'calib.yaml'")

    monkeypatch.setattr("mvis.calibration.cam_run.run_cam", other)
    monkeypatch.setattr("mvis.calibration.cam_run.configure_process", lambda: None)
    with pytest.raises(FileExistsError):
        cli.calibrate_cam(str(cfg), str(tmp_path / "results"))
