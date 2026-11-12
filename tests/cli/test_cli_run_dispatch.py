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
"""CLI dispatch tests for the symmetric camera commands (config-cleanup M6, Stage B).

`mvis cam` runs camera-only (CamRunConfig); `mvis run` dispatches on the
config's top-level `mode:` (cam | cam-imu). calibrate_cam()/calibrate_cam_imu() (or
cam_run.run_cam) are monkeypatched so no real data is needed.
"""

import os
import sys
import tempfile

import pytest

from mvis.cli import main as cli
from mvis.core.config import CamRunConfig, CamImuRunConfig


def _yaml(text):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        return f.name


_CAM = "mode: cam\ninput:\n  cameras:\n    - {camera_id: 0}\nsolve: {max_iterations: 9}\n"
_CAMIMU = "mode: cam-imu\nsolve: {state_hz: 8.0}\n"


def test_run_dispatch_cam(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "calibrate_cam", lambda p: seen.update(path=p))
    p = _yaml(_CAM)
    try:
        cli._run_dispatch(p)
    finally:
        os.unlink(p)
    assert seen["path"] == p


def test_run_dispatch_cam_imu(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "calibrate_cam_imu",
                        lambda c: seen.update(cfg=c) or {"ok": True})
    p = _yaml(_CAMIMU)
    try:
        cli._run_dispatch(p)
    finally:
        os.unlink(p)
    assert isinstance(seen["cfg"], CamImuRunConfig)


def test_run_dispatch_defaults_to_cam(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "calibrate_cam", lambda p: seen.update(path=p))
    p = _yaml("input:\n  cameras:\n    - {camera_id: 0}\n")   # no mode: -> cam
    try:
        cli._run_dispatch(p)
    finally:
        os.unlink(p)
    assert seen["path"] == p


def test_run_dispatch_unknown_mode_raises():
    p = _yaml("mode: lidar\n")
    try:
        with pytest.raises(SystemExit):
            cli._run_dispatch(p)
    finally:
        os.unlink(p)


def _spy_run_cam(monkeypatch, rc=0):
    from mvis.calibration import cam_run
    seen = {}

    def fake(config_path, output=None, **kw):
        seen.update(path=config_path, output=output, **kw)
        return rc
    monkeypatch.setattr(cam_run, "run_cam", fake)
    return seen


@pytest.mark.parametrize("argv, output", [([], None), (["-o", "/x/run"], "/x/run")])
def test_cam_command_runs_the_pipeline_driver(monkeypatch, argv, output):
    """G8: `mvis cam` runs cam_run.run_cam (the regression's driver) on the config path,
    with the config's output_file as the extra result file."""
    seen = _spy_run_cam(monkeypatch)
    p = _yaml(_CAM + "output: {path: out/calib_cam0.yaml}\n")
    monkeypatch.setattr(sys, "argv", ["mvis", "cam", "-c", p, *argv])
    try:
        cli.main()
        want = CamRunConfig.from_yaml(p).output_file
    finally:
        os.unlink(p)
    assert seen == {"path": p, "output": output, "result_file": want}
    assert want.endswith("calib_cam0.yaml")


def test_cam_command_exits_nonzero_when_the_solve_fails(monkeypatch):
    _spy_run_cam(monkeypatch, rc=1)
    p = _yaml(_CAM)
    try:
        with pytest.raises(SystemExit) as e:
            cli.calibrate_cam(p)
    finally:
        os.unlink(p)
    assert e.value.code == 1


def test_version_flag(capsys, monkeypatch):
    import sys
    import pytest
    from mvis import __version__
    from mvis.cli.main import main
    monkeypatch.setattr(sys, "argv", ["mvis", "--version"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"mvis {__version__}"
