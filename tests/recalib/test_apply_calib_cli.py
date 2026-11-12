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
"""`mvis apply-calib` flags for the apply_calibration keywords: --no-imu-intrinsics, --no-td,
--no-t-ga and --strict (and their config keys). Defaults keep the old behaviour."""
import subprocess
import sys

import numpy as np
import pytest
import yaml

from mvis.recalib.apply import apply_calibration
from mvis.recalib.calib_spec import load_rig_calib

from .test_apply_options import _bag, _imu_stamps, _read, _stamp, T0

_MAP = {"/imu0": "imu0", "/cam0/image_raw": "cam0"}
_CALIB = {
    "cameras": [{"name": "cam0", "intrinsics": {"fx": 20.0, "fy": 21.0, "cx": 15.5, "cy": 16.0},
                 "distortion": {"model": "radtan", "coeffs": [-0.2, 0.05, 0.001, 0.0]},
                 "td": 0.0123, "rolling_shutter_readout": 0.0, "rostopic": "/cam0/image_raw"}],
    "imus": [{"name": "imu0", "role": "base", "model": "rpng", "topic": "/imu0",
              "b_g": [1.0, 2.0, 3.0], "b_a": [0.5, 0.5, 0.5], "t_ga": 1.5e-3,
              "imu_intr_xi": [0.01] * 6 + [0.0] * 18}],
}


@pytest.fixture
def calib(tmp_path):
    path = tmp_path / "calib.yaml"
    path.write_text(yaml.safe_dump(_CALIB))
    return str(path)


@pytest.fixture
def raw_bag(tmp_path, ts):
    path = str(tmp_path / "raw.bag")
    _bag(path, ts, _imu_stamps(8), images=[(T0 + 2_000_000, "mono8", 4, 4, 4, np.arange(16)),
                                           (T0 + 22_000_000, "mono8", 4, 4, 4, np.arange(16))])
    return path


def _captured(monkeypatch, argv):
    """Run the CLI in-process with apply_calibration stubbed; return the keywords it got."""
    import mvis.recalib.apply as apply_mod
    from mvis.cli import main as cli
    got = {}

    def fake(in_bag, out_bag, rig, mapping, base_imu_topic, **kw):
        got.update(kw)
        return {"out_bag": out_bag, "n_imu": 0, "n_cam": 0, "n_pass": 0, "provenance": "p"}
    monkeypatch.setattr(apply_mod, "apply_calibration", fake)
    monkeypatch.setattr(sys, "argv", ["mvis", "apply-calib", *argv])
    cli.main()
    return got


_DEFAULTS = dict(remove_bias=True, undistort=False, imu_intrinsics=True, td=True, t_ga=True,
                 strict=False)


def test_defaults_keep_old_behaviour(monkeypatch, calib, raw_bag, tmp_path):
    got = _captured(monkeypatch, ["--calib", calib, "--in", raw_bag,
                                  "--out", str(tmp_path / "o.bag")])
    assert got == _DEFAULTS


def test_flags_switch_each_keyword(monkeypatch, calib, raw_bag, tmp_path):
    got = _captured(monkeypatch, ["--calib", calib, "--in", raw_bag,
                                  "--out", str(tmp_path / "o.bag"), "--no-imu-intrinsics",
                                  "--no-td", "--no-t-ga", "--strict"])
    assert got == {**_DEFAULTS, "imu_intrinsics": False, "td": False, "t_ga": False,
                   "strict": True}


def test_config_keys_and_flag_override(monkeypatch, calib, raw_bag, tmp_path):
    cfg = tmp_path / "recalib.yaml"
    cfg.write_text(yaml.safe_dump({"calib": calib, "input": raw_bag,
                                   "output": str(tmp_path / "o.bag"), "imu_intrinsics": False,
                                   "td": False, "t_ga": True, "strict": True}))
    assert _captured(monkeypatch, ["-c", str(cfg)]) == {
        **_DEFAULTS, "imu_intrinsics": False, "td": False, "strict": True}
    assert _captured(monkeypatch, ["-c", str(cfg), "--no-t-ga"])["t_ga"] is False


def _run_cli(*args):
    return subprocess.run([sys.executable, "-m", "mvis.cli.main", "apply-calib", *args],
                          capture_output=True, text=True)


@pytest.mark.parametrize("flags, kw", [
    ([], {}),
    (["--no-imu-intrinsics", "--no-td", "--no-t-ga"],
     dict(imu_intrinsics=False, td=False, t_ga=False)),
    (["--no-td", "--strict"], dict(td=False, strict=True)),
])
def test_cli_bag_matches_library(calib, raw_bag, tmp_path, flags, kw):
    """The CLI writes the same bytes as apply_calibration with the matching keywords."""
    out = tmp_path / "cli.bag"
    r = _run_cli("--calib", calib, "--in", raw_bag, "--out", str(out), *flags)
    assert r.returncode == 0, r.stderr
    rig = load_rig_calib(calib)
    rig._src_path = calib
    ref = tmp_path / "lib.bag"
    apply_calibration(raw_bag, str(ref), rig, _MAP, "/imu0", **kw)
    assert out.read_bytes() == ref.read_bytes()
    prov = yaml.safe_load(open(str(out) + ".recalib.yaml"))
    assert prov["corrections"] == {"remove_bias": True, "undistort": False,
                                   "imu_intrinsics": kw.get("imu_intrinsics", True),
                                   "td": kw.get("td", True), "t_ga": kw.get("t_ga", True)}


def test_cli_no_td_keeps_stamps_and_bias_only(calib, raw_bag, tmp_path, ts):
    out = tmp_path / "out.bag"
    r = _run_cli("--calib", calib, "--in", raw_bag, "--out", str(out),
                 "--no-td", "--no-t-ga", "--no-imu-intrinsics")
    assert r.returncode == 0, r.stderr
    imu = _read(str(out), ts, "/imu0")
    assert [(t, _stamp(m)) for t, m in imu] == [(s + 7_000_000, s) for s in _imu_stamps(8)]
    a = imu[3][1].linear_acceleration
    assert (a.x, a.y, a.z) == pytest.approx((3.0 - 0.5, -0.5, 9.81 - 0.5))   # bias only


def test_cli_strict_rejects_missing_topic(calib, tmp_path, ts):
    raw = str(tmp_path / "imu_only.bag")
    _bag(raw, ts, _imu_stamps(3))                       # mapped camera topic absent
    out = tmp_path / "out.bag"
    assert _run_cli("--calib", calib, "--in", raw, "--out", str(out)).returncode == 0
    strict_out = tmp_path / "strict.bag"
    r = _run_cli("--calib", calib, "--in", raw, "--out", str(strict_out), "--strict")
    assert r.returncode != 0 and "absent" in r.stderr
    assert not strict_out.exists()
