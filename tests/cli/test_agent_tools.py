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
"""Agent-facing tools: `mvis verdict`, `mvis check`, `mvis inspect` (mvis-calibrate skill)."""
import json
import os

import numpy as np
import pytest
import yaml

from mvis.analysis.verdict import assess
from mvis.doctor.check_config import check_config
from mvis.doctor.inspect import inspect

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cam_imu_json(tmp_path, **over):
    d = dict(ok=True, reproj_rms_px=0.3, reproj_median_px=0.2, reproj_p95_px=0.6,
             cams=[dict(name="cam0", td_ms=1.0, reproj=0.3, reproj_median=0.2,
                        reproj_p95=0.6, p_CinI_mm=[10.0, 0.0, 0.0], rotvec=[0, 0, 0])],
             imu_model="calyx", Dw_diag=[1.0, 1.01, 0.99], Da_diag=[1.0, 1.0, 1.0],
             imus=[], diagnostics=dict(iterations=12, max_iterations=40,
                                       cam_chi2_dof=0.8, imu_chi2_dof=0.9))
    d.update(over)
    p = tmp_path / "cam_imu.json"
    p.write_text(json.dumps(d))
    return str(p)


def _ids(v, status):
    return {c.id for c in v.checks if c.status == status}


def test_verdict_green(tmp_path):
    assert assess(_cam_imu_json(tmp_path)).grade == "GREEN"


def test_verdict_ok_true_but_diverged_is_red(tmp_path):
    v = assess(_cam_imu_json(tmp_path, reproj_median_px=40.0, reproj_p95_px=171.0))
    assert v.grade == "RED"
    assert "reproj.median" in _ids(v, "fail")


def test_verdict_iteration_cap_warns(tmp_path):
    p = _cam_imu_json(tmp_path, diagnostics=dict(iterations=40, max_iterations=40))
    v = assess(p)
    assert v.grade == "YELLOW" and "solve.converged" in _ids(v, "warn")


def test_verdict_reference_lever(tmp_path):
    p = _cam_imu_json(tmp_path)
    v = assess(p, {"label": "CAD", "lever_mm": {"cam0": [40.0, 0.0, 0.0]}})
    assert "ref.lever" in _ids(v, "warn")


def test_verdict_camera_only(tmp_path):
    d = {"cameras": [{"id": 0, "intrinsics": {"fx": -5.0, "fy": 500.0, "cx": 320, "cy": 240},
                      "distortion": {"coeffs": [0.1, 0.0, 0.0, 0.0]}, "image_size": [640, 480]}],
         "reprojection_errors": {"per_camera": [
             {"camera": 0, "rms": 0.4, "median": 0.3, "p95": 0.7, "num_observations": 4000}]}}
    p = tmp_path / "calib.yaml"
    p.write_text(yaml.safe_dump(d))
    v = assess(str(p))
    assert v.grade == "RED" and "intr.focal_positive" in _ids(v, "fail")


def _write_cfg(tmp_path, d):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


def _cam_cfg(tmp_path, **board):
    img = tmp_path / "imgs"
    img.mkdir(exist_ok=True)
    b = dict(board_type="apriltag", rows=6, cols=6, tag_size=0.088, tag_spacing=0.3,
             tag_family="t36h11")
    b.update(board)
    return dict(mode="cam", input={"cameras": [dict(camera_id=0, source_type="images",
                                                    source_path=str(img))]}, board=b)


def _fails(found):
    return {f.id for f in found if f.status == "fail"}


def test_check_clean_cam_config(tmp_path):
    assert not _fails(check_config(_write_cfg(tmp_path, _cam_cfg(tmp_path))))


def test_check_missing_mode_fails(tmp_path):
    d = _cam_cfg(tmp_path)
    del d["mode"]
    assert "config.mode" in _fails(check_config(_write_cfg(tmp_path, d)))


@pytest.mark.parametrize("board,fid", [
    (dict(tag_family="tag36h11"), "board.tag_family"),
    (dict(tag_spacing=26.4), "board.tag_spacing"),          # mm instead of ratio
    (dict(tag_size=88.0), "board.tag_size"),                # mm instead of m
])
def test_check_board_mistakes(tmp_path, board, fid):
    assert fid in _fails(check_config(_write_cfg(tmp_path, _cam_cfg(tmp_path, **board))))


def test_check_unknown_key_fails(tmp_path):
    d = _cam_cfg(tmp_path)
    d["board"]["tag_sise"] = 0.088
    assert "config.unknown_key" in _fails(check_config(_write_cfg(tmp_path, d)))


def test_check_cam_imu_needs_prior(tmp_path):
    bag = tmp_path / "x.bag"
    bag.write_bytes(b"")
    d = dict(mode="cam-imu", input=dict(format="ros1", path=str(bag), cam_topics=["/c"],
                                        imu_topics=["/i"]),
             board=_cam_cfg(tmp_path)["board"], output=dict(path=str(tmp_path / "o.json")))
    assert "camera_input.path" in _fails(check_config(_write_cfg(tmp_path, d)))


@pytest.mark.parametrize("out_path,warns", [("cam_imu.json", True), ("res/cam_imu.json", False)])
def test_check_cam_imu_bare_output_path_warns(tmp_path, out_path, warns):
    """A bare cam-IMU output.path writes the summary and its sidecars into the cwd: a warning,
    not a failure."""
    bag = tmp_path / "x.bag"
    bag.write_bytes(b"")
    d = dict(mode="cam-imu", input=dict(format="ros1", path=str(bag), cam_topics=["/c"],
                                        imu_topics=["/i"]),
             board=_cam_cfg(tmp_path)["board"], output=dict(path=out_path))
    found = check_config(_write_cfg(tmp_path, d))
    got = [f for f in found if f.id == "output.path"]
    assert [f.status for f in got] == (["warn"] if warns else [])
    assert "output.path" not in _fails(found)


def _euroc(tmp_path, accel_scale=1.0):
    from PIL import Image
    root = tmp_path / "seq" / "mav0"
    (root / "cam0" / "data").mkdir(parents=True)
    (root / "imu0").mkdir()
    t0 = 1_000_000_000_000
    with open(root / "cam0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for i in range(5):
            name = f"{t0 + i * 50_000_000}.png"
            Image.fromarray(np.zeros((48, 64), np.uint8)).save(root / "cam0" / "data" / name)
            f.write(f"{t0 + i * 50_000_000},{name}\n")
    with open(root / "imu0" / "data.csv", "w") as f:
        f.write("#t,wx,wy,wz,ax,ay,az\n")
        for i in range(400):
            f.write(f"{t0 + i * 5_000_000},0.01,0.0,0.0,0.0,0.0,{9.81 * accel_scale}\n")
    return str(tmp_path / "seq")


def test_inspect_euroc(tmp_path):
    rep = inspect(_euroc(tmp_path))
    assert rep.format == "euroc" and "cam-imu" in rep.supported_modes
    cam = next(s for s in rep.streams if s.kind == "camera")
    imu = next(s for s in rep.streams if s.kind == "imu")
    assert cam.info["width"] == 64 and cam.rate_hz == pytest.approx(20.0)
    assert imu.rate_hz == pytest.approx(200.0)
    assert not [f for f in rep.findings if f.status == "fail"]


def test_inspect_accel_in_g_fails(tmp_path):
    rep = inspect(_euroc(tmp_path, accel_scale=1 / 9.81))
    assert any(f.id == "imu.accel_units" and f.status == "fail" for f in rep.findings)


@pytest.mark.parametrize("name", ["cam.yaml", "cam_imu.yaml"])
def test_skill_templates_have_no_unknown_keys(tmp_path, name):
    """Drift guard: the mvis-calibrate templates only use keys the parser knows."""
    import re
    src = open(os.path.join(REPO, ".claude/skills/mvis-calibrate/templates", name)).read()
    numeric = {"rows", "cols", "tag_size", "tag_spacing"}
    src = re.sub(r"^(\s*-?\s*(\w+)):\s*<[^>\n]*>", lambda m: f"{m.group(1)}: "
                 + ("6" if m.group(2) in ("rows", "cols") else
                    "0.1" if m.group(2) in numeric else "x"), src, flags=re.M)
    src = re.sub(r"\[<[^\]]*\]", "[/a]", src)
    p = tmp_path / name
    p.write_text(src)
    found = check_config(str(p))
    assert not [f for f in found if f.id in ("config.unknown_key", "config.parse", "config.mode")], \
        [(f.id, f.message) for f in found]


# --- Stage 0: installation verification (`mvis check --env [--smoke]`) ---

from mvis.doctor import check_config as cc


def test_config_needs(tmp_path):
    d = _cam_cfg(tmp_path)
    assert cc.config_needs(_write_cfg(tmp_path, d)) == {"apriltag"}
    d = dict(mode="cam-imu", input=dict(format="ros1", path="x.bag"),
             board=dict(board_type="aruco"))
    assert cc.config_needs(_write_cfg(tmp_path, d)) == {"aruco", "ros1"}


@pytest.mark.parametrize("needs,status", [(set(), "warn"), ({"ros1"}, "fail")])
def test_env_missing_extra(monkeypatch, needs, status):
    real = cc.importlib.import_module

    def fake(name, *a, **k):
        if name == "rosbags":
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(cc.importlib, "import_module", fake)
    f = next(f for f in cc.check_env(needs) if f.id == "env.extra.ros1")
    assert f.status == status and "pymvis[ros1]" in f.message


def test_env_cv2_version_clash(monkeypatch):
    vers = {"opencv-python": "4.8.0", "opencv-contrib-python": "4.13.0"}

    def version(d):
        if d in vers:
            return vers[d]
        raise cc._md.PackageNotFoundError(d)
    monkeypatch.setattr(cc._md, "version", version)
    f = next(f for f in cc._check_cv2(False) if f.id == "env.cv2")
    assert f.status == "warn"


def test_env_smoke_solve_recovers_intrinsics():
    f = cc.smoke_solve()
    assert f.status == "pass", f.message


@pytest.mark.parametrize("ver,status", [("4.2", "pass"), ("4.2.2", "pass"), ("4.3.0", "warn")])
def test_env_gtsam_version(monkeypatch, ver, status):
    real = cc._version
    monkeypatch.setattr(cc, "_version", lambda m: ver if m == "gtsam" else real(m))
    f = next(f for f in cc.check_env() if f.id == "env.gtsam")
    assert f.status == status


def _agent_argv(cmd, tmp_path):
    if cmd == "inspect":
        return ["inspect", _euroc(tmp_path)]
    if cmd == "check":
        return ["check", "-c", _write_cfg(tmp_path, _cam_cfg(tmp_path))]
    return ["verdict", _cam_imu_json(tmp_path)]


@pytest.mark.parametrize("cmd,key,ok", [
    ("inspect", "format", ("euroc",)),
    ("check", "status", ("pass", "warn")),
    ("verdict", "grade", ("GREEN",)),
])
def test_out_creates_missing_run_folder(tmp_path, monkeypatch, capsys, cmd, key, ok):
    """`mvis <cmd> ... --out <run>/x.json` works before `<run>/` exists.

    Driven through main() so the argparse `--out` wiring is covered too."""
    import sys
    from mvis.cli.main import main
    out = tmp_path / "agent_runs" / "new-run" / f"{cmd}.json"
    monkeypatch.setattr(sys, "argv", ["mvis", *_agent_argv(cmd, tmp_path), "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code in (0, 1)
    assert json.loads(out.read_text())[key] in ok


def _allowlist():
    with open(os.path.join(REPO, ".claude", "settings.json")) as f:
        return json.load(f)["permissions"]["allow"]


def test_settings_allowlist_matches_cli(monkeypatch, capsys):
    """Drift guard: every pre-approved `mvis ...` command parses.

    `Bash(mvis <sub>:*)` must be a real subcommand (`<sub> --help` exits 0);
    an exact rule `Bash(mvis <args>)` must itself exit 0."""
    import re
    import sys
    from mvis.cli.main import main
    rules = [m.groups() for r in _allowlist()
             if (m := re.fullmatch(r"Bash\(mvis ([^:)]*)(:\*)?\)", r))]
    assert rules
    for args, prefix in rules:
        argv = ["mvis", *args.split()] + (["--help"] if prefix else [])
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code in (0, None), f"allowlisted `{' '.join(argv)}` exits {e.value.code}"


def test_skill_creates_solve_log_folders():
    """Stage 5 redirects logs into <run>/05_solve/attempt-NN/; the shell needs
    the folder before mvis starts, so the skill must create it."""
    with open(os.path.join(REPO, ".claude", "skills", "mvis-calibrate", "SKILL.md")) as f:
        skill = f.read()
    stage5 = skill[skill.index("## Stage 5"):skill.index("## Stage 6")]
    assert "> <run>/05_solve/attempt-01/" in stage5
    assert stage5.index("mkdir -p <run>/05_solve/attempt-01") < stage5.index("mvis run -c")
