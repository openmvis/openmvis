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
"""End-to-end HTML generation for both calibration reports (html-reports H6).

Builds a complete camera-only report (generate_calibration_report) and a
complete cam-IMU report (write_cam_imu_report) from small synthetic results
with every interactive feature switched on — two cameras, real thumbnails
with an index, detections, config + result files — and checks the written
page is well formed and self-consistent:

  * balanced block tags (div / details / aside / table / select)
  * every inline script parses (node --check; skipped without node)
  * every progress spec parses, its trace indices exist in its figure, and
    its views / panel / co-visibility / packed corner pixels agree
  * the section layout: 3D section, folded Data Quality (cam-only), Result
    files + Configuration closing the report, OpenMVIS branding
"""

import base64
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser

import numpy as np
import pytest
import gtsam

from mvis.analysis.cam_imu_report import write_cam_imu_report
from mvis.analysis.generate_report import generate_calibration_report
from mvis.core.types import (
    CalibrationResult, CameraParams, ExtrinsicTransform, FrameObservations,
    Observation, ReprojectionErrorStats)
from mvis.detection.board_factory import create_board
from mvis.io.detection_pipeline import write_thumbnail
from types import SimpleNamespace as NS

W, H, F = 320, 240, 300.0
BOARD = create_board(NS(board_type="apriltag", rows=2, cols=3, tag_size=0.05,
                        tag_spacing=0.3, tag_family="t36h11"))


# --------------------------------------------------------------------- checks
class _Balance(HTMLParser):
    BLOCK = ("div", "details", "aside", "table", "select", "summary")

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag not in self.BLOCK:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1] if self.stack else None}>")
        else:
            self.stack.pop()


def _assert_balanced(html):
    # scripts / styles hold "<" in code; strip them before tag balancing
    body = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S)
    p = _Balance()
    p.feed(body)
    assert not p.errors, p.errors[:5]
    assert not p.stack, f"unclosed: {p.stack[-5:]}"


def _assert_scripts_parse(html, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available for the inline-script syntax check")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts
    for k, js in enumerate(scripts):
        if len(js) > 1_000_000:          # the inlined plotly.js bundle
            continue
        f = tmp_path / f"s{k}.js"
        f.write_text(js)
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:500]


def _figure_traces(html, div_id):
    """Trace list of the plotly figure in div `div_id` (from its newPlot call)."""
    m = re.search(r'Plotly\.newPlot\(\s*"' + re.escape(div_id) + r'",\s*', html)
    assert m, f"no newPlot for {div_id}"
    j = m.end()
    data, _ = json.JSONDecoder().raw_decode(html[j:])
    return data


def _arr_len(a):
    """Length of a plotly array: a list, or a packed {"bdata", "dtype"} dict."""
    if isinstance(a, dict) and "bdata" in a:
        return len(base64.b64decode(a["bdata"])) // np.dtype(a["dtype"]).itemsize
    return len(a)


def _progress_specs(html):
    out = []
    for m in re.finditer(r'mvisProgress\("([^"]+)",', html):
        spec, _ = json.JSONDecoder().raw_decode(html[m.end():])
        out.append((m.group(1), spec))
    return out


def _check_spec(html, div_id, spec, n_views):
    traces = _figure_traces(html, div_id)
    n = len(spec["xyz"])
    assert n >= 2 and len(spec["R"]) == n
    assert spec["t"] is None or len(spec["t"]) == n
    idx = [spec["done"], spec["cur"], *spec["glyph"].values()]
    assert all(0 <= k < len(traces) for k in idx)
    assert traces[spec["glyph"]["frustum"]]["name"] == "current frustum"
    assert traces[spec["glyph"]["image"]]["type"] == "surface"
    views = spec["views"]
    assert len(views) == n_views
    for v in views:
        raw = base64.b64decode(v["img"])
        assert len(raw) == len(v["img_idx"]) * v["iw"] * v["ih"]
        assert v["img_idx"][-1] == n - 1
        p = v["panel"]
        assert len(p["uvo"]) == n + 1 and len(p["cov"]) == n == len(p["n"])
        uv = np.frombuffer(base64.b64decode(p["uvb"]), "<i2")
        assert len(uv) == p["uvo"][-1]
        assert p["n"][-1] * 2 == len(uv)            # n counts every packed corner
        for jpg in p["jpg"]:
            assert base64.b64decode(jpg)[:2] == b"\xff\xd8"
        assert len(v["ids"]) == n
    if n_views > 1:
        assert 0 <= spec["covis"]["trace"] < len(traces)
        assert len(spec["covis"]["color"]) == _arr_len(traces[spec["covis"]["trace"]]["x"])
        assert html.count('class="mvis-card') == n_views
        assert f'<option value="{n_views - 1}">' in html
    return views


def _section_ids(html):
    return re.findall(r'<div class="section(?: section-alt)?" id="([^"]+)"', html)


# ------------------------------------------------------------------- fixtures
def _project(T_board_cam):
    """Pixel detections of every board point visible from this pose."""
    T_cb = np.linalg.inv(T_board_cam)
    obs = []
    for pid, p in enumerate(BOARD.all_points_3d()):
        q = T_cb[:3, :3] @ p + T_cb[:3, 3]
        if q[2] <= 0:
            continue
        u, v = F * q[0] / q[2] + W / 2, F * q[1] / q[2] + H / 2
        if 0 <= u < W and 0 <= v < H:
            obs.append(Observation(pid, np.array([u, v])))
    return obs


def _ref_poses(n=24):
    cx, cy = BOARD.all_points_3d()[:, :2].mean(0)
    poses = []
    for k in range(n):
        a = 2 * np.pi * k / n
        T = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.15 * np.sin(a), 0.15 * np.cos(a), 0.0])),
                        gtsam.Point3(cx + 0.05 * np.sin(a), cy + 0.03 * np.cos(a), -0.45)).matrix()
        poses.append(T)
    return poses


def _frames_and_thumbs(tmp_path, cid, poses, times):
    frames = []
    for k, (T, t) in enumerate(zip(poses, times)):
        obs = _project(T)
        frames.append(FrameObservations(camera_id=cid, timestamp=float(t),
                                        observations=obs, image_size=(W, H)))
        write_thumbnail(tmp_path / "thumbnails", BOARD, cid, k,
                        np.full((H, W), 90, np.uint8), obs, float(t), reset=(k == 0))
    return frames


def _cam(cid):
    return CameraParams(camera_id=cid, intrinsics=np.array([F, F, W / 2, H / 2]),
                        distortion=np.zeros(4), distortion_model="radtan",
                        image_size=(W, H))


# ---------------------------------------------------------------------- tests
def test_cam_only_report_two_cameras_end_to_end(tmp_path):
    poses = _ref_poses()
    times = 100.0 + 0.1 * np.arange(len(poses))
    T_ext = np.eye(4); T_ext[0, 3] = -0.04          # ext T_to_from (ref <- cam1)
    cam1_poses = [T @ np.linalg.inv(T_ext) for T in poses]
    frames = {0: _frames_and_thumbs(tmp_path, 0, poses, times),
              1: _frames_and_thumbs(tmp_path, 1, cam1_poses, times + 0.001)}
    result = CalibrationResult(
        cameras=[_cam(0), _cam(1)],
        extrinsics=[ExtrinsicTransform(1, 0, T_ext, np.zeros(3), T_ext[:3, 3])],
        reprojection_errors=[ReprojectionErrorStats(
            camera_id=c, rms=0.3, mean=0.25, median=0.2, p90=0.5, p95=0.6,
            max=0.9, num_observations=500) for c in (0, 1)],
        overall_rms=0.3, board_poses=poses, board_pose_times=list(times),
        reference_camera=0)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("board:\n  board_type: apriltag\nsolve:\n  max_iterations: 15\n")
    calib = tmp_path / "run_calib.yaml"
    calib.write_text("cam0:\n  fx: 300.0\n")
    out = tmp_path / "report.html"
    generate_calibration_report(
        result=result, all_frames=frames, board=BOARD, calibrations={},
        image_sizes={0: (W, H), 1: (W, H)}, title="e2e / cam-only",
        output_path=str(out), thumb_dir=tmp_path / "thumbnails",
        calibration_info={"config_path": str(cfg), "result_files": [str(calib)],
                          "corner_subsample": "top_left"})
    html = out.read_text()

    _assert_balanced(html)
    _assert_scripts_parse(html, tmp_path)
    from tests.io.test_cam_imu_report import view_specs
    specs = view_specs(html)
    assert set(specs) == {"scene", "rig"} and 'mvisProgress("' not in html
    scene, rig = specs["scene"], specs["rig"]
    assert scene["v"] == 2 and scene["origin"] == "camera" and len(scene["xyz"]) == len(poses)
    assert scene["t"][-1] == pytest.approx(times[-1] - times[0])
    assert [c["name"] for c in scene["cams"]] == ["cam0", "cam1"] and scene["cams"][1]["p"] == [0.04, 0.0, 0.0]
    assert all(any(ids) for ids in scene["cams"][1]["ids"])          # cam1 matched every pose
    assert all(c["jpg"] for c in scene["cams"]) and len(scene["board"]["pts"]) == 24
    assert rig["origin"] == "cam0" and [c["p"] for c in rig["cams"]] == [[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]
    assert html.count("MVIS report views: the calibration process player") == 1   # the player once

    assert "<h1>OpenMVIS</h1>" in html and "Calibration Process" in html
    ids = _section_ids(html)
    assert ids.index("sec-3d") < ids.index("sec-results") < ids.index("sec-config")
    assert "sec-appendix-a" not in ids or ids.index("sec-config") < ids.index("sec-appendix-a")
    q = html[html.index('id="sec-quality"'):]
    assert q.index('<details class="appendix">') < q.index("</h2>") + 200   # folded
    assert "Corner Coverage" not in html
    cfg_part = html[html.index('id="sec-config"'):]
    assert "max_iterations: 15" in cfg_part and 'class="copy-btn"' in cfg_part
    assert "run_calib.yaml</code> &middot; YAML" in html


def test_cam_imu_report_two_cameras_end_to_end(tmp_path):
    from tests.helpers.cam_imu_report_fakes import _fake_calib, _fake_result
    r = _fake_result(aux=True)
    r.cameras.append(dict(r.cameras[0], calib=_fake_calib(trans=(0.09, -0.02, 0.01))))
    n = 30
    X = gtsam.symbol_shorthand.X
    B0 = gtsam.symbol("b", 0)
    vals = gtsam.Values()
    for k in range(n):
        a = 2 * np.pi * k / n
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.1 * np.sin(a), 0.1, 0.0])),
                                      gtsam.Point3(0.1 * np.cos(a), 0.05 * np.sin(a), 0.0)))
    # board 0.5 m ahead along world +z, facing the rig
    vals.insert(B0, gtsam.Pose3(gtsam.Rot3.Rx(np.pi), gtsam.Point3(-0.05, 0.05, 0.5)))
    r.values, r.T_BinW = vals, vals.atPose3(B0)
    r.t_states = 50.0 + 0.05 * np.arange(n)
    cam_data = []
    for c, e in enumerate(r.cameras):
        cal = e["calib"]
        poses = [r.T_BinW.inverse().compose(vals.atPose3(X(k))).compose(cal.T).matrix()
                 for k in range(n)]
        frames = _frames_and_thumbs(tmp_path / f"c{c}", c, poses, r.t_states - cal.td)
        cam_data.append(dict(frames=frames, image_size=(W, H),
                             thumb_dir=tmp_path / f"c{c}" / "thumbnails"))
    res = tmp_path / "cam_imu.json"
    res.write_text('{"ok": true}')
    out = tmp_path / "cam_imu_report.html"
    write_cam_imu_report(r, html_path=str(out), board=BOARD,
                         config="solve:\n  max_iterations: 40\n",
                         cam_data=cam_data, result_files=[str(res)])
    html = out.read_text()

    _assert_balanced(html)
    _assert_scripts_parse(html, tmp_path)
    from tests.io.test_cam_imu_report import view_specs
    specs = view_specs(html)
    assert set(specs) == {"scene", "rig"}
    scene, rig = specs["scene"], specs["rig"]
    assert scene["v"] == 2 and scene["unit"] == "s" and len(scene["xyz"]) == n
    assert scene["t"][0] == 0.0 and abs(scene["t"][-1] - 0.05 * (n - 1)) < 1e-6
    assert [c["p"] for c in scene["cams"]] == [[0.05, -0.02, 0.01], [0.09, -0.02, 0.01]]
    for c in scene["cams"]:
        assert len(c["uvo"]) == n + 1 and len(c["ids"]) == n and len(c["outline"]) == n
        assert c["jpg"] and len(c["jpg"]) == len(c["jpg_idx"])      # frames stay in the report
    assert [c["name"] for c in rig["cams"]] == ["cam0", "cam1"] and all(len(c["rim"]) >= 16 for c in rig["cams"])
    side = json.loads((tmp_path / "cam_imu_views.json").read_text())
    assert all("jpg" not in c for c in side["scene"]["cams"]) and "/9j/" not in json.dumps(side)
    assert side["scene"]["cams"][0]["uvb"] == scene["cams"][0]["uvb"]

    assert "<h1>OpenMVIS</h1>" in html and "Calibration process" in html
    ids = _section_ids(html)
    assert ids[-2:] == ["sec-results", "sec-config"]
    assert "cam_imu.json</code> &middot; JSON" in html
    assert "max_iterations: 40" in html[html.index('id="sec-config"'):]
