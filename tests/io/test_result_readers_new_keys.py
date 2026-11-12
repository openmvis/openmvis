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
"""Readers of the result files tolerate the board-shape keys and the readout sigma: `board_shape`,
`board_point_bias`, `cameras[].rolling_shutter_readout_sigma` (cam-only YAML), `board.shape` and the
`<out>_board_shape.yaml` sidecar (cam-IMU). A reader's output must not change when they are present."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from mvis.analysis.verdict import assess
from mvis.cli.main import apply_board_heights, write_board_shape
from mvis.core.types import CalibrationResult, CameraParams, ReprojectionErrorStats
from mvis.io.output_writer import OutputWriter

REPO = Path(__file__).parents[2]
N = 8
BOARD_SHAPE = {"model": "pose + per-corner height", "heights_m": [0.001] * N, "rms_mm": 0.1,
               "peak_to_peak_mm": 0.5}
BIAS = {"median_px": 0.2, "max_px": 0.9, "n_points_seen": N}
HEIGHTS = {"heights_m": np.linspace(-1e-3, 1e-3, N), "offset_m": np.zeros(N), "initial_m": np.zeros(N),
           "sigma_m": None, "n_obs": np.full(N, 5), "rms_mm": 0.6, "p2p_mm": 2.0, "offset_rms_mm": 0.6,
           "n_seen": N}


def _result(new_keys):
    cam = CameraParams(camera_id=0, intrinsics=np.array([500., 500., 320., 240.]), distortion=np.zeros(4),
                       distortion_model="radtan", image_size=(640, 480), rolling_shutter_readout=0.03,
                       rolling_shutter_readout_sigma=1e-4 if new_keys else None)
    kw = dict(board_heights=np.linspace(-1e-3, 1e-3, N),
              board_point_bias=np.column_stack([np.full(N, .1), np.zeros(N), np.full(N, 5.)])) if new_keys else {}
    return CalibrationResult(cameras=[cam], extrinsics=[], reference_camera=0, overall_rms=0.3,
                             reprojection_errors=[ReprojectionErrorStats(0, 0.3, 0.3, 0.3)], **kw)


def test_writer_emits_the_keys_only_when_there_is_something_to_say(tmp_path):
    new, old = tmp_path / "n.yaml", tmp_path / "o.yaml"
    OutputWriter.write_calibration_yaml(_result(True), str(new))
    OutputWriter.write_calibration_yaml(_result(False), str(old))
    a, b = yaml.safe_load(new.read_text()), yaml.safe_load(old.read_text())
    assert {"board_shape", "board_point_bias"} <= set(a) and not {"board_shape", "board_point_bias"} & set(b)
    assert a["cameras"][0]["rolling_shutter_readout_sigma"] == 1e-4
    assert "rolling_shutter_readout_sigma" not in b["cameras"][0]


def test_regenerate_html_loads_a_yaml_with_the_keys(tmp_path):
    from mvis.analysis.regenerate_html import _result_from_yaml
    new, old = tmp_path / "n.yaml", tmp_path / "o.yaml"
    OutputWriter.write_calibration_yaml(_result(True), str(new))
    OutputWriter.write_calibration_yaml(_result(False), str(old))
    a, b = _result_from_yaml(new), _result_from_yaml(old)
    assert a.cameras[0].rolling_shutter_readout == b.cameras[0].rolling_shutter_readout == 0.03
    assert a.overall_rms == b.overall_rms


def test_regenerate_html_does_not_take_the_sidecar_for_a_result(tmp_path, monkeypatch):
    """A cam-IMU run dir has only the sidecar outside the *_calib pattern; it must stay a skip."""
    from mvis.analysis import regenerate_html as rh
    read = []
    monkeypatch.setattr(rh, "_result_from_yaml", lambda p: read.append(p) or _result(False))
    (tmp_path / "cam_imu_calib.yaml").write_text("cameras: []\n")
    write_board_shape(HEIGHTS, str(tmp_path / "cam_imu_board_shape.yaml"), "x.bag")
    assert rh.regenerate_one(tmp_path) is False
    assert not read and not list(tmp_path.glob("*.html"))


def test_cam_stats_reads_the_same_numbers(tmp_path, monkeypatch):
    from mvis.regression import cam_stats
    from mvis.regression.manifest import RunSpec
    monkeypatch.setattr(cam_stats, "RESULTS_ROOT", tmp_path)
    got = {}
    for new_keys, coll in ((True, "n"), (False, "o")):
        d = tmp_path / "ds" / coll
        d.mkdir(parents=True)
        OutputWriter.write_calibration_yaml(_result(new_keys), str(d / f"ds_{coll}.yaml"))
        s = cam_stats.collect_stats(RunSpec(mode="cam", dataset="ds", collection=coll, config=""))
        got[new_keys] = {k: v for k, v in s.items() if k != "report_paths"}
    assert got[True]["overall_rms"] == 0.3
    assert got[True] == got[False]


def _summary(tmp_path, with_shape):
    d = dict(ok=True, reproj_rms_px=0.3, reproj_median_px=0.2, reproj_p95_px=0.6,
             cams=[dict(name="cam0", td_ms=1.0, reproj=0.3, reproj_median=0.2, reproj_p95=0.6,
                        p_CinI_mm=[10.0, 0.0, 0.0], rotvec=[0, 0, 0])],
             imu_model="calyx", Dw_diag=[1.0, 1.01, 0.99], Da_diag=[1.0, 1.0, 1.0], imus=[],
             diagnostics=dict(iterations=12, max_iterations=40, cam_chi2_dof=0.8, imu_chi2_dof=0.9),
             board=dict(T_BinW=np.eye(4).tolist(), g_inB=[0, 0, -1], normal_tilt_deg=0.0, g_inplane_deg=0.0))
    if with_shape:
        d["board"]["shape"] = write_board_shape(HEIGHTS, str(tmp_path / "cam_imu_board_shape.yaml"), "x.bag")
    p = tmp_path / ("s.json" if with_shape else "p.json")
    p.write_text(json.dumps(d))
    return p, d


def test_cam_imu_gate_metrics_and_verdict_ignore_board_shape(tmp_path):
    from mvis.regression.cam_imu_adapter import CamImuAdapter
    ps, a = _summary(tmp_path, True)
    pp, b = _summary(tmp_path, False)
    assert a["board"]["shape"]["n_seen"] == N
    assert CamImuAdapter().extract_metrics(a) == CamImuAdapter().extract_metrics(b)
    def rows(p):
        return [(c.id, c.status) for c in assess(str(p)).checks]
    assert rows(ps) == rows(pp)


def test_sidecar_is_a_valid_heights_file(tmp_path):
    from types import SimpleNamespace
    p = tmp_path / "cam_imu_board_shape.yaml"
    write_board_shape(HEIGHTS, str(p), "x.bag")
    pts = np.zeros((N, 3))
    out = apply_board_heights(pts, SimpleNamespace(heights_file=str(p)))
    assert np.allclose(out[:, 2], HEIGHTS["heights_m"])
    assert yaml.safe_load(p.read_text())["board_shape"]["sigma_m"] is None


def test_cam_imu_html_and_text_report_with_heights_in_diagnostics():
    from mvis.analysis.cam_imu_report import cam_imu_html_report, cam_imu_text_report
    from tests.helpers.cam_imu_report_fakes import _fake_result
    r = _fake_result()
    r.diagnostics = dict(r.diagnostics, board_heights=HEIGHTS)
    assert "sec-summary" in cam_imu_html_report(r)
    assert cam_imu_text_report(r)


def test_mvisdb_estimates_hold_parameters_only():
    """Attaching a result ignores the diagnostics: same estimate states with and without them."""
    from mvis.mvisdb.estimates import estimates_from_calibration_result as to_es
    a, b = to_es(_result(True)), to_es(_result(False))
    assert a.sensor_states[0].intrinsics.readout_s == 0.03
    assert repr(a) == repr(b)


def _gen_stats(results, baselines, out):
    env = dict(os.environ, MVIS_BASELINES=str(baselines))
    subprocess.run([sys.executable, str(REPO / "scripts/regression_report/gen_stats.py"), "--results", str(results),
                    "--out", str(out)], check=True, env=env, capture_output=True)
    return json.loads(out.read_text())


def test_gen_stats_output_does_not_depend_on_the_new_keys(tmp_path):
    """One camera-only RS family and one cam-IMU family, written with and without the keys."""
    base = tmp_path / "baselines"
    for mode in ("cam", "cam_imu"):
        (base / mode).mkdir(parents=True)
    rs, imu = ["stereo_rs_bag1", "stereo_rs_bag2"], ["imu1", "imu2"]
    for coll in rs:
        (base / "cam" / f"mvis_elp__{coll}.json").write_text('{"reproj_rms_px": 0.3}')
    for coll in imu:
        (base / "cam_imu" / f"tum_vi_raw__{coll}.json").write_text('{"reproj_rms_px": 0.3}')
    stats = []
    for new_keys in (False, True):
        root = tmp_path / f"r{int(new_keys)}"
        for coll in rs:
            d = root / "mvis_elp" / coll
            d.mkdir(parents=True)
            OutputWriter.write_calibration_yaml(_result(new_keys), str(d / f"mvis_elp_{coll}_calib.yaml"))
        for coll in imu:
            d = root / "tum_vi_raw" / coll
            d.mkdir(parents=True)
            summary = dict(t_ga_ms=1.0, cams=[dict(p_CinI_mm=[1.0, 0, 0])], board={"T_BinW": []})
            cy = dict(imus=[dict(role="base")], cameras=[
                dict(intrinsics=dict(fx=500., fy=500., cx=320., cy=240.), distortion=dict(coeffs=[0.1]),
                     rolling_shutter_readout=0.03)])
            if new_keys:
                summary["board"]["shape"] = write_board_shape(HEIGHTS, str(d / "cam_imu_board_shape.yaml"), "x.bag")
                summary["board"]["shape"]["file"] = "cam_imu_board_shape.yaml"
            (d / "cam_imu.json").write_text(json.dumps(summary))
            (d / "cam_imu_calib.yaml").write_text(yaml.safe_dump(cy))
        stats.append(_gen_stats(root, base, tmp_path / f"s{int(new_keys)}.json"))
    assert stats[0]["cam/mvis_elp/stereo_rs"]["stats"]["cam0.tr_ms"][0] == pytest.approx(30.0)
    assert stats[0]["cam_imu/tum_vi_raw/imu"]["stats"]["cam0.fx"][0] == 500.0
    assert stats[0] == stats[1]


def test_board_shape_writer_creates_a_missing_output_folder(tmp_path):
    # cam-IMU writes the board shape before cam_imu.json, into a folder that may not exist yet
    path = tmp_path / "fresh" / "row" / "cam_imu_board_shape.yaml"
    out = write_board_shape(HEIGHTS, str(path), "x.bag")
    assert path.exists() and out["file"] == str(path)
    assert len(yaml.safe_load(path.read_text())["board_shape"]["heights_m"]) == N
