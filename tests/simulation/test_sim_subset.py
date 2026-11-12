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
"""mvis.simulation.run.sims_to_cams and mvis.simulation.synth.run_subset (solve stubbed)."""
import json
from types import SimpleNamespace

import gtsam
import numpy as np

from mvis.simulation import synth
from mvis.simulation.from_result import GroundTruth, gt_to_json
from mvis.simulation.run import sims_to_cams

CFG = "examples/regression/mvis_4imus_3cams/cam_imu_april_01.yaml"


def test_sims_to_cams_pins_intrinsics_and_estimates_readout_only_when_rolling():
    sim = dict(t_cam=[0.0], ids_list=[[1]], uv_list=[[[1.0, 2.0]]], cal_vec=[400.0], is_fisheye=False,
               model="radtan", image_size=(8, 6), readout=0.0, extra="dropped")
    gs, rs = sims_to_cams([sim, {**sim, "readout": 0.02}])
    assert gs["pin_intrinsics"] is True and gs["rolling_mode"] == 0
    assert gs["estimate_readout"] is False and rs["estimate_readout"] is True
    assert "extra" not in gs and "readout" not in gs
    assert gs["model"] == "radtan" and gs["image_size"] == (8, 6) and gs["cal_vec"] == [400.0]


def _gt_file(path):
    ts = 1.0 + np.arange(90) / 30.0
    states = []
    for t in ts:
        M = np.eye(4)
        M[:3, :3] = gtsam.Rot3.Rz(0.4 * np.sin(t)).matrix()
        M[:3, 3] = [0.2 * np.sin(t), 0.2 * np.cos(t), 0.0]
        states.append(M.tolist())
    Tb = np.eye(4)
    Tb[:3, 3] = [-0.5, -0.5, 1.2]
    cam = dict(intr8=[400., 400., 320., 240., 0., 0., 0., 0.], T_CtoI=np.eye(4).tolist(), td=0.0,
               is_fisheye=False, image_size=[640, 480], frame_times=ts[5:-5:3].tolist())
    imu = dict(name="base", is_base=True, xi24=[0.0] * 24, T_IstoI=None, td=0.0, sample_times=ts.tolist())
    gt = GroundTruth(cams=[cam, {**cam, "td": 0.001}],
                     imus=[imu, {**imu, "name": "aux", "is_base": False, "T_IstoI": np.eye(4).tolist()},
                           {**imu, "name": "aux2", "is_base": False, "T_IstoI": np.eye(4).tolist()}],
                     b_g=[0.0] * 3, b_a=[0.0] * 3, T_BinW=Tb.tolist(), t_states=ts.tolist(), T_states=states,
                     imu_model="calyx", g_inW=[0.0, 0.0, -9.81])
    gt_to_json(gt, str(path))


def _stub_solve(monkeypatch):
    seen = {}

    def recover(cams, pts, t_imu, w_m, a_m, aux, solve):
        seen.update(cams=cams, aux=aux, solve=solve, t_imu=t_imu, pts=pts)
        return SimpleNamespace(diagnostics={"reproj_rms_px": 0.5})

    monkeypatch.setattr(synth, "recover", recover)
    monkeypatch.setattr(synth, "zscore_table", lambda gt, res: [{"pass_": True}, {"pass_": False}])
    return seen


def test_run_subset_uses_cam0_and_the_first_imus(tmp_path, monkeypatch):
    gt_path = tmp_path / "gt.json"
    _gt_file(gt_path)
    seen = _stub_solve(monkeypatch)
    calls = []
    out_path = tmp_path / "sub" / "out.json"
    gt, out = synth.run_subset(str(gt_path), CFG, str(out_path), n_imus=2, seed=3, sigma_px=0.5,
                               aux_imu_stride=7, on_synth=lambda *a: calls.append(a))
    assert len(gt.cams) == 1 and [m["name"] for m in gt.imus] == ["base", "aux"]
    assert len(seen["cams"]) == 1 and len(seen["aux"]) == 1 and len(seen["t_imu"]) > 0
    assert seen["solve"].aux_imu_stride == 7
    assert len(calls) == 1 and calls[0][0].solve.aux_imu_stride == 7
    assert out == {"seed": 3, "sigma_px": 0.5, "reproj_rms_px": 0.5, "rows": [{"pass_": True}, {"pass_": False}],
                   "n_fail": 1}
    assert json.loads(out_path.read_text()) == out


def test_run_subset_keeps_the_config_stride_and_a_base_only_rig(tmp_path, monkeypatch):
    gt_path = tmp_path / "gt.json"
    _gt_file(gt_path)
    seen = _stub_solve(monkeypatch)
    default = synth.parse_cam_imu_config(CFG).solve.aux_imu_stride
    gt, _ = synth.run_subset(str(gt_path), CFG, str(tmp_path / "out.json"), n_imus=1)
    assert len(gt.imus) == 1 and seen["aux"] == []
    assert seen["solve"].aux_imu_stride == default
    assert seen["pts"].shape[1] == 3 and len(seen["pts"]) > 0  # the config's board
