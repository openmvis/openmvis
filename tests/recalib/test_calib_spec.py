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
# tests/recalib/test_calib_spec.py
import numpy as np, yaml
import pytest
import gtsam
from types import SimpleNamespace as NS
from mvis.imu import ImuModel
from mvis.io.cam_imu_output import build_cam_imu_dict


def _fake_calib():
    """Minimal camera calib duck-type accepted by _cam_block."""
    return NS(
        model="radtan",
        intr=np.array([400., 400., 320., 240., -0.28, 0.07, 5e-4, -4e-4]),
        td=0.0,
        tr=0.0,
        T=gtsam.Pose3(),
    )


class _Result:  # minimal duck-typed stand-in for a calib result
    def __init__(self, xi, two_cams=False, with_aux_imu=False):
        cams = [dict(calib=_fake_calib(), image_size=[640, 480], reproj_rms_px=0.4)]
        if two_cams:
            cams.append(dict(calib=_fake_calib(), image_size=[640, 480], reproj_rms_px=0.5))
        self.cameras = cams
        self.calib = None
        self.imu_model = "rpng"
        self.b_g = np.array([0.01, -0.02, 0.03])
        self.b_a = np.array([0.1, 0.2, -0.3])
        self.imu_intr_xi = xi
        self.imus = [dict(name="imu1", role="aux", td=0.005)] if with_aux_imu else []
        self.ok = True
        self.diagnostics = {"reproj_rms_px": 0.4}

def test_imu_intr_xi_is_persisted_and_reconstructs():
    rng = np.random.default_rng(0)
    xi = rng.normal(scale=1e-3, size=24)
    d = build_cam_imu_dict(_Result(xi))
    base = d["imus"][0]
    assert "imu_intr_xi" in base and len(base["imu_intr_xi"]) == 24
    assert base["model"] == "rpng"
    # round-trips back to the same absolute model
    ii = ImuModel.from_tangent(np.array(base["imu_intr_xi"]), base["model"])
    ref = ImuModel.from_tangent(xi, "rpng")
    np.testing.assert_allclose(ii.Dw(), ref.Dw(), atol=1e-12)
    np.testing.assert_allclose(ii.Tg(), ref.Tg(), atol=1e-12)


def test_load_rig_calib_reconstructs_base_imu(tmp_path):
    from mvis.recalib.calib_spec import RigCalib, ImuCalib, MVIS, load_rig_calib

    rng = np.random.default_rng(42)
    xi = rng.normal(scale=1e-3, size=24)

    # Write a real MVIS result YAML to disk using build_cam_imu_dict
    # (yaml.safe_dump works on the full dict — confirmed: all values are plain Python types)
    d = build_cam_imu_dict(_Result(xi))
    path = tmp_path / "calib.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))

    rig = load_rig_calib(str(path))

    # rig structure
    assert isinstance(rig, RigCalib)
    assert isinstance(rig.base_imu, ImuCalib)
    assert rig.aux_imus == []
    assert len(rig.cams) == 1

    # base IMU round-trip
    base = rig.base_imu
    assert base.name == "imu0"
    assert base.role == "base"
    assert base.model == "rpng"
    assert base.td == 0.0
    np.testing.assert_allclose(base.b_g, np.array([0.01, -0.02, 0.03]), atol=1e-12)
    np.testing.assert_allclose(base.b_a, np.array([0.1, 0.2, -0.3]), atol=1e-12)
    ref = ImuModel.from_tangent(xi, "rpng")
    np.testing.assert_allclose(base.intr.Dw(), ref.Dw(), atol=1e-12)
    np.testing.assert_allclose(base.intr.Tg(), ref.Tg(), atol=1e-12)

    # MVIS round-trip
    cam = rig.cams[0]
    assert isinstance(cam, MVIS)
    assert cam.name == "cam0"
    assert cam.model == "radtan"
    np.testing.assert_allclose(cam.K, [400.0, 400.0, 320.0, 240.0], atol=1e-12)
    np.testing.assert_allclose(cam.dist, [-0.28, 0.07, 5e-4, -4e-4], atol=1e-12)
    assert cam.td == 0.0
    assert cam.tr == 0.0


def test_missing_imu_intr_xi_raises(tmp_path):
    from mvis.recalib.calib_spec import load_rig_calib

    d = {
        "imus": [{
            "name": "imu0",
            "role": "base",
            "model": "rpng",
            "b_g": [0.0, 0.0, 0.0],
            "b_a": [0.0, 0.0, 0.0],
            # imu_intr_xi intentionally absent
        }],
        "cameras": [],
    }
    path = tmp_path / "stale.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))

    with pytest.raises(ValueError, match="imu_intr_xi"):
        load_rig_calib(str(path))


def test_aux_imu_without_xi_loads_identity(tmp_path):
    """An aux-IMU block lacking imu_intr_xi loads with identity intrinsics.

    The output writer emits AUX-IMU blocks (role: aux) with only
    name/role/td/p_IsinI_m and NO imu_intr_xi. These must load (not crash):
    aux IMUs are re-stamped, not intrinsically corrected, so identity
    intrinsics + zero bias is the correct, honest reconstruction.
    """
    from mvis.recalib.calib_spec import load_rig_calib

    rng = np.random.default_rng(7)
    xi = rng.normal(scale=1e-3, size=24)  # valid base intrinsics
    d = {
        "imus": [
            {  # base: full intrinsics
                "name": "imu0", "role": "base", "model": "rpng",
                "b_g": [0.0, 0.0, 0.0], "b_a": [0.0, 0.0, 0.0],
                "imu_intr_xi": [float(v) for v in xi],
            },
            {  # aux: extrinsics + td only, no imu_intr_xi
                "name": "imu1", "role": "aux",
                "td": 0.012, "p_IsinI_m": [0.1, 0.0, 0.0],
            },
        ],
        "cameras": [],
    }
    path = tmp_path / "multi_imu.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))

    rig = load_rig_calib(str(path))

    assert rig.base_imu.name == "imu0"
    assert len(rig.aux_imus) == 1
    aux = rig.aux_imus[0]
    assert aux.name == "imu1"
    assert aux.role == "aux"
    assert aux.td == pytest.approx(0.012)
    # identity intrinsics
    np.testing.assert_allclose(aux.intr.Dw(), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(aux.intr.Da(), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(aux.b_g, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(aux.b_a, np.zeros(3), atol=1e-12)


def test_aux_imu_with_xi_is_honored(tmp_path):
    """If an aux block DOES carry imu_intr_xi, use it (not identity)."""
    from mvis.recalib.calib_spec import load_rig_calib

    rng = np.random.default_rng(11)
    xi_base = rng.normal(scale=1e-3, size=24)
    xi_aux = rng.normal(scale=1e-3, size=24)
    d = {
        "imus": [
            {"name": "imu0", "role": "base", "model": "rpng",
             "b_g": [0.0, 0.0, 0.0], "b_a": [0.0, 0.0, 0.0],
             "imu_intr_xi": [float(v) for v in xi_base]},
            {"name": "imu1", "role": "aux", "model": "rpng",
             "imu_intr_xi": [float(v) for v in xi_aux]},
        ],
        "cameras": [],
    }
    path = tmp_path / "multi_imu_xi.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))

    rig = load_rig_calib(str(path))
    ref = ImuModel.from_tangent(xi_aux, "rpng")
    np.testing.assert_allclose(rig.aux_imus[0].intr.Dw(), ref.Dw(), atol=1e-12)


# ---------------------------------------------------------------------------
# Task-1: per-sensor topic round-trip + backward compat
# ---------------------------------------------------------------------------

def test_topic_round_trip(tmp_path):
    """Topics written via build_cam_imu_dict are loaded back by load_rig_calib."""
    from mvis.io.cam_imu_output import build_cam_imu_dict, write_cam_imu_calibration
    from mvis.recalib.calib_spec import load_rig_calib

    rng = np.random.default_rng(99)
    xi = rng.normal(scale=1e-3, size=24)
    result = _Result(xi, two_cams=True, with_aux_imu=True)

    path = tmp_path / "calib_topics.yaml"
    write_cam_imu_calibration(
        result, str(path),
        cam_topics=["/cam0/image_raw", "/cam1/image_raw"],
        imu_topics=["/imu0", "/imu1"],
    )

    rig = load_rig_calib(str(path))

    assert rig.base_imu.topic == "/imu0"
    assert rig.aux_imus[0].topic == "/imu1"
    assert rig.cams[0].topic == "/cam0/image_raw"
    assert rig.cams[1].topic == "/cam1/image_raw"


def test_topic_backward_compat(tmp_path):
    """Writing without topics leaves every sensor's .topic as None."""
    from mvis.io.cam_imu_output import write_cam_imu_calibration
    from mvis.recalib.calib_spec import load_rig_calib

    rng = np.random.default_rng(7)
    xi = rng.normal(scale=1e-3, size=24)
    result = _Result(xi, two_cams=True, with_aux_imu=True)

    path = tmp_path / "calib_no_topics.yaml"
    write_cam_imu_calibration(result, str(path))

    rig = load_rig_calib(str(path))

    assert rig.base_imu.topic is None
    assert rig.aux_imus[0].topic is None
    for cam in rig.cams:
        assert cam.topic is None


# ---------------------------------------------------------------------------
# Task-1 (correct-3cams): aux-IMU intrinsics + biases serialization
# ---------------------------------------------------------------------------

def test_aux_imu_intrinsics_serialized_and_loaded(tmp_path):
    import numpy as np, yaml
    from mvis.imu import ImuModel
    from mvis.io.cam_imu_output import build_cam_imu_dict
    from mvis.recalib.calib_spec import load_rig_calib

    rng = np.random.default_rng(5)
    base_xi = rng.normal(scale=1e-3, size=24)
    aux_xi = rng.normal(scale=1e-2, size=24)             # nontrivial aux intrinsics
    aux_bg, aux_ba = np.array([0.02, -0.01, 0.03]), np.array([0.2, -0.1, 0.05])

    r = _Result(base_xi)                                 # existing helper (base IMU + 1 cam)
    r.imus = [dict(name="imu1", td=0.004,
                   intr_xi24=aux_xi, b_g_s=aux_bg, b_a_s=aux_ba)]
    d = build_cam_imu_dict(r, cam_topics=["/cam0/image_raw"],
                           imu_topics=["/imu0", "/imu1"])
    aux = d["imus"][1]
    assert aux["role"] == "aux" and aux["model"] == "rpng"
    assert len(aux["imu_intr_xi"]) == 24
    assert aux["topic"] == "/imu1"
    np.testing.assert_allclose(aux["b_g"], aux_bg); np.testing.assert_allclose(aux["b_a"], aux_ba)

    p = tmp_path / "calib.yaml"; p.write_text(yaml.safe_dump(d, sort_keys=False))
    rig = load_rig_calib(str(p))
    a = rig.aux_imus[0]
    assert a.topic == "/imu1" and a.td == 0.004
    np.testing.assert_allclose(a.b_g, aux_bg); np.testing.assert_allclose(a.b_a, aux_ba)
    ref = ImuModel.from_tangent(aux_xi, "rpng")
    np.testing.assert_allclose(a.intr.Dw(), ref.Dw(), atol=1e-12)
    np.testing.assert_allclose(a.intr.Da(), ref.Da(), atol=1e-12)

def test_aux_imu_without_intrinsics_back_compat(tmp_path):
    # An aux dict lacking intr_xi24 still writes the old shape and loads as identity.
    import numpy as np, yaml
    from mvis.io.cam_imu_output import build_cam_imu_dict
    from mvis.recalib.calib_spec import load_rig_calib
    r = _Result(np.zeros(24)); r.imus = [dict(name="imu1", td=0.001)]
    d = build_cam_imu_dict(r, imu_topics=["/imu0", "/imu1"])
    assert "imu_intr_xi" not in d["imus"][1]
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(d, sort_keys=False))
    rig = load_rig_calib(str(p))
    np.testing.assert_allclose(rig.aux_imus[0].intr.Dw(), np.eye(3), atol=1e-12)  # identity
