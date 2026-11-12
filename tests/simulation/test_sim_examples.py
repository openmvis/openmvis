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
"""Simulator examples (mvis.simulation.examples): every example loads with a usable setup and
perturbed truth, and one simulates end to end."""
import tomllib
from pathlib import Path

import gtsam
import numpy as np
import pytest

from mvis.simulation.examples import (EXAMPLES, PACKAGED_DIR, REPO_DIR, available, example_dir,
                                      load_example, simulate)
from mvis.simulation.twin import PERTURB_SEED, perturb_rng_seed

ROOT = Path(__file__).resolve().parents[2]
SMALL = "twin_gs"            # the packaged one


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_loads(name):
    ex = load_example(name)
    s = ex.setup
    assert s["name"] == name
    t = np.loadtxt(ex.path / "trajectory.txt")
    assert t.shape[1] == 7 and t[0, 0] == 0.0
    assert np.allclose(np.diff(t[:, 0]), 0.005, atol=1e-6), "trajectory is not 200 Hz"
    assert abs(t[-1, 0] - s["duration_s"]) < 1e-3
    assert ex.traj.end_time - ex.traj.start_time > 5.0
    assert ex.board_points.shape == ex.board_points_flat.shape and len(ex.board_points) > 20
    assert np.allclose(ex.board_points[:, :2], ex.board_points_flat[:, :2])
    for cam in s["cameras"]:
        assert len(cam["intrinsics"]) == 8 and len(cam["image_size"]) == 2
        assert cam["shutter"] == ("rolling" if cam["readout_s"] else "global")
        T = np.asarray(cam["T_CtoI"], float)
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
        assert cam["rate_hz"] > 1 and cam["corner_sigma_px"] > 0
    imu = s["imu"]
    assert len(imu["intrinsics_xi24"]) == 24 and imu["rate_hz"] > 50 and imu["t_ga_s"] != 0
    assert set(imu["noise"]) == {"sigma_w", "sigma_a", "sigma_bg", "sigma_ba"}
    for a in s.get("aux_imus") or []:
        assert len(a["intrinsics_xi24"]) == 24 and np.asarray(a["T_IstoI"]).shape == (4, 4)


@pytest.mark.parametrize("name", EXAMPLES)
def test_truth_is_perturbed(name):
    """Decision D3: no real device calibration ships; each truth carries its seeded perturbation."""
    p = load_example(name).setup["truth"]["perturbation"]
    assert p["seed"] == PERTURB_SEED
    assert p["rng_seed"] == perturb_rng_seed(p["seed"], p["name"])


@pytest.mark.parametrize("name", EXAMPLES)
def test_no_private_strings(name):
    text = " ".join(f.read_text() for f in example_dir(name).iterdir() if f.suffix in (".yaml", ".md"))
    assert "/home/" not in text
    # the examples ship in the public tree; the maintainers' generator, twin drivers and
    # regression manifests do not, so the text must not send a reader to them
    assert "scripts/" not in text and "examples/regression" not in text
    if name.startswith("twin_"):
        assert "gilabs" not in text.lower() and "gi labs" not in text.lower()


def test_listing():
    assert available()[:len(EXAMPLES)] == list(EXAMPLES)
    assert available()[0] == "virig_4imus_3cams"
    with pytest.raises(FileNotFoundError):
        load_example("no_such_example")


def test_one_example_is_packaged():
    assert [p.name for p in PACKAGED_DIR.iterdir() if p.is_dir()] == [SMALL]
    assert example_dir(SMALL) == PACKAGED_DIR / SMALL
    assert not (REPO_DIR / SMALL).exists()


def test_every_example_has_a_licence():
    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text())
    paths = {p for a in reuse["annotations"] for p in a["path"]}
    for name in EXAMPLES:
        rel = example_dir(name).relative_to(ROOT).as_posix()
        assert f"{rel}/**" in paths, f"{rel} has no REUSE.toml entry"
        head = (example_dir(name) / "setup.yaml").read_text().splitlines()[:4]
        lic = next(a["SPDX-License-Identifier"] for a in reuse["annotations"] if f"{rel}/**" in a["path"])
        assert f"# SPDX-License-Identifier: {lic}" in head


def _cam_pose_W(ex, cam, t):
    R_GtoI, p = ex.traj.get_pose(t + float(cam["td_s"]))
    T_ItoW = gtsam.Pose3(gtsam.Rot3(np.asarray(R_GtoI).T), gtsam.Point3(*p))
    return T_ItoW.compose(gtsam.Pose3(np.asarray(cam["T_CtoI"], float)))


def test_simulate_short_window():
    ex = load_example(SMALL)
    t0 = ex.traj.start_time + 1.0
    d = simulate(ex, seed=0, noise=False, t0=t0, t1=t0 + 2.0)
    s = ex.setup
    assert len(d["cams"]) == len(s["cameras"]) and d["aux_imus"] == []
    assert abs(len(d["t_imu"]) - 2.0 * s["imu"]["rate_hz"]) <= 2
    assert d["w_m"].shape == d["a_m"].shape == (len(d["t_imu"]), 3)
    for cam, c in zip(s["cameras"], d["cams"]):
        assert len(c["t_cam"]) > 0.8 * 2.0 * cam["rate_hz"]
        assert all(len(i) == len(u) for i, u in zip(c["ids_list"], c["uv_list"]))
        # noise-free corners are the board projected at the camera's true pose (global shutter)
        k = len(c["t_cam"]) // 2
        T_CtoW = _cam_pose_W(ex, cam, float(c["t_cam"][k]))
        P_W = ex.board_points_world()[c["ids_list"][k][0]]
        P_C = np.asarray(T_CtoW.transformTo(gtsam.Point3(*P_W)), float)
        uv = gtsam.Cal3Fisheye(*cam["intrinsics"][:2], 0.0, *cam["intrinsics"][2:]).uncalibrate(P_C[:2] / P_C[2])
        assert np.allclose(uv, c["uv_list"][k][0], atol=1e-6)
    noisy = simulate(ex, seed=0, t0=t0, t1=t0 + 2.0)
    assert np.std(noisy["w_m"] - d["w_m"]) > 0


def test_simulate_aux_and_rolling_shutter():
    ex = load_example("elp_t265")
    t0 = ex.traj.start_time + 10.0
    d = simulate(ex, seed=1, t0=t0, t1=t0 + 1.5)
    assert [a["name"] for a in d["aux_imus"]] == [a["name"] for a in ex.setup["aux_imus"]]
    assert [c["estimate_readout"] for c in d["cams"]] == [False, True]
    assert all(len(c["t_cam"]) > 10 for c in d["cams"])
