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
"""`mvis.core.eigen_cache`: a GTSAM solve must not depend on the core it
starts on (G0, 2026-11-08).

Eigen reads the CPU cache sizes once, on the first large product, and blocks
every later product with them; a hybrid CPU reports other sizes on its P- and
E-cores, so the same graph gave other last bits depending on where the
process happened to start (monado_mgc/stereo01 pass-1 fx flipped between
269.98568948862805 and 269.985689488628).  The subprocess test starts a
small bundle adjustment on a P-core and on an E-core and needs equal bytes
with the fix (and, as its sensitivity check, unequal bytes without it); it
runs only on a hybrid CPU.  The unit tests run everywhere.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mvis.core import eigen_cache as E

linux_affinity = pytest.mark.skipif(not hasattr(os, "sched_getaffinity"),
                                    reason="no sched_getaffinity")


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(E, "STATUS", None)
    return E


@linux_affinity
def test_warm_up_runs_on_the_lowest_allowed_cpu_and_affinity_is_restored(fresh):
    before = os.sched_getaffinity(0)
    seen = []
    st = fresh.fix_eigen_cache_sizes(environ={}, warm_up=lambda: seen.append(os.sched_getaffinity(0)))
    assert seen == [{min(before)}]
    assert os.sched_getaffinity(0) == before
    assert st["pinned"] and st["cpu"] == min(before)
    assert "lowest allowed CPU" in fresh.describe()


@linux_affinity
def test_runs_once_per_process(fresh):
    calls = []
    first = fresh.fix_eigen_cache_sizes(environ={}, warm_up=lambda: calls.append(1))
    again = fresh.fix_eigen_cache_sizes(environ={}, warm_up=lambda: calls.append(1))
    assert calls == [1] and again is first


@linux_affinity
def test_env_override_off_and_bad_cpu(fresh, monkeypatch):
    calls = []
    st = fresh.fix_eigen_cache_sizes(environ={E.ENV_VAR: "off"}, warm_up=lambda: calls.append(1))
    assert not st["pinned"] and not calls
    monkeypatch.setattr(E, "STATUS", None)
    bad = max(os.sched_getaffinity(0)) + 1000
    st = fresh.fix_eigen_cache_sizes(environ={E.ENV_VAR: str(bad)}, warm_up=lambda: calls.append(1))
    assert not st["pinned"] and not calls and "not in the allowed" in st["reason"]


@linux_affinity
def test_a_failing_warm_up_never_raises_and_restores_affinity(fresh):
    before = os.sched_getaffinity(0)

    def boom():
        raise RuntimeError("x")
    st = fresh.fix_eigen_cache_sizes(environ={}, warm_up=boom)
    assert not st["pinned"] and "RuntimeError" in st["reason"]
    assert os.sched_getaffinity(0) == before


def test_real_warm_up_solves():
    E._warm_up()                       # the GTSAM call itself works in this build


# --- hybrid CPU: start core must not matter ---------------------------------

def _cpus(kind):
    p = Path(f"/sys/devices/{kind}/cpus")
    if not p.exists():
        return set()
    out = set()
    for part in p.read_text().strip().split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


_CHILD = r"""
import os, sys, hashlib
start = int(sys.argv[1]); full = os.sched_getaffinity(0)
os.sched_setaffinity(0, {start})       # the process starts on this core
import numpy as np, gtsam
from mvis.core.eigen_cache import fix_eigen_cache_sizes
os.sched_setaffinity(0, full)          # ... with the whole machine allowed
st = fix_eigen_cache_sizes()
os.sched_setaffinity(0, {start})       # and the solve runs there too
rng = np.random.default_rng(0)
K = gtsam.Cal3_S2(500, 500, 0, 320, 240)
g = gtsam.NonlinearFactorGraph(); v = gtsam.Values()
noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
X = gtsam.symbol_shorthand.X; L = gtsam.symbol_shorthand.L
pts = rng.uniform(-1, 1, (60, 3)) + np.array([0, 0, 5])
for i in range(30):
    pose = gtsam.Pose3(gtsam.Rot3.Rodrigues(*(rng.standard_normal(3) * 0.05)),
                       rng.standard_normal(3) * 0.3)
    cam = gtsam.PinholeCameraCal3_S2(pose, K)
    for j, p in enumerate(pts):
        g.add(gtsam.GenericProjectionFactorCal3_S2(
            cam.project(p) + rng.standard_normal(2) * 0.5, noise, X(i), L(j), K))
    v.insert(X(i), pose.retract(rng.standard_normal(6) * 0.01))
for j, p in enumerate(pts):
    v.insert(L(j), p + rng.standard_normal(3) * 0.05)
g.add(gtsam.PriorFactorPose3(X(0), v.atPose3(X(0)), gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))
g.add(gtsam.PriorFactorPoint3(L(0), v.atPoint3(L(0)), gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)))
d = g.linearize(v).optimize()
print(hashlib.sha1(np.concatenate([d.at(X(i)) for i in range(30)]).tobytes()).hexdigest())
"""


def _solve_from(cpu, eigen_env):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **eigen_env)
    if not eigen_env:
        env.pop(E.ENV_VAR, None)
    r = subprocess.run([sys.executable, "-c", _CHILD, str(cpu)], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@linux_affinity
def test_hybrid_cpu_start_core_does_not_change_the_solve():
    allowed = os.sched_getaffinity(0)
    p_cores, e_cores = _cpus("cpu_core") & allowed, _cpus("cpu_atom") & allowed
    if not (p_cores and e_cores):
        pytest.skip("not a hybrid CPU (or one core type not allowed)")
    p, e = min(p_cores), min(e_cores)
    off = {E.ENV_VAR: "off"}
    if _solve_from(p, off) == _solve_from(e, off):
        pytest.skip("this CPU's core types give equal bytes even unfixed: nothing to check")
    assert _solve_from(p, {}) == _solve_from(e, {})


def test_camera_only_driver_fixes_the_cache_sizes_before_solving(monkeypatch):
    """scripts/run_calibration.py and `python -m mvis.calibration.cam_run` (the regression
    rows) both enter cam_run.main, which must fix the sizes before run_cam solves."""
    import mvis.core.eigen_cache as E
    from mvis.calibration import cam_run
    order = []
    monkeypatch.setattr(E, "fix_eigen_cache_sizes", lambda: order.append("fix") or {})
    monkeypatch.setattr(cam_run, "configure_process", lambda: None)
    monkeypatch.setattr(cam_run, "run_cam", lambda *a, **k: order.append("solve") or 0)
    assert cam_run.main(["--config", "x.yaml"]) == 0
    assert order == ["fix", "solve"]
