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
"""`mvis.cli.blas_threads` (speed-up T0): the `mvis` command line pins the BLAS
thread count to 1 before numpy / OpenCV / scipy are imported, `MVIS_BLAS_THREADS`
overrides it, an exported OPENBLAS/OMP/MKL_NUM_THREADS is never overridden, and
`import mvis` as a library leaves the environment alone.

The policy is unit-tested on a fake environment; the effect is checked in fresh
subprocesses (the pin only works before the first numpy import, which pytest has
long done) by reading the loaded OpenBLAS's thread count through the S0 helper
`tests/_blas_threads.py` (ctypes; `threadpoolctl` is not in the env).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mvis.cli import blas_threads as B

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
_ALL_VARS = B.USER_VARS + (B.ENV_VAR, "OPENBLAS_MAIN_FREE")

# what the subprocess reports after `import mvis.cli.main`
_PROBE = """
import json, os, sys
assert "numpy" not in sys.modules, "numpy imported before the CLI in the probe"
import mvis.cli.main                      # the console script's import
import mvis.cli.blas_threads as b
from tests._blas_threads import get_num_threads, num_threads_per_lib
print("JSON" + json.dumps(dict(
    status=b.STATUS,
    env={k: os.environ.get(k) for k in %r},
    numpy_threads=get_num_threads(),
    per_lib=num_threads_per_lib())))
""" % (list(_ALL_VARS),)

_PROBE_LIB = """
import json, os, sys
import mvis, mvis.core.config             # library use: never touches mvis.cli
from tests._blas_threads import get_num_threads
print("JSON" + json.dumps(dict(
    cli_loaded="mvis.cli" in sys.modules,
    env={k: os.environ.get(k) for k in %r},
    numpy_threads=get_num_threads())))
""" % (list(_ALL_VARS),)


def _run(code, extra_env=None, argv=None):
    """Run `code` (or argv) in a fresh interpreter with every thread variable
    removed from the environment, plus `extra_env`."""
    env = {k: v for k, v in os.environ.items() if k not in _ALL_VARS}
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(REPO)])
    env["PYTHONNOUSERSITE"] = "1"
    env.update(extra_env or {})
    cmd = [sys.executable] + (argv if argv is not None else ["-c", code])
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, f"{cmd}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    return p


def _probe(extra_env=None, code=_PROBE):
    out = _run(code, extra_env).stdout
    line = [l for l in out.splitlines() if l.startswith("JSON")][-1]
    return json.loads(line[4:])


def _skip_without_openblas(rep):
    if rep["numpy_threads"] is None:
        pytest.skip("no OpenBLAS loaded in the subprocess (cannot read the thread count)")


# ---------------------------------------------------------------------------
# policy (pure function on a fake environment)
# ---------------------------------------------------------------------------
def test_default_pins_one_thread():
    env = {}
    st = B.pin_blas_threads(env, modules={})
    assert st["pinned"] and st["threads"] == 1
    assert {k: env[k] for k in B.THREAD_VARS} == {k: "1" for k in B.THREAD_VARS}
    assert env["OPENBLAS_MAIN_FREE"] == "1"
    assert st["set"] == list(B.THREAD_VARS) + ["OPENBLAS_MAIN_FREE"]


def test_explicit_value_is_applied_even_over_exported_vars():
    env = {B.ENV_VAR: "4", "OMP_NUM_THREADS": "8", "OPENBLAS_MAIN_FREE": "0"}
    st = B.pin_blas_threads(env, modules={})
    assert st["pinned"] and st["threads"] == 4
    assert all(env[k] == "4" for k in B.THREAD_VARS)
    assert env["OPENBLAS_MAIN_FREE"] == "0"        # setdefault only
    assert "OPENBLAS_MAIN_FREE" not in st["set"]


@pytest.mark.parametrize("raw", ["0", "", "  "])
def test_zero_or_empty_means_blas_default(raw):
    env = {B.ENV_VAR: raw}
    st = B.pin_blas_threads(env, modules={})
    assert not st["pinned"] and st["threads"] is None
    assert set(env) == {B.ENV_VAR}
    assert "BLAS default" in st["reason"]


@pytest.mark.parametrize("raw", ["-1", "two", "1.5"])
def test_invalid_value_does_not_pin_or_raise(raw):
    env = {B.ENV_VAR: raw}
    st = B.pin_blas_threads(env, modules={})
    assert not st["pinned"] and set(env) == {B.ENV_VAR}
    assert "not a non-negative integer" in st["reason"]


@pytest.mark.parametrize("var", B.USER_VARS)
def test_exported_thread_var_is_never_overridden(var):
    env = {var: "3"}
    st = B.pin_blas_threads(env, modules={})
    assert not st["pinned"]
    assert env == {var: "3"}                       # nothing added, nothing changed
    assert var in st["reason"] and B.ENV_VAR in st["reason"]


def test_exported_empty_var_does_not_count_as_a_choice():
    env = {"OMP_NUM_THREADS": ""}
    st = B.pin_blas_threads(env, modules={})
    assert st["pinned"] and env["OPENBLAS_NUM_THREADS"] == "1"


@pytest.mark.parametrize("mod", ["numpy", "scipy", "cv2"])
def test_no_pin_once_a_blas_library_is_imported(mod):
    env = {}
    st = B.pin_blas_threads(env, modules={mod: object()})
    assert not st["pinned"] and env == {}
    assert mod in st["reason"] and "already imported" in st["reason"]


def test_in_process_import_after_numpy_left_env_alone():
    """This pytest process imported numpy long before `mvis.cli`; the package
    import at the top of this module must not have set anything."""
    assert B.STATUS is not None and not B.STATUS["pinned"]
    assert "already imported" in B.STATUS["reason"]
    assert "pin_blas_threads() did not run" not in B.describe()


def test_describe():
    assert B.describe(dict(pinned=True, threads=1, reason="pinned to 1 (x)", set=[])) \
        == "BLAS threads: pinned to 1 (x)"
    assert B.describe(None).startswith("BLAS threads: not pinned") or B.STATUS is not None


# ---------------------------------------------------------------------------
# effect in a fresh process (the console script's import path)
# ---------------------------------------------------------------------------
def test_cli_import_pins_numpy_openblas_to_one_thread():
    rep = _probe()
    assert rep["status"]["pinned"] and rep["status"]["threads"] == 1
    assert rep["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert rep["env"]["OMP_NUM_THREADS"] == "1"
    assert rep["env"]["MKL_NUM_THREADS"] == "1"
    assert rep["env"]["OPENBLAS_MAIN_FREE"] == "1"
    _skip_without_openblas(rep)
    assert rep["numpy_threads"] == 1
    # every OpenBLAS build mapped into the process (numpy's, opencv's, scipy's)
    assert all(n == 1 for n in rep["per_lib"].values()), rep["per_lib"]


def test_library_import_keeps_the_default_threads():
    lib = _probe(code=_PROBE_LIB)
    assert not lib["cli_loaded"]
    assert all(v is None for v in lib["env"].values()), lib["env"]
    _skip_without_openblas(lib)
    if (os.cpu_count() or 1) < 2:
        pytest.skip("single-core host: BLAS default is 1 thread anyway")
    assert lib["numpy_threads"] > 1, "library import must not pin (OpenBLAS default)"


def test_mvis_blas_threads_overrides_the_default():
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs >= 2 cores")
    rep = _probe({B.ENV_VAR: "2"})
    assert rep["status"]["threads"] == 2 and rep["env"]["OPENBLAS_NUM_THREADS"] == "2"
    _skip_without_openblas(rep)
    assert rep["numpy_threads"] == 2


def test_mvis_blas_threads_zero_keeps_blas_default():
    rep = _probe({B.ENV_VAR: "0"})
    lib = _probe(code=_PROBE_LIB)
    assert not rep["status"]["pinned"]
    assert rep["env"]["OPENBLAS_NUM_THREADS"] is None
    _skip_without_openblas(rep)
    assert rep["numpy_threads"] == lib["numpy_threads"]   # = plain numpy default


@pytest.mark.parametrize("var", ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"])
def test_exported_var_wins_in_a_fresh_process(var):
    if (os.cpu_count() or 1) < 3:
        pytest.skip("needs >= 3 cores")
    rep = _probe({var: "3"})
    assert not rep["status"]["pinned"] and var in rep["status"]["reason"]
    assert rep["env"][var] == "3"
    other = [v for v in B.THREAD_VARS if v != var]
    assert all(rep["env"][v] is None for v in other), rep["env"]
    _skip_without_openblas(rep)
    assert rep["numpy_threads"] == 3


def test_cli_help_documents_the_variable_and_logs_the_pin():
    p = _run(None, argv=["-m", "mvis.cli.main", "--help"])
    assert B.ENV_VAR in p.stdout and "doc/blas_threads.md" in p.stdout
    # a subcommand reaches basicConfig + the status line; `check` on a missing
    # file is the cheapest one that does
    p = subprocess.run([sys.executable, "-m", "mvis.cli.main", "check", "-c",
                        "/nonexistent/mvis_t0.yaml"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=300,
                       env={**{k: v for k, v in os.environ.items() if k not in _ALL_VARS},
                            "PYTHONPATH": str(SRC), "PYTHONNOUSERSITE": "1"})
    assert "BLAS threads: pinned to 1 (default" in p.stderr + p.stdout, p.stderr[-2000:]
