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
"""Read / pin the BLAS thread count of this process (cam-IMU speed-up S0).

numpy's bundled OpenBLAS runs its kernels on every core by default, and the
reduction order then depends on the thread count: two runs of the same solve
are bitwise identical only with ONE BLAS thread (S0b, bag 2 C: 6032 dump
entries differ at <= 1e-13 with 32 threads, all equal with 1). Every bitwise
gate therefore has to run single-threaded. `OPENBLAS_NUM_THREADS=1` in the
environment does that, but a test cannot rely on the caller having set it, so
this module talks to the loaded OpenBLAS libraries directly
(`openblas_get/set_num_threads` through ctypes, the way `threadpoolctl` does;
that package is not in the env).

The mvis process maps SEVERAL OpenBLAS builds (numpy's `libopenblas64_`,
opencv's `libopenblasp`, scipy's `libscipy_openblas`): `set_num_threads` pins
all of them, `get_num_threads` reports numpy's (the one the solver's numpy
calls run on), falling back to the first found.

    get_num_threads()      -> int | None (None: no OpenBLAS found in the process)
    set_num_threads(n)     -> bool (False: cannot set numpy's)
    blas_threads(n)        context manager: pin to n inside, restore after
    describe()             -> short string for the golden's metadata

Only OpenBLAS is handled (that is what the mvis env ships); with another BLAS
`get_num_threads()` returns None and the callers fall back to the
`OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` environment variables or skip.
"""
import contextlib
import ctypes
import os
import platform
import re

import numpy as np

_LIBS = None          # [(path, get, set, config|None)], numpy's first
_PREFIXES = ("openblas_", "scipy_openblas_")
_SUFFIXES = ("", "64_")


def _bind(lib, name):
    for pre in _PREFIXES:
        for suf in _SUFFIXES:
            fn = getattr(lib, pre + name + suf, None)
            if fn is not None:
                return fn
    return None


def _find_openblas():
    """Load every OpenBLAS shared object mapped into this process."""
    global _LIBS
    if _LIBS is not None:
        return _LIBS
    _LIBS = []
    np.zeros(2) @ np.zeros(2)                 # make sure numpy's BLAS is loaded
    paths = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and re.search(r"openblas", os.path.basename(parts[-1]), re.I):
                    if parts[-1] not in paths:
                        paths.append(parts[-1])
    except OSError:
        return _LIBS
    for p in paths:
        try:
            lib = ctypes.CDLL(p)
        except OSError:
            continue
        g, s = _bind(lib, "get_num_threads"), _bind(lib, "set_num_threads")
        if g is None or s is None:
            continue
        g.restype = ctypes.c_int
        s.argtypes = [ctypes.c_int]
        s.restype = None
        c = _bind(lib, "get_config")
        if c is not None:
            c.restype = ctypes.c_char_p
        _LIBS.append((p, g, s, c))
    _LIBS.sort(key=lambda e: 0 if "numpy" in e[0] else 1)      # numpy's first
    return _LIBS


def loaded_libs():
    """Paths of the OpenBLAS libraries found (numpy's first)."""
    return [e[0] for e in _find_openblas()]


def get_num_threads():
    """numpy's OpenBLAS thread count, or None when no OpenBLAS is loaded."""
    libs = _find_openblas()
    return int(libs[0][1]()) if libs else None


def num_threads_per_lib():
    """{path: thread count} for every OpenBLAS found (diagnostics)."""
    return {p: int(g()) for p, g, _s, _c in _find_openblas()}


def env_num_threads():
    """Thread count requested through the environment (None if unset)."""
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(k)
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return None


def set_num_threads(n: int) -> bool:
    """Pin every loaded OpenBLAS to `n` threads; False if numpy's did not take it."""
    libs = _find_openblas()
    if not libs:
        return False
    for _p, _g, s, _c in libs:
        s(int(n))
    return int(libs[0][1]()) == int(n)


@contextlib.contextmanager
def blas_threads(n):
    """Run the block with OpenBLAS pinned to `n` threads (restored after).
    Yields the thread count actually in effect (None: could not pin)."""
    before = num_threads_per_lib()
    if n is None or not before:
        yield get_num_threads()
        return
    ok = set_num_threads(n)
    try:
        yield (int(n) if ok else get_num_threads())
    finally:
        for p, _g, s, _c in _find_openblas():
            s(before[p])


def describe() -> str:
    """One line naming numpy's BLAS build, numpy and the CPU architecture (golden metadata)."""
    cfg = ""
    libs = _find_openblas()
    if libs and libs[0][3] is not None:
        try:
            cfg = libs[0][3]().decode(errors="replace")
        except Exception:                      # pragma: no cover - defensive
            cfg = "openblas"
    return (f"{cfg or 'no-openblas'} ({len(libs)} OpenBLAS lib(s) loaded); "
            f"numpy {np.__version__}; {platform.machine()}")
