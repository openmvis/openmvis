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
"""Checked-in config generators rewrite their committed configs byte for byte.

A generator under examples/ opts in by defining ``regenerate(out_dir)``, which
writes every config it owns into ``out_dir`` and returns
``{committed path: regenerated path}`` (``None`` for a committed file it no longer
writes). Generators that are not shipped are simply not found.
"""
import filecmp
import glob
import importlib.util
import os

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GENS = sorted(glob.glob(os.path.join(_ROOT, "examples", "**", "gen_*.py"), recursive=True))


def _load(path):
    spec = importlib.util.spec_from_file_location(
        "_gen_" + os.path.relpath(path, _ROOT).replace(os.sep, "_")[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("gen", _GENS, ids=lambda p: os.path.relpath(p, _ROOT))
def test_generator_rewrites_committed_configs(gen, tmp_path, capsys):
    mod = _load(gen)
    if not hasattr(mod, "regenerate"):
        pytest.skip("generator has no regenerate()")
    pairs = mod.regenerate(str(tmp_path / "out"))
    assert pairs, "regenerate() wrote nothing"
    rel = lambda p: os.path.relpath(p, _ROOT)                       # noqa: E731
    dropped = [rel(c) for c, g in pairs.items() if g is None]
    assert not dropped, f"committed but no longer generated: {dropped}"
    new = [rel(c) for c in pairs if not os.path.exists(c)]
    assert not new, f"generated but not committed: {new}"
    differ = [rel(c) for c, g in pairs.items() if not filecmp.cmp(c, g, shallow=False)]
    assert not differ, f"committed configs differ from the generator output: {differ}"
