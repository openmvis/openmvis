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
"""Import-boundary guards: the structure is sqlite-free.

Rules under test:
  R1  sqlite3 appears ONLY in manager.py, _rows.py and migrations/ under mvisdb.
  R2  mvis.mvisdb.mvisdatabase imports nothing from io/ or detection/.
  R3  importing the structure never loads the manager.
The AST checker walks function bodies too, and is itself tested red/green.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MVISDB = REPO / "src" / "mvis" / "mvisdb"
# persistence machinery may speak sqlite: the manager, the row-fold, and the
# migrations package (numbered up(conn) modules). Nothing else.
SQLITE_ALLOWED = {"manager.py", "_rows.py"}
SQLITE_ALLOWED_DIRS = {"migrations"}


def _imports_of(source: str) -> set[str]:
    """Every module name imported anywhere in the source, function bodies included."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
            if node.module.startswith("mvis."):
                names.add(node.module)
    return names


def test_checker_catches_violations_red_green():
    bad = "def f():\n    import sqlite3\n    return sqlite3\n"
    assert "sqlite3" in _imports_of(bad)          # red
    assert "sqlite3" not in _imports_of("import numpy\n")  # green


def test_r1_sqlite_only_in_manager_and_rows():
    offenders = []
    for py in MVISDB.rglob("*.py"):
        if py.name in SQLITE_ALLOWED or py.parent.name in SQLITE_ALLOWED_DIRS:
            continue
        if "sqlite3" in _imports_of(py.read_text()):
            offenders.append(py.name)
    assert offenders == [], f"sqlite3 imported outside manager/_rows: {offenders}"


def test_r2_structure_imports_no_io_or_detection():
    banned_prefixes = ("mvis.io", "mvis.detection", "mvis.board", "mvis.calibration")
    for name in ("mvisdatabase.py", "spec.py", "__init__.py"):
        imports = _imports_of((MVISDB / name).read_text())
        hits = [i for i in imports if i.startswith(banned_prefixes)]
        assert hits == [], f"{name} imports {hits}"


def test_r3_importing_structure_never_loads_manager_or_sqlite():
    code = (
        "import sys\n"
        "import mvis.mvisdb.mvisdatabase\n"
        "import mvis.mvisdb.spec\n"
        "assert 'mvis.mvisdb.manager' not in sys.modules, 'manager loaded'\n"
        "assert 'sqlite3' not in sys.modules, 'sqlite3 loaded by the structure'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0, r.stderr
