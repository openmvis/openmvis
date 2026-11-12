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
"""Every shipped source file carries the OpenMVIS licence header: the GPL-3.0-only notice
block followed by the SPDX lines on code (src/, scripts, tools, tests, examples/*.py); the two
SPDX lines alone on YAML configs and workflows. Data files (JSON, plain text, npz) are exempt."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = {".py", ".sh", ".sql", ".js", ".css", ".yaml", ".yml"}
# scripts and tools shipped in the public export, beside src/
SHIPPED = ["scripts/setup_env.sh", "scripts/run_calibration.py", "scripts/visualize_result.py",
           "scripts/recalib_batch.py", "scripts/test.sh"]
SHIPPED_DIRS = ["tools/ros1_recalib", "tests", "examples", ".github/workflows"]
TITLE = "OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark"
NOTICE_LINE = "the Free Software Foundation, version 3 of the License."
SPDX_ONLY = {".yaml", ".yml"}      # configs and workflows carry the two SPDX lines only
LICENSE = "SPDX-License-Identifier: GPL-3.0-only"
HEADER_LINES = 30       # the GPL notice block sits above the two SPDX lines


def _files():
    out = [p for p in (ROOT / "src").rglob("*") if p.is_file() and p.suffix in CODE
           and "__pycache__" not in p.parts]
    out += [ROOT / s for s in SHIPPED if (ROOT / s).exists()]
    for d in SHIPPED_DIRS:
        out += [p for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in CODE
                and "__pycache__" not in p.parts and not _is_data(p)]
    return sorted(set(out))


def _is_data(p):
    rel = p.relative_to(ROOT).parts
    # tests/data, fixtures, and the generated regression manifests (compared verbatim by their tests)
    return "data" in rel or "fixtures" in rel or rel[:2] == ("examples", "regression")


def _header(path):
    lines = path.read_text(encoding="utf-8").splitlines()[:HEADER_LINES]
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    return lines


def test_src_is_found():
    assert len([p for p in _files() if p.is_relative_to(ROOT / "src")]) > 100


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_file_has_spdx_header(path):
    head = _header(path)
    at = [i for i, ln in enumerate(head) if "SPDX-FileCopyrightText: " in ln]
    assert at, f"no SPDX copyright line in the first {HEADER_LINES} lines"
    assert any(LICENSE in ln for ln in head[at[0] + 1:]), f"no '{LICENSE}' after the copyright line(s)"


@pytest.mark.parametrize("path", [p for p in _files() if p.suffix not in SPDX_ONLY],
                         ids=lambda p: p.relative_to(ROOT).as_posix())
def test_code_file_has_gpl_notice_block(path):
    head = _header(path)
    title = [i for i, ln in enumerate(head) if TITLE in ln]
    assert title, f"no '{TITLE}' line in the first {HEADER_LINES} lines"
    assert any(NOTICE_LINE in ln for ln in head[title[0]:]), "GPL-3.0-only notice text missing"
    assert not any("any later version" in ln for ln in head), "notice must be GPL-3.0-only"
    spdx = [i for i, ln in enumerate(head) if "SPDX-License-Identifier" in ln]
    assert spdx and spdx[0] > title[0], "SPDX lines go after the notice block"
