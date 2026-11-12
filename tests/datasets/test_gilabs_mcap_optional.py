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
"""The MCAP reader is an optional extra: the module imports without it and its CLI says what to install."""
import subprocess
import sys

BLOCK = "import sys; sys.modules['mcap'] = None; sys.modules['mcap.reader'] = None; "


def _run(code, *argv):
    return subprocess.run([sys.executable, "-c", BLOCK + code, *argv], capture_output=True, text=True, timeout=120)


def test_module_imports_without_the_extra():
    r = _run("import mvis.datasets.converters.gilabs_mcap")
    assert r.returncode == 0, r.stderr


def test_cli_without_the_extra_names_the_extra():
    r = _run("import sys; sys.argv = ['x', '--mcap', 'a.mcap', '--out', 'b']; "
             "from mvis.datasets.converters.gilabs_mcap import main; main()")
    assert r.returncode != 0
    assert "pymvis[mcap]" in r.stderr
