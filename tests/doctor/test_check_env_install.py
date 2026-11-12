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
"""`mvis check --env` on an installed wheel vs. a source checkout: the checkout/worktree and
git-sha checks only apply when mvis is imported from a source tree (pyproject.toml two
levels up); a pip-installed package must report PASS, not 'another checkout/worktree?'."""
import subprocess

import pytest

import mvis
from mvis.doctor import check_config as cc


def _by_id(findings):
    return {f.id: f for f in findings}


def _site_packages(tmp_path):
    pkg = tmp_path / "venv" / "lib" / "python3.11" / "site-packages" / "mvis"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    return pkg / "__init__.py"


def _checkout(tmp_path, name="repo"):
    repo = tmp_path / name
    (repo / "src" / "mvis").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'pymvis'\n")
    (repo / "src" / "mvis" / "__init__.py").write_text("")
    return repo


def test_installed_wheel_passes_from_any_cwd(tmp_path):
    f = _by_id(cc._check_install(str(_site_packages(tmp_path)), "9.9.9", str(tmp_path / "elsewhere")))
    assert set(f) == {"env.mvis_checkout"}
    assert f["env.mvis_checkout"].status == "pass"
    assert "installed package pymvis" in f["env.mvis_checkout"].message
    assert "another checkout" not in f["env.mvis_checkout"].message


def test_installed_wheel_reports_the_package_version(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_version", lambda m: "")
    f = _by_id(cc._check_install(str(_site_packages(tmp_path)), "9.9.9", str(tmp_path)))
    assert "pymvis 9.9.9" in f["env.mvis_checkout"].message


def test_checkout_under_cwd_passes(tmp_path):
    repo = _checkout(tmp_path)
    f = _by_id(cc._check_install(str(repo / "src" / "mvis" / "__init__.py"), "0", str(repo / "sub")))
    assert f["env.mvis_checkout"].status == "pass"


def test_checkout_elsewhere_warns(tmp_path):
    repo = _checkout(tmp_path)
    other = _checkout(tmp_path, "other")
    f = _by_id(cc._check_install(str(repo / "src" / "mvis" / "__init__.py"), "0", str(other)))
    assert f["env.mvis_checkout"].status == "warn"
    assert "another checkout/worktree" in f["env.mvis_checkout"].message


def test_checkout_git_sha(tmp_path):
    repo = _checkout(tmp_path)
    init = str(repo / "src" / "mvis" / "__init__.py")
    f = _by_id(cc._check_install(init, "0", str(repo)))
    assert f["env.git_sha"].status == "warn" and "not a git checkout" in f["env.git_sha"].message
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    try:
        subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "commit", "-qm", "x"],
                       check=True, env=env)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not usable here")
    f = _by_id(cc._check_install(init, "0", str(repo)))
    assert f["env.git_sha"].status == "pass" and len(f["env.git_sha"].message) >= 7


def test_check_env_from_this_checkout_passes():
    """The real thing: run from the source tree, the install findings are PASS."""
    f = _by_id(cc.check_env())
    assert f["env.mvis_checkout"].status == "pass", f["env.mvis_checkout"].message
    assert "env.git_sha" in f
    assert mvis.__file__  # imported from somewhere; the status above says it is this tree
