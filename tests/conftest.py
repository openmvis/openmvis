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
"""Shared test fixtures and the tier markers (see doc/testing.md)."""

import re
from pathlib import Path

import numpy as np
import pytest

# helpers hold asserts too (assert_close, ...); keep pytest's detailed failure messages
pytest.register_assert_rewrite("tests.helpers")

_HERE = Path(__file__).parent


def _slow_list():
    path = _HERE / "slow_tests.txt"
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def pytest_collection_modifyitems(config, items):
    """Mark the listed slow tests, so the unit tier needs no edits across the test files,
    and skip the tests whose maintainer-only files are absent."""
    slow = _slow_list()
    root = _HERE.parent
    for item in items:
        node = re.sub(r"\[.*$", "", item.nodeid)
        rel = Path(item.fspath).relative_to(root).as_posix()
        if rel + node[node.index("::"):] in slow:
            item.add_marker(pytest.mark.slow)
    _skip_maintainer_only(items)


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden", action="store_true", default=False,
        help="rewrite tests/data/golden_camimu_multi_once.npz from the live "
             "code (cam-IMU speed-up S0 golden; also re-measures its noise floor)")


@pytest.fixture
def sample_intrinsics():
    """Typical camera intrinsics."""
    return np.array([500.0, 500.0, 320.0, 240.0])


@pytest.fixture
def sample_dist_radtan():
    """Small radtan distortion coefficients."""
    return np.array([-0.1, 0.01, 0.001, -0.001])


@pytest.fixture
def sample_dist_equidistant():
    """Small equidistant distortion coefficients."""
    return np.array([0.01, -0.001, 0.0001, -0.00001])


# --- Maintainer-only files ---------------------------------------------------
# These tests exercise files that live only in the maintainers' checkout
# (regression configs, experiment and report scripts, agent-skill templates).
# The public release does not ship those files; the tests then skip with the
# reason instead of failing.
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parent.parent
_REG_3CAMS = "examples/regression/mvis_4imus_3cams/cam_imu_april_01.yaml"
_REG_MIC = "examples/regression/tum_vi_mic/cam_imu_mic05.yaml"
_REGRESSION = "src/mvis/regression"  # the maintainers' regression engine, not in the public package
_NEEDS = {  # "<file>::<test name prefix>" (or just "<file>") -> required paths
    "test_camonly_rotation_rad.py::test_cam_stats_extrinsics_list_carries_rot_rad": [_REGRESSION],
    "test_kb_kernel.py::test_regression_": [_REGRESSION],
    "test_result_readers_new_keys.py::test_cam_stats_reads_the_same_numbers": [_REGRESSION],
    "test_result_readers_new_keys.py::test_cam_imu_gate_metrics_and_verdict_ignore_board_shape":
        [_REGRESSION],
    "test_removed_kb8.py::test_stats_mark_one_old_kb8_row_instead_of_failing_the_batch": [_REGRESSION],
    "test_camera_model_registry.py::test_cam_adapter_key_mismatch_fails": [_REGRESSION],
    "test_omni_p2.py::test_cam_imu_gate_fails_on_key_set_mismatch": [_REGRESSION],
    "test_fileutil.py::test_write_baseline_bytes": [_REGRESSION],
    "test_grade_ladders.py::test_grade_equals_the_old_ladders": [_REGRESSION],
    "test_agent_tools.py::test_skill_templates_have_no_unknown_keys": [".claude/skills/mvis-calibrate"],
    "test_agent_tools.py::test_settings_allowlist_matches_cli": [".claude/settings.json"],
    "test_agent_tools.py::test_skill_creates_solve_log_folders": [".claude/skills/mvis-calibrate/SKILL.md"],
    "test_cam_imu_batch.py::test_input_bags_defaults_empty": [_REG_3CAMS],
    "test_cam_imu_batch.py::test_batch_plan_": [_REG_3CAMS],
    "test_imu_correlation.py::test_studies_": ["studies/tga_correlation_lever.py"],
    "test_report_misc_formatting.py::test_sim_recovery_2imu1cam": ["scripts/sim_recovery_2imu1cam.py"],
    "test_report_misc_formatting.py::test_render_final": ["demo_video/render_final.py"],
    "test_roundtrip_report.py::test_run_roundtrip_rejects_non_ros1": [_REG_MIC],
    "test_result_readers_new_keys.py::test_gen_stats_": ["scripts/regression_report/gen_stats.py"],
    "test_sim_noise.py::test_sim_aruco_grounded_": ["examples/experiments/sim_aruco_grounded.py"],
    "test_sim_subset.py::test_run_subset_": [_REG_3CAMS],
}


def _skip_maintainer_only(items):
    for item in items:
        fname = _Path(str(item.fspath)).name
        for key, needs in _NEEDS.items():
            f, _, prefix = key.partition("::")
            if fname != f or not item.name.startswith(prefix):
                continue
            missing = [n for n in needs if not (_REPO / n).exists()]
            if missing:
                item.add_marker(pytest.mark.skip(
                    reason=f"needs maintainer-only file(s): {', '.join(missing)}"))
