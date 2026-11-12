#!/usr/bin/env bash
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
# Run the MVIS tests by tier. See doc/testing.md.
#   scripts/test.sh unit          fast tests, while coding
#   scripts/test.sh solver        five real solves, with unit this is what CI runs
#   scripts/test.sh integration   digital twins through the full solver, before committing solver changes
#   scripts/test.sh full          everything, before a merge
# Extra arguments go to pytest, e.g. scripts/test.sh unit -k imu
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${1:-}"
[ $# -gt 0 ] && shift
case "$TIER" in
  unit)        ARGS=(tests -m "not slow and not integration") ;;
  solver)      ARGS=(
    "tests/cam_imu/test_pipeline_cam_imu.py::test_pipeline_recovers_calibration[0.0]"
    "tests/integration/test_digital_twins.py::test_imu_factor_recovers_truth_without_noise[twin_gs]"
    tests/integration/test_digital_twins.py::test_t_ga_term_removes_the_gs_bias
    tests/cam_imu/test_pipeline_vision_imu.py::test_t_ga_term_solves_end_to_end
    tests/cam_imu/test_golden_camimu_multi.py::test_linearized_fit_matches_the_exact_factor
  ) ;;
  integration) ARGS=(tests/integration) ;;
  full)        ARGS=(tests) ;;
  *) echo "usage: $0 unit|solver|integration|full [pytest args]" >&2; exit 2 ;;
esac
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
# a worktree must test its own code, not the editable install
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m pytest -q -p no:cacheprovider "${ARGS[@]}" "$@"
