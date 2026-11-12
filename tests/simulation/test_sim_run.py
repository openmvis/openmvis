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
"""Task 5 — the one shared sim->recover->report pipeline (sim-cleanup plan)."""
from mvis.simulation.run import run_sim_recovery
from mvis.simulation.multi_board import SimConfig


def test_run_sim_recovery_smoke():
    cfg = SimConfig.realistic_single_board(seed=1, duration=22.0)   # short window
    gt, res, err = run_sim_recovery(cfg)
    assert set(("lever_mm", "Da", "reproj_px")).issubset(err)
    assert err['ok'] is True
    assert err['lever_mm']['norm'] < 10.0          # loose smoke gate, not a tolerance study
