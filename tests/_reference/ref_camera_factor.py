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
"""Frozen copy of `camera_factor._rs_row_offset` (S0, never edited)."""

from __future__ import annotations


def _rs_row_offset(row: float, h: int, mode: int) -> float:
    """Dimensionless rolling-shutter row factor ρ: capture time = td + ρ·tr.

    Modes match MVIS / `simulation.camera_measurements._rs_row_time`:
      0: frame stamped at top    -> ρ = row/H
      1: frame stamped at middle -> ρ = row/H − 0.5
      2: frame stamped at bottom -> ρ = (row − H)/H
    """
    if mode == 0:
        return row / h
    if mode == 1:
        return row / h - 0.5
    return (row - h) / h
