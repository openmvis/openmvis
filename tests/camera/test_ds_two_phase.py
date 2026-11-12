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
"""Tests for the Double Sphere two-phase optimization strategy.

Bug discovered 2026-05-01: MVIS's DS optimizer let all 6 params
(fx, fy, cx, cy, xi, alpha) move freely from initialization → for some
bags it converged to a non-canonical basin in the DS parameter-equivalence
class (positive xi, fx ~2× larger than Usenko 2018 paper). Cross-bag
stddev was 5–29 % on the same camera.

Fix (now `omni_schedule.solve_phased`, used by pipeline Stage 2 and `mvis cam`):
- Phase 1: freeze (xi, alpha) at canonical Usenko init [-0.2, 0.6]; let
  fx/fy/cx/cy + board poses settle.
- Phase 2: warm-start from Phase 1, unfreeze all params for refinement.
- Initializer default for DS changed from [0.0, 0.5] to [-0.2, 0.6]
  (xi=0.0 was on the boundary of the wrong basin).

These tests are structural guards on the pipeline + initializer source —
they fire if the fix is reverted, even before a regression run would catch
it. Heavier end-to-end validation lives in
`tests/camera/test_double_sphere_paper.py` (forward projection / Jacobian /
sign convention) and the `tum_fisheye_*_ds` regression rows.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


# ----- Initializer default xi/alpha -----

def test_ds_initializer_default_is_canonical_basin():
    """The DS init in `_default_dist` MUST start in the canonical basin
    (xi=-0.2, alpha=0.6 per Usenko/Basalt). xi=0.0 sits on the basin
    boundary and lets the optimizer drift to a wrong (positive xi) minimum
    where fx inflates ~2×."""
    # The defaults live in the camera-model registry; the initializer must
    # seed from it.
    from mvis.calibration import initializer
    from mvis.core.camera_models import default_dist
    src = inspect.getsource(initializer)
    assert "default_dist(distortion_model)" in src, (
        "Initializer no longer seeds wide-FOV distortion from the registry "
        "default_dist — the canonical DS init may have been lost."
    )
    xi, alpha = default_dist("double_sphere")
    assert xi < 0, (
        f"DS init xi={xi} is non-negative. Typical fisheye lenses have "
        "xi in [-0.3, -0.15]; positive xi is the wrong DS basin."
    )
    assert -0.5 <= xi <= -0.05, (
        f"DS init xi={xi} outside [-0.5, -0.05]. Usenko/Basalt default "
        "is -0.2; significant deviation may push the optimizer out of "
        "the canonical basin."
    )
    assert 0.4 <= alpha <= 0.8, (
        f"DS init alpha={alpha} outside [0.4, 0.8]. Paper defaults are "
        "around 0.55-0.65; far values can stall the Phase-1 optimizer."
    )


# ----- Two-phase invocation (omni_schedule.solve_phased since G1) -----

def test_pipeline_and_cli_route_double_sphere_through_solve_phased():
    """Pipeline Stage 2 and `mvis cam` MUST give double_sphere the two-phase
    schedule: P1 frozen distortion, P2 all free, warm-started from P1. The
    schedule itself (call order, options, warm start) is tested in
    tests/calibration/test_solve_phased.py."""
    from mvis.calibration import omni_schedule
    from mvis.calibration import pipeline_cam
    import mvis.cli.main as cli
    assert "double_sphere" in omni_schedule.PHASED_MODELS
    assert "solve_phased(" in inspect.getsource(pipeline_cam)
    assert "solve_phased(" in inspect.getsource(cli.calibrate)


# ----- End-to-end DS-paper sanity (cheap, no optimizer involved) -----

def test_ds_default_init_matches_paper_for_typical_fisheye():
    """The default xi/alpha init should be close to the median Usenko paper
    values across the 5 lens families (xi ≈ -0.20 to -0.27, alpha ≈ 0.55
    to 0.64). This is a redundant check on top of the explicit init test
    above — it ties the constant to its paper provenance."""
    from mvis.core.camera_models import default_dist   # initializer seeds from it
    xi, alpha = default_dist("double_sphere")
    # Usenko Table 3 medians: xi ≈ -0.20, alpha ≈ 0.59
    assert abs(xi - (-0.20)) < 0.10, (
        f"DS init xi={xi} drifted >0.10 from paper median -0.20")
    assert abs(alpha - 0.60) < 0.10, (
        f"DS init alpha={alpha} drifted >0.10 from paper median 0.60")
