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
"""Camera-only calibration of one config: a thin wrapper over mvis.calibration.cam_run.

Usage:
    python scripts/run_calibration.py --config examples/configs/example_cam_run.yaml
    python scripts/run_calibration.py --config examples/configs/example_cam_run.yaml --output results/x

Flags: python scripts/run_calibration.py --help. The regression's camera-only rows run
the same driver as python -m mvis.calibration.cam_run.
"""

import sys
from pathlib import Path

# this checkout's src, not an installed mvis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mvis.calibration.cam_run import (  # noqa: E402,F401  (old names of this script)
    build_parser, configure_process, main, parse_config as _parse_config,
    run_calibration_with_ref, run_cam)

configure_process()

if __name__ == "__main__":
    sys.exit(main() or 0)
