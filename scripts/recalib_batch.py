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
"""Apply one calib of record across a directory of raw MVIS bags."""
import argparse
import glob
import os
import sys

from mvis.recalib.calib_spec import load_rig_calib
from mvis.recalib.apply import apply_calibration
from mvis.recalib.sensor_map import MappingDeriveError, resolve_mapping


def main():
    p = argparse.ArgumentParser(
        description="Apply one calibration across every *.bag in a directory."
    )
    p.add_argument("--calib", required=True,
                   help="Path to the MVIS result YAML (rig calibration).")
    p.add_argument("--map",
                   help="Path to the topic-map YAML (base_imu_topic + mapping). "
                        "Optional if the calib YAML contains rostopic/topic fields.")
    p.add_argument("--in-dir", required=True,
                   help="Directory containing raw *.bag files.")
    p.add_argument("--out-dir", required=True,
                   help="Directory for corrected bags (created if absent).")
    p.add_argument("--keep-bias", action="store_true",
                   help="Skip bias removal (default: remove bias).")
    p.add_argument("--undistort", action="store_true",
                   help="Undistort images (default: off).")
    a = p.parse_args()

    rig = load_rig_calib(a.calib)
    rig._src_path = a.calib

    try:
        base_imu_topic, mapping = resolve_mapping(rig, a.map)
    except MappingDeriveError as e:
        sys.exit(
            f"recalib_batch: cannot auto-derive mapping: {e}\n"
            "Pass --map or re-export the calib with the current writer."
        )

    os.makedirs(a.out_dir, exist_ok=True)

    bags = sorted(glob.glob(os.path.join(a.in_dir, "*.bag")))
    fails = []

    for b in bags:
        out = os.path.join(a.out_dir, os.path.basename(b))
        try:
            s = apply_calibration(
                b, out, rig, mapping, base_imu_topic,
                remove_bias=not a.keep_bias,
                undistort=a.undistort,
            )
            print(
                f"OK  {os.path.basename(b)}: "
                f"{s['n_imu']} imu / {s['n_cam']} cam / {s['n_pass']} pass"
            )
        except Exception as e:  # noqa: BLE001 — batch resilience
            print(f"FAIL {os.path.basename(b)}: {e}", file=sys.stderr)
            fails.append(b)

    print(f"\n{len(bags) - len(fails)}/{len(bags)} bags corrected.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
