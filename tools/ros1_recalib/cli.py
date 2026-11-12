# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
# Copyright (C) 2026 Yuantai-Z
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
# SPDX-FileCopyrightText: 2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-only
"""Configuration and CLI for the standalone ROS1 correction tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_OPERATIONS = ("imu_intrinsics", "remove_bias", "td", "t_ga", "undistort")
_CONFIG_KEYS = {"calib", "input", "output", "corrections", "mapping", "base_imu_topic"}


def _yaml_mapping(path: Path) -> dict:
    import yaml

    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _path_value(value, name: str, relative_to: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: a nonempty file path is required")
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else relative_to / path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply saved MVIS calibration to a ROS1 bag and write a new bag.",
        epilog="No calibration is solved. Existing output files are never overwritten.",
    )
    parser.add_argument("-c", "--config", type=Path, help="YAML configuration file")
    parser.add_argument("--calib", help="MVIS camera–IMU calibration result YAML")
    parser.add_argument("--input", help="original ROS1 bag")
    parser.add_argument("--output", help="new corrected ROS1 bag")
    parser.add_argument("--map", type=Path, dest="map_file",
                        help="YAML containing mapping and base_imu_topic")
    help_text = {
        "imu_intrinsics": "apply the inverse IMU intrinsic model (default: on)",
        "remove_bias": "remove saved static IMU biases (default: on)",
        "td": "shift sensor header and record times onto the base IMU clock (default: on)",
        "t_ga": "align accelerometer samples to gyro time (default: on)",
        "undistort": "undistort supported mono8 camera images (default: off)",
    }
    for operation in _OPERATIONS:
        parser.add_argument(
            "--" + operation.replace("_", "-"),
            action=argparse.BooleanOptionalAction, default=None,
            help=help_text[operation],
        )
    parser.add_argument("--json", action="store_true", help="print a JSON output summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = _yaml_mapping(args.config) if args.config else {}
        extra = set(cfg) - _CONFIG_KEYS
        if extra:
            raise ValueError(f"unknown configuration keys: {sorted(extra, key=str)}")
        corrections = cfg.get("corrections", {})
        if not isinstance(corrections, dict):
            raise ValueError("corrections must be a mapping of operation names to booleans")
        extra = set(corrections) - set(_OPERATIONS)
        if extra:
            raise ValueError(f"unknown corrections: {sorted(extra, key=str)}")
        for key, value in corrections.items():
            if type(value) is not bool:
                raise ValueError(f"corrections.{key} must be true or false")

        config_dir = args.config.resolve().parent if args.config else Path.cwd()
        paths = {}
        for key in ("calib", "input", "output"):
            cli_value = getattr(args, key)
            value = cli_value if cli_value is not None else cfg.get(key)
            paths[key] = _path_value(value, key, Path.cwd() if cli_value is not None else config_dir)

        options = dict(corrections)
        for operation in _OPERATIONS:
            value = getattr(args, operation)
            if value is not None:
                options[operation] = value

        mapping = cfg.get("mapping")
        base_topic = cfg.get("base_imu_topic")
        if args.map_file:
            map_cfg = _yaml_mapping(args.map_file)
            if set(map_cfg) - {"mapping", "base_imu_topic"} or "mapping" not in map_cfg:
                raise ValueError("--map expects mapping and optional base_imu_topic")
            mapping = map_cfg["mapping"]
            base_topic = map_cfg.get("base_imu_topic")
        if mapping is not None and (
            not isinstance(mapping, dict) or not mapping
            or not all(isinstance(k, str) and k and isinstance(v, str) and v
                       for k, v in mapping.items())
        ):
            raise ValueError("mapping must contain nonempty topic-to-sensor name pairs")
        if base_topic is not None and (not isinstance(base_topic, str) or not base_topic):
            raise ValueError("base_imu_topic must be a nonempty topic name")

        from .export import CorrectionOptions, export_ros1
        from rosbags.rosbag1 import ReaderError, WriterError

        try:
            result = export_ros1(
                paths["calib"], paths["input"], paths["output"],
                options=CorrectionOptions(**options), mapping=mapping, base_imu_topic=base_topic,
            )
        except (ReaderError, WriterError) as exc:
            raise ValueError(f"ROS1 bag I/O failed: {exc}") from exc
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f"ros1-recalib: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Wrote {result['out_bag']} "
              f"({result['n_imu']} IMU, {result['n_cam']} camera, "
              f"{result['n_pass']} passthrough messages)")
        print(f"Provenance: {result['provenance']}")
    return 0
