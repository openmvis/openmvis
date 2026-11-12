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
# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
# SPDX-License-Identifier: GPL-3.0-only
"""Validated, no-overwrite front end of `mvis.recalib.apply.apply_calibration`.

The corrections themselves (IMU model, bias, t_ga, td restamp, undistortion, output
connections) are the library's; this module adds the tool's calibration checks, its
`strict` input policy, staging beside the output and an atomic, never-overwriting publish
of the bag and its `.recalib.yaml` sidecar. The library streams the bag (two passes when
td or t_ga is on), so memory does not grow with the image data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import hashlib
import logging
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from mvis.recalib.apply import apply_calibration
from mvis.recalib.calib_spec import load_rig_calib
from mvis.recalib.sensor_map import build_topic_plan, derive_mapping

logger = logging.getLogger(__name__)

# os.link errors meaning "this filesystem cannot hard-link" (exFAT/FAT32, some
# FUSE/SMB mounts), as opposed to the target already existing.
_NO_HARDLINK_ERRNOS = {errno.EPERM, errno.EXDEV, errno.EMLINK,
                       getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                       errno.EOPNOTSUPP}


@dataclass(frozen=True)
class CorrectionOptions:
    imu_intrinsics: bool = True
    remove_bias: bool = True
    td: bool = True
    t_ga: bool = True
    undistort: bool = False


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _load_calibration(path):
    contents = path.read_bytes()
    try:
        doc = yaml.safe_load(contents)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid calibration YAML: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("imus"), list) or not doc["imus"]:
        raise ValueError("calibration must contain a nonempty 'imus' list")
    cameras = doc.get("cameras", [])
    if not isinstance(cameras, list):
        raise ValueError("calibration 'cameras' must be a list")
    names, bases = set(), 0
    for block in doc["imus"]:
        if not isinstance(block, dict):
            raise ValueError("each IMU calibration must be a mapping")
        role = block.get("role", "base")
        if role not in ("base", "aux"):
            raise ValueError(f"unsupported IMU role: {role!r}")
        bases += role == "base"
        name = block.get("name", "imu0")
        _sensor_name(name, names)
        model = block.get("model", "rpng")
        if not isinstance(model, str) or model.lower() not in ("rpng", "kalibr", "calyx"):
            raise ValueError(f"unsupported IMU model for {name}")
        if "imu_intr_xi" in block:
            _array(block["imu_intr_xi"], (24,), f"{name}.imu_intr_xi")
        for field in ("b_g", "b_a"):
            _array(block.get(field, [0., 0., 0.]), (3,), f"{name}.{field}")
        for field in ("td", "t_ga"):
            _array(block.get(field, 0.), (), f"{name}.{field}")
    if bases != 1:
        raise ValueError("calibration must contain exactly one base IMU")
    for i, block in enumerate(cameras):
        if not isinstance(block, dict):
            raise ValueError("each camera calibration must be a mapping")
        name = block.get("name", f"cam{i}")
        _sensor_name(name, names)
        try:
            intr, dist = block["intrinsics"], block["distortion"]
            _array([intr[k] for k in ("fx", "fy", "cx", "cy")], (4,), f"{name}.K")
            coeffs = _array(dist.get("coeffs", []), None, f"{name}.distortion")
            if coeffs.ndim != 1 or not isinstance(dist["model"], str):
                raise ValueError(f"invalid distortion calibration for {name}")
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid camera calibration for {name}") from exc
        for field in ("td", "rolling_shutter_readout"):
            _array(block.get(field, 0.), (), f"{name}.{field}")
    rig = load_rig_calib(str(path))
    rig._src_path = str(path)
    for imu in [rig.base_imu, *rig.aux_imus]:
        for label, matrix in (("Dw", imu.intr.Dw()), ("Da", imu.intr.Da()),
                              ("Tg", imu.intr.Tg()), ("R", imu.intr.R)):
            _array(matrix, (3, 3), f"{imu.name}.{label}")
    return rig, hashlib.sha256(contents).hexdigest()


def _sensor_name(name, names):
    if not isinstance(name, str) or not name.strip() or name in names:
        raise ValueError(f"sensor names must be nonempty and unique: {name!r}")
    names.add(name)


def _topic_plan(rig, mapping, base_imu_topic):
    if mapping is None:
        derived_base, mapping = derive_mapping(rig)
        base_imu_topic = base_imu_topic or derived_base
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be a nonempty topic-to-sensor mapping")
    if any(not isinstance(t, str) or not t.strip() or not isinstance(n, str)
           for t, n in mapping.items()):
        raise ValueError("mapping topics and sensor names must be nonempty strings")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("each sensor may map to only one input topic")
    if base_imu_topic is None:
        candidates = [t for t, name in mapping.items() if name == rig.base_imu.name]
        if len(candidates) != 1:
            raise ValueError("mapping must identify the base IMU topic")
        base_imu_topic = candidates[0]
    build_topic_plan(rig, mapping, base_imu_topic)     # validate before any bag I/O
    return dict(mapping), base_imu_topic


def _publish_one(staged, target):
    """Publish ``staged`` at ``target`` without ever replacing a file.

    A same-filesystem hard link is atomic and refuses an existing target,
    including one created after the preflight.  Filesystems without hard
    links (exFAT/FAT32, some FUSE/SMB mounts) fall back to an exclusive
    create plus copy, which still refuses an existing target but is not
    atomic: a reader could see a partially written file.
    """
    try:
        os.link(staged, target)
        return
    except FileExistsError:
        raise
    except OSError as exc:
        if exc.errno not in _NO_HARDLINK_ERRNOS:
            raise
        logger.warning("hard link unsupported at %s (%s); publishing by "
                       "exclusive copy (not atomic)", target.parent, exc)
    with open(staged, "rb") as src, open(target, "xb") as dst:
        try:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        except BaseException:
            dst.close()
            target.unlink()
            raise


def _publish(staged_bag, staged_sidecar, output, sidecar):
    published = []
    try:
        for staged, target in ((staged_bag, output), (staged_sidecar, sidecar)):
            _publish_one(staged, target)
            published.append(target)
    except BaseException:
        # Only targets this call created are in `published`.
        for target in reversed(published):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        raise


def export_ros1(calib_path, input_path, output_path, *, options=CorrectionOptions(),
                mapping=None, base_imu_topic=None) -> dict:
    """Export a corrected bag and ``<output>.recalib.yaml`` without overwrites.

    ``mapping`` is topic -> sensor name; omitted mappings use stored topics.
    All five operations are independent. With ``td=False``, original header
    and bag record timestamps are preserved. With ``td=True``, both become
    ``original_header + td``. Disabling every option preserves message bytes.
    Temporary artifacts are staged beside the output and removed on failure.
    """
    if not isinstance(options, CorrectionOptions) or any(type(v) is not bool for v in asdict(options).values()):
        raise ValueError("options must be CorrectionOptions with boolean fields")
    calib = Path(calib_path).expanduser().absolute()
    source = Path(input_path).expanduser().absolute()
    output = Path(output_path).expanduser().absolute()
    sidecar = Path(str(output) + ".recalib.yaml")
    for path in (calib, source):
        if not path.is_file():
            raise ValueError(f"input file does not exist: {path}")
    for target in (output, sidecar):
        if any(target.resolve() == path.resolve() or
               (target.exists() and os.path.samefile(target, path)) for path in (calib, source)):
            raise ValueError("output must not alias the input bag or calibration")
        if os.path.lexists(target):
            raise FileExistsError(f"refusing to overwrite existing output: {target}")
    rig, calib_hash = _load_calibration(calib)
    mapping, base_imu_topic = _topic_plan(rig, mapping, base_imu_topic)
    provenance = {
        "tool": "ros1_recalib", "output_bag": str(output), "calib_path": str(calib),
        "calib_sha256": calib_hash, "requested_options": asdict(options),
        "timeline": "header + td on mapped topics" if options.td else "original timestamps",
        "limitations": ["IMU covariance/orientation unchanged",
                        "CameraInfo unchanged", "no rolling-shutter per-row unwarp"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ros1-recalib-", dir=output.parent) as staging:
        staged_bag = Path(staging) / "output.bag"
        summary = apply_calibration(
            str(source), str(staged_bag), rig, mapping, base_imu_topic,
            **asdict(options), strict=True, provenance_extra=provenance)
        _publish(staged_bag, Path(summary["provenance"]), output, sidecar)
    counts = {k: summary[k] for k in ("n_imu", "n_cam", "n_pass")}
    return {"output_path": str(output), "out_bag": str(output), "provenance": str(sidecar),
            **counts, "corrected_topics": summary["corrected_topics"],
            "passthrough_topics": summary["passthrough_topics"],
            "operation_counts": summary["operation_counts"]}
