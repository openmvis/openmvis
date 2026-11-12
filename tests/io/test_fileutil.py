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
"""mvis.io.fileutil: the shared sha256 and JSON writer give the bytes the copies they replace gave."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from mvis.io.fileutil import compute_sha256, write_json


@pytest.mark.parametrize("n", [0, 1, (1 << 20) - 1, 1 << 20, (1 << 20) + 7, 3 * (1 << 20) + 5])
def test_compute_sha256_matches_the_old_readers(tmp_path, n):
    p = tmp_path / "f.bin"
    p.write_bytes(np.random.default_rng(n).integers(0, 256, n, dtype=np.uint8).tobytes())
    whole = hashlib.sha256(p.read_bytes()).hexdigest()           # twin_regression.file_sha256
    h = hashlib.sha256()                                         # restamp_sensor_clock._sha256
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert compute_sha256(p) == whole == h.hexdigest()
    from mvis.datasets.download import compute_sha256 as old_name
    from mvis.recalib.provenance import _sha256 as prov
    from mvis.recalib.restamp_sensor_clock import _sha256 as restamp
    assert old_name(p) == prov(str(p)) == restamp(str(p)) == whole
    try:
        from mvis.regression.twin_regression import file_sha256  # maintainers' package
    except ImportError:
        return
    assert file_sha256(p) == whole


def test_provenance_sha256_of_a_missing_file_is_none(tmp_path):
    from mvis.recalib.provenance import _sha256
    assert _sha256(str(tmp_path / "missing")) is None


@pytest.mark.parametrize("indent,sort_keys", [(1, False), (2, True)])
def test_write_json_bytes_and_parent_folders(tmp_path, indent, sort_keys):
    d = {"z": [1.5, None, -0.0, 1e-300], "a": {"k": "v", "b": True}, "m": 3}
    p = write_json(tmp_path / "x" / "y" / "out.json", d, indent=indent, sort_keys=sort_keys)
    assert p.read_text() == json.dumps(d, indent=indent, sort_keys=sort_keys)


def test_write_baseline_bytes(tmp_path, monkeypatch):
    from mvis.regression import baseline
    monkeypatch.setattr(baseline, "ref_path", lambda m, d, c: tmp_path / m / f"{d}__{c}.json")
    metrics = {"cam1.fx": 500.123456789, "a": None, "b": [1, 2]}
    p = baseline.write_baseline("cam", "ds", "coll", metrics)
    assert p == tmp_path / "cam" / "ds__coll.json"
    assert p.read_text() == json.dumps(metrics, indent=2, sort_keys=True)


def test_light_io_helpers_do_not_load_numpy_cv2_or_yaml():
    """mvis.io loads its classes lazily, so the stdlib helpers (and mvis.datasets.download,
    which re-exports compute_sha256) stay free of numpy, cv2 and yaml."""
    import subprocess
    import sys
    code = ("import sys, mvis.io.fileutil, mvis.io.imu_reader, mvis.datasets.download\n"
            "print(sorted(m for m in ('numpy', 'cv2', 'yaml') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_mvis_io_classes_resolve_lazily():
    import mvis.io
    from mvis.io.data_source import DataSource
    from mvis.io.image_source import ImageSource
    from mvis.io.output_writer import OutputWriter
    assert (mvis.io.DataSource, mvis.io.ImageSource, mvis.io.OutputWriter) == (
        DataSource, ImageSource, OutputWriter)
    with pytest.raises(AttributeError):
        mvis.io.no_such_name  # noqa: B018
