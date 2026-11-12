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
"""mvis.io.safe_yaml: safe loading of result YAMLs that may carry old numpy / tuple tags."""
import numpy as np
import pytest
import yaml

from mvis.io.safe_yaml import load_yaml_file_safe, load_yaml_safe

VALUES = {
    "f64": np.float64(1.5), "i64": np.int64(-3), "b": np.bool_(True), "tup": (1, 2),
    "arr": np.arange(6, dtype=np.float32).reshape(2, 3),
    "fortran": np.asfortranarray(np.arange(6.0).reshape(2, 3)),
    "T": np.eye(4), "objs": np.array([1, "x"], dtype=object),
}


def test_numpy_tags_load_like_unsafe_load():
    text = yaml.dump(VALUES)                    # python tags, as old result files wrote them
    assert "!!python/object/apply:numpy" in text
    got, ref = load_yaml_safe(text), yaml.unsafe_load(text)
    for k in VALUES:
        a, b = got[k], ref[k]
        assert type(a) is type(b), k
        assert np.asarray(a).dtype == np.asarray(b).dtype, k
        assert np.array_equal(np.asarray(a), np.asarray(b)), k


def test_plain_yaml_is_safe_load(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("cameras:\n- id: 0\n  intrinsics: {fx: 1.0}\nlist: [1, 2.5, null]\n")
    assert load_yaml_file_safe(p) == yaml.safe_load(p.read_text())


@pytest.mark.parametrize("text", [
    '!!python/object/apply:os.system ["true"]',
    '!!python/name:os.system ""',
    '!!python/object/new:subprocess.Popen [["true"]]',
])
def test_other_python_tags_are_refused(text):
    with pytest.raises(yaml.YAMLError):
        load_yaml_safe(text)
