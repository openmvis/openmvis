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
"""Unit tests for mvis.datasets.catalog — packaged catalog + selection."""
import pytest

from mvis.datasets.catalog import (KNOWN_CONVERTERS, DatasetCatalog,
                                   SelectionItem)


@pytest.fixture(scope="module")
def catalog():
    return DatasetCatalog.load()


def test_catalog_loads_and_validates(catalog):
    keys = catalog.keys()
    assert len(keys) >= 16
    for expected in ["mvis_4imus_3cams", "mvis_4imus_3cams_aruco",
                     "tum_vi_raw", "tum_vi_standard_512", "tum_mic",
                     "monado_mgc", "monado_moc", "tum_event",
                     "uzh_fpv_davis", "uzh_fpv_snapdragon", "tum_fisheye",
                     "penncosyvio", "tum_rgbd_bag", "fisheye_dataset",
                     "looper_insight9", "vi_sensor_rigs_5imus"]:
        assert expected in keys


def test_every_dataset_is_complete(catalog):
    for d in catalog:
        assert d.title and d.sources
        labels = [s.sequence for s in d.sources]
        assert len(labels) == len(set(labels)), f"{d.key}: duplicate sequences"
        for step in d.conversion_chain:
            assert step.converter in KNOWN_CONVERTERS


def test_restricted_licenses_are_never_hosted(catalog):
    from mvis.datasets.catalog import RESTRICTED_BUCKETS
    assert "partner" not in RESTRICTED_BUCKETS     # hosted with permission
    for d in catalog:
        if d.license_bucket in RESTRICTED_BUCKETS:
            assert d.hosting_policy == "recipe_only", d.key


def test_hosted_sets_have_their_licence_and_attribution(catalog):
    """F49/F51/F129: Looper is hosted (CC BY-NC 4.0, gated, with permission);
    every entry states its licence and every hosted one its attribution."""
    looper = catalog.get("looper_insight9")
    assert looper.hosting_policy == "host_rosbag"
    assert looper.license == "CC BY-NC 4.0"
    for key in ("mvis_4imus_3cams", "mvis_4imus_3cams_aruco",
                "mvis_4imus_4cams", "vi_sensor_rigs_5imus"):
        assert catalog.get(key).license == "CC BY-NC 4.0", key
    for key in ("tum_vi_raw", "tum_mic", "monado_mgc", "monado_moc",
                "tum_event", "tum_rgbd_bag"):
        assert catalog.get(key).license == "CC BY 4.0", key
    assert catalog.get("uzh_fpv_davis").license == "CC BY-NC-SA 3.0"
    assert catalog.get("fisheye_dataset").license == "GPL-2.0"
    for d in catalog:
        a = d.attribution
        assert a is not None and a.creator and a.license and a.source_url, d.key
        if d.hosting_policy != "recipe_only":
            assert a.license_url and a.citation, d.key


def test_catalog_notes_carry_only_dataset_facts(catalog):
    """F127/F128: no approval bookkeeping, TODOs or maintainer-disk state."""
    import re
    for d in catalog:
        assert not re.search(r"TODO|FIXME|approved|not local|legacy local|local files",
                             d.notes, re.I), (d.key, d.notes)


def test_catalog_sources_use_https():
    import json
    from pathlib import Path
    import mvis.datasets as ds
    raw = json.loads((Path(ds.__file__).parent / "dataset_catalog.json").read_text())
    for d in raw["datasets"]:
        for s in d["sources"]:
            assert s["url"] is None or s["url"].startswith("https://"), (d["key"], s["url"])


def test_hosted_on_hub_follows_policy_and_source_host(catalog):
    assert catalog.get("tum_event").hosted_on_hub            # conversions
    assert catalog.get("mvis_4imus_3cams").hosted_on_hub     # bags on the Hub
    assert catalog.get("looper_insight9").hosted_on_hub
    assert not catalog.get("tum_vi_raw").hosted_on_hub       # bags at origin
    assert not catalog.get("tum_rgbd_bag").hosted_on_hub
    assert not catalog.get("uzh_fpv_davis").hosted_on_hub    # recipe only


def test_license_falls_back_to_bucket_without_attribution():
    from mvis.datasets.catalog import CalibrationDataset, DataSource
    d = CalibrationDataset(
        key="t", title="t", suites=["camera_only"], license_bucket="cc_by",
        hosting_policy="recipe_only", native_format="rosbag",
        sources=[DataSource("s", "s.bag")])
    assert d.license == "cc_by"                    # raw bucket, nothing nicer


def test_suite_filtering(catalog):
    cam_only = {d.key for d in catalog.filter(suite="camera_only")}
    cam_imu = {d.key for d in catalog.filter(suite="camera_imu")}
    assert "tum_fisheye" in cam_only and "tum_fisheye" not in cam_imu
    assert "mvis_4imus_3cams_aruco" in cam_imu
    assert "mvis_4imus_3cams_aruco" not in cam_only
    assert "tum_rgbd_bag" not in cam_imu


def test_sequence_suite_tags(catalog):
    raw = catalog.get("tum_vi_raw")
    assert [s.sequence for s in raw.sequences(suite="camera_imu")] == \
        ["imu1", "imu2", "imu3", "imu4"]
    mgc = catalog.get("monado_mgc")
    tags = {s.sequence: s.suite for s in mgc.sources}
    assert tags["MGC01_camcalib01_1"] == "camera_only"
    assert tags["MGC04_imucalib01_1"] == "camera_imu"
    assert tags["MGC07_magcalib_1"] == "other"   # excluded from both suites
    assert all(s.suite != "other"
               for s in mgc.sequences(suite="camera_imu"))


def test_selection_inline_and_builtin(catalog):
    items = catalog.select(["mvis_4imus_3cams/april_01", "tum_vi_raw/imu1",
                            "tum_mic/MIC05"])
    assert items == [
        SelectionItem("mvis_4imus_3cams", ("april_01",)),
        SelectionItem("tum_vi_raw", ("imu1",)),
        SelectionItem("tum_mic", ("MIC05",))]
    smoke = catalog.select("smoke")
    assert all(len(i.sequences) == 1 for i in smoke)
    # every sequence referenced by every built-in selection must exist
    for spec in DatasetCatalog.BUILTIN_SELECTIONS.values():
        catalog.select(spec)


def test_selection_rejects_unknown_names(catalog):
    with pytest.raises(KeyError, match="valid:"):
        catalog.select(["no_such_dataset"])
    with pytest.raises(KeyError, match="valid:"):
        catalog.select(["tum_mic/NOPE"])
    with pytest.raises(KeyError, match="valid:"):
        catalog.select("no_such_builtin")


def test_whole_dataset_selection_means_all_sequences(catalog):
    (item,) = catalog.select(["tum_vi_standard_512"])
    assert item.sequences == ()


def test_local_path_follows_mvis_datasets(monkeypatch, tmp_path):
    """F11: a source's local_path is under $MVIS_DATASETS when set, else ~/Datasets."""
    import os
    from pathlib import Path
    from mvis.datasets.catalog import DataSource, catalog_root
    src = DataSource("s", "tum_vi/bags/s.bag")
    monkeypatch.delenv("MVIS_DATASETS", raising=False)
    assert catalog_root() == Path.home() / "Datasets"
    assert src.local_path == Path.home() / "Datasets" / "tum_vi/bags/s.bag"
    monkeypatch.setenv("MVIS_DATASETS", str(tmp_path))
    assert src.local_path == tmp_path / "tum_vi/bags/s.bag"
    monkeypatch.setenv("MVIS_DATASETS", "~/elsewhere")          # a leading ~ expands
    assert src.local_path == Path(os.path.expanduser("~/elsewhere")) / "tum_vi/bags/s.bag"


def test_pinned_datasets_root_wins_over_env(monkeypatch, tmp_path):
    import mvis.datasets.catalog as catalog_module
    monkeypatch.setenv("MVIS_DATASETS", "/nowhere")
    monkeypatch.setattr(catalog_module, "DATASETS_ROOT", tmp_path)
    assert catalog_module.DataSource("s", "a.bin").local_path == tmp_path / "a.bin"
