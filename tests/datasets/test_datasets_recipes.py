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
"""Tests for recipe_export + huggingface upload planning (no network)."""
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mvis.datasets import converters as _converters
from mvis.datasets.catalog import DatasetCatalog
from mvis.datasets.huggingface import (build_dataset_card, plan_upload,
                                       upload_dataset_to_huggingface)
from mvis.datasets.recipe_export import (EMBEDDED_MODULES,
                                         build_standalone_convert_script,
                                         export_standalone_recipes)


# ---- recipe export --------------------------------------------------------

def test_exported_script_embeds_lib_functions_verbatim():
    script = build_standalone_convert_script()
    for name in ("convert_euroc_folder_to_rosbag",
                 "convert_penncosyvio_sequence_to_rosbag",
                 "convert_image_folder_to_rosbag",
                 "convert_rosbag_to_mcap",
                 "extract_archive"):
        body = inspect.getsource(getattr(_converters, name))
        assert body in script, f"{name} drifted from the lib source"


def test_exported_script_compiles_and_has_no_relative_imports(tmp_path):
    (script, manifest, _sources) = export_standalone_recipes(tmp_path)
    compile(script.read_text(), str(script), "exec")   # syntax-valid
    assert "from ." not in script.read_text()
    data = json.loads(manifest.read_text())
    assert set(data["datasets"]) == set(DatasetCatalog.load().keys())
    # recipe-only sets are marked not hosted
    assert data["datasets"]["uzh_fpv_davis"]["hosted_converted"] is False
    assert data["datasets"]["tum_event"]["hosted_converted"] is True
    assert data["datasets"]["looper_insight9"]["hosted_converted"] is True
    assert data["datasets"]["tum_vi_raw"]["hosted_converted"] is False   # bags at origin
    assert data["datasets"]["tum_event"]["license"] == "CC BY 4.0"


def test_exported_sources_md_carries_attribution(tmp_path):
    """F51: SOURCES.md is generated from the catalog, so the card's
    reference to it is true and the CC BY attribution ships with the code."""
    (_script, _manifest, sources) = export_standalone_recipes(tmp_path)
    assert sources.name == "SOURCES.md"
    text = sources.read_text()
    catalog = DatasetCatalog.load()
    for d in catalog:
        assert f"`{d.key}`" in text
        assert d.attribution.creator in text, d.key
        assert d.attribution.source_url in text, d.key
    assert "https://creativecommons.org/licenses/by/4.0/" in text
    assert "https://creativecommons.org/licenses/by-nc/4.0/" in text
    assert "Monado SLAM Dataset" in text          # citation present
    assert "approved" not in text                  # no bookkeeping leaks


def test_exported_script_runs_euroc_conversion(tmp_path, euroc_like_folder):
    (script, _, _) = export_standalone_recipes(tmp_path)
    out = tmp_path / "out.bag"
    result = subprocess.run(
        [sys.executable, str(script), "euroc_folder",
         str(euroc_like_folder), str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0


@pytest.fixture()
def euroc_like_folder(tmp_path):
    import numpy as np
    from PIL import Image as PILImage
    mav0 = tmp_path / "mav0"
    (mav0 / "cam0" / "data").mkdir(parents=True)
    rows = ["#timestamp [ns],filename"]
    for i in range(2):
        ts = 1_000_000_000 + i * 100_000_000
        PILImage.fromarray(np.full((6, 8), 30 * i, np.uint8)).save(
            mav0 / "cam0" / "data" / f"{ts}.png")
        rows.append(f"{ts},{ts}.png")
    (mav0 / "cam0" / "data.csv").write_text("\n".join(rows) + "\n")
    return mav0


def test_embedded_module_list_covers_registry():
    # every registry converter except the partner-only mcap route is embedded
    exported = set(EMBEDDED_MODULES)
    assert exported == {
        "archive", "euroc_folder_to_rosbag", "tumvie_to_euroc_folder",
        "penncosyvio_to_rosbag", "image_folder_to_rosbag",
        "rosbag_mono16_to_mono8", "rosbag_to_mcap"}


# ---- huggingface upload planning ------------------------------------------

def test_plan_upload_refuses_recipe_only():
    catalog = DatasetCatalog.load()
    with pytest.raises(ValueError, match="recipe_only"):
        plan_upload(catalog.get("uzh_fpv_davis"))


def test_upload_calls_hf_cli_per_file_and_card(tmp_path, monkeypatch):
    catalog = DatasetCatalog.load()
    dataset = catalog.get("monado_moc")
    source = dataset.get_sequence("MOC02_imucalib_1")
    # fake a converted bag + sidecar in the expected bags/ dir
    bags_dir = source.local_path.parent / "bags"
    monkeypatch.setattr(
        "mvis.datasets.huggingface.converted_bag_dirs",
        lambda s: [tmp_path / "bags"])
    fake_bags = tmp_path / "bags"
    fake_bags.mkdir()
    (fake_bags / "MOC02_imucalib_1.bag").write_bytes(b"x" * 128)
    (fake_bags / "MOC02_imucalib_1.provenance.json").write_text(
        json.dumps({"source_sha256": "ab" * 32}))
    calls = []
    report = upload_dataset_to_huggingface(
        dataset, sequences=["MOC02_imucalib_1"],
        runner=lambda cmd, check: calls.append(cmd))
    assert report == {"MOC02_imucalib_1": "ok"}
    uploaded = [c[3] for c in calls]                 # local path argument
    assert any(p.endswith(".bag") for p in uploaded)
    assert any(p.endswith(".provenance.json") for p in uploaded)
    assert calls[-2][4] == "converted/monado_moc/README.md"
    assert calls[-1][4] == "recipes/SOURCES.md"   # attribution file refreshed
    assert all(c[:3] == ["hf", "upload", "calibench-vi/datasets"]
               for c in calls)


def test_upload_dry_run_reports_not_converted(monkeypatch, tmp_path):
    catalog = DatasetCatalog.load()
    dataset = catalog.get("monado_moc")
    monkeypatch.setattr(
        "mvis.datasets.huggingface.converted_bag_dirs",
        lambda s: [tmp_path / "empty"])
    report = upload_dataset_to_huggingface(
        dataset, sequences=["MOC05_imucalib_2"], dry_run=True,
        runner=lambda *a, **k: pytest.fail("dry run must not upload"))
    assert report == {"MOC05_imucalib_2": "not_converted"}


def test_virig_mcap_twin_lands_beside_hosted_bag(monkeypatch, tmp_path):
    catalog = DatasetCatalog.load()
    dataset = catalog.get("mvis_4imus_3cams")     # native rosbag, hosted
    source = dataset.get_sequence("april_01")
    twin = tmp_path / Path(source.local_relative_path).with_suffix(".mcap").name
    twin.mkdir()                                   # ROS2 folder bag
    monkeypatch.setattr(
        "mvis.datasets.huggingface.DataSource.local_path",
        property(lambda s: tmp_path / Path(s.local_relative_path).name))
    calls = []
    report = upload_dataset_to_huggingface(
        dataset, sequences=["april_01"],
        runner=lambda cmd, check: calls.append(cmd))
    assert report == {"april_01": "ok"}
    (call, sources) = calls                        # no card for rosbag-native
    assert call[4] == ("virig-4imus-3cams/"
                       "virig_4imus_3cams_april_01.mcap")
    assert sources[4] == "recipes/SOURCES.md"


def test_dataset_card_lists_files_and_license(tmp_path):
    catalog = DatasetCatalog.load()
    dataset = catalog.get("tum_event")
    bag = tmp_path / "imu_calibA1.bag"
    bag.write_bytes(b"y" * 2048)
    card = build_dataset_card(dataset, [bag], version_stamp="2026-08-29")
    assert "CC BY 4.0" in card                     # cc_by bucket
    assert "https://creativecommons.org/licenses/by/4.0/" in card
    assert "TUM-VIE: The TUM Stereo Visual-Inertial Event Dataset" in card
    assert "visual-inertial-event-dataset" in card  # source URL
    assert "recipes/SOURCES.md" in card
    assert "imu_calibA1.bag" in card
    assert "convert_tumvie_vidata_to_euroc_folder" in card


def test_sources_md_upload_is_separate_from_dataset_uploads(tmp_path, monkeypatch):
    from mvis.datasets.huggingface import upload_sources_markdown
    catalog = DatasetCatalog.load()
    dataset = catalog.get("mvis_4imus_3cams")
    twin = tmp_path / "virig_4imus_3cams_april_01.mcap"
    twin.mkdir()
    monkeypatch.setattr(
        "mvis.datasets.huggingface.DataSource.local_path",
        property(lambda s: tmp_path / Path(s.local_relative_path).name))
    calls = []
    upload_dataset_to_huggingface(
        dataset, sequences=["april_01"], with_sources=False,
        runner=lambda cmd, check: calls.append(cmd))
    assert [c[4] for c in calls] == ["virig-4imus-3cams/virig_4imus_3cams_april_01.mcap"]
    seen = []

    def runner(cmd, check):
        seen.append(cmd)
        assert "Monado SLAM Dataset" in Path(cmd[3]).read_text()
    assert upload_sources_markdown(runner=runner) == "recipes/SOURCES.md"
    assert seen[0][4] == "recipes/SOURCES.md"


def test_looper_is_uploadable_as_hosted_partner_data():
    catalog = DatasetCatalog.load()
    plan = plan_upload(catalog.get("looper_insight9"), sequences=["bag_1"])
    assert list(plan) == ["bag_1"]


def test_select_upload_card_keeps_other_sequences(tmp_path, monkeypatch):
    """A --select push must not truncate the folder card to the pushed subset."""
    catalog = DatasetCatalog.load()
    dataset = catalog.get("monado_moc")
    monkeypatch.setattr(
        "mvis.datasets.huggingface.converted_bag_dirs",
        lambda s: [tmp_path / "bags"])
    fake_bags = tmp_path / "bags"
    fake_bags.mkdir()
    for seq in ("MOC01_camcalib_1", "MOC02_imucalib_1"):
        (fake_bags / f"{seq}.bag").write_bytes(b"x" * 128)
        (fake_bags / f"{seq}.provenance.json").write_text(
            json.dumps({"source_sha256": "ab" * 32}))

    cards = []

    def runner(cmd, check):
        if cmd[4].endswith("README.md"):
            cards.append(Path(cmd[3]).read_text())

    upload_dataset_to_huggingface(
        dataset, sequences=["MOC02_imucalib_1"], runner=runner)

    assert len(cards) == 1
    # the pushed sequence AND the one merely present on disk both appear
    assert "`MOC02_imucalib_1.bag`" in cards[0]
    assert "`MOC01_camcalib_1.bag`" in cards[0]


def test_dataset_card_licence_for_own_and_partner_recordings(tmp_path):
    # VI-Rig (own) and Looper (partner) are CC BY-NC 4.0 and gated, not CC BY.
    catalog = DatasetCatalog.load()
    for key in ("mvis_4imus_3cams", "looper_insight9"):
        card = build_dataset_card(catalog.get(key), [], version_stamp="2026-11-08")
        assert "CC BY-NC 4.0" in card, key
        assert "gated" in card, key


def test_catalog_urls_are_pinned_to_a_revision():
    import json
    import re
    from pathlib import Path
    import mvis.datasets as ds
    text = (Path(ds.__file__).parent / "dataset_catalog.json").read_text()
    assert "/resolve/main/" not in text
    for url in re.findall(r'"url": "([^"]+huggingface\.co[^"]+)"', text):
        assert re.search(r"/resolve/[0-9a-f]{40}/", url), url
    json.loads(text)
