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
"""Unit tests for mvis.datasets.download — local http.server fixture, no
external network."""
import functools
import hashlib
import http.server
import threading

import pytest

from mvis.datasets.download import (compute_sha256, download_file,
                                    verify_local_files)

PAYLOAD = b"0123456789abcdef" * 4096  # 64 KiB
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(scope="module")
def file_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("served")
    (root / "data.bin").write_bytes(PAYLOAD)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_fresh_download_and_checksum(file_server, tmp_path):
    out = tmp_path / "data.bin"
    download_file(f"{file_server}/data.bin", out,
                  expected_sha256=PAYLOAD_SHA,
                  expected_size_bytes=len(PAYLOAD))
    assert out.read_bytes() == PAYLOAD


def test_skip_when_already_present(file_server, tmp_path):
    out = tmp_path / "data.bin"
    out.write_bytes(PAYLOAD)
    before = out.stat().st_mtime_ns
    download_file(f"{file_server}/data.bin", out,
                  expected_sha256=PAYLOAD_SHA,
                  expected_size_bytes=len(PAYLOAD))
    assert out.stat().st_mtime_ns == before  # untouched


def test_resume_from_truncation(file_server, tmp_path):
    out = tmp_path / "data.bin"
    out.write_bytes(PAYLOAD[: len(PAYLOAD) // 2])
    download_file(f"{file_server}/data.bin", out,
                  expected_sha256=PAYLOAD_SHA,
                  expected_size_bytes=len(PAYLOAD))
    assert out.read_bytes() == PAYLOAD


def test_corrupt_file_is_refetched(file_server, tmp_path):
    out = tmp_path / "data.bin"
    out.write_bytes(b"x" * len(PAYLOAD))  # right size, wrong content
    download_file(f"{file_server}/data.bin", out,
                  expected_sha256=PAYLOAD_SHA,
                  expected_size_bytes=len(PAYLOAD))
    assert out.read_bytes() == PAYLOAD


def test_checksum_mismatch_raises(file_server, tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        download_file(f"{file_server}/data.bin", tmp_path / "bad.bin",
                      expected_sha256="0" * 64,
                      expected_size_bytes=len(PAYLOAD))


def test_verify_local_files_states(tmp_path, monkeypatch):
    import mvis.datasets.catalog as catalog_module
    monkeypatch.setattr(catalog_module, "DATASETS_ROOT", tmp_path)
    from mvis.datasets.catalog import CalibrationDataset, DataSource
    (tmp_path / "ok.bin").write_bytes(PAYLOAD)
    (tmp_path / "short.bin").write_bytes(PAYLOAD[:10])
    (tmp_path / "corrupt.bin").write_bytes(b"y" * len(PAYLOAD))
    dataset = CalibrationDataset(
        key="t", title="t", suites=["camera_only"], license_bucket="own",
        hosting_policy="recipe_only", native_format="rosbag",
        sources=[
            DataSource("ok", "ok.bin", None, PAYLOAD_SHA, len(PAYLOAD)),
            DataSource("short", "short.bin", None, PAYLOAD_SHA, len(PAYLOAD)),
            DataSource("corrupt", "corrupt.bin", None, PAYLOAD_SHA,
                       len(PAYLOAD)),
            DataSource("absent", "absent.bin", None, PAYLOAD_SHA,
                       len(PAYLOAD)),
            DataSource("unverifiable", "ok.bin", None, None, None),
        ])
    report = verify_local_files(dataset)
    assert report == {"ok": "ok", "short": "size_mismatch_partial",
                      "corrupt": "checksum_mismatch", "absent": "missing",
                      "unverifiable": "unverified"}


def test_compute_sha256(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(PAYLOAD)
    assert compute_sha256(f) == PAYLOAD_SHA


def test_hf_token_from_env_and_file(tmp_path, monkeypatch):
    from mvis.datasets.download import hf_token
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    assert hf_token() is None
    (tmp_path / "token").write_text("hf_file\n")
    assert hf_token() == "hf_file"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hf_token() == "hf_env"


def test_token_sent_only_to_huggingface(monkeypatch, tmp_path):
    import mvis.datasets.download as D
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    seen = []

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n): return b""

    monkeypatch.setattr(D._OPENER, "open",
                        lambda req: seen.append((req.full_url, req.get_header("Authorization"))) or Resp())
    D.download_file("https://huggingface.co/datasets/x/resolve/main/a.bin", tmp_path / "a.bin")
    D.download_file("https://example.org/a.bin", tmp_path / "b.bin")
    D.download_file("http://huggingface.co/datasets/x/resolve/main/a.bin", tmp_path / "c.bin")
    D.download_file("https://hf.co/datasets/x/resolve/main/a.bin", tmp_path / "d.bin")
    D.download_file("https://nothf.co/datasets/x/resolve/main/a.bin", tmp_path / "e.bin")
    assert seen == [("https://huggingface.co/datasets/x/resolve/main/a.bin", "Bearer hf_secret"),
                    ("https://example.org/a.bin", None),
                    # plain http would send the token in cleartext: never
                    ("http://huggingface.co/datasets/x/resolve/main/a.bin", None),
                    # the short Hub host the catalog also accepts; a look-alike is not it
                    ("https://hf.co/datasets/x/resolve/main/a.bin", "Bearer hf_secret"),
                    ("https://nothf.co/datasets/x/resolve/main/a.bin", None)]


def test_redirect_to_other_host_drops_token():
    import urllib.request
    from mvis.datasets.download import _DropAuthOnHostChange
    req = urllib.request.Request("https://huggingface.co/a", headers={"Authorization": "Bearer t"})
    h = _DropAuthOnHostChange()
    same = h.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/b")
    other = h.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.net/b")
    plain = h.redirect_request(req, None, 302, "Found", {}, "http://huggingface.co/b")
    assert same.get_header("Authorization") == "Bearer t"
    assert other.get_header("Authorization") is None
    assert plain.get_header("Authorization") is None     # same host, not https


def test_gated_403_gives_actionable_error(monkeypatch, tmp_path):
    import io
    import urllib.error
    import mvis.datasets.download as D
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    def deny(req):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO())
    monkeypatch.setattr(D._OPENER, "open", deny)
    with pytest.raises(PermissionError, match="gated.*hf auth login"):
        D.download_file("https://huggingface.co/datasets/x/resolve/main/a.bin", tmp_path / "a.bin")


def _dataset_with(tmp_path, *sources):
    from mvis.datasets.catalog import CalibrationDataset
    return CalibrationDataset(
        key="t", title="t", suites=["camera_only"], license_bucket="own",
        hosting_policy="recipe_only", native_format="rosbag",
        sources=list(sources))


def test_download_dataset_skips_unverified_sources_by_default(
        file_server, tmp_path, monkeypatch, capsys):
    import mvis.datasets.catalog as catalog_module
    from mvis.datasets.catalog import DataSource
    from mvis.datasets.download import download_dataset
    monkeypatch.setattr(catalog_module, "DATASETS_ROOT", tmp_path)
    dataset = _dataset_with(
        tmp_path,
        DataSource("pinned", "pinned.bin", f"{file_server}/data.bin",
                   PAYLOAD_SHA, len(PAYLOAD)),
        DataSource("loose", "loose.bin", f"{file_server}/data.bin", None, None),
        DataSource("private", "private.bin", None, None, None))
    report = download_dataset(dataset, progress=False)
    assert report == {"pinned": "ok", "loose": "unverified_skipped",
                      "private": "no_public_url"}
    assert (tmp_path / "pinned.bin").exists()
    assert not (tmp_path / "loose.bin").exists()
    err = capsys.readouterr().err
    assert "no sha256" in err and "--allow-unverified" in err

    report = download_dataset(dataset, progress=False, allow_unverified=True)
    assert report["loose"] == "ok"
    assert (tmp_path / "loose.bin").read_bytes() == PAYLOAD
    assert "without an integrity check" in capsys.readouterr().err
