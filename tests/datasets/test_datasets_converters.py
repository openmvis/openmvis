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
"""Unit tests for mvis.datasets.converters — tiny synthetic fixtures."""
import numpy as np
import pytest
from PIL import Image as PILImage
from rosbags.rosbag1 import Reader

from mvis.datasets.converters import (convert_euroc_folder_to_rosbag,
                                      convert_image_folder_to_rosbag,
                                      convert_rosbag_mono16_to_mono8,
                                      convert_rosbag_to_mcap,
                                      convert_tumvie_vidata_to_euroc_folder,
                                      extract_archive)


def _write_png(path, value, size=(8, 6)):
    PILImage.fromarray(np.full(size[::-1], value, np.uint8)).save(path)


@pytest.fixture()
def euroc_folder(tmp_path):
    """3-frame, 2-camera EuRoC folder with 5 IMU samples."""
    mav0 = tmp_path / "mav0"
    stamps = [1_000_000_000, 1_100_000_000, 1_200_000_000]
    for cam in ("cam0", "cam1"):
        (mav0 / cam / "data").mkdir(parents=True)
        rows = ["#timestamp [ns],filename"]
        for i, ts in enumerate(stamps):
            _write_png(mav0 / cam / "data" / f"{ts}.png", 10 * i + 5)
            rows.append(f"{ts},{ts}.png")
        (mav0 / cam / "data.csv").write_text("\n".join(rows) + "\n")
    (mav0 / "imu0").mkdir()
    imu_rows = ["#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z"]
    for k in range(5):
        imu_rows.append(f"{1_000_000_000 + k * 50_000_000},"
                        "0.1,0.2,0.3,0.0,0.0,9.81")
    (mav0 / "imu0" / "data.csv").write_text("\n".join(imu_rows) + "\n")
    return mav0


def _topics(bag):
    with Reader(bag) as reader:
        return {c.topic: c.msgcount for c in reader.connections}


def test_euroc_folder_to_rosbag_round_trip(euroc_folder, tmp_path):
    bag = tmp_path / "out.bag"
    stats = convert_euroc_folder_to_rosbag(euroc_folder, bag)
    assert stats == {"images": 6, "imu_samples": 5,
                     "cameras": ["cam0", "cam1"]}
    assert _topics(bag) == {"/cam0/image_raw": 3, "/cam1/image_raw": 3,
                            "/imu0": 5}
    # stamps preserved exactly
    with Reader(bag) as reader:
        times = [t for _, t, _ in reader.messages()]
    assert min(times) == 1_000_000_000 and max(times) == 1_200_000_000


def test_euroc_folder_without_imu(euroc_folder, tmp_path):
    import shutil
    shutil.rmtree(euroc_folder / "imu0")
    bag = tmp_path / "out.bag"
    stats = convert_euroc_folder_to_rosbag(euroc_folder, bag)
    assert stats["imu_samples"] == 0
    assert "/imu0" not in _topics(bag)


def test_euroc_folder_imu_only_raw_csv(euroc_folder, tmp_path):
    # e.g. Monado MOC13_imustatic: imu0/ only, no cameras, and the IMU
    # samples live in data.raw.csv instead of data.csv
    import shutil
    for cam in ("cam0", "cam1"):
        shutil.rmtree(euroc_folder / cam)
    (euroc_folder / "imu0" / "data.csv").rename(
        euroc_folder / "imu0" / "data.raw.csv")
    bag = tmp_path / "out.bag"
    stats = convert_euroc_folder_to_rosbag(euroc_folder, bag)
    assert stats == {"images": 0, "imu_samples": 5, "cameras": []}
    assert _topics(bag) == {"/imu0": 5}


def test_image_folder_to_rosbag_deterministic(tmp_path):
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(3):
        _write_png(images / f"img_{i}.png", 40 * i)
    bag_a, bag_b = tmp_path / "a.bag", tmp_path / "b.bag"
    stats = convert_image_folder_to_rosbag(images, bag_a, frame_rate_hz=2.0)
    convert_image_folder_to_rosbag(images, bag_b, frame_rate_hz=2.0)
    assert stats["timestamps"] == "synthetic"
    assert bag_a.read_bytes() == bag_b.read_bytes()   # byte-identical re-run
    with Reader(bag_a) as reader:
        times = [t for _, t, _ in reader.messages()]
    assert times == [0, 500_000_000, 1_000_000_000]   # 2 Hz from epoch 0


def test_tumvie_vidata_microseconds_to_nanoseconds(tmp_path):
    seq = tmp_path / "seq-vi_data"
    for side in ("left", "right"):
        d = seq / f"{side}_images"
        d.mkdir(parents=True)
        for i in range(2):
            _write_png(d / f"{i:05d}.jpg", 100 + i)
        (d / f"image_timestamps_{side}.txt").write_text(
            "# timestamps [us]\n1000000.25\n1050000.75\n")
    (seq / "imu_data.txt").write_text(
        "# header\n1000000.5 0.1 0.2 0.3 0.0 0.0 9.81 30.0\n")
    euroc = convert_tumvie_vidata_to_euroc_folder(seq)
    cam0 = (euroc / "cam0" / "data.csv").read_text().splitlines()
    assert cam0[1].split(",")[0] == "1000000250"      # exact us -> ns
    assert cam0[2].split(",")[0] == "1050000750"
    imu = (euroc / "imu0" / "data.csv").read_text().splitlines()
    assert imu[1].split(",")[0] == "1000000500"


def test_mono16_reencode_and_restamp(euroc_folder, tmp_path):
    # build a bag with one mono16 image whose header clock differs from
    # bag-record time, then convert
    from mvis.datasets.converters.euroc_folder_to_rosbag import (
        TYPESTORE, _Image, _header)
    from rosbags.rosbag1 import Writer
    src = tmp_path / "m16.bag"
    with Writer(src) as writer:
        conn = writer.add_connection("/cam0/image_raw", _Image.__msgtype__,
                                     typestore=TYPESTORE)
        array = (np.arange(48, dtype=np.uint16) * 1000).reshape(6, 8)
        msg = _Image(header=_header(5_000_000_000, "cam0"), height=6, width=8,
                     encoding="mono16", is_bigendian=0, step=16,
                     data=array.view(np.uint8).reshape(-1))
        writer.write(conn, 9_999, TYPESTORE.serialize_ros1(
            msg, _Image.__msgtype__))          # bag-record time != header
    out = tmp_path / "m8.bag"
    stats = convert_rosbag_mono16_to_mono8(
        src, out, image_topics=["/cam0/image_raw"],
        keep_topics=["/cam0/image_raw"])
    assert stats == {"converted": 1, "passthrough": 0}
    with Reader(out) as reader:
        ((_, record_time, raw),) = list(reader.messages())
        assert record_time == 5_000_000_000    # re-stamped to header clock
        message = TYPESTORE.deserialize_ros1(
            raw, "sensor_msgs/msg/Image")
        assert message.encoding == "mono8"
        # the shared mono16 rule (max > 4095: upper byte), as every MVIS reader decodes the raw bag
        np.testing.assert_array_equal(
            np.frombuffer(bytes(message.data), np.uint8).reshape(6, 8), (array >> 8).astype(np.uint8))


def test_rosbag_to_mcap_message_parity(euroc_folder, tmp_path):
    bag = tmp_path / "out.bag"
    convert_euroc_folder_to_rosbag(euroc_folder, bag)
    mcap = convert_rosbag_to_mcap(bag)
    assert mcap.exists() and mcap.suffix == ".mcap"
    from rosbags.rosbag2 import Reader as Reader2
    with Reader2(mcap.parent / mcap.name) as reader:
        count = sum(1 for _ in reader.messages())
    assert count == 6 + 5                       # images + imu samples


@pytest.fixture()
def penncosyvio_sequence(tmp_path):
    """Tiny mp4 (5 frames) + timestamps/IMU in PennCOSYVIO layout."""
    av = pytest.importorskip("av")
    seq = tmp_path / "af"
    seq.mkdir()
    with av.open(str(seq / "seq_af.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 16, 16
        stream.pix_fmt = "yuv420p"
        for i in range(5):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), 40 * i, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return seq


def test_penncosyvio_surplus_stamps_dropped(penncosyvio_sequence, tmp_path):
    # 7 stamps for 5 frames (starting negative, as in the real data):
    # frame i pairs with stamp i, surplus trailing stamps are reported
    from mvis.datasets.converters import convert_penncosyvio_sequence_to_rosbag
    stamps = [(-1.0 + i / 30.0) for i in range(7)]
    (penncosyvio_sequence / "timestamps.txt").write_text(
        "\n".join(f"{t:.6f}" for t in stamps) + "\n")
    bag = tmp_path / "af.bag"
    stats = convert_penncosyvio_sequence_to_rosbag(penncosyvio_sequence, bag)
    assert stats["images"] == 5
    assert stats["unmatched_trailing_stamps"] == 2
    with Reader(bag) as reader:
        times = [t for _, t, _ in reader.messages()]
    assert len(times) == 5
    offset_ns = int(stats["timestamp_offset_applied_s"] * 1e9)
    assert times[0] == int(round((stamps[0]) * 1e9)) + offset_ns


def test_penncosyvio_too_few_stamps_raises(penncosyvio_sequence, tmp_path):
    from mvis.datasets.converters import convert_penncosyvio_sequence_to_rosbag
    (penncosyvio_sequence / "timestamps.txt").write_text(
        "0.0\n0.033\n0.066\n")                 # 3 stamps for 5 frames
    with pytest.raises(ValueError, match="ambiguous"):
        convert_penncosyvio_sequence_to_rosbag(
            penncosyvio_sequence, tmp_path / "af.bag")


def test_extract_archive_zip_and_skip(tmp_path):
    import zipfile
    payload = tmp_path / "content"
    payload.mkdir()
    (payload / "a.txt").write_text("x")
    archive = tmp_path / "c.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(payload / "a.txt", "content/a.txt")
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "content" / "a.txt").read_text() == "x"
    (dest / "content" / "a.txt").write_text("modified")
    extract_archive(archive, dest)              # skip: already extracted
    assert (dest / "content" / "a.txt").read_text() == "modified"


def test_extract_archive_tar_and_skip(tmp_path):
    import tarfile
    payload = tmp_path / "content"
    payload.mkdir()
    (payload / "a.txt").write_text("x")
    archive = tmp_path / "c.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload / "a.txt", "content/a.txt")
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "content" / "a.txt").read_text() == "x"
    (dest / "content" / "a.txt").write_text("modified")
    extract_archive(archive, dest)              # skip: already extracted
    assert (dest / "content" / "a.txt").read_text() == "modified"


@pytest.mark.parametrize("member", ["../escaped.txt", "content/../../escaped.txt",
                                    "/tmp/escaped.txt"])
def test_extract_archive_refuses_tar_path_traversal(tmp_path, member):
    """A crafted tar must not write outside the destination directory."""
    import io
    import tarfile
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("content/ok.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
        info = tarfile.TarInfo(member)
        info.size = 4
        tf.addfile(info, io.BytesIO(b"evil"))
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="outside"):
        extract_archive(archive, dest)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (dest / "content" / "ok.txt").exists()   # nothing extracted


def test_extract_archive_refuses_tar_symlink_out_of_tree(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("content/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        tf.addfile(info)
        info = tarfile.TarInfo("content/link/payload.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="outside"):
        extract_archive(archive, tmp_path / "out")


def test_extract_archive_refuses_path_through_symlink_member(tmp_path):
    """A symlink member to '..' followed by a path through it escapes a
    lexical check; the member list itself must be refused."""
    import io
    import tarfile
    archive = tmp_path / "chain.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("content/d")
        info.type = tarfile.SYMTYPE
        info.linkname = ".."              # stays inside out/ ... lexically
        tf.addfile(info)
        info = tarfile.TarInfo("content/d/../../evil.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="symlink member"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
