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
"""ROS2 bag data source (optional dependency: rosbag2_py and rclpy from a ROS 2 install).

Frames are decoded by io.image_decode from the message bytes, as in the ROS1 readers;
cv_bridge is not needed."""

from typing import Iterator

import numpy as np

from mvis.io.data_source import DataSource
from mvis.io.image_decode import ros_image_to_gray

_COMPRESSED = "sensor_msgs/msg/CompressedImage"


class Ros2BagSource(DataSource):
    """Load images from a ROS2 bag file.

    Requires rosbag2_py and rclpy (not cv_bridge).
    """

    def __init__(self, bag_path: str, topics: list[str]):
        """
        Args:
            bag_path: Path to ROS2 bag directory.
            topics: List of image topic names (one per camera).
        """
        try:
            import rosbag2_py  # noqa: F401
        except ImportError:
            raise ImportError(
                "ROS2 bag support requires 'rosbag2_py'. "
                "Install via ROS2 workspace."
            )
        self._bag_path = bag_path
        self._topics = topics
        self._topic_to_cam = {t: i for i, t in enumerate(topics)}

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CompressedImage, Image
        storage_options = rosbag2_py.StorageOptions(
            uri=self._bag_path, storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        msg_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        jpeg_warned: set[str] = set()

        topic_filter = rosbag2_py.StorageFilter(topics=self._topics)
        reader.set_filter(topic_filter)

        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            if topic not in self._topic_to_cam:
                continue

            cam_id = self._topic_to_cam[topic]
            compressed = msg_types.get(topic) == _COMPRESSED
            msg = deserialize_message(data, CompressedImage if compressed else Image)
            # header.stamp, as in the ROS1 path; record time only if it was never set
            stamp = msg.header.stamp
            timestamp = (stamp.sec + stamp.nanosec * 1e-9
                         if (stamp.sec or stamp.nanosec) else timestamp_ns * 1e-9)
            yield ros_image_to_gray(msg, topic, jpeg_warned), timestamp, cam_id

    def num_cameras(self) -> int:
        """Number of cameras this source provides."""
        return len(self._topics)
