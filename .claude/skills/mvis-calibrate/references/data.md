# Stage 1 — data formats and qualification

| Format | Detected from | Camera-only | Cam-IMU |
|---|---|---|---|
| ROS1 bag | `*.bag` | yes (`source_type: rosbag_topic`) | yes (`input.format: ros1`) |
| ROS2 bag (sqlite or MCAP) | `metadata.yaml`, `*.db3`, `*.mcap` | yes (`source_type: ros2bag`) | yes (`input.format: ros2`) |
| EuRoC mav0 | `mav0/` or `cam*/data.csv` | yes (`source_type: euroc`) | yes (`input.format: euroc`) |
| Image folder | image files | yes (`source_type: images`) | **no** (no IMU) |
| Images + CSV | image files and `filename,timestamp` rows | yes (`source_type: timestamped_images`) | **no** (no IMU) |

Bags are read with `rosbags` (the `ros1` extra), no ROS installation. Image
topics may be `sensor_msgs/Image` or PNG / JPEG `sensor_msgs/CompressedImage`.
There is no video source: ask the user to extract frames to an image folder.
Catalogued datasets have converters in `mvis datasets`; do not write one.

## `mvis inspect` findings

| id | Meaning | What to do |
|---|---|---|
| `imu.accel_units` FAIL (`|a|≈1`) | accelerometer in g | Data must be m/s². Re-export, don't scale silently. |
| `imu.gyro_units` FAIL | gyro looks like deg/s | Data must be rad/s. |
| `imu.accel_units` WARN (`|a|≈0`) | gyro-only IMU | Mark it gyro-only in Stage 2 (`input.imus[i].gyro_only`). |
| `imu.stamps_monotonic` FAIL | stamps go backwards | Recording or driver problem; this IMU cannot be used as is. |
| `imu.gaps` WARN | dropped IMU samples | Fine if rare; many gaps weaken the IMU-related estimates. |
| `time.cam_imu_clock` FAIL | camera and IMU first stamps > 5 s apart | Different clocks. A time offset cannot absorb this. |
| `data.duration` WARN | < 30 s | Too short for cam-IMU (motion window is 80 s by default, 60 s in the Fast preset). |
| `camera.lossy` WARN | JPEG `CompressedImage` topic | Readable, but JPEG can shift corners. Prefer a raw or PNG recording; say so in the report if there is none. |
| `camera.first_image` FAIL | image cannot be decoded, or an unsupported `CompressedImage` format | MVIS reads raw `Image` and PNG or JPEG `CompressedImage`. Re-record or convert. |

## What a good calibration recording looks like
- Board fully visible, sharp, filling a good share of the image, seen over the
  whole image area (corners and edges, not just the centre).
- Cam-IMU: ≥ 60–90 s of motion that excites **all** rotation axes and all
  translation directions, smooth (no motion blur), with short still periods at
  start/end. The guide: [calibration motion](https://openmvis.com/guides/calibration-motion/).
- The VI-Rig recordings are the reference example of this — see
  [datasets.md](datasets.md).
