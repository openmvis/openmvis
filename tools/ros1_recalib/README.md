# ROS1 calibration application tool

Apply an existing MVIS camera–IMU calibration to a ROS1 bag and write a new bag
plus a provenance YAML. The corrections are the library's
`mvis.recalib.apply.apply_calibration`, the same code `mvis apply-calib` and
`scripts/recalib_batch.py` run; this tool adds per-operation switches on the command
line, stricter input checks (`strict=True`: missing or empty mapped topics, wrong
message types, unsupported undistortion encodings and duplicate IMU stamps on a `t_ga`
topic are errors, not passthrough) and a staged, never-overwriting publish.

From this checkout, using a Python environment with the MVIS dependencies and
`rosbags` installed:

```bash
python tools/ros1_recalib/run.py --calib calibration.yaml \
  --input raw.bag --output corrected.bag
```

The script also works by absolute path from another directory. It uses this
checkout's numerical model code without changing the installed MVIS package.
Python 3.11 or newer is supported. In a separate environment, the repository's
`pip install -e '.[ros1]'` supplies the dependencies. This tool is a dedicated
entry point within the repository, not a separately installable distribution;
GTSAM remains required by the existing IMU calibration model loader.

## Configuration

Copy [example.yaml](example.yaml), set its paths and operations, then run:

```bash
python tools/ros1_recalib/run.py --config correction.yaml
```

Paths in the configuration are relative to that file; command-line paths are
relative to the working directory. CLI flags override the configuration, with
an explicit `--no-...` form for every operation:

```bash
python tools/ros1_recalib/run.py --config correction.yaml --no-remove-bias --no-td
```

| Configuration key | CLI flag | Default | Operation |
| --- | --- | --- | --- |
| `imu_intrinsics` | `--[no-]imu-intrinsics` | true | Apply the calibrated IMU intrinsic model. |
| `remove_bias` | `--[no-]remove-bias` | true | Remove saved static gyro/accel biases. |
| `td` | `--[no-]td` | true | Set header and record times to header time + sensor `td`. |
| `t_ga` | `--[no-]t-ga` | true | Interpolate raw acceleration at gyro time + `t_ga` before other corrections. |
| `undistort` | `--[no-]undistort` | false | Undistort supported mono8 camera images. |

The operations are independent. Disabling intrinsic correction while enabling
bias removal subtracts the saved biases without applying scale, skew or frame
rotation. Disabling `td` preserves original header and bag record timestamps.
The tool does not estimate parameters, transform extrinsics or unwarp rolling
shutter. Acceleration interpolation clamps to the first/last sample at the
recording boundaries; duplicate source header timestamps are rejected when
interpolation is needed.

Input calibration is the MVIS camera–IMU result YAML, including its IMU model
and intrinsic parameters. Summary JSON, camera-only results and Kalibr files
are not accepted. Mapped IMU topics use `sensor_msgs/Imu`. Mapped camera
topics use `sensor_msgs/Image`; `sensor_msgs/CompressedImage` is accepted
when undistortion is off (a td restamp touches only the header). Topics can be inferred from the calibration or supplied
as `mapping` and `base_imu_topic` in the configuration (or a `--map` YAML).

## Output and limitations

- Input data and existing output files are preserved. Export is prepared in a
  temporary directory; bag and provenance are published only after processing
  succeeds, with rollback if publication fails. Publication uses a hard link
  (atomic, never replaces a file); on filesystems without hard links (exFAT,
  FAT32, some network mounts) it falls back to an exclusive-create copy, which
  still never replaces a file but is not atomic.
- Mapped topics must exist and have the expected message types. Unmapped
  messages retain their original serialized bytes and connection definitions,
  including custom message schemas and latching metadata.
- Image undistortion supports `mono8`/`8UC1` (including row padding) with radtan
  or equidistant models. An explicitly requested unsupported encoding/model is
  an error. Camera-info messages are not rewritten; consumers must not treat
  their original distortion parameters as corrected-image calibration.
- The output is globally sorted by record time. The bag is streamed: with `td`
  or `t_ga` on, a first pass reads the mapped topics (the IMU series for `t_ga`,
  the corrected stamps for `td`) and a second pass writes each message once no
  unread message can sort before it. Memory holds the IMU series, one integer
  per mapped message and the messages inside the reordering window (about the
  largest gap between a record time and its corrected stamp), not the bag.
  ROS2, MCAP and direct EuRoC output are outside the initial scope.
- `--json` prints a machine-readable summary. The `.recalib.yaml` sidecar records
  the calibration hash, operation choices, message counts and corrected topics.
