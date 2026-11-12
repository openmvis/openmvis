# Changelog

All notable changes to MVIS are recorded here. Versions follow
[Semantic Versioning](https://semver.org/); 0.x releases may change the config schema.

## [0.1.0] — unreleased

First public release of MVIS, the calibration toolbox of the OpenMVIS project,
as the Python package `pymvis` (import and CLI `mvis`). Linux and macOS,
Python 3.11 and 3.12. Everything below describes the toolbox as it ships; there
is no earlier public version to compare against.

### Calibration

- Camera calibration (`mvis cam`, `mvis run` with `mode: cam`): single- and
  multi-camera intrinsics, distortion, extrinsics, camera-camera time offset and
  rolling-shutter readout, on a GTSAM factor graph with analytical Jacobians,
  quality gating, an outlier-driven second pass with a pathology guardrail, and
  a `corner_accounting` block in every result (corners used, past the fold,
  otherwise invalid, frames without a pose, outliers, largest incidence angle).
- Camera-IMU calibration (`mvis cam-imu`, `mode: cam-imu`): joint discrete solve
  for camera-IMU extrinsics, time offset, readout, camera intrinsics, IMU
  intrinsics (`kalibr`, `rpng` or `calyx` model; `Dw`/`Da`, optional `Tg`),
  biases, the base IMU's gyro-accel time offset `t_ga`, and aux-IMU extrinsics and
  time offsets, for single or multi cam/IMU rigs. Per-IMU `gyro_only` mode for an
  IMU with an unreliable accelerometer. `input.bags` calibrates a whole dataset
  family from one config.
- Eight camera models, usable in both modes and mixed within one rig: `radtan`,
  `equidistant`, `kb4`, `fov`, `double_sphere`, `eucm`, `ucm` and `omni_radtan`
  (experimental). `kb4` (Kannala-Brandt, theta from atan2) is valid past 90 deg up
  to its first fold and is what `fisheye` and `kannala_brandt` mean; name
  `equidistant` for the gtsam model that stops at 90 deg. `kb4` exports to Kalibr
  as `pinhole` + `equidistant` (same polynomial; the export warns when the fit used
  corners past 90 deg). `equidistant` does not project points at z <= 0 and warns
  to switch to `kb4` when a fit has such corners.
- `estimate:` block (see `doc/estimate_config.md`) to choose what is solved and
  what is pinned, including the board shape (`board_points`), on both modes. With
  `board_points` off the board is held at its given shape.
- Defaults: camera-only `corner_subsample: two_anti` (two diagonal corners per
  AprilTag; one corner moves the principal point by up to 0.8 px), cam-IMU `all`;
  `estimate_t_ga: true` in cam-IMU runs; the `mvis` CLI pins BLAS to one thread
  (`doc/blas_threads.md`).

### Input and output

- Inputs: image folders, timestamped images, ROS1 bags (`sensor_msgs/Image` and
  PNG or JPEG `sensor_msgs/CompressedImage`), ROS2 bags and EuRoC / ASL folders.
  Every reader decodes frames through one rule (`mvis.io.image_decode`): mono16
  is shifted or min-max stretched, colour becomes grey with OpenCV luma, image
  folders keep the sensor's stored orientation, and an unknown encoding or a frame
  of the wrong size stops with an error.
- Boards: AprilGrid (`aprilgrid`), chessboard, ArUco.
- Kalibr interop (`mvis.io.kalibr`): read and write `camchain` and
  `camchain-imucam`, load AprilGrid targets from Kalibr YAML.
- Output: YAML calibration files, JSON results and self-contained HTML reports for
  both modes (`doc/cam_imu_report.md`), optional Rerun 3D view
  (`scripts/visualize_result.py`).
- `mvis datasets`: catalog, download and verification of the Calibench datasets
  on Hugging Face (VI-Rig first, then TUM-VI and the converted third-party sets).

### Tools

- `mvis apply-calib`: apply a cam-IMU calibration back onto a raw ROS1 bag
  (IMU intrinsics and bias removed, streams re-stamped by `td` on the header
  clock, accel shifted by `t_ga`, optional undistortion), streaming so memory does
  not grow with the bag; `--no-imu-intrinsics`, `--no-td`, `--no-t-ga`,
  `--keep-bias`, `--undistort` and `--strict` switch each step. The same
  implementation (`mvis.recalib.apply.apply_calibration`) drives
  `scripts/recalib_batch.py` and the stand-alone `tools/ros1_recalib`.
- `mvis roundtrip`: calibrate, apply, recalibrate and grade the collapse of the
  IMU intrinsics and biases with a PASS/FAIL report.
- `mvis check` (environment, smoke solve, config lint), `mvis inspect` (dataset
  summary) and `mvis verdict` (grade a finished result), all with `--json`.
- Simulator (`mvis.simulation`) with digital twins of real recordings and the
  simulator examples in `data/sim_examples/` (`mvis.simulation.examples`,
  `load_example` / `simulate`); the wheel ships `twin_gs`.

### Licence

GPL-3.0-only, with a commercial licence for closed-source products
(`COMMERCIAL-LICENSE.md`).
