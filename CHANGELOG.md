# Changelog

All notable changes to MVIS are recorded here. Versions follow
[Semantic Versioning](https://semver.org/); 0.x releases may change the config schema.

## [Unreleased]

## [0.4.0] — 2026-11-24

Changes since 0.3.0. A config that ran on 0.3.0 runs unchanged: without `gravity:` or `site:`
the solve keeps 9.81, as before.

### Added

- Cam-IMU report (HTML, text, `cam_imu.json` `gravity_site`) names the gravity used and its site: city,
  latitude, longitude, altitude. `site:` accepts `longitude_deg` (a label; g ignores it). Regression rows that
  keep 9.81 (VI-Rig, ELP, ELP+T265, TUM-VI) now give `gravity: 9.81` plus their site, so g is unchanged.
- Local gravity in camera + IMU solves: `gravity:` (m/s²) or `site: {latitude_deg, altitude_m}`
  (WGS84 normal gravity). The accelerometer scale absorbs g_true / g_used, so the fixed 9.81
  read it 0.23 % high in Shenzhen and up to 0.30 % at the equator. Neither key: 9.81, results
  unchanged. The value used is recorded in the results, the calibration YAML, the report and
  the run database (whose world frame used to say 9.80665). Without either key the run log
  warns that 9.81 is in use (`mvis check` still passes). `doc/local_gravity.md` lists latitude and altitude
  for common recording places.
- Regression families T13 and T14, the first on a ChArUco board: ALG Tech's AVS111C-Y
  global-shutter and AVS115D-Y rolling-shutter stereo modules with their IMU, one
  recording each, camera + IMU in both IMU models, at Shenzhen's gravity (`site:`). They
  are full tier only (`Family.seed_level`), so the fast and mid gates are unchanged; full
  grows from 176 to 180 rows. Kalibr and Basalt have no comparison rows for them.
- Regression family T11 (Looper Insight-9, recorded in Shenzhen) runs at the local gravity
  (`site:`); its 20 baselines were re-blessed. Unlike T13 / T14, where only the accelerometer
  scale moved (by exactly 9.7879 / 9.81), the Looper rig's weak focal / depth constraint lets
  gravity reach the focal lengths too: fx -0.11 %, lever up to +0.4 mm, accel scale x 0.9977-0.9988;
  reprojection unchanged. A 9.81 control on the same code reproduced the old baselines.
- Regression family T12 (Monado MGC, HP Reverb G2) runs at Córdoba's gravity (`site:`; the recording
  place, confirmed by the dataset's first author); its 8 baselines were re-blessed. Accel scale
  x 0.9984-0.9992; focal length <= 0.04 px, lever <= 0.05 mm, td <= 0.006 ms; reprojection unchanged.
- Board and multi-camera cam-IMU runs report failed init checks (`g_norm_ok`, `scale_ok`): a log
  warning, a line in the text report, and an `init_gates` block in `cam_imu.json` (gravity norm, scale,
  pass/fail per check). They judge the seed, not the solve, so `ok` still follows the solve only.

### Fixed

- Gyro-only auxiliary IMUs: the time offset was applied twice (the window was already shifted by the
  seed td), so a 50 ms offset came out as 0.35 ms; it now comes out as 49.85 ms. No example or
  regression config uses `gyro_only`.
- Cam-IMU report: the bias values labelled "mean" are the first state's and now say so; the bias
  evolution plot shows the per-state biases, each axis minus its own mean (in the legend), instead of
  a straight first-to-last line.

## [0.3.0] — 2026-11-22

Changes since 0.2.0. A config that ran on 0.2.0 runs unchanged.

### Added

- ChArUco boards (`board_type: charuco`): chessboard corners named by ArUco markers, so
  partly visible boards are used. `rows`/`cols` count squares; the dictionary
  (`aruco_dict`, or OpenCV's `dictionary`) is required. Corners are refined with
  `cornerSubPix` (`charuco_refine`, on by default), since OpenCV's own are about 0.49 px
  off. All corners are used (`corner_subsample` has nothing to thin). `mvis doctor`
  checks the board block. The full results YAML records the ChArUco layout. Needs
  OpenCV >= 4.7 (the `aruco` extra now asks for it). See `doc/board_charuco.md`.

## [0.2.0] — 2026-11-19

Changes since 0.1.0. Results files gain fields; nothing is removed or renamed, and a
config that ran on 0.1.0 runs unchanged.

### Added

- ROS2 bags are read with `rosbags`, with no ROS install: the bag folder, or one
  `.db3` / `.mcap` file, in camera-only (`source_type: ros2bag`) and cam-IMU
  (`input.format: ros2`) runs, and `mvis inspect` lists their streams. Times are
  the header stamps, as for ROS1. `mvis roundtrip` and `apply-calib` stay
  ROS1-only. The `ros1` extra needs `rosbags>=0.11.6`, which reads ROS2 bags whose
  type hashes are empty (bags written by converters, such as the Hilti-Trimble
  2026 calibration bags).
- Reprojection statistics per camera and overall: mean, median, standard
  deviation, P90, P95 and max, also for the inlier population (corners within
  3 px), in the results (`cam_imu.json`, the camera-only result) and both reports.
- Camera-only runs with `estimate.board_points` on a corner subsample warn that the
  dropped corners keep their prior heights; `corner_subsample: all` estimates every
  corner at about 1.8x the solve time.
- The Kalibr `camchain-imucam` export names the topic each camera was read from
  (`rostopic`), not `/cam<N>/image_raw`.
- Dataset catalog: the Hilti-Trimble SLAM Challenge 2026 calibration (recipe only).
- Examples: WHUVID ZED2 configs (camera-only and cam-IMU, global and rolling shutter).

### Changed

- Reports redesigned, camera-only and cam-IMU alike: a Summary with "Calibration
  parameters by sensor" folds, per-camera tabs, graded cells, aligned numeric
  columns with one precision per unit, a lettered and closed Appendix, a print /
  PDF view, a Board Shape section whenever the board shape was estimated, and IMU
  Residuals per state block (rotation, position, velocity, gyro and accel bias
  steps). The rig table shows reprojection median and P95; the calyx `Dw` shows as
  R · U; a σ that rounds to zero (a pinned parameter) is left out; a sub-µs readout
  shows as global shutter. The "Within 3.0 px" table is gone.
- Reports name the installed version (they said v0.1.0 whatever the version).
- AprilGrid detection skips the upscale pass when the native pass already decodes
  the board, counting only this board's tags.

### Fixed

- A camera-only solve that runs out of iterations, or stops at its damping cap,
  with most of its cost still predicted to go is solved again with diagonal damping
  from its start, as the cam-IMU solve does after a stuck stop
  (`mvis.calibration.lm_rescue`); every other solve is unchanged. The rescue report
  is written to the result as `lm_rescue`, and `mvis verdict` grades a result RED
  when LM is still stuck after the rescue (`lm_rescue.failed`).
- `mvis roundtrip` runs `apply-calib` through the interpreter it runs in, not the
  `mvis` found on `PATH`.

## [0.1.0] — 2026-11-15

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
