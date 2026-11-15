# Stage 4 — model choices

## Camera model (`distortion_model`)
| Lens | Model | Notes |
|---|---|---|
| Normal / pinhole (FOV < ~100°) | `radtan` | default for pinhole lenses |
| Fisheye (~100–180°) | `kb4` (Kannala–Brandt polynomial, θ from atan2, valid up to the first fold) | default for fisheye; `fisheye` and `kannala_brandt` mean kb4. Check `corner_accounting` and `valid_domain` in the result. Exports to Kalibr as pinhole-equi, which drops rays past 90° (the export warns) |
| Fisheye, every corner under ~85°, equidistant asked for | `equidistant` (same polynomial on x/z, stops at 90°) | the same pixels as kb4 there; name it explicitly |
| Wide fisheye, alternative | `eucm` | prefer over `double_sphere` when an alternative is needed |
| > 180° / catadioptric | `omni_radtan` | expert use; check convergence carefully |
Also available: `double_sphere`, `fov`, `ucm` (unified model: EUCM with β = 1,
one distortion parameter α; Kalibr `omni-none`). There is no `kb8`. Pick one
model per camera; justify any non-default choice in `04_configs/rationale.md`.
Maths and Jacobians: [camera model](https://openmvis.com/concepts/camera-model/).

## Board (`board:`)
- AprilGrid: `board_type: apriltag`, `rows`, `cols` (tags), `tag_size` in **metres**
  (measured on the print, not the PDF), `tag_spacing` = gap / tag size
  (**ratio**, e.g. 0.3), `tag_family`:
  `t36h11` = Kalibr AprilGrid (2-bit border, most printed grids),
  `t36h11b1` = standard apriltag3 (1-bit border).
- Chessboard: `board_type: chessboard`, `rows`/`cols` = **inner corners**,
  `square_size` in metres.
- ArUco boards exist (`aruco`, `aruco_features`) but are not covered by this
  agent yet.
- **Board shape** (`estimate.board_points`): off by default, the board is held
  at its given shape. Turn it on (camera-only and cam-IMU alike) when the board
  is not known to be flat — a print on foam board, a bent panel; the VI-Rig
  rows estimate it in every solve. A `board.heights_file` fixes a measured
  shape instead. Guide: [board setup](https://openmvis.com/guides/board-setup/).

## Corners per tag (`solve.corner_subsample`)
- Camera-only default `two_anti`: two diagonal corners per tag, about 20 %
  faster than `all` with the same fit; `all` keeps every corner. Single
  corners (`top_left` …) shift the principal point by up to 0.8 px — do not use.
- Cam-IMU default `all`; the Fast preset sets `two_anti`.

## IMU
- `estimate.imu.model: calyx`, `estimate.imu.intrinsics: dw_da` (default).
- Noise `solve.sigma_g / sigma_bg / sigma_a / sigma_ba`: from the datasheet
  (continuous-time densities: rad/s/√Hz, rad/s²/√Hz, m/s²/√Hz, m/s³/√Hz) or
  Allan variance; measured at rest the density is usually 2–3× the datasheet.
  Do not copy a Kalibr `imu.yaml` into `sigma_*` unchanged. Defaults are
  generic — say so if used.
- Gyro-only IMU: `input.imus: [{...}, {gyro_only: true}]` by `imu_topics` order.
- `solve.vision_pose: imu` (default): the camera pose in the vision factor is
  propagated from the IMU; `solve.estimate_t_ga: true` (default) estimates the
  base IMU's gyro-accel offset with a 10 ms prior.

## Solver
- `solve.rms_gate_px: 3` (a diverged solve then reports `ok: false`; the code
  default is 0 = off, the template sets 3).
- `input.window_mode: motion`, `segment_length: 80` (auto-picks the most
  exciting 80 s; 60 in the Fast preset).
- Non-base cameras: `camera_input.cameras[i].factor_hz` (10 Hz default, 4 Hz
  in the Fast preset); the base camera keeps its full rate.
- `input.cache_dir: <run>/cache` in BOTH `cam.yaml` and `cam_imu.yaml` (same dir):
  detection cache, and for ROS1 bags a shared per-camera store (`streams/`) so the
  cam-IMU solve reuses the camera-only detections.
