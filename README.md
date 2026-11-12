# MVIS

A Python toolbox for **camera** and **camera-IMU** calibration of single- and
multi-sensor rigs, built on GTSAM factor graphs with native SO(3)/SE(3) Lie
group support. OpenMVIS is the project, MVIS is the calibration toolbox, and
`pymvis` is its Python package (import and CLI `mvis`).
Website: [openmvis.com](https://openmvis.com) · datasets:
[Calibench](https://huggingface.co/calibench-vi) · results on real recordings:
[openmvis.com](https://openmvis.com).

## Features
- **Single- & multi-camera calibration** with joint optimization of
  intrinsics, distortion, extrinsics, time offsets, rolling-shutter
  readout, and board poses.
- **Camera models**: 8 distortion models — pinhole with radial-tangential,
  `kb4` (Kannala-Brandt polynomial with theta from atan2, valid up to its first fold, past
  90 deg; what `fisheye` and `kannala_brandt` mean; exports to Kalibr as pinhole-equi),
  equidistant (the same polynomial on x/z, gtsam Cal3Fisheye, stops at 90 deg; the same
  pixels as kb4 when every corner is under 90 deg), FOV (Devernay),
  Double Sphere (Usenko/Basalt), EUCM (Khomutenko), UCM (unified, alpha form;
  Kalibr omni-none), and omni_radtan (MEI +
  radtan with all of Kalibr's xi, k1, k2, p1, p2; experimental). Custom analytical Jacobians for non-GTSAM models. The per-model
  math (projection + Jacobians + unprojection) lives in one place
  (`camera/CameraModel`), so the same models are available in **both camera-only
  and cam-IMU calibration** (board + feature mode), including mixed-model
  multi-camera rigs. The
  cam-IMU path is validated on the TUM-VI raw IMU bags — omni_radtan and
  equidistant reproduce the Kalibr cam-IMU extrinsics (lever ≤2 mm, stereo
  baseline ≤0.15 mm, time offset ≤1 ms); recipe in
  `examples/configs/tumvi_cam_imu/`.
- **Camera-IMU calibration**: joint discrete solve for cam-IMU extrinsics
  (`T_CtoI`), time offset (`td`), rolling-shutter readout, IMU intrinsics, and
  biases, for single or multi cam/IMU rigs and all distortion models. Pose /
  velocity / bias state nodes are placed at a configurable **`state_hz`**
  (default 30 Hz).
- **Batch cam-IMU sweep** (`input.bags`): one cam-IMU config lists many sequences
  (`input.bags: [bag1, bag2, …]` — ROS1 bags *or* EuRoC `mav0` dirs, each inheriting
  the config's format) and `calibrate_cam_imu_batch` calibrates each into a
  discoverable per-bag `bag<NN>/cam_imu.json` — a whole dataset family from **one**
  config, no N hand-written files. This is the basis for the repeatability study
  (spread of extrinsics / intrinsics across sequences).
- **IMU intrinsics**: three selectable models (`imu_intrinsics_model:
  kalibr | rpng | calyx`) for gyro/accel scale + misalignment (`Dw`/`Da`),
  optional g-sensitivity (`Tg`), and biases. Calyx parameterizes the gyro
  correction as a full 3×3 matrix (post-hoc decomposed into rotation ×
  upper-triangular). On real data (TUM-VI + MIC, cam-intrinsics pinned or freed)
  all three recover identical extrinsics/intrinsics — the model is a convention,
  not a quality lever.
- **Per-IMU gyro-only mode** (`input.imus: [{gyro_only: true}]`): calibrate a chosen
  **aux** IMU from its **gyro alone** (extrinsic rotation + `td` + `Dw` + gyro bias;
  lever / `Da` / `Tg` pinned), skipping an unreliable or interpolated accelerometer.
  Built for the **T265 (Bosch BMI055)**, whose 62.5 Hz accel interpolated to 200 Hz
  injects a gyro-accel timing offset that corrupts the T265's `td` — gyro-only makes
  the t265 `td` collapse in a correct→recalibrate round-trip (+8.6→−9.0 ms becomes
  +0.9→+0.4 ms) and halves the residual lever drift. Not allowed on the base IMU (which
  needs the accel for gravity + metric scale). The calib self-documents via a
  `gyro_only: true` flag per IMU.
- **Gyro↔accel time offset `t_ga`** (`solve.estimate_t_ga: true`): estimate the
  base IMU's *internal* accel-vs-gyro timing offset **inside the ACI factor** —
  the combined IMU stamp is taken as the gyro clock, and the accel stored at
  stamp `t` is modeled as physically sampled at `t − t_ga`. Implemented as one
  extra Jacobian column in the preintegration (`∂a/∂t_ga = ȧ` from per-step
  finite differences, coupled into the gyro row through the g-sensitivity `Tg`)
  plus an optional 6th scalar factor key with a staged prior (`sigma_t_ga`) —
  the 5-key factor is untouched when off. Sim: injected {−2, +1, +5} ms recover
  with |z| ≤ 1.3 at σ = 0.024 ms, no td/Dw/Da cross-contamination. Real data:
  the **T265 BMI055 measures t_ga ≈ −6.2 ms, stable to ~0.2 ms across bags**
  (≈16σ) — directly quantifying the driver's accel-interpolation offset that
  gyro-only mode works around. Reported as `t_ga` in the result JSON/calib
  YAML/text report; `mvis apply-calib` **corrects** it by re-interpolating
  the accel channel at `t + t_ga` (header clock, before the inverse intrinsic
  model), and `roundtrip` checks `t_ga → 0` (t265: −6.32 → +0.53 ms, cam `td`
  −7.6 → −0.37 ms, relative fe2−fe1 td 2.69 → 0.006 ms).
- **Calibration-corrected dataset export** (`mvis apply-calib`): apply a
  saved cam-IMU calibration back onto a raw ROS1 bag and re-record it
  *calibration-corrected* — IMU intrinsics (`Dw`/`Da`/`Tg`) + static bias removed
  via the inverse model, all camera/aux-IMU streams re-stamped onto a single
  base-IMU timeline by each sensor's `td` (**on the header clock** — record-time
  re-stamping would bake each topic's differential receive latency into the
  corrected bag), the base-IMU accel channel time-shifted by `t_ga` when
  calibrated, optional image undistort, plus a
  provenance sidecar; a batch runner applies one calib across a dataset family.
  The calib records each sensor's **source topic**, so the tool **auto-derives the
  topic→sensor mapping** — no hand-written map. Run it config-driven like the
  calibrator (`mvis apply-calib -c recalib.yaml` bundling `calib`/`input`/
  `output`) or by flags (`--calib … --in … [--out …]`; output defaults to
  `<bag>.corrected.bag`). Each correction can be switched off (`--keep-bias`,
  `--no-imu-intrinsics` for bias only, `--no-td`, `--no-t-ga`; `--undistort` turns
  pixel undistortion on), and `--strict` rejects bad input instead of passing it
  through. The bag is streamed, so memory does not grow with the image data.
  Validated on TUM MIC05 + TUM-VI raw (bit-exact correction; the corrected bag
  recalibrates to near-identity IMU intrinsics). See
  `src/mvis/recalib/` and `examples/configs/recalib_example.yaml`.
- **Round-trip validation** (`mvis roundtrip -c <cam_imu_config>`): one command runs
  calibrate → `apply-calib` → recalibrate → a **PASS/FAIL report**. A correct calibration
  must, on its own corrected bag, drive every IMU `Dw`/`Da` → identity and `b_g`/`b_a` → 0
  (**collapse**) while leaving the camera/IMU extrinsics + **stereo baseline** unchanged
  (**preserve**); the residual `td` is reported as a *warning* (observability-dependent, not a
  failure). The corrected bag gets a distinct name (fresh detection cache) and is auto-deleted
  (`--keep-corrected` to keep); exits non-zero on FAIL (CI-usable). This is the automatic
  version of the model-mismatch round-trip check. Confirmed on 3cams (all 4 IMUs' `Da` →
  identity, baseline 69.86 mm stable) and Monado/MIC05 (`Da` 1.06 → 1.00, `b_a` −0.25 → −0.01,
  baseline 133.8 mm stable).
- **Board detection**: chessboard, AprilGrid (Kalibr 2-bit border via
  `aprilgrid`), ArUco markers.
- **Input sources**: image directories, timestamped images,
  ROS1/ROS2 bags, EuRoC / ASL format (used by Monado SLAM / TUM-VI MIC).
- **Kalibr interop** (`mvis.io.kalibr`): one surface to read/write Kalibr
  `camchain` / `camchain-imucam` — `read_camchain` (intrinsics + distortion across
  pinhole/ds/eucm/fov/omni, omni-none = ucm), `read_camchain_imucam` (also recovers each camera's
  `T_cam_imu` + stereo `T_cn_cnm1`), and `write_camchain_imucam` to export a cam-IMU
  result as a round-trippable Kalibr camchain-imucam. AprilGrid **targets** load via
  `apriltag_board.from_kalibr_yaml`.
- **GTSAM optimizer**: factor graph with Levenberg-Marquardt and Huber
  robust loss. A single unified factor handles pose interpolation (SLERP),
  extrinsic composition, time offset, rolling shutter, and projection in
  one place — the same projection code is also used by the HTML report,
  so residuals are guaranteed consistent.
- **Outlier-driven two-pass calibration** with pathology guardrail:
  pass 2 re-runs on filtered observations, and automatically reverts to
  pass 1 if the optimizer slid into a pathological basin (non-positive
  focal, blown-out distortion, or P95 regression).
- **Corner subsampling** (AprilGrid, camera-only default `two_anti`): keep 2
  diagonal corners per tag, about 20% faster than `all` with the same fit and
  principal point within about 0.1 px. `all` keeps every corner. Single corners
  (`top_left` and others) are fastest, but they shift `cx` and `cy` by up to
  about 0.8 px in a direction set by the corner, and reprojection error doesn't
  show it.
- **Multi-scale adaptive AprilTag detection**: probe scale=1.0, then
  pick up to 2 adaptive extra scales based on observed module size
  (upscales for far/small tags, downscales for huge tags). Subpixel
  refinement on the original-resolution image. ~1.4 detection passes
  on average; catches small tags missed by single-scale detection.
- **Detection-quality gating**: pooled-pose filter (drops ref frames
  whose `B(i)` would be poorly constrained across cameras) and
  per-corner observability filter (drops corners with <5 views pooled
  across cameras). Both run between Stage 1 and Stage 2.
- **Sub-stage timing**: hierarchical Timer with metadata, serialized
  to a `timings:` block in every calibration YAML. Lets you ask
  questions like "is `top_left` 4× faster overall, or only 4× faster
  in Stage 4?".
- **Professional HTML reports** — self-contained single-file reports with:
  - Navigable TOC sidebar (auto-highlights active section, responsive)
  - Summary stat cards, intrinsics/distortion tables, extrinsics + 3D plot
  - Temporal calibration (time offset + rolling shutter, merged section)
  - Reprojection error analysis: per-frame robust (median+P95), per-tag,
    error vector histograms, dual-panel scatter, corner heatmaps
  - Calibration data quality: motion excitation metrics with assessment
    grades, rotation axis spread, angular velocity timeline, feature
    coverage heatmap with convex hull
  - Detection statistics: per-tag bar charts, tag tracking history,
    sample images
  - Pipeline timing breakdown (stacked bar + per-stage table)
  - Collapsible appendices (lowest-50 detection frames, per-tag heatmap)
  - Print-friendly CSS (hides nav, clean layout)
  - All matplotlib plots as base64 PNG, 3D plots as inline Plotly JS
- **Rerun 3D visualization** — interactive viewer showing board points,
  camera trajectory, multi-camera rig with extrinsics, and per-camera
  detection thumbnails. Set `rerun_visualization: true` in the config (needs the
  `viz` extra) or view a finished run post-hoc:
  `python scripts/visualize_result.py <result dir>`
- **Pure Python**: installs with `pip`, no C++ build or ROS workspace. Linux and
  macOS (x86-64 / arm64), Python 3.11 and 3.12 — GTSAM publishes no Windows
  wheels and none below cp311, and `numpy<2` has no 3.13 wheel, so those are out
  of reach until upstream ships them.

## Installation

From source (Python 3.11 or 3.12, Linux or macOS):

```bash
git clone https://github.com/openmvis/openmvis && cd openmvis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[apriltag,aruco,ros1,dev]"
mvis check --env --smoke                 # verify the install
```

`pip install pymvis` (and `pip install "pymvis[apriltag,ros1]"` for AprilGrid
boards + ROS1 bag input) becomes available with the v0.1.0 release on PyPI.
The import and the CLI are `mvis`.

Optional extras: `apriltag` (AprilGrid), `aruco` (opencv-contrib, pinned
`<4.12` like opencv-python), `ros1` (ROS1 bags, no ROS needed), `mcap` (MCAP
episodes; also installs PyAV, which the PennCOSYVIO converter needs), `viz`
(Rerun viewer), `dev` (pytest).

**Scope.** MVIS is research code, maintained on a best-effort basis. Linux and macOS
are supported; the ROS1 recalibration tool in `tools/ros1_recalib` is provided as is.
MVIS builds on GTSAM, NumPy, SciPy and OpenCV. Kalibr and Basalt are only called as
external programs for comparisons; neither is distributed with MVIS.

> **GTSAM + NumPy**: GTSAM 4.2 (pip) is built against the NumPy 1.x ABI, and
> NumPy 2 segfaults when constructing `gtsam.Pose3(Rot3, np.array)`. The
> `numpy<2` ceiling is pinned in `pyproject.toml`, so a plain install is safe;
> only override it if you know GTSAM has a NumPy-2 build. OpenCV 4.12+ wheels
> require `numpy>=2`, so pymvis pins `opencv-python` (and `opencv-contrib-python`
> for the `aruco` extra) to `<4.12`; `cv2` stays at 4.11.x.

## Quick start

### 1. Get a calibration dataset

**Start with VI-Rig**: multi-camera, multi-IMU calibration recordings (pinhole +
fisheye, global and rolling shutter, AprilGrid and ArUco) made for calibration
research, on Hugging Face at
[calibench-vi/datasets](https://huggingface.co/datasets/calibench-vi/datasets).
They are gated: accept the terms on the Hub, then `hf auth login` (or
`huggingface-cli login` on older clients, or set `HF_TOKEN`) before downloading.

```bash
mvis datasets list --dataset mvis_4imus_3cams
mvis datasets download --select mvis_4imus_3cams/april_01    # ~7 GB
```

Sources the catalog has no sha256 for (the PennCOSYVIO `c1`, `c2` and `visensor`
archives, the ImprovedOcamCalib zip) are skipped unless you pass
`--allow-unverified`.

Datasets download to `$MVIS_DATASETS` (default `~/Datasets`); every example
config refers to them as `${MVIS_DATASETS}/...`. **TUM-VI** is the second
reference: `examples/configs/tumvi_cam_imu/` reproduces the published Kalibr
cam-IMU calibration (see its README).
The VI-Rig data (`virig-*`) are **CC BY-NC 4.0** with gated access (accept the
terms on the Hub); commercial licences on request (licensing@openmvis.com). If you use them,
please cite the MVIS paper:

```bibtex
@article{Yang2024MVIS,
  author  = {Yang, Yulin and Geneva, Patrick and Huang, Guoquan},
  title   = {Multi-visual-inertial system: Analysis, calibration, and estimation},
  journal = {The International Journal of Robotics Research},
  year    = {2024},
  volume  = {43},
  number  = {13},
  pages   = {1995--2026},
  doi     = {10.1177/02783649241245726}
}
```

### 2. Write a config

Camera-only example (`examples/configs/example_*.yaml` has more):
```yaml
mode: cam                       # cam (camera-only) or cam-imu
board:
  board_type: apriltag          # or chessboard
  rows: 6
  cols: 6
  tag_size: 0.088               # meters (chessboard: use square_size instead)
  tag_spacing: 0.3
  tag_family: t36h11

cameras:
  - camera_id: 0
    source_type: images         # or rosbag_topic, euroc (video: extract frames first)
    source_path: /path/to/images
    distortion_model: radtan    # or equidistant, double_sphere, eucm, ucm, fov, omni_radtan, kb4
    shutter_type: global        # or rolling (enables RS readout calibration)

optimizer:
  max_iterations: 15
  pixel_sigma: 1.0
  huber_threshold: 1.345

# Defaults (can be omitted — shown for reference):
corner_subsample: two_anti    # two_anti | two_diag | all | top_left (sectioned configs may
                              # also write solve.corner_subsample; this key wins)
rerun_outliers: true          # outlier-driven rerun after joint opt
optimize_time_offset: false   # true for unsynchronized cameras
rerun_visualization: false    # true launches the Rerun viewer (`viz` extra; off in batch)

output_file: calibration_result.yaml
```

Camera-IMU configs (`mode: cam-imu`) for VI-Rig are in `examples/configs/tga_mvis/`
and `examples/configs/tga_t265*/`, for TUM-VI in `examples/configs/tumvi_cam_imu/`.

### 3. Run
```bash
mvis check -c config.yaml       # lint the config first
mvis run -c config.yaml
```

The `mvis` command pins the BLAS thread count to **1** before numpy loads: the
solvers are per-factor Python on small matrices, so extra OpenBLAS threads only
spin (~590 s of `sys` time per cam-IMU solve on a 32-core host, no wall-time gain)
and make results depend on the thread count. `MVIS_BLAS_THREADS=<n>` changes it,
`0` keeps the BLAS default, an exported `OPENBLAS/OMP/MKL_NUM_THREADS` is never
overridden, and `import mvis` as a library leaves the environment alone — see
`doc/blas_threads.md`.

## Supported Input Formats

MVIS supports six input formats. Detection is streaming (1 image at a
time) so memory usage is bounded regardless of dataset size.

### Image directory (`source_type: images`)

A folder of image files sorted by filename. For stereo, one folder per camera.

```yaml
cameras:
  - camera_id: 0
    source_type: images
    source_path: /path/to/cam0_images
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`.

### ROS1 bag (`source_type: rosbag_topic`)

ROS1 bag file with image topics. Multiple cameras from one bag file. Topics can be
`sensor_msgs/Image` or `sensor_msgs/CompressedImage` (PNG or JPEG). A mono8 PNG topic
gives the same pixels as raw; JPEG is lossy and can shift corners, so MVIS warns once
per topic. `compressedDepth` and video codecs are rejected with an error.

```yaml
cameras:
  - camera_id: 0
    source_type: rosbag_topic
    source_path: /path/to/data.bag
    topic: /cam0/image_raw
  - camera_id: 1
    source_type: rosbag_topic
    source_path: /path/to/data.bag
    topic: /cam1/image_raw
```

Requires `pip install rosbags` (pure Python, no ROS installation).

### EuRoC / ASL format (`source_type: euroc`)

Standard visual-inertial dataset format (TUM-VI, EuRoC MAV, Monado SLAM).

```
cam0/
  data.csv          # "#timestamp [ns],filename"
  data/
    1403715273262142976.png
    ...
```

```yaml
cameras:
  - camera_id: 0
    source_type: euroc
    source_path: /path/to/cam0
```

### Other formats

- **Timestamped images** (`source_type: timestamped_images`): image directory + CSV with `filename,timestamp` rows.
- **ROS2 bag** (`source_type: ros2bag`): ROS2 SQLite-based bag directory; same image message types as ROS1.

### Calibration boards

| Board | Config | Required fields |
|-------|--------|-----------------|
| AprilTag grid | `board_type: apriltag` | `rows`, `cols`, `tag_size` (m), `tag_spacing` (ratio), `tag_family` |
| Chessboard | `board_type: chessboard` | `rows`, `cols` (inner corners), `square_size` (m) |
| ArUco | `board_type: aruco` | `rows`, `cols`, `marker_size`, `marker_separation`, `aruco_dict` |

## Architecture

```
src/mvis/
  core/         Types (Observation, CameraParams, CalibrationResult), config, timing
  camera/       Camera models (one CameraModel class per model: RadTan, Equidistant,
                FOV, DS, EUCM, UCM, omni_radtan, kb4) — single home for projection/Jacobian/
                unprojection math, shared by camera-only and cam-IMU calibration
  detection/    Boards (chessboard, AprilGrid, ArUco), detectors, frame quality,
                corner_subsample.py
  io/           DataLoader (unified), DetectionPipeline (streaming), data sources,
                detection cache, image decoding, cam-IMU loader, Kalibr/Basalt interop
  imu/          IMU intrinsics models (kalibr, rpng, calyx), resampling, Allan analysis
  calibration/  Initializer (OpenCV PnP), optimizer, camera-only and cam-IMU pipelines,
                outlier_rejection.py, factors/ (unified camera factor, IMU factors,
                camera_projection.py, custom_projections.py: analytical Jacobians)
  simulation/   Synthetic cam-IMU data, digital twins, simulator examples
  recalib/      apply-calib and roundtrip (apply a calibration back onto a bag)
  datasets/     Calibench catalog, download and conversion (`mvis datasets`)
  doctor/       `mvis check`, `mvis inspect`, `mvis verdict`
  mvisdb/       Capture database (detections, estimates, exports)
  analysis/     HTML reports, plotting, Rerun 3D viz, motion analysis
  cli/          Command-line interface (`mvis`), BLAS thread pin
```

### Calibration pipeline

```
Detection → Quality gating → Init → Per-cam opt (+ RS) → Alignment → Joint opt → Outlier rerun
```

| Stage | What it does |
|---|---|
| **Detection** | Adaptive multi-scale AprilTag (auto max_scale by image size, FoV clipping guard, regional crop for far boards) |
| **Quality gating** | Corner subsample (two_anti, about 20% faster than all), pooled-pose filter (≥20% tags across cams), per-corner ≥5 views, per-camera corner ≥3 views, min 6 tags/frame, auto-fallback on sparse data |
| 1 | Per-camera initialization: vanishing-point focal length + RANSAC PnP board poses (reprojection error gate) + intrinsic sanity check |
| 2 | Per-camera GTSAM refinement (intrinsics + distortion + poses + RS readout if `shutter_type: rolling`) with Huber robust loss |
| 3 | Trajectory alignment: extrinsics (RANSAC) + time offsets (dual PnP + M5 homography cross-correlation) |
| 4 | Joint multi-camera factor graph — unified factor with SLERP interpolation, all parameters |
| 5 | (optional) Outlier-driven rerun + pathology guardrail (reverts to pass-1 on fx<0 etc.) |

### Engineering techniques for robustness

| Technique | What it prevents |
|---|---|
| Adaptive multi-scale detection + FoV guard | Missed tags at extreme distances; wasted upscales on clipped boards |
| Median quad side (not max) | Oblique tags triggering wrong scale decisions |
| Corner subsample with auto-fallback | 3× speedup safely — auto-reverts to `all` on sparse data |
| Min 6 tags/frame filter | Degenerate PnP poses from sparse tag detections |
| Per-camera corner observability (≥3 views) | Under-constrained corners in per-camera Stage 2 graphs |
| RANSAC PnP + reprojection error gate | Outlier tag detections producing bad initial poses |
| Stage 1 intrinsic sanity check | OpenCV init divergence on degenerate geometry |
| Vanishing-point focal init | OpenCV init failure on wide fisheye |
| Huber robust loss (1.345) | Outlier observations corrupting the fit |
| SLERP pose interpolation | Quantization error from nearest-frame matching |
| Dual dt estimation (PnP + M5 homography) | Cross-FOV cameras where single-method dt fails |
| Weak anchor priors on B variables | GTSAM elimination failure on disconnected pose nodes |
| Two-stage outlier rejection | Bad frames (motion blur) + bad individual observations |
| Pass-2 pathology guardrail | Optimizer divergence after warm-started rerun |
| Shared projection model | Factor residuals matching report error-analysis exactly |
| Dual YAML output | `_calib.yaml` (deploy) + `_full.yaml` (replay with board poses) |
### Key technical decisions

| Decision | Choice | Why |
|---|---|---|
| Language | Pure Python | `pip install`, cross-platform, GTSAM has full Python API |
| Optimizer | GTSAM | Native SO(3)/SE(3), factor graph, built-in camera models |
| Rotation | SO(3) via `gtsam.Rot3` | 3-param tangent space, no unit-norm constraint |
| Extrinsics | SE(3) via `gtsam.Pose3` | Unified 6-DoF rigid transform |
| Factor design | One unified factor / observation | Consistent projection between optimizer and report |
| Error metric | Median + P95 (not RMS) | Robust under long-tailed residual distributions |
| Detection | Adaptive multi-scale + FoV guard | Maximize tag recovery across all board distances |
| Analytical Jacobians (SLERP chain rule) | 32 finite-difference evals → 1 forward + chain rule | ~4× speedup, correct RS Jacobians |
| kb4 for fisheye, chosen by incidence angle | equidistant when every corner is under 90° (same pixels); DS/EUCM/FOV available; **DS now paper-validated** via two-phase optimization (Usenko 2018 within ≤2 % per lens, see `tests/camera/test_double_sphere_paper.py`) | DS originally had fx-xi equivalence-class drift; Phase-1 freeze of (xi, α) at canonical init anchors the optimizer in the right basin |
| Unclamped SLERP λ | Dead zone at bracket edges killing td/tr gradients | Enables proper RS readout calibration |
| RANSAC rotation retry | Low inlier rate (<70%) on cross-model cameras | Auto-retry with tighter threshold |
| Diverged pose detect-remove-reopt | Fisheye poses diverging in GTSAM (partial board at FOV edge) | Z-depth check + quality gate + graph rebuild |
| Board pose sanity check | GTSAM pushing weakly-constrained poses to infinity | Catch and exclude in output |
| Pass-2 guardrail | Revert to pass 1 on pathology | Avoids silently shipping a diverged calibration |
| Stage 2 input pre-filter | Score frames against Stage 1's OpenCV calibration; drop high-residual ones (MAD-based) before GTSAM optimization | Prevents wrong-basin convergence in joint multi-cam optimization. Found 2026-05-02: time-offset ref-cam boost + diversity-filter skip can feed Stage 2 a frame set with ~13% redundant/degenerate frames that pull intrinsics off by 18 % (virig_flir_t265 bag1 fx 508→454). Radtan-only; other models pass through. |

## Testing
```bash
scripts/test.sh unit          # fast tests, while coding
scripts/test.sh full          # everything (about 20 min on one core)
```

More than 3,000 tests covering types, config, camera models, board detection,
I/O, initialization, single- and multi-camera optimization, **cam-IMU
calibration**, outlier rejection, the simulator, and the reports; the tiers are
in `doc/testing.md`. Tests that need real recordings skip when the data is
absent; point `MVIS_DATASETS` at your datasets to run them.

The maintainers also run a regression suite over real recordings (rows T1–T12
spanning global/rolling shutter, pinhole/fisheye, mono/stereo/multi-camera and
1–4 IMUs). Its results are published on [openmvis.com](https://openmvis.com).

## Comparison with Kalibr and Basalt

Based on analysis of [Kalibr](https://github.com/ethz-asl/kalibr) (Furgale
et al. 2013/2015) and [Basalt](https://gitlab.com/VladyslavUsenko/basalt)
(Usenko et al. 2018/2020) source code, documentation, and papers.

### Feature Comparison

| Aspect | MVIS | Kalibr | Basalt |
|---|---|---|---|
| **Install** | `pip install` (no ROS) | C++ build + ROS workspace | C++ build (no ROS) |
| **Platforms** | Linux / macOS | Linux (ROS only) | Linux / macOS |
| **Language** | Python + GTSAM | C++ / Python | C++ (headers-only core) |
| **Optimizer** | GTSAM factor graph (LM) | Custom `aslam_optimizer` | Custom LM + Gauss-Newton |
| **Lie groups** | Native SO(3)/SE(3) via GTSAM | Manual parameterization | Custom Lie group exp/log |
| **Trajectory** | Per-frame pose + SLERP | B-spline (order 6, 70 knots/s) | B-spline SE(3) (order 5) |
| **Inputs** | Images, ROS1/2 bags, EuRoC (video: extract frames first) | ROS1 bags only | ROS bags, EuRoC |
| **Boards** | AprilGrid, ArUco, Chessboard | AprilGrid, Checker, Circle | AprilGrid only |
| **Camera models** | RadTan, Equidistant, `kb4` (theta from atan2), FOV, DS, EUCM, UCM, omni_radtan (MEI) | RadTan, Equi, FOV, Omni, OmniRadtan, EUCM, DS | DS, EUCM, UCM, Pinhole, KB4 |
| **RS calibration** | Single-cam + joint multi-cam readout | Single-cam only (tested 2026-04-30: best config gives readout +45 % vs MVIS; default config gives non-physical readout) | Not supported |
| **RS detection** | Config-based (`shutter_type`) | Manual flag | N/A |
| **Time offset (cam-cam)** | Supported | Not supported | Not supported |
| **Time offset (cam-IMU)** | Supported (td; ≤1 ms vs Kalibr on TUM-VI) | Supported | Supported |
| **IMU calibration** | Cam-IMU extrinsics + td + RS readout + IMU intrinsics (Dw/Da scale/misalignment) + biases | Cam-IMU extrinsics + temporal | Biases, scale, gravity, td |
| **Multi-camera** | N cameras (no FOV overlap needed) | Chain (FOV overlap required) | Stereo only |
| **Extrinsic model** | Per-camera T_cam_ref (independent) | Chained T_cn_cnm1 | T_i_c (IMU-to-camera) |
| **Outlier rejection** | Two-pass: P99 + warm-start rerun | Single-pass Blake-Zisserman | Single-pass Huber |
| **Output** | YAML + self-contained HTML report | YAML + TXT + PDF | JSON (cereal) |
| **Error metrics** | Median + P95 | Mean/std of squared error | Reprojection error |
| **GUI** | HTML report + optional Rerun | ROS RViz | Pangolin (OpenGL) |

### MVIS Advantages

- **No ROS / C++ dependency** — `pip install`, works everywhere
- **RS calibration** — single-camera and multi-camera readout optimization
  jointly with all parameters (Kalibr RS is single-cam only; Basalt
  doesn't support RS at all)
- **Camera-camera time offsets** — calibrate unsynchronized cameras
  with cross-correlation dt estimation and joint optimization
  (both Kalibr and Basalt assume hardware sync between cameras)
- **No FOV overlap required** — cameras can point different directions
  (Kalibr requires chained FOV overlap; Basalt is stereo only)
- **Two-pass outlier rejection** — Stage 4 optimize → Stage 5 filter +
  warm-start rerun (both Kalibr and Basalt use single-pass robust kernels)
- **Single-camera RS calibration** — works with just one RS camera
  (validated: ELP single-cam RS=14.93ms, matches stereo baseline)
- **Professional HTML reports** — self-contained with TOC, interactive 3D,
  data quality analysis (Kalibr: PDF; Basalt: JSON only)
- **Kalibr comparison** — on the maintainers' regression recordings (sweep of
  2026-04-30) Kalibr fails on RS cameras (single + multi), mixed-model+RS+dt
  rigs, cam-cam dt without IMU and small calibration tags, where MVIS converges.
  Kalibr supports every MVIS model; kb4 exports as pinhole-equi, exact below 90°.
  See `doc/comparison_kalibr.md` and `doc/comparison_basalt.md` (dated
  snapshots; current results are on [openmvis.com](https://openmvis.com)).

### Kalibr Advantages

- **Continuous-time B-spline IMU** — handles arbitrary motion without discrete
  poses (MVIS's cam-IMU is discrete per-frame states + SLERP
  interpolation, validated on TUM-VI)
- **`omni-none` catadioptric model** — pure omni without radial-tangential
  distortion (MVIS only ships `omni_radtan`/MEI; experimental)
- **Mature ecosystem** — widely used since 2013, hundreds of citations

### Basalt Advantages

- **C++ performance** — fastest optimization iterations
- **Double Sphere model** — closed-form projection AND unprojection, fastest Jacobians
- **IMU calibration** — biases, scale/misalignment matrix, gravity estimation
- **Staged optimization** — gradually enables temporal offset and IMU scale
  to prevent divergence
- **SLAM integration** — calibration feeds directly into real-time VIO

## Documentation

- `doc/camera_models.html` — camera models: projection, unprojection, Jacobians
- `doc/estimate_config.md` — the `estimate:` config block (what is solved, what is pinned)
- `doc/cam_imu_base_camera.md` — choosing the base camera in cam-IMU calibration
- `doc/blas_threads.md` — why the CLI pins one BLAS thread, `MVIS_BLAS_THREADS`, library use
- `doc/cam_imu_report.md` — reading the cam-IMU HTML report
- `doc/testing.md` — the test tiers (`scripts/test.sh`)
- `doc/comparison_kalibr.md` — numerical comparison with Kalibr (snapshot of 2026-04-30)
- `doc/comparison_kalibr_regression.md` — Kalibr on the maintainers' regression recordings (snapshot of 2026-05-01)
- `doc/comparison_basalt.md` — Basalt architecture and comparison (snapshot of 2026-04)
- `examples/configs/tumvi_cam_imu/README.md` — TUM-VI cam-IMU recipe
- `data/sim_examples/README.md` — simulator examples: real motions with a full rig setup (`mvis.simulation.examples`)

## Citation

If you use MVIS, please cite the MVIS paper (see `CITATION.cff`):

```bibtex
@article{Yang2024MVIS,
  author  = {Yang, Yulin and Geneva, Patrick and Huang, Guoquan},
  title   = {Multi-visual-inertial system: Analysis, calibration, and estimation},
  journal = {The International Journal of Robotics Research},
  year    = {2024},
  volume  = {43},
  number  = {13},
  pages   = {1995--2026},
  doi     = {10.1177/02783649241245726}
}
```

## Contributing

Issues and pull requests are welcome; see
[CONTRIBUTING.md](https://github.com/openmvis/openmvis/blob/main/CONTRIBUTING.md).

## License

MVIS is released under
[GPL-3.0-only](https://github.com/openmvis/openmvis/blob/main/LICENSE): free for
everyone, including companies and commercial use, as long as software that
includes or is derived from MVIS is released under the GPL. For use in
closed-source products, a commercial licence is available; see
[COMMERCIAL-LICENSE.md](https://github.com/openmvis/openmvis/blob/main/COMMERCIAL-LICENSE.md)
(licensing@openmvis.com). Calibration results you compute with MVIS are yours to
publish, including commercially.
