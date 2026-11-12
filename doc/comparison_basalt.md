# Basalt Calibration — Architecture Comparison with MVIS

> **Snapshot of April 2026.** This document compares the Basalt source of that
> time with the MVIS design; the tables were updated for the shipped feature set
> but the numbers and the Basalt reading were not redone. Where it says "KB4" for
> Basalt, that is Basalt's Kannala-Brandt model; MVIS's `kb4` (theta from atan2,
> valid past 90 deg) and `equidistant` (stops at 90 deg) give the same pixels for
> corners under 90 deg. Current MVIS results on real recordings are on
> [openmvis.com](https://openmvis.com).

## Overview

[Basalt](https://gitlab.com/VladyslavUsenko/basalt) is a visual-inertial
SLAM framework by Usenko et al. (TUM) with integrated camera and cam-IMU
calibration tools. It uses custom LM optimizers (not GTSAM/Ceres) and
continuous-time B-spline trajectory representation.

**Key source files:**
- `include/basalt/calibration/cam_calib.h` — camera calibration pipeline
- `include/basalt/calibration/cam_imu_calib.h` — cam-IMU calibration pipeline
- `include/basalt/optimization/poses_optimize.h` — LM for camera-only (~315 lines)
- `include/basalt/optimization/spline_optimize.h` — B-spline for cam-IMU (~612 lines)
- `include/basalt/optimization/spline_linearize.h` — multi-residual Jacobians
- `src/calibrate.cpp` — camera calibration executable
- `src/calibrate_imu.cpp` — cam-IMU calibration executable

---

## Feature Comparison

| Feature | Basalt | MVIS | Notes |
|---------|--------|----------|-------|
| **Language** | C++ (headers-only core) | Python | Basalt faster; MVIS easier to extend |
| **Optimizer** | Custom LM (camera), custom Gauss-Newton (spline) | GTSAM factor graph (LM) | Both use LM; Basalt rolls its own |
| **Camera models** | Double Sphere, EUCM, Pinhole, KB4 | RadTan, Equidistant, `kb4`, FOV, DS, EUCM, UCM, omni_radtan | Basalt has DS as default |
| **Trajectory** | B-spline SE(3) (continuous, order N=5) | Discrete per-frame poses + SLERP | B-spline smoother; SLERP simpler |
| **IMU calibration** | Yes: biases, scale/misalign, gravity, temporal offset | Yes: cam-IMU extrinsics, td, readout, scale/misalignment (Dw/Da), biases, gyro-accel offset `t_ga`; discrete states, not B-spline | Both; different trajectory models |
| **Temporal offset** | Optimized via B-spline (cam_time_offset_ns) | Cross-correlation + joint optimization | Both support cam-cam td |
| **Rolling shutter** | Not mentioned in calibration code | Supported (per-camera readout joint opt) | MVIS advantage |
| **Robust kernel** | Huber (default thresh=4.0 px) | Huber (default thresh=1.345) | MVIS tighter |
| **Target** | AprilGrid only | AprilGrid, chessboard, ArUco | MVIS more flexible |
| **Input** | ROS bags, EuRoC | ROS1/ROS2 bags, EuRoC, image folders | MVIS more flexible |
| **GUI** | Pangolin (OpenGL) with real-time viz | Rerun (optional), HTML report | Different approaches |
| **Parallelism** | TBB for Jacobian assembly | GTSAM sparse solver | Both parallel |
| **Outlier rejection** | Huber only (single pass) | Two-pass: P99 filter + warm-start rerun | MVIS more robust |
| **Output** | calibration.json (cereal) | YAML + self-contained HTML report | MVIS more informative |
| **Multi-camera** | Stereo (T_i_c extrinsics) | N cameras (independent extrinsics) | MVIS more flexible |

---

## Camera Calibration Pipeline Comparison

### Basalt Camera Calibration

```
loadDataset() → detectCorners() → initCamIntrinsics() → initCamPoses()
     → initCamExtrinsics() → initOptimization() → optimize() [iterative LM]
```

- GUI-driven (Pangolin checkboxes/buttons)
- Corner detection: AprilTag with sub-pixel refinement + adaptive radii
- Intrinsics init: geometry-based from corner pattern
- Pose init: PnP per frame
- Extrinsics init: from shared pose observations
- Optimizer: Custom LM with damping (lambda 1e-6 to 100)
- Jacobian assembly: TBB-parallel over AprilGrid measurements
- Sparse accumulator for block-diagonal approximation

### MVIS Camera Calibration

```
Detection → Quality Gating → Init → Per-cam Opt → RS Detect
     → Alignment → Joint Opt → Outlier Rerun
```

- Config-driven (YAML) + CLI
- Multi-scale adaptive AprilTag detection
- Vanishing-point focal init (fisheye) or OpenCV (radtan)
- RANSAC PnP with reprojection error gate
- Per-camera Stage 2 refinement before joint
- Joint Stage 4 with SLERP interpolation
- Stage 5 outlier-driven rerun with pathology guardrail
- Diverged pose detection and removal

### Key Differences

1. **Basalt is simpler in pipeline** — single optimization pass, no
   outlier rerun, no quality gating cascade. Relies on Huber kernel.

2. **MVIS has more stages** — multi-stage pipeline with quality
   gates between stages, two-pass outlier rejection, warm-start rerun.

3. **Basalt is GUI-centric** — user clicks buttons in order. MVIS
   is fully automated (config → result).

---

## Camera-IMU Calibration (Basalt)

### Pipeline

```
loadDataset() → detectCorners() → initCamPoses()
     → initCamImuTransform() → initOptimization()
     → optimize(corners) → setOptCamTimeOffset(true)
     → setOptImuScale(true) → optimize(imu+corners)
     → [optional] initMocap() → setOptMocap(true)
```

### B-Spline Trajectory Model

- Order N=5 (quintic), knot spacing ~10ms
- Continuous SE(3) trajectory on Lie group
- Provides smooth position, orientation, velocity, angular velocity,
  acceleration at any timestamp
- B-spline basis functions weight N knots per query point
- Jacobians w.r.t. knot poses are analytically derived

### IMU Model

- **Accelerometer bias**: 3D constant vector
- **Gyroscope bias**: 3D constant vector
- **Scale/misalignment**: 3×3 matrix (optional, `opt_imu_scale`)
- **Gravity**: 3D vector (estimated during optimization)
- **Noise model**: Continuous-time (gyro_noise_std, accel_noise_std,
  gyro_bias_std, accel_bias_std) — Kalibr-compatible

### Cost Function (Multi-Residual)

The spline optimizer jointly minimizes:
1. **Reprojection residuals**: corner detections → project through
   camera model → error (same as camera-only)
2. **Gyro residuals**: measured angular velocity vs. B-spline angular
   velocity at IMU timestamps
3. **Accel residuals**: measured acceleration vs. B-spline linear
   acceleration + gravity at IMU timestamps
4. **Pose residuals**: initial pose estimates as soft constraints
5. **Mocap residuals**: optional motion capture ground truth

### Temporal Offset

- `cam_time_offset_ns`: shift between camera and IMU clocks
- Optimized by perturbing the timestamp used to query the B-spline
  for camera observations
- Gradients flow through B-spline evaluation w.r.t. time

---

## Design differences

### Where MVIS follows Basalt

1. **Staged optimization**: Basalt first optimizes with corners only,
   then enables temporal offset, then IMU scale. MVIS lets the IMU
   intrinsics be opened in steps (`estimate.imu.intrinsics`: `pinned`,
   `dw_da`, `dw_da_r`, `full`) and holds `t_ga` under a staged prior
   (`sigma_t_ga`).

2. **Scale/misalignment matrix**: a 3×3 transform per sensor captures axis
   misalignment and scale factor errors in MEMS IMUs (MVIS `Dw`/`Da`, with
   three parameterizations: `kalibr`, `rpng`, `calyx`).

### Where MVIS differs on purpose

1. **B-spline trajectory**: MVIS keeps discrete per-frame states; in the
   cam-IMU solve each corner's pose is the anchor state propagated through
   the raw IMU samples to its capture time (`vision_pose: imu`), so no
   spline model error enters.

2. **Custom optimizer**: GTSAM already provides LM with factor graphs.

3. **GUI-driven flow**: MVIS is config-driven, which suits batch processing
   and regression testing.

4. **Gravity**: MVIS aligns the world frame with gravity at initialization and
   keeps it fixed; Basalt estimates the gravity vector.

### MVIS Advantages Over Basalt for Calibration

1. **Rolling shutter calibration** — Basalt doesn't calibrate RS readout
2. **Multi-camera (>2)** — Basalt is stereo only
3. **Camera-camera time offset** — Basalt requires hw-sync between cameras
4. **Two-pass outlier rejection** — more robust than single Huber pass
5. **Automated quality gating** — no manual intervention needed
6. **Comprehensive HTML reports** — Basalt outputs only JSON
7. **Regression testing** — the maintainers' regression suite over real recordings (results on openmvis.com), Basalt has none

### Basalt Advantages Over MVIS

1. **B-spline trajectory** — continuous-time, handles arbitrary motion
2. **Gravity estimation** — gravity direction is a free variable
3. **C++ performance** — faster optimization iterations
4. **Real-time SLAM integration** — calibration feeds directly into VIO

---

## References

- Usenko et al., "The Double Sphere Camera Model," 3DV 2018
- Usenko et al., "Visual-Inertial Mapping with Non-Linear Factor
  Recovery," IEEE RA-L 2020
- Basalt source: <https://gitlab.com/VladyslavUsenko/basalt>
- Basalt docs: <https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md>
