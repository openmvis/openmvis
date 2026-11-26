# TUM-VI cam-IMU — all distortion models on real data

Runs the joint camera-IMU calibration on the **TUM-VI raw IMU-calibration bags**
(`dataset-calib-imu{1,2,3,4}.bag`, stereo 1024×1024 fisheye + `/imu0`) for every
distortion model, and validates against the published Kalibr cam-IMU results.

Dataset layout assumed (configs read `${MVIS_DATASETS}`, default `~/Datasets`):

```
${MVIS_DATASETS}/tum_vi_calib/
  raw/dataset-calib-imu{1..4}.bag            # cam-IMU sequences
  raw/dataset-calib-cam1-camchain.yaml       # Kalibr omni/radtan cam intrinsics
  tumvi_camchain.yaml                        # Kalibr equidistant cam intrinsics
  standard_1024/dataset-calib-cam1_1024_16.bag  # board bag for camera-only fits
  omni-radtan-1024/ , pinhole-equi-1024/     # Kalibr cam-IMU reference results
```

## Models with a ready camchain (one step)

`omni_radtan` and `equidistant` ship with a Kalibr camchain for these cameras, so
the cam-IMU run is a single step (stereo, intrinsics pinned):

```bash
mvis run -c examples/configs/tumvi_cam_imu/cam_imu_omni.yaml   # omni_radtan
mvis run -c examples/configs/tumvi_cam_imu/cam_imu_equi.yaml   # equidistant
```

To sweep all four bags, override `input.path` with each
`dataset-calib-imu{1,2,3,4}.bag`.

## Models needing a fit first (two steps)

`double_sphere`, `eucm`, `fov` have no shipped camchain, so calibrate the camera
intrinsics first, then feed them to the cam-IMU solve. The wide-FOV **stereo**
camera-only fits diverge on this dataset (the fx–xi / alpha basin problem), so the
recipe fits **cam0 only** — which is stable — and runs a single-camera cam-IMU:

```bash
# double_sphere (likewise eucm, fov); from the repo root
mvis run -c examples/configs/tumvi_cam_imu/mvis_cam0_ds.yaml   # -> results/tumvi_cam_imu/calib_ds_cam0.yaml
mvis run -c examples/configs/tumvi_cam_imu/cam_imu_ds.yaml     # reads it as camera_input
```

Outputs go to `results/tumvi_cam_imu/` relative to the working directory, so run
both steps from the same place.

## Results (imu1, 45 s window)

Solved with MVIS 0.4.1 at the config's `gravity: 9.8077` (2026-11-25). Against the same run at 9.81, gravity
moves only the accel scale, by about −0.02 %. The earlier table (td ≈ −0.7 ms, lever ~1 mm shorter) came from
older MVIS code, not from gravity.

| model | mode | reproj RMS | cam0 lever (mm) | baseline | td |
|---|---|---|---|---|---|
| omni_radtan | stereo, pinned | 0.123 px | [46.1, −69.7, −44.0] | 100.8 mm | +0.16 ms |
| equidistant | stereo, pinned | 0.136 px | [45.4, −71.3, −44.4] | 100.8 mm | +0.16 ms |
| double_sphere | cam0, pinned | 0.124 px | [45.3, −69.5, −44.9] | — | +0.17 ms |
| eucm | cam0, pinned | 0.124 px | [45.3, −69.6, −44.9] | — | +0.17 ms |
| fov | cam0, pinned | 0.163 px | [45.0, −66.7, −44.6] | — | +0.17 ms |
| **Kalibr (omni)** | reference | 0.11 px | **[46.2, −70.9, −43.7]** | **100.94 mm** | ~0 |

Every model lands on the same physical lever (~[45, −70, −44] mm) and time offset
(~+0.17 ms); omni_radtan / equidistant reproduce Kalibr's lever to ≤1.5 mm and the
stereo baseline to ≤0.15 mm. The omni_radtan run agrees across all four bags
(baseline 100.8 mm, lever within ~1 mm). Across the four bags the freed IMU
intrinsics (`Dw`/`Da`) are consistent to <1 %.

## Notes

- **Pin omni/ds/eucm intrinsics in the cam-IMU step.** Freeing them
  (`free_intrinsics: true`) lets the fx–xi (omni/ds) / alpha (eucm) gauge drift
  along its equivalence class on a single sequence while reprojection stays flat;
  the recovered extrinsics/td remain correct, but the intrinsics are not uniquely
  pinned. Freeing the IMU intrinsics (`imu_intrinsics: dw_da`) is stable.
- Detections are cached under `/tmp/mvis_tumvi_detcache`, so re-runs (e.g.
  pinned vs free intrinsics) only repeat the solve.
