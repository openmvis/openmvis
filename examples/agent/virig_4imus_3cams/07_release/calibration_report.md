# Calibration report — VI-Rig 3-cam (april_01)

## 1. Summary
- **Grade: GREEN**, independently cross-checked (camera intrinsics and stereo baseline vs the Kalibr calibration shipped with the dataset).
- Rig: 3 global-shutter cameras (Blackfly pinhole + RealSense T265 stereo fisheye), 4 IMUs.
- Data: `virig_4imus_3cams_april_01.bag` (ROS1, 130.8 s); 80 s motion window used.
- Date: 2026-10-17 (first run, MVIS f556f0c + branch `yulin/mvis-agent`); re-solved 2026-11-26 on MVIS 0.4.1.

## 2. Sensor suite
| Sensor | Topic | Role | Lens / type | Rate |
|---|---|---|---|---|
| Blackfly | /blackfly/image_raw | **base camera** | pinhole, global shutter, 720×540 | 30 Hz |
| T265-1 | /t265/fisheye1/image_raw | camera | fisheye, global shutter, 848×800 | 30 Hz |
| T265-2 | /t265/fisheye2/image_raw | camera | fisheye, global shutter, 848×800 | 30 Hz |
| imu | /imu/data | **base IMU** | gyro + accel | 400 Hz |
| gx3_25 | /gx3_25/data | aux IMU | gyro + accel | 500 Hz |
| gx3_35 | /gx3_35/imu/data | aux IMU | gyro + accel | 100 Hz |
| t265_imu | /t265/imu | aux IMU | gyro + accel | 200 Hz |

## 3. Calibrated vs fixed parameters
- **Estimated:** camera intrinsics (camera-only stage), camera–camera extrinsics, camera–IMU extrinsics T_C→I and td for all 3 cameras, base-IMU intrinsics (calyx Dw + Da) and gyro–accel offset t_ga, aux-IMU extrinsics, td, intrinsics and t_ga.
- **Fixed:** camera intrinsics in the cam-IMU stage (from camera-only), rolling-shutter readout (all global shutter), IMU noise densities.

## 4. Models and settings
- Blackfly `radtan`; T265 `equidistant` (KB4). AprilGrid 6×6, tag 0.088 m, spacing ratio 0.3, `t36h11`.
- IMU model `calyx`, staging `dw_da`, `estimate_t_ga: true`; solver max 40 iterations, `rms_gate_px: 3`.

## 5. Results
Re-solved 2026-11-26 on MVIS 0.4.1 at Newark's gravity (`gravity: 9.8013` in `04_configs/cam_imu.yaml`). Against the first run (2026-10-17) the levers move about 5 mm in y and td by about −1.5 ms; that comes from the 2026-11-02 solver defaults (the t_ga rule and linearized vision-IMU factor, then `vision_pose: imu`), not from gravity, which moves only the accel scale (about −0.07 %).
### Cameras (intrinsics: fx, fy, cx, cy px; distortion)
| Camera | fx, fy, cx, cy | distortion | reproj median / P95 (px) |
|---|---|---|---|
| Blackfly | 449.46, 449.43, 370.33, 281.28 | -0.3284, 0.1002, -0.0003, -0.0002 | 0.275 / 0.776 |
| T265-1 | 286.80, 286.79, 415.70, 403.75 | -0.0036, 0.0380, -0.0384, 0.0072 | 0.234 / 0.742 |
| T265-2 | 286.43, 286.20, 414.68, 398.12 | 0.0006, 0.0233, -0.0221, 0.0020 | 0.244 / 0.742 |

### Camera → base IMU (T_C→I) and time offset
| Camera | p_CinI (mm) | log(R) (rad) | td (ms) |
|---|---|---|---|
| Blackfly | 35.07, -78.77, 27.00 | 0.0074, -2.2198, 2.2193 | -4.336 |
| T265-1 | 67.94, -109.83, -27.83 | -0.0165, -2.2349, 2.2070 | -5.140 |
| T265-2 | 4.26, -109.62, -27.65 | -0.0224, -2.2306, 2.2080 | -7.777 |

T265-1 → Blackfly baseline: 71.1 mm

T265-2 → Blackfly baseline: 69.9 mm

### IMUs
| IMU | p_IsinI (mm) | log(R) (rad) | td (ms) | t_ga (ms) | Dw diag | Da diag | χ²/dof |
|---|---|---|---|---|---|---|---|
| imu (base) | 0 | 0 | 0 | -0.43 ± 0.28 | 0.9976, 0.9998, 0.9984 | 0.9991, 0.9973, 1.0023 | 0.007 |
| gx3_25 | -122.64, -47.01, -3.13 | -2.2380, -2.2045, 0.0238 | +0.005 | +0.12 ± 0.29 | 1.0017, 1.0023, 0.9969 | 0.9976, 1.0002, 1.0023 | 0.007 |
| gx3_35 | -81.97, -47.24, -3.56 | -2.2177, -2.2244, 0.0157 | -1.543 | +0.06 ± 0.31 | 1.0002, 1.0011, 0.9952 | 0.9971, 0.9994, 1.0021 | 0.007 |
| t265 | 56.27, -106.93, -31.23 | 1.5839, 0.0068, -0.0020 | +2.904 | -6.66 ± 0.28 | 0.9999, 0.9994, 1.0042 | 1.0193, 1.0209, 1.0137 | 0.016 |

Overall reprojection: median 0.249 px, P95 0.752 px, RMS 0.609 px; converged in 5 iterations.

## 6. Verification
`mvis verdict`: **GREEN** — 26/26 checks pass (full list in `06_verification.json`).

| Cross-check vs the Kalibr camchain shipped with the dataset | MVIS | Kalibr | Δ |
|---|---|---|---|
| Blackfly fx (px) | 449.46 | 445.92 | +0.8 % |
| T265-1 fx (px) | 286.80 | 285.17 | +0.6 % |
| T265-1 → Blackfly baseline (mm) | 71.1 | 71.3 | -0.2 |

- Consistent with the earlier MVIS reference for this rig: aux-IMU levers 95–131 mm (ref 93–130 mm), aux td −1.5…+2.9 ms, Dw/Da ≈ I.
- Plausibility: all IMU scales within 0.5 % of 1 except the T265 accelerometer (≈2 %); T265 IMU t_ga −6.7 ms (internal gyro/accel sampling offset of that device).

## 7. Limitations
- IMU noise densities are MVIS generic defaults (no datasheet values). IMU χ²/dof ≈ 0.007–0.016 means they are larger than the real noise: estimates are unaffected, but reported σ (e.g. t_ga ± 0.3 ms) are conservative.
- Kalibr cross-check covers camera intrinsics and one stereo baseline only; no external reference for camera–IMU levers or td (no CAD values supplied).
- Single recording; a repeatability run over april_02–10 would quantify spread.

## 8. Conventions
- T_C→I maps points from the camera frame into the base-IMU frame; p_CinI is the camera origin in the IMU frame.
- Rotations are log(R) rotation vectors in rad. Translations in mm.
- td: an image stamped t_cam corresponds to IMU time t_cam + td.

## 9. Files
- `cam_imu_calib.yaml` — full MVIS cam-IMU result
- `cam_imu_camchain-imucam.yaml` — Kalibr-format export
- `cam_imu.json` — summary
- `camera_only_calibration.yaml` — camera-only stage
- `cam_imu_report.html` — interactive report
- `06_verification.json` — verdict

---
Calibrated with MVIS (f556f0c).
