# Calibration report — VI-Rig 3-cam (april_01)

## 1. Summary
- **Grade: GREEN**, independently cross-checked (camera intrinsics and stereo baseline vs the Kalibr calibration shipped with the dataset).
- Rig: 3 global-shutter cameras (Blackfly pinhole + RealSense T265 stereo fisheye), 4 IMUs.
- Data: `virig_4imus_3cams_april_01.bag` (ROS1, 130.8 s); 80 s motion window used.
- Date: 2026-10-17. MVIS f556f0c + branch `yulin/mvis-agent`.

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
### Cameras (intrinsics: fx, fy, cx, cy px; distortion)
| Camera | fx, fy, cx, cy | distortion | reproj median / P95 (px) |
|---|---|---|---|
| Blackfly | 449.46, 449.43, 370.33, 281.28 | -0.3284, 0.1002, -0.0003, -0.0002 | 0.281 / 0.768 |
| T265-1 | 286.80, 286.79, 415.70, 403.75 | -0.0036, 0.0380, -0.0384, 0.0072 | 0.238 / 0.752 |
| T265-2 | 286.43, 286.20, 414.68, 398.12 | 0.0006, 0.0233, -0.0221, 0.0020 | 0.249 / 0.751 |

### Camera → base IMU (T_C→I) and time offset
| Camera | p_CinI (mm) | log(R) (rad) | td (ms) |
|---|---|---|---|
| Blackfly | 35.66, -83.66, 26.84 | 0.0081, -2.2206, 2.2185 | -2.878 |
| T265-1 | 68.75, -114.72, -28.00 | -0.0159, -2.2356, 2.2059 | -3.663 |
| T265-2 | 5.07, -114.50, -27.79 | -0.0218, -2.2313, 2.2070 | -6.292 |

T265-1 → Blackfly baseline: 71.2 mm

T265-2 → Blackfly baseline: 69.8 mm

### IMUs
| IMU | p_IsinI (mm) | log(R) (rad) | td (ms) | t_ga (ms) | Dw diag | Da diag | χ²/dof |
|---|---|---|---|---|---|---|---|
| imu (base) | 0 | 0 | 0 | -1.83 ± 0.22 | 1.0035, 1.0032, 1.0023 | 1.0002, 0.9982, 1.0031 | 0.036 |
| gx3_25 | -120.65, -46.43, -2.94 | -2.2374, -2.2049, 0.0245 | +1.476 | +0.11 ± 0.21 | 1.0040, 1.0039, 0.9991 | 0.9988, 1.0011, 1.0029 | 0.009 |
| gx3_35 | -79.93, -45.47, -3.48 | -2.2170, -2.2251, 0.0166 | -0.104 | -0.01 ± 0.22 | 1.0033, 1.0032, 0.9980 | 0.9986, 1.0003, 1.0029 | 0.010 |
| t265 | 57.71, -109.21, -31.49 | 1.5853, 0.0071, -0.0023 | +4.819 | -6.06 ± 0.20 | 1.0011, 1.0013, 1.0065 | 1.0204, 1.0218, 1.0139 | 0.019 |

Overall reprojection: median 0.254 px, P95 0.757 px, RMS 0.619 px; converged in 6 iterations.

## 6. Verification
`mvis verdict`: **GREEN** — 26/26 checks pass (full list in `06_verification.json`).

| Cross-check vs the Kalibr camchain shipped with the dataset | MVIS | Kalibr | Δ |
|---|---|---|---|
| Blackfly fx (px) | 449.46 | 445.92 | +0.8 % |
| T265-1 fx (px) | 286.80 | 285.17 | +0.6 % |
| T265-1 → Blackfly baseline (mm) | 71.2 | 71.3 | -0.1 |

- Consistent with the earlier MVIS reference for this rig: aux-IMU levers 92–129 mm (ref 93–130 mm), aux td −0.1…+4.8 ms, Dw/Da ≈ I.
- Plausibility: all IMU scales within 0.7 % of 1 except the T265 accelerometer (≈2 %); T265 IMU t_ga −6.1 ms (internal gyro/accel sampling offset of that device).

## 7. Limitations
- IMU noise densities are MVIS generic defaults (no datasheet values). IMU χ²/dof ≈ 0.01–0.04 means they are larger than the real noise: estimates are unaffected, but reported σ (e.g. t_ga ± 0.2 ms) are conservative.
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
