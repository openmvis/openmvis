# Calibration report — TUM-VI raw 1024 (calib-cam1 + calib-imu1)

## 1. Summary
- **Grade: YELLOW, accepted** — 16/17 checks pass; the one warning (gyro scale 9.4 %) is a repeatable property of this IMU (§6, §7) and was accepted by the user. Independently cross-checked against TUM's published Kalibr calibration and by repeatability over the four TUM-VI IMU recordings.
- Rig: TUM-VI sensor head — stereo 1024×1024 fisheye cameras (global shutter, hardware-triggered), Bosch BMI160 IMU.
- Data: TUM-VI raw `dataset-calib-cam1.bag` (camera-only, 109 s) and `dataset-calib-imu1.bag` (cam-IMU, 51.9 s, whole recording used); ROS1.
- Date: 2026-10-18. MVIS 96af17d (branch `yulin/mvis-agent`).

## 2. Sensor suite
| Sensor | Topic | Role | Lens / type | Rate |
|---|---|---|---|---|
| cam0 | /cam0/image_raw | **base camera** | fisheye (~195°), global shutter, 1024×1024 mono16 | 20 Hz |
| cam1 | /cam1/image_raw | camera | fisheye (~195°), global shutter, 1024×1024 mono16 | 20 Hz |
| imu0 | /imu0 | **base IMU** | Bosch BMI160, gyro + accel | 200 Hz |

## 3. Calibrated vs fixed parameters
- **Estimated:** camera intrinsics (camera-only stage), cam1→cam0 extrinsics, camera–IMU extrinsics T_C→I and td for both cameras, IMU intrinsics (calyx Dw + Da) and gyro–accel offset t_ga.
- **Fixed:** camera intrinsics in the cam-IMU stage (from camera-only); camera–camera time offset (hardware trigger); rolling-shutter readout (global shutter); IMU noise densities (TUM-published BMI160 values: σ_g 1.6e-4, σ_bg 2.2e-5, σ_a 2.8e-3, σ_ba 8.6e-4).

## 4. Models and settings
- Both cameras pinhole + `equidistant` (KB4) — the model TUM-VI publishes, so the cross-check compares like with like. AprilGrid 6×6, tag 0.088 m, spacing ratio 0.3, `t36h11`.
- IMU model `calyx`, staging `dw_da`, `estimate_t_ga: true`; cam1 factors at 8 Hz; solver max 40 iterations, `rms_gate_px: 3`.

## 5. Results
Re-solved 2026-11-25 on MVIS 0.4.1 at Garching's gravity (`gravity: 9.8077` in `04_configs/cam_imu.yaml`); against 9.81 only the accel scale moves (−0.02 %). The repeatability table in §6 is from the original runs at 9.81.
### Cameras
| Camera | fx, fy, cx, cy (px) | distortion (k1..k4) | reproj median / P95 (px) |
|---|---|---|---|
| cam0 | 382.04, 382.16, 510.47, 514.26 | 0.0054, -0.0032, 0.0005, -0.0003 | 0.107 / 0.261 |
| cam1 | 381.14, 381.26, 505.39, 510.52 | 0.0071, -0.0053, 0.0019, -0.0006 | 0.105 / 0.265 |

### Camera → IMU (T_C→I) and time offset
| Camera | p_CinI (mm) | log(R) (rad) | td (ms) |
|---|---|---|---|
| cam0 | 46.43, -69.06, -46.02 | -0.0535, -2.1761, 2.2528 | +0.176 |
| cam1 | -54.22, -66.48, -49.73 | -0.0528, -2.2297, 2.2002 | +0.179 |

cam1 → cam0 baseline: 100.75 mm (cam-IMU stage), 100.86 mm (camera-only stage).

### IMU
| IMU | Dw diag | Da diag | gyro misalignment log(R) (rad) | t_ga (ms) | χ²/dof |
|---|---|---|---|---|---|
| imu0 | 0.9460, 1.0942, 1.0181 | 1.0074, 1.0071, 0.9732 | 0.0079, -0.0082, -0.0004 | -1.58 ± 0.03 | 0.02 |

Overall reprojection: median 0.106 px, P95 0.263 px, RMS 0.187 px; converged in 5 iterations (solve 259 s, run 6.8 min).

## 6. Verification
`mvis verdict`: **YELLOW** — 16/17 checks pass (full list in `06_verification.json`).

**Cross-tool — TUM's published Kalibr calibration (pinhole-equi-1024):**
| Quantity | MVIS | TUM Kalibr | Δ |
|---|---|---|---|
| cam0 fx (px) | 382.04 | 380.81 | +0.32 % |
| cam1 fx (px) | 381.14 | 379.29 | +0.49 % |
| principal points (px) | | | ≤ 0.24 |
| stereo baseline (mm) | 100.75 | 100.95 | -0.20 |
| stereo rotation (mrad) | | | 1.1 |
| cam0 lever p_CinI (mm) | 46.43, -69.06, -46.02 | 45.38, -72.68, -45.07 | 3.9 |
| cam1 lever p_CinI (mm) | -54.22, -66.48, -49.73 | -55.52, -71.10, -47.83 | 5.2 |
| camera → IMU rotation (mrad) | | | 9.8 (both cameras) |

**Repeatability — the same calibration on all four TUM-VI IMU recordings** (MVIS regression runs, generic IMU noise):
| Recording | gyro scale Dw diag | cam0 lever (mm) | td (ms) | t_ga (ms) |
|---|---|---|---|---|
| imu1 | 0.9470, 1.0943, 1.0191 | 46.5, -68.8, -45.2 | 0.28 | -1.70 |
| imu2 | 0.9465, 1.0942, 1.0192 | 46.0, -68.7, -45.4 | 0.05 | -1.52 |
| imu3 | 0.9464, 1.0941, 1.0188 | 45.9, -68.5, -45.6 | 0.40 | -1.69 |
| imu4 | 0.9469, 1.0944, 1.0190 | 45.3, -67.9, -45.4 | 0.16 | -1.45 |

## 7. Limitations
- **WARN imu.dw_scale (accepted by the user 2026-10-18):** gyro y-axis scale 1.094 and x-axis 0.946 exceed the 5 % plausibility bound. Repeats to ±0.0003 over four independent recordings, IMU χ²/dof 0.02 with the published noise (0.52 before the χ² fix that scores the IMU at the solved t_ga), sub-0.3 px camera residuals: a real scale error of the raw (uncorrected) BMI160 gyro, which a solver must estimate rather than assume 1.
- Camera → IMU rotation differs from TUM's Kalibr by 9.8 mrad (0.56°), identically for both cameras, so it is a difference in the IMU-frame definition, not a camera error. TUM's calibration likely refers to its own corrected IMU frame; applying MVIS's estimated gyro misalignment does not remove the offset. Not resolved.
- Lever arms differ from TUM's Kalibr by 3.8 / 5.1 mm, mostly along y (−69 vs −73 mm), consistently over all four recordings: a systematic difference between the tools, within the 10 mm consistency bound.
- Data available locally as ROS1 bags; the EuRoC (mav0) export of TUM-VI was not exercised.

## 8. Conventions
- T_C→I maps points from the camera frame into the IMU frame; p_CinI is the camera origin in the IMU frame.
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
Calibrated with MVIS (96af17d).
