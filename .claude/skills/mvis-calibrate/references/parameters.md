# Stage 3 — what can be calibrated (defaults)

| Group | Parameters | Default | Config switch |
|---|---|---|---|
| Camera intrinsics | focal, principal point, distortion | estimate in camera-only; **refined again** in cam-IMU from the camera-only prior (pin only with a reason, and say so up front) | `estimate.camera.intrinsics` |
| Camera–camera extrinsics | T_Ci→C0 | estimate (multi-camera) | camera-only solve |
| Camera–IMU extrinsics | T_C→I (rotation + lever arm) | estimate | `estimate.camera.extrinsics` |
| Camera–IMU time offset | td per camera | estimate | `estimate.camera.td` |
| Rolling-shutter readout | tr | only for rolling-shutter cameras | `estimate.camera.tr` |
| Board shape | per-corner heights | **fixed** (flat or `heights_file`); estimate when the board is not known flat | `estimate.board_points` |
| IMU intrinsics | gyro/accel scale + misalignment (Dw, Da) | estimate, staging `dw_da`, model `calyx` | `estimate.imu.intrinsics`, `estimate.imu.model` |
| Gyro–accel time offset | t_ga | **estimate (keep on)** | `solve.estimate_t_ga` |
| IMU–IMU | T_Ij→I0, inter-IMU td | estimate (aux IMUs) | `estimate.imu.extrinsics`, `estimate.imu.td` |
| IMU noise | σ_g, σ_bg, σ_a, σ_ba (continuous-time) | **fixed** — datasheet or Allan variance | `solve.sigma_g` … |

Notes
- Base IMU is the gauge: its extrinsic and td are not estimated; everything is
  relative to it.
- `kalibr` IMU model (full Dw+Da+Tg+R) is only for comparison with Kalibr.
- Rolling-shutter base cameras are weak: prefer a global-shutter base.
- Mixed global/rolling-shutter or unsynchronised cameras: camera-only per
  camera (mono), extrinsics and td from the cam-IMU solve.
- Record nominal values for Stage 6 in the cam-IMU config (CAD, the datasheet,
  or the Device wiki page):

```yaml
reference:
  label: CAD
  lever_mm: {cam0: [x, y, z]}      # camera position in the base-IMU frame, mm
  baseline_mm: 64.0                # |cam1 - cam0|
```
