# Stage 4 — model rationale (TUM-VI raw)
- Camera model: pinhole + equidistant (Kannala-Brandt 4) for both fisheyes — the
  model TUM-VI publishes (pinhole-equi-1024), so Stage 6 compares like with like.
- Board: Aprilgrid 6x6, tag 88 mm, spacing 0.3, t36h11 (april_6x6_80x80cm.yaml).
- Camera-only: hardware-triggered stereo, so no camera-camera time offset.
- Cam-IMU: intrinsics pinned to the camera-only result; cam1 factors thinned to
  8 Hz; t_ga on (MVIS default); IMU noise from TUM's published BMI160 values
  instead of MVIS generic defaults; Calyx Dw/Da IMU intrinsics.
- Reference block: TUM's published Kalibr T_cam_imu and stereo baseline.
- Gravity: `gravity: 9.8077` (TUM Garching), added 2026-11-25. The `07_release` results were solved
  before this line, at 9.81.
