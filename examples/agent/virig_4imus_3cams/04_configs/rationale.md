# Stage 4 — model and setting rationale
- Blackfly: `radtan` — standard pinhole lens.
- T265-1/2: `equidistant` (KB4) — fisheye lenses; default fisheye model.
- Board: AprilGrid 6×6, tag 0.088 m, spacing ratio 0.3, `t36h11` (Kalibr grid).
- Camera-only: 15 Hz factor rate, 100 init frames, Huber 1.345.
- Cam-IMU: motion window 80 s, max 40 iterations, `rms_gate_px: 3`, IMU `calyx`
  + `dw_da`, `estimate_t_ga` on (default), all 4 IMUs (base `/imu/data`).
- IMU noise: MVIS generic defaults (no datasheet supplied) — limitation noted.
