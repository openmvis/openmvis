# Run log — TUM-VI raw 1024 (worked example 2)
- Stage 0: `mvis check --env --smoke` PASS (numpy 1.26.4, gtsam, aprilgrid, rosbags, cv2 4.13; smoke fx 499.1/500; mvis 96af17d yulin/mvis-agent).
- Stage 1: `mvis inspect` PASS on both bags — ROS1. calib-cam1: 109.0 s, 2 cams 1024² mono16 @4 Hz, imu0 199.4 Hz. calib-imu1: 51.9 s (≥30 s gate), 2 cams @20 Hz, imu0 199.4 Hz, accel m/s². Plan said EuRoC mav0; only the ROS1 bags are local, so format generality is not shown by this example.
- Stage 2: cam0 base (global, fisheye), cam1 aux; imu0 (BMI160) base; hardware-triggered.
- Stage 3: intrinsics pinned in cam-IMU; IMU noise from TUM's published BMI160 values; reference = TUM's published Kalibr calibration (pinhole-equi-1024).
- Stage 4: pinhole-equi both cams; `mvis check --env -c` PASS (cam); cam_imu only missing the prior (expected before 5a).
- Stage 5a: camera-only GREEN (median 0.24/0.25 px, p95 0.60/0.61 px, 14.6k obs/cam), 3.5 min. vs TUM's published Kalibr (pinhole-equi-1024): fx +0.32 % / +0.49 %, principal point ≤ 0.24 px, baseline 100.86 vs 100.95 mm, stereo rotation 0.0480 vs 0.0469 rad (1.1 mrad).
- Stage 5b: cam-IMU finished in 6.8 min (solve 200 s), 5 iterations.
- Stage 6: verdict YELLOW (16/17): WARN imu.dw_scale (gyro y 1.094, x 0.946) — repeatable to ±0.0003 over imu1–4, IMU χ² 0.52 with published noise → real raw-BMI160 scale; explained; accepted by the user 2026-10-18 (gate passed). Cross-tool vs TUM Kalibr: fx +0.3/+0.5 %, levers 3.8/5.1 mm, baseline −0.19 mm; common 9.8 mrad cam→IMU rotation offset (IMU-frame definition, unresolved).
- Stage 7: release written.
