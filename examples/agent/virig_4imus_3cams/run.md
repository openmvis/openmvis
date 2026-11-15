# Run log — VI-Rig 3-cam april_01 (worked example 1)
- Stage 0: env PASS (numpy 1.26.4, gtsam ok, mvis f556f0c + yulin/mvis-agent).
- Stage 1: `mvis inspect` PASS — ROS1, 130.8 s, 3 cams @30 Hz, 4 IMUs, accel units m/s².
- Stage 2: base camera Blackfly (global, pinhole), base IMU /imu/data; 2 aux cams, 3 aux IMUs.
- Stage 3: full rig; intrinsics pinned in cam-IMU; IMU noise generic (waiver: no datasheet).
- Stage 4: configs from regression example; `mvis check` PASS (cam).
- Stage 5a: camera-only GREEN (median 0.56–0.62 px, p95 1.5–1.7 px), 26 min. vs Kalibr camchain (dataset): Blackfly fx +0.8 %, T265-1 fx +0.6 %, baseline 71.6 vs 71.3 mm. Committed regression prior (c8a2771) is ~3.5 % low in focal vs Kalibr — stale.
- Stage 5b: cam-IMU finished in 17.5 min, 6 iterations.
- Stage 6: verdict GREEN (26/26), cross-tool vs Kalibr camchain; consistent with prior MVIS reference for this rig.
- Stage 7: release written; home path in the HTML report's embedded config replaced by <DATA>.
