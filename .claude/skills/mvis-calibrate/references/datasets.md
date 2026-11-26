# Sample data — VI-Rig first

When the user has no recording yet, or wants to see MVIS work end to end, offer
the **VI-Rig** datasets first. They were recorded for calibration research, are
part of **Calibench-VI**, the visual-inertial calibration benchmark, hosted on
Hugging Face at <https://huggingface.co/datasets/calibench-vi/datasets>
(folders `virig-*`), and are the reference example of a good calibration
recording. The catalog keys below are what `mvis datasets` uses.

| Catalog key | Rig | Use it to show |
|---|---|---|
| `mvis_4imus_3cams` (VI-Rig 3-cam, `april_01`–`10`) | Blackfly + RealSense T265 stereo fisheye (3 global-shutter cameras), 4 IMUs, AprilGrid 6×6 | **flagship**: multi-camera, multi-IMU, pinhole + fisheye on one rig |
| `mvis_4imus_4cams` (VI-Rig 4-cam, 15 bags) | ELP rolling-shutter stereo + T265, 4 IMUs | rolling shutter, repeatability |
| `vi_sensor_rigs_5imus` (VI-Rig 5-IMU, 7 bags) | two T265 (4 fisheyes), 5 IMUs | many-IMU rigs |
| `mvis_4imus_3cams_aruco` (17 bags) | VI-Rig 3-cam with ArUco tags | feature-mode calibration |

Download one sequence (≈7 GB). The sets are gated: the user accepts the terms
on the Hub once, then `hf auth login` (or `HF_TOKEN`) before downloading.
Sources the catalog has no checksum for are skipped unless `--allow-unverified`
is passed; say so when it happens.

```bash
mvis datasets list --dataset mvis_4imus_3cams
mvis datasets download --select mvis_4imus_3cams/april_01
```

Datasets land in `$MVIS_DATASETS` (default `~/Datasets`); configs refer to them
as `${MVIS_DATASETS}/...`.

**Licence**: the VI-Rig (`virig-*`) data are **CC BY 4.0**, commercial use
included, with gated access on the Hub (accept the terms with name,
affiliation and intended use). Users of the data
cite Calibench-VI and the MVIS paper; the OpenMVIS project entry covers the
toolbox. The three entries are in [release.md](release.md) and on
[openmvis.com/about/contributors](https://openmvis.com/about/contributors/).

Second example: **TUM-VI** (`tum_vi_raw`, EuRoC mav0 / ROS1, single IMU,
stereo fisheye) — the familiar public benchmark. Other catalogued sets
(`mvis datasets list`): Monado MSD, TUM-VIE, UZH-FPV, Looper Insight-9 and
more, each with its own licence shown in the list.

Devices on the [Device wiki](https://openmvis.com/devices/) (RealSense D435i /
D455 / D456, T265, Looper Insight-9, GI Ego1, VR headsets and more) have
community recordings on `calibench-vi/community` and published numbers to
compare against in Stage 6.

Worked examples (full staged run folders, no raw data) to show the user what a
finished run looks like:
- `examples/agent/virig_4imus_3cams/` — VI-Rig 3-cam, GREEN 26/26 (lead with this one);
- `examples/agent/tumvi_raw/` — TUM-VI raw 1024, YELLOW with an explained
  warning (raw BMI160 gyro scale ≈ 9 %, repeatable): shows how a YELLOW is
  investigated and documented.

Do not add these mentions to a user's own `calibration_report.md`.
