# MVIS calibration agent (`mvis-calibrate`)

A Claude Code skill that takes a user from a raw recording to a verified,
released calibration. It lives in this repository and is versioned with the
code it describes:

| Path | What it is |
|---|---|
| `.claude/skills/mvis-calibrate/SKILL.md` | the procedure: Stage 0 + seven gated stages |
| `.claude/skills/mvis-calibrate/references/` | data, parameters, models, verification, release, sample datasets |
| `.claude/skills/mvis-calibrate/templates/` | `cam.yaml`, `cam_imu.yaml` config templates with the Fast / Full presets |
| `.claude/agents/mvis-triage.md` | read-only subagent that diagnoses a non-GREEN result |
| `src/mvis/doctor/`, `src/mvis/analysis/verdict.py`, `src/mvis/recalib/` | the deterministic tools below |

## Starting it

Install MVIS (`pip install "pymvis[apriltag,ros1]"`, Python 3.11 or 3.12,
Linux or macOS; or a source checkout with `pip install -e .`), start Claude
Code in the folder where the run should live and say "calibrate my rig, data
at …", or run `/mvis-calibrate`. With a source checkout the skill is picked up
from `.claude/`; with a pip install, copy `.claude/` from the repository into
the working directory (or into `~/.claude/` for every project).

Run folders: `results/agent_runs/<dataset>-<stamp>/` inside a checkout,
`./mvis_runs/<dataset>-<stamp>/` anywhere else. Both are git-ignored here.

The `mvis` CLI pins BLAS to one thread itself; the project
`.claude/settings.json` sets the same environment and pre-approves the
read-only and solve `mvis` commands when they are written as plain `mvis …`
(an absolute path or `conda run -n mvis mvis …` still prompts). `--out`
creates missing parent folders.

## Tools

Correctness lives in the tools, not in the model. Each prints text or `--json`,
can write `--out`, and exits 0 = pass/GREEN, 1 = warn/YELLOW, 2 = fail/RED.
Every finding has a stable id.

| Command | Purpose |
|---|---|
| `mvis check --env --smoke` | installation verification: numpy < 2, GTSAM, OpenCV wheel clashes / ArUco, optional extras, `mvis` on PATH, wheel or checkout imported, git sha when a checkout, ~1 s synthetic solve |
| `mvis inspect <dataset>` | data qualification: format (ROS1 / ROS2 / EuRoC / images), streams, rates, IMU units and stamps, camera–IMU overlap, supported modes |
| `mvis check [--env] -c <config>` | strict config lint (missing `mode:`, unknown keys, board units, missing prior); with `--env`, fails on extras that config needs |
| `mvis run -c <config>` | the calibration (`mode: cam` or `mode: cam-imu`) |
| `mvis verdict <result> [-c <cam_imu config>]` | GREEN / YELLOW / RED: fit, convergence, χ²/dof, intrinsics and IMU plausibility, time offsets, consistency with the user's reference values |
| `mvis roundtrip -c <cam_imu config>` | independent check: calibrate, apply the calibration to the bag, recalibrate, PASS/FAIL on the collapse of IMU intrinsics and biases |

## Stages

| # | Stage | Deliverable | Gate |
|---|---|---|---|
| 0 | Installation verification | `00_env.json` | no FAIL, smoke solve passes |
| 1 | Data qualification | `01_dataset_report.json` | no FAIL; mode supported; cam-IMU ≥ 30 s with an IMU |
| 2 | Sensor suite identification | `02_sensor_suite.yaml` | every sensor maps to a stream; one base camera / IMU |
| 3 | Calibration parameter specification | `03_calibration_spec.yaml` | every estimated parameter observable |
| 4 | Model selection and configuration | `04_configs/` | `mvis check` clean; preset chosen; user approves |
| 5 | Calibration execution | `05_solve/attempt-NN/` | outputs written |
| 6 | Verification and validation | `06_verification.json` | GREEN, or YELLOW with every warning explained and accepted |
| 7 | Calibration release | `07_release/` | grade and independent-check status stated; citation block included |

Camera-IMU is always two solves: camera-only first, then cam-IMU with the
camera result as the prior (intrinsics refined again by default). Mixed
global/rolling-shutter or unsynchronised rigs get one camera-only solve per
camera. A result without an independent check (cross-tool, Device wiki
numbers, repeatability, round trip, hold-out) is reported as unverified.

Presets for a new dataset (Stage 4): **Full** (80 s window, non-base cameras
at 10 Hz, all corners) for the release calibration; **Fast** (60 s, 4 Hz, two
corners per tag) for a first pass at about a third of the solve time. The 60 s
window alone loosened the VI-Rig camera levers from ± 1.57 to ± 2.53 mm, which
is why the published rows use 80 s with 4 Hz and two corners.

## Worked examples

**1. VI-Rig 3-cam** — [`examples/agent/virig_4imus_3cams/`](../examples/agent/virig_4imus_3cams/07_release/calibration_report.md).
Blackfly (pinhole) + RealSense T265 stereo fisheye, four IMUs, AprilGrid 6×6.
GREEN 26/26 (median 0.25 px, P95 0.76 px). Cross-checked against the Kalibr
calibration shipped with the dataset: focal +0.8 % / +0.6 %, stereo baseline
71.2 vs 71.3 mm. Exercises every stage, including multi-camera, multi-IMU and
two lens classes on one rig.

**2. TUM-VI raw 1024** — [`examples/agent/tumvi_raw/`](../examples/agent/tumvi_raw/07_release/calibration_report.md).
Stereo fisheye + BMI160, TUM's published IMU noise. YELLOW 16/17: the raw
gyro scale is ≈ 9 % off 1, repeatable to ±0.0003 over the four TUM-VI IMU
recordings — shown as an example of how a YELLOW is investigated and
documented. Against TUM's published Kalibr calibration: focal +0.3 % / +0.5 %,
lever arms 3.8 / 5.1 mm, baseline −0.19 mm.

Both examples were produced with a source checkout before v0.1.0; their
`00_env.json` therefore records a git sha, not a wheel version. The public
repository ships them without the 5 MB `cam_imu_report.html` files.

## Sample data

The VI-Rig datasets, part of Calibench-VI, are the recommended practice data
and the reference for what a good calibration recording looks like:
[calibench-vi/datasets](https://huggingface.co/datasets/calibench-vi/datasets).

| Catalog key | Rig |
|---|---|
| `mvis_4imus_3cams` | VI-Rig 3-cam: Blackfly + T265 stereo fisheye, 4 IMUs, AprilGrid (flagship) |
| `mvis_4imus_3cams_aruco` | the same rig with ArUco tags |
| `mvis_4imus_4cams` | ELP rolling-shutter stereo + T265, 4 IMUs |
| `vi_sensor_rigs_5imus` | two T265 (four fisheyes), 5 IMUs |

```bash
mvis datasets list --dataset mvis_4imus_3cams
mvis datasets download --select mvis_4imus_3cams/april_01
```

Licence: VI-Rig (`virig-*`) data are CC BY-NC 4.0 with gated access (accept
the terms on the Hub, then `hf auth login`); commercial licences on request
(licensing@openmvis.com). Cite Calibench-VI, the MVIS paper and the OpenMVIS
project; the entries are in the skill's
[release.md](../.claude/skills/mvis-calibrate/references/release.md) and on
[openmvis.com/about/contributors](https://openmvis.com/about/contributors/).

Devices with a page on the [Device wiki](https://openmvis.com/devices/) have
community recordings and published numbers the agent compares against in
Stage 6.

## Limits (current)

- Camera-IMU input must be a ROS1 bag, a ROS2 bag (`format: ros2`) or a EuRoC
  mav0 folder; image+CSV dumps are camera-only.
- `mvis run` (mode `cam`) does not accept `source_type: video`; extract frames.
- ArUco boards are not covered by the agent yet.
- The round trip needs a ROS1 bag.
