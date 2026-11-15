---
name: mvis-calibrate
description: Calibrate a camera or camera-IMU rig from the user's own recording with MVIS (pymvis) — a gated procedure: installation verification, then seven stages (data qualification, sensor suite, parameters, models and configs, solve, verification, release). Use when a user wants to calibrate their rig, their dataset or a device from the Device wiki.
---

# MVIS calibration procedure

You take the user from a raw recording to a verified, released calibration.
Verify the installation (Stage 0), then work through the seven stages **in order**. Each stage has a deliverable written
to the run folder and an **exit gate**; do not advance until the gate passes or
the user explicitly accepts a waiver (record it in `run.md`).

Correctness lives in the tools, not in your judgement: `mvis inspect`,
`mvis check`, `mvis verdict` and `mvis roundtrip` produce findings with stable
ids. Explain those findings; never override a FAIL by argument.

The manual is [openmvis.com](https://openmvis.com): link its guides instead of
re-explaining them ([board setup](https://openmvis.com/guides/board-setup/),
[calibration motion](https://openmvis.com/guides/calibration-motion/),
[camera](https://openmvis.com/guides/camera-calibration/),
[camera-IMU](https://openmvis.com/guides/camera-imu-calibration/),
[rolling shutter](https://openmvis.com/guides/rolling-shutter/),
[multi-sensor rigs](https://openmvis.com/guides/multi-sensor-rigs/),
[round trip](https://openmvis.com/guides/roundtrip/),
[configuration reference](https://openmvis.com/reference/configuration/),
[Device wiki](https://openmvis.com/devices/)).

## Hard rules

- Run calibrations only with `mvis run -c <config>` and a config that has an
  explicit `mode:` (`cam` or `cam-imu`). Never use `mvis calibrate`,
  `scripts/`, `studies/` or `scripts/run_regression*.py`.
- Never edit `src/`, `tests/` or `examples/regression/` while calibrating.
- Camera-IMU is always two solves: camera-only first, then cam-IMU with the
  camera result as `camera_input.path`. Mixed global/rolling-shutter or
  unsynchronised rigs: one camera-only solve **per camera** (mono), then the
  extrinsics and time offsets come from the cam-IMU solve; a joint camera-only
  pair only for hardware-synchronised cameras of the same shutter type.
- `ok: true` is not evidence. Only `mvis verdict` decides fit quality.
- Long solves: run in the background. The `mvis` CLI pins BLAS to one thread
  itself (`MVIS_BLAS_THREADS` changes it); in a source checkout
  `.claude/settings.json` sets the same. Do not prefix thread variables onto
  commands. Write commands as plain `mvis …`: only that form matches the
  permission allowlist; an absolute path to `mvis` or a `conda run -n mvis
  mvis …` wrapper asks for permission on every call. A native segfault (≈3–5 %
  of runs) is retried once unchanged before anything is changed.
- Read every solve log for `WARNING` lines (e.g. "No co-visible frames … Using
  identity extrinsic") and report them to the user with the result, even when
  the verdict is GREEN.
- After changing the board or detector settings, clear the detection cache
  (`input.cache_dir`), or stale detections are reused.
- Rotations are reported as the log(R) rotation vector in rad (never
  roll/pitch/yaw degrees); translations in mm; time offsets in ms with the sign
  convention stated. Reprojection numbers are per-corner norms in px.
- Put no absolute home paths into anything under `07_release/`.

## Stage 0 — Installation verification

Objective: prove this installation can run a calibration before touching data.

First find the environment the user is in; do not assume conda or a checkout:

```bash
which mvis                           # which env's mvis is on PATH
mvis --version                       # pymvis version
```

- `mvis` found: use it as is, whatever made the env (pip venv, uv, conda).
- Not found, and `conda env list` shows `mvis`: ask the user to
  `conda activate mvis` and restart Claude Code (preferred). `conda run -n mvis
  mvis …` also works, but it is not on the allowlist: tell the user every call
  will ask for permission, and do not leave a long solve waiting on a prompt.
- Neither: stop and point to [Installation](https://openmvis.com/getting-started/installation/)
  (`pip install "pymvis[apriltag,ros1]"`, Python 3.11 or 3.12, Linux or macOS).

Then choose the run folder `<run>` and create it with its `run.md` log (one line
per stage: gate result, decisions, waivers):

- inside a source checkout of MVIS: `results/agent_runs/<dataset>-<YYYYMMDD-HHMM>/`;
- otherwise (pip install, any working directory): `./mvis_runs/<dataset>-<YYYYMMDD-HHMM>/`.

```bash
mvis check --env --smoke --out <run>/00_env.json
```

It checks numpy < 2, GTSAM, OpenCV (clashing wheels, ArUco), the optional
extras (`apriltag` for AprilTag boards, `ros1` for ROS1 bags), that `mvis`
on PATH is this env, whether a source checkout or the installed wheel is
imported (worktrees!), and runs a ~1 s synthetic camera solve that must
recover known intrinsics. Missing extras are warnings here; Stage 4 re-runs
`mvis check --env -c <config>`, which fails on the extras that config needs.

If a check fails, point the user to the exact fix in the finding and to
[Installation](https://openmvis.com/getting-started/installation/). Do not
`pip install` or modify the env yourself unless the user asks; the usual fix is
`pip install "pymvis[apriltag,ros1]"` in the same env.

**Gate:** no FAIL (in particular `env.smoke` passes). Record in `run.md` the
`pymvis` version and, when a checkout is imported, its git sha — they go into
the release.

```
<run>/
├── run.md                      # stage log, decisions, waivers
├── 00_env.json                 # Stage 0: mvis check --env --smoke
├── 01_dataset_report.json      # Stage 1
├── 02_sensor_suite.yaml        # Stage 2
├── 03_calibration_spec.yaml    # Stage 3
├── 04_configs/                 # Stage 4: cam.yaml, cam_imu.yaml, rationale.md
├── 05_solve/attempt-NN/        # Stage 5: outputs + logs per attempt
├── 06_verification.json        # Stage 6
└── 07_release/                 # Stage 7
```

If the user has no data yet, offer the VI-Rig sample first — see
[references/datasets.md](references/datasets.md). If the user's device is on
the [Device wiki](https://openmvis.com/devices/), read its page first: it has
the board, the recording recipe and reference numbers for Stage 6.

## Stage 1 — Data qualification

Objective: prove the recording can support the calibration before any solve.

```bash
mvis inspect <dataset> --out <run>/01_dataset_report.json
```

Read [references/data.md](references/data.md). Explain to the user: format and
which modes it supports, every camera (rate, resolution, encoding), every IMU
(rate, accel/gyro units), duration, findings.

**Gate:** status not FAIL; the requested mode is in `supported_modes`; for
cam-IMU, at least one IMU and ≥ 30 s of recording. ROS1, ROS2 and EuRoC mav0
support both modes; an image folder or image+CSV dump supports camera-only
only. Do not write an ad-hoc converter; `mvis datasets` ships converters for
the catalogued sets, nothing else.

## Stage 2 — Sensor suite identification

Objective: fix which sensors are calibrated and their roles.

From the Stage 1 streams, confirm with the user (one question at a time, each
with a proposed default). End every turn that needs an answer with that one
question, last and in bold, so it is not lost in a long reply. If the user says
you skipped something, re-read your own earlier messages before agreeing; if
you did show it, point to it politely instead of withdrawing a valid answer.
- cameras in scope; lens class per camera (pinhole / fisheye / >180°); shutter
  (global / rolling);
- IMUs in scope; gyro-only or full;
- **base camera** (global shutter, highest rate, best view of the board) and
  **base IMU** (`imu_topics[0]`, the best-quality IMU);
- synchronisation (hardware-triggered, shared clock, unknown). Mixed shutter
  types or unsynchronised cameras decide the camera-only strategy (hard rules).

Derive defaults from the data (`01_dataset_report.json`), the user, the
references and the Device wiki page when there is one, not from the worked
examples. `examples/agent/` shows the *format* of each deliverable; use its
*values* only when `mvis inspect` identifies the recording as that sample
dataset, and say so.

Write `02_sensor_suite.yaml` (topic, role, lens, shutter, frame name per sensor).
**Gate:** every sensor maps to a Stage 1 stream; exactly one base camera (and
one base IMU for cam-IMU).

## Stage 3 — Calibration parameter specification

Objective: decide what is estimated, what is fixed, and from which prior.
Use the default table in [references/parameters.md](references/parameters.md);
deviate only with a reason. Ask for the user's nominal values (CAD lever arm,
stereo baseline, rotation) — they go into the cam-IMU config's `reference:`
block and are compared in Stage 6. For a device on the Device wiki, the
published numbers are the reference when the user has none.

Write `03_calibration_spec.yaml`. **Gate:** every estimated parameter is
observable from this recording (e.g. rolling-shutter readout only for rolling
shutter cameras; IMU noise has a stated source).

## Stage 4 — Model selection and configuration

Objective: turn the spec into validated configs.
- Choose camera, board and IMU models with
  [references/models.md](references/models.md).
- **New dataset, cam-IMU: ask the user to pick a preset** and say what each
  costs (both are in the cam-IMU template):

  | Preset | Motion window | Non-base cameras | Corners per tag | Use for |
  |---|---|---|---|---|
  | Fast | 60 s | 4 Hz | two (`two_anti`) | a first pass; about a third of Full's solve time |
  | Full | 80 s | 10 Hz | all | the release calibration |

  On the VI-Rig 3-cam rig the 60 s window alone loosened the camera levers
  (`cam0` ± 1.57 → ± 2.53 mm), which is why the published regression rows use
  80 s with 4 Hz and two corners, a mix of the two. Record the choice in
  `rationale.md`.
- Start from [templates/cam.yaml](templates/cam.yaml) and
  [templates/cam_imu.yaml](templates/cam_imu.yaml); write
  `04_configs/cam.yaml`, `04_configs/cam_imu.yaml` and `04_configs/rationale.md`
  (each non-default choice and why).
- Lint until clean:

```bash
mvis check --env -c <run>/04_configs/cam.yaml  # also fails on extras this config needs
mvis check -c <run>/04_configs/cam_imu.yaml    # after the camera prior exists
```

**Gate:** `mvis check` reports no FAIL; the user approves the configs.

## Stage 5 — Calibration execution

Objective: produce the estimates with provenance.

```bash
mkdir -p <run>/05_solve/attempt-01          # the shell opens the log before mvis starts
mvis run -c <run>/04_configs/cam.yaml       > <run>/05_solve/attempt-01/cam.log 2>&1
mvis verdict <cam result yaml>              # quick look before cam-IMU
mvis run -c <run>/04_configs/cam_imu.yaml   > <run>/05_solve/attempt-01/cam_imu.log 2>&1
```

Tell the user the expected duration before starting. Measured references
(one solve at a time, no other load, Full preset):

| Run | Camera-only | Cam-IMU |
|---|---|---|
| TUM-VI raw, 2 cams 1024², ~440 frames/cam | 3.5 min | 6.8 min |
| VI-Rig 3-cam, 3 cams, ~1 960 frames/cam @15 Hz | 26 min (desktop) – 50 min (2019 laptop) | 17.5 – 23 min |

Board detection is ≈ 1–4 min per camera per 2 000 frames (fisheye and large
images are slower); the optimisation scales with the number of corner
observations, at roughly 30–65 s per iteration for ~170 k observations. The
board is detected once per recording when the camera-only and cam-IMU configs set
the **same** `input.cache_dir` (ROS1 bags): both read and fill its shared per-camera
store (`<cache_dir>/streams/`), so the cam-IMU solve skips detection. Without a
shared `cache_dir`, or for EuRoC / image sources, budget the detection time twice.
Say the solve is single-threaded: a faster CPU, not more cores, shortens it. Present the estimates in the fixed table of
[references/release.md](references/release.md). Every config change starts a new
`attempt-NN/` (create it with `mkdir -p` first) with a one-line reason in `run.md`.

**Gate:** both runs finished and wrote their outputs.

## Stage 6 — Verification and validation

Objective: decide, with evidence, whether the result can be trusted.

```bash
mvis verdict <run>/05_solve/attempt-NN/cam_imu.json -c <run>/04_configs/cam_imu.yaml \
    --out <run>/06_verification.json
```

Follow [references/verification.md](references/verification.md): fit quality,
physical plausibility, consistency with the user's reference values, and an
independent check where possible. Offer them in this order: **cross-tool** when
a Kalibr (or other tool's) calibration of the same data exists (ask the user;
datasets often ship one) — cheapest and strongest; **Device wiki** when the
sensor has a page there (compare intrinsics, lever arm and time offset with the
published numbers and their spread); then **repeatability** over a second
recording; then **round trip** (`mvis roundtrip -c <cam_imu.yaml> --out
<run>/06_roundtrip/`, ROS1 only, slowest: it rewrites the bag, re-detects the
board and re-solves cam-IMU — longer than the cam-IMU solve itself; 40+ min on
VI-Rig 3-cam). If there is no independent check, the result is **unverified** — say so.

Non-GREEN: delegate to the `mvis-triage` subagent with the run folder path. It
returns a ranked diagnosis and one change; apply it as a new attempt (back to
Stage 4/5). **Gate:** GREEN, or YELLOW with every warning explained and
accepted by the user.

## Stage 7 — Calibration release

Objective: hand over a self-describing calibration package in `07_release/`:
the MVIS result files, the Kalibr camchain export where the camera model allows
it, `06_verification.json`, and `calibration_report.md` written from
[references/release.md](references/release.md), including the citation block.

If the user wants the result public, point to the Calibench-VI submission path
([openmvis/calibench-submissions](https://github.com/openmvis/calibench-submissions)):
results and community datasets appear on the openmvis.com Device wiki and
leaderboard after review.

**Gate:** you state the final grade and whether it was independently verified.
