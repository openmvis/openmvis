# The `estimate:` block — choosing what to calibrate

`estimate:` is the canonical, symmetric way to choose **which quantities are
calibrated**, on both the camera-only (`CalibrationConfig` / `CamRunConfig`) and
cam-IMU (`CamImuRunConfig`) configs. It is the **source of truth**: each key is
pushed down into the low-level solve knobs and **wins** over a conflicting legacy
key (with a warning). Omit it and the legacy fields behave exactly as before.

## Shape

```yaml
estimate:
  camera:                 # rig-wide camera defaults (+ per-camera override)
    intrinsics: false     # focal / principal point
    distortion: true      # distortion params (camera-only granularity)
    tr:         false     # rolling-shutter readout (RS cameras)
    td:         true      # camera <-> base-IMU time offset (cam-IMU)
    extrinsics: true      # cam-cam (camera-only) / cam-IMU T_C->I (cam-IMU)
  imu:                    # rig-wide IMU defaults (one setting for every IMU)
    intrinsics: dw_da     # bool (true=full/false=pinned) OR pinned|dw_da|dw_da_r|full
    model:      calyx     # IMU model family: kalibr | rpng | calyx
    extrinsics: true      # aux-IMU -> base-IMU T_IstoI (base IMU = gauge)
    td:         true      # aux-IMU <-> base-IMU time offset
  board_points: false     # board shape: per-corner heights (camera-only AND cam-IMU)
```

Each key is **tri-state**: omit it to inherit the pipeline default; `true`/`false`
to estimate / hold fixed. The pattern mirrors `factor_hz` — `estimate.camera` /
`estimate.imu` are rig-wide defaults; a per-sensor `estimate:` on a camera entry
overrides only the keys that differ. The IMU block is rig-wide only: the per-IMU
entries of `input.imus` carry `gyro_only` (calibrate an aux IMU from its gyro
alone), not an `estimate:` override.

## Two orthogonal IMU axes

- `imu.intrinsics` — **how much** to estimate (staging: `pinned` → `dw_da` →
  `dw_da_r` → `full`; or a bool, `true`=`full`, `false`=`pinned`).
- `imu.model` — **which** model family (`kalibr` / `rpng` / `calyx`). Rig-wide
  (one family for the whole rig today).

Defaults: `imu.intrinsics: dw_da`, `imu.model: calyx`.

## Mapping to the low-level knobs

| `estimate.` key | camera-only | cam-IMU |
|---|---|---|
| `camera.intrinsics` | `optimizer.optimize_intrinsics` | `camera_input.free_intrinsics` |
| `camera.distortion` | `optimizer.optimize_distortion` ¹ | folded into `camera.intrinsics` |
| `camera.tr` | `optimizer.optimize_rolling_shutter` | `(camera_input/solve).estimate_readout` |
| `camera.td` | `optimize_time_offset` | `solve.estimate_td` (pin = `sig_C[6]`) |
| `camera.extrinsics` | `optimizer.optimize_extrinsics` ¹ | `solve.estimate_extrinsics` (pin = `sig_C[0:6]`) |
| `imu.intrinsics` | — ² | `solve.imu_intrinsics` |
| `imu.model` | — ² | `solve.imu_intrinsics_model` |
| `imu.extrinsics` | — | `solve.estimate_aux_extrinsics` (pin = `_As[0:6]`) |
| `imu.td` | — | `solve.estimate_aux_td` (pin = `_As[6]`) |
| `board_points` | `optimizer.optimize_board_points` ¹ | — ² |

¹ Camera-only: per-corner heights along the board normal (3-row gauge, 3 mm
prior), via the `mvis cam` CLI and `scripts/run_calibration.py` (the latter since
2026-11-01). ² Cam-IMU: `solve.estimate_board_heights`, seeded from
`board.heights_file` when given.

**Gauge and prior.** The heights are offsets along the board normal, and both solvers
fix what the board pose already carries with a 3-row gauge: the mean height and the two
tilts (first moments about the board x and y axes) are held at 0, so the pose keeps the
mean plane and the heights hold only the shape. Each height has a weak zero-mean prior,
`board_height_sigma`, 3 mm (settable as `solve.board_height_sigma` on cam-IMU; the
camera-only configs keep the default), so corners never observed stay at their start. The start is flat or `board.heights_file`, and the
cam-IMU sidecar `<out>_board_shape.yaml` is a valid `heights_file`.

**One board only.** On cam-IMU, `board_points: true` with several boards, or in feature
mode, raises `ValueError` (`estimate board heights: board mode with ONE board only`).

**Default and the VI-Rig rule.** The code default is `false` (flat board). For
the VI-Rig recordings (ELP, T265, 4 IMUs + 3 cameras), camera-only and cam-IMU,
set `estimate.board_points: true`: that AprilGrid is a saddle (2.4 mm rms, 11 mm
peak to peak), and estimating it halves the residual and puts the camera-only
focal length on the IMU-anchored value (ELP fx 425.3 → 428.4 px, median residual
0.53 → 0.31 px, +2–3 % solve time). Other rigs keep the default unless their
board is measured non-flat.

A `false` on a cam-IMU spatial-temporal key adds a `_TINY`-sigma prior holding the
quantity at its seed; the pinned component's posterior marginal collapses to ~0
(verified in `tests/config/test_estimate_pins.py`).

## Examples

- `examples/configs/example_estimate.yaml` — cam-IMU, full `estimate:` block.
- `examples/configs/example_cam_run.yaml` — camera-only, with an `estimate:` block.
