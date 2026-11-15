# Stage 6 — verification and validation

`mvis verdict` grades each check pass / warn / fail; overall RED if any fail,
YELLOW if any warn, GREEN otherwise.

| Check id | Threshold (pass / warn) | Meaning when it trips |
|---|---|---|
| `solve.ok` | true | Solver's own gates failed (init or rms gate). |
| `solve.converged` | iterations < max | Hit the iteration cap; may not be converged. Raise `max_iterations` once. |
| `reproj.median`, `reproj.p95` | 1.41 / 2.83 px, 2.83 / 5.66 px | Poor fit; heavy p95 alone = outlier frames (blur, partial board). |
| `fit.cam_chi2_dof`, `fit.imu_chi2_dof`, `fit.aux_imu_chi2_dof` | 1.5 / 3.0 | Model or noise mismatch. IMU χ² high → noise values too small or bad IMU. |
| `intr.focal_positive`, `intr.distortion_finite` | — | Diverged camera model: wrong model or bad init. |
| `intr.focal_aspect` | ≤ 2 % | Non-square pixels are rare: board size wrong or weak coverage. |
| `intr.principal_point` | ≤ 10 % off centre | Weak edge coverage. |
| `time.td` | ≤ 50 ms | Real acquisition latency (USB cameras) or bad stamps — ask. |
| `imu.dw_scale`, `imu.da_scale` | ≤ 5 % | Plausible for raw consumer IMUs; confirm on a second recording. |
| `ref.lever`, `ref.baseline` | ≤ 10 mm, ≤ 5 mm | Disagrees with the user's CAD values: wrong reference, frame convention, or board scale. |

## Beyond the fit
Reprojection error cannot reveal a wrong board scale or a wrong IMU unit — the
fit absorbs them. So also:
1. **Physical plausibility**: focal length vs lens spec (f_px ≈ f_mm / pixel
   pitch), baseline and lever arm vs the rig drawing, td sign and size.
2. **Independent check** (pass `--independent-check` to `mvis verdict`):
   - `cross-tool`: compare with another tool's calibration of the same data
     (e.g. a Kalibr camchain shipped with the dataset): focal within ~1 %,
     baseline within a few mm;
   - **Device wiki**: for a sensor with a page on
     [openmvis.com/devices](https://openmvis.com/devices/), compare focal,
     principal point, lever arm and td with the published values and their
     spread across recordings; a difference beyond that spread needs a reason
     (different unit, firmware, lens) and goes into the report;
   - `repeatability`: same config on 2+ recordings (`input.bags`), compare
     extrinsics/td spread;
   - `roundtrip`: `mvis roundtrip -c cam_imu.yaml --out <run>/06_roundtrip/`
     (ROS1 input) — calibrate, apply, recalibrate; PASS when the IMU intrinsics
     and biases collapse and the extrinsics and baseline hold. Guide:
     [round trip](https://openmvis.com/guides/roundtrip/);
   - `holdout`: evaluate on a recording not used for the solve.
   Without one, the release says **unverified**.

## Escalation
Non-GREEN → delegate to the `mvis-triage` subagent with the run folder. Apply
its single proposed change as a new attempt.
