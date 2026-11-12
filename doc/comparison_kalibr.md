# MVIS vs Kalibr Comparison

> **Snapshot of 2026-04-30.** The numbers below were measured on the
> maintainers' regression recordings with the MVIS pre-release build of that
> date; the Kalibr batch runner and the recordings are not part of this
> repository. MVIS defaults have changed since (see `CHANGELOG.md`), so treat
> the figures as a dated record, not as the current state. Current MVIS
> results on real recordings are on [openmvis.com](https://openmvis.com).
>
> Aggregate of the full 228-row sweep: Kalibr **116/228 OK (50.9 %)** with the
> first runner, **135/228 OK (59 %)** after the runner fixes of 2026-05-01;
> MVIS **228/228 OK (100 %)**. The 4-dataset deep-dive below dates to
> 2026-04-24 (93 runs) and is kept for the qualitative comparisons; the
> 228-row tally and per-family breakdown are in
> `doc/comparison_kalibr_regression.md`. All 7 Kalibr models
> (`pinhole-radtan`, `pinhole-equi`, `pinhole-fov`, `omni-none`,
> `omni-radtan`, `eucm-none`, `ds-none`) were exercised in the sweep.

## Setup (2026-04-24 snapshot)

- **Kalibr**: Docker image `kalibr:noetic` (ROS Noetic, catkin build)
- **MVIS**: pre-release build of 2026-04-24, GTSAM factor graph, analytical Jacobians
- **Datasets**: 4 priority-1 datasets from the maintainers' regression suite
- **Kalibr settings**: `--bag-freq 4`, `--dont-show-report`

## Results

### euroc_mav_april/stereo (pinhole-radtan, Kalibr's canonical dataset)

| Param | Kalibr | MVIS | GT (Kalibr wiki) | K→GT | C→GT |
|---|:-:|:-:|:-:|:-:|:-:|
| cam0 fx | 459.34 | 462.08 | 458.65 | **+0.15%** | +0.75% |
| cam0 fy | 458.01 | 460.80 | 457.30 | **+0.16%** | +0.77% |
| cam0 cx | 366.82 | 371.67 | 367.22 | **-0.11%** | +1.21% |
| cam0 cy | 248.52 | 245.35 | 248.38 | **+0.06%** | -1.22% |
| cam1 fx | 458.12 | 460.56 | 457.59 | **+0.12%** | +0.65% |
| cam1 fy | 456.69 | 459.32 | 456.13 | **+0.12%** | +0.70% |
| cam1 cx | 379.26 | 382.41 | 380.00 | **-0.19%** | +0.64% |
| cam1 cy | 255.39 | 251.44 | 255.24 | **+0.06%** | -1.49% |

**Winner: Kalibr** — 0.1-0.2% from GT vs MVIS's 0.6-1.5%.

Note: GT is from Kalibr's own calibration output on this exact bag, so this
comparison is inherently biased toward Kalibr. MVIS uses a different
optimization approach (per-frame poses vs B-spline).

### mvis_t265/bag1 (pinhole-equi, T265 fisheye stereo)

| Param | Kalibr | GT (factory) | K→GT |
|---|:-:|:-:|:-:|
| cam0 fx | 282.80 | 286.64 | -1.34% |
| cam0 fy | 282.10 | 286.55 | -1.55% |
| cam1 fx | 284.54 | 286.16 | -0.57% |
| cam1 fy | 283.72 | 286.01 | -0.80% |

MVIS used different camera IDs (cam2/cam3) for this dataset in the snapshot, so
the two were not compared directly. Factory GT may not be accurate
(community-reported values).

### virig_flir/bag1 (pinhole-radtan, single Blackfly camera)

| Param | Kalibr | MVIS |
|---|:-:|:-:|
| fx | 444.15 | 451.55 |
| fy | 442.72 | 451.32 |
| cx | 370.52 | 362.00 |
| cy | 281.98 | 267.72 |

~1.6% focal length difference. No independent GT available.

Note: Kalibr's focal length initialization failed on this dataset (vanishing
point method returned NaN). Required manual focal length via stdin pipe
(`KALIBR_MANUAL_FOCAL_LENGTH_INIT` env var + echo pipe). MVIS handled
initialization automatically.

### tum_vi_standard_1024/cam1 (equidistant fisheye stereo)

**Kalibr FAILED** with `pinhole-equi` model — optimizer diverged on all attempts
(including with `--mi-tol -1` and `--bag-freq 2`).

Kalibr succeeded only with `omni-radtan` model (different projection model):

| Param | Kalibr (omni) | MVIS (equidist) | GT (equidist) |
|---|:-:|:-:|:-:|
| cam0 | xi=1.78, fu=1063 | fx=382.27 | fx=380.81 |
| cam1 | xi=1.78, fu=1056 | fx=380.99 | fx=379.29 |

Models are different — not directly comparable. MVIS matches GT to 0.4-0.5%.

## Summary

| Dataset | Kalibr | MVIS | Notes |
|---|:-:|:-:|---|
| euroc_mav_april | **0.1-0.2%** from GT | 0.6-1.5% | GT from Kalibr (biased) |
| mvis_t265 | 0.6-1.6% from factory | — | Different cam IDs |
| virig_flir | Works (manual init) | Works | No GT, ~1.6% diff |
| tum_vi_1024 | **FAILED** (equidist) | **0.4-0.5%** from GT | MVIS advantage |

## Full Batch Results (93 runs, 2026-04-24 snapshot)

**Overall: 50 OK, 43 FAIL (54 % success rate).** Superseded by the
2026-04-30 sweep (116/228, then 135/228 after the runner fixes). See
`doc/comparison_kalibr_regression.md` for the per-family table.

MVIS succeeds on all 93 datasets (plus 39 more with DS/EUCM/FOV models
that Kalibr doesn't support).

### Success/failure by dataset family

| Dataset Family | Kalibr OK | Total | Fail Rate | Failure Mode |
|---|:-:|:-:|:-:|---|
| euroc_mav (radtan stereo) | 2 | 2 | 0% | — |
| tum_rs (equi stereo) | 4 | 4 | 0% | — |
| virig_t265 (equi stereo) | 5 | 5 | 0% | — |
| mvis_t265 (equi stereo) | 4 | 5 | 20% | 1 needed omni fallback |
| mvis_t265_equidistant | 5 | 5 | 0% | (1 used omni fallback) |
| hilti_2022 (equi) | 6 | 6 | 0% | — |
| hilti_2023 (equi) | 6 | 6 | 0% | — |
| tum_vi_1024 (equi stereo) | 4 | 8 | **50%** | pinhole-equi diverges |
| tum_vi_512 (equi stereo) | 5 | 8 | **38%** | pinhole-equi diverges |
| tum_vi_raw (equi stereo) | 5 | 8 | **38%** | pinhole-equi diverges |
| tum_vi_1024_equidistant | 4 | 8 | **50%** | pinhole-equi diverges |
| tum_mic (equi stereo) | **0** | **6** | **100%** | Small tags, init fails |
| virig_flir (radtan single) | 2 | 5 | **60%** | Focal length init NaN |
| mvis_elp_single (radtan RS) | **0** | **5** | **100%** | RS single-cam init fails |
| mvis_elp (radtan RS stereo) | 1 | 5 | **80%** | RS stereo diverges |
| mvis_elp_t265 (mixed+RS+dt) | **0** | **5** | **100%** | Cross-model + RS + dt too complex |
| virig_flir_t265 (mixed) | 1 | 5 | **80%** | Cross-model, most fail |

### Key failure patterns

1. **pinhole-equi optimizer divergence** (30 runs): Kalibr's incremental batch
   optimizer fails on ~40-50% of TUM-VI equidistant datasets. Some succeed with
   omni-radtan fallback, but the model is different (not directly comparable).

2. **Rolling shutter cameras** (15 runs): All mvis_elp (single + stereo) and
   mvis_elp_t265 fail. Kalibr requires `kalibr_calibrate_rs_cameras` for RS,
   which is a separate tool with limited functionality.

3. **Focal length initialization** (8 runs): Kalibr's vanishing-point focal
   length init fails on some cameras, requiring manual stdin input.

4. **Small calibration targets** (6 runs): All tum_mic (30mm tags) fail —
   Kalibr can't initialize on small tags with fisheye distortion.

5. **Cross-model + time offset** (10 runs): All mvis_elp_t265 and most
   virig_flir_t265 fail — Kalibr doesn't support cam-cam time offset.

## Robustness Comparison

| Aspect | Kalibr | MVIS |
|---|---|---|
| **Success rate** | **54%** (50/93) | **100%** (93/93 + 39 more) |
| **pinhole-radtan stereo** | Works | Works |
| **pinhole-radtan single** | 40% fail (init) | Works |
| **pinhole-equi stereo** | ~60% OK, ~40% diverge | Works |
| **Rolling shutter** | All fail | Works (single + multi) |
| **Cross-model + dt** | All fail | Works |
| **Small tags (30mm)** | All fail | Works |
| **Focal length init** | Fails on some (needs manual) | Auto-init works |
| **Camera models** | 7 (radtan, equi, fov, omni, omni-radtan, eucm, ds) | 6 at the time of the sweep (radtan, equi, fov, ds, eucm, omni_radtan); 8 now (+ ucm, kb4) |
| **Setup** | Docker + ROS workspace | `pip install` |
| **Avg time per run** | 122s (OK runs) | 300-1800s |
| **Timing analysis** | None | Per-stage hierarchical timer |

## Key Takeaway

**MVIS is significantly more robust**: 100 % success rate vs Kalibr's
59 % (2026-04-30 sweep, 228 rows, runner fixes of 2026-05-01 included). Kalibr
fails on RS cameras (single + multi),
mixed-model+RS+dt rigs, cam-cam dt without IMU, small targets, on ~50 % of
TUM-VI equidistant collections, and on a subset of TUM fisheye lens families
across **every** model (suggesting a lens-level / detector issue, not a
Kalibr capability gap). Kalibr is slightly more accurate on its own canonical
dataset (EuRoC, where GT comes from Kalibr).

In this sweep MVIS converged on every setup involving fisheye cameras, rolling
shutter, time offset calibration or mixed hardware, where Kalibr often did not.
