# Kalibr vs MVIS — Regression Suite Comparison

> **Snapshot of 2026-05-01.** A full 228-row sweep of the maintainers'
> regression recordings (sweep of 2026-04-30, runner fixes of 2026-05-01) with
> the MVIS pre-release build of that date. The recordings, the baselines and the
> Kalibr batch runner are not part of this repository, and MVIS defaults have
> changed since (see `CHANGELOG.md`), so treat the figures as a dated record.
> Current MVIS results on real recordings are on
> [openmvis.com](https://openmvis.com).

**Result: Kalibr 135/228 OK (59 %); MVIS 228/228 OK.**
(After the runner fixes of 2026-05-01: docker `-i` + manual-focal retry-prompt
pipe + `--approx-sync 0.05` for multi-cam + RS-tool config + auto-mono16. Of the
175 GS-runnable rows, 124 OK; plus 11/11 RS = 135 total.)

This document records **what Kalibr can and cannot do on the maintainers'
regression recordings**, and how its results compare to MVIS on the runs both
tools complete. It complements `doc/comparison_kalibr.md` (which holds the
2026-04-24 numerical snapshot).

## TL;DR

| Capability | Kalibr | MVIS |
|---|:-:|:-:|
| GS multi-cam radtan | ✓ | ✓ |
| GS multi-cam equidistant | ⚠ ~60% (40% diverge) | ✓ |
| omni-radtan / MEI | ✓ | ✓ (experimental) |
| FOV / DS / EUCM | ✓ (`pinhole-fov`, `ds-none`, `eucm-none`) | ✓ — DS paper-validated against Usenko 2018 (within 2 % on all 5 paper lenses) |
| RS **single-cam** | ⚠ converges, but readout 4× wrong (non-physical) | ✓ (RS readout matches across runs) |
| RS **multi-cam** | ✗ no native tool | ✓ |
| Mixed distortion models in one rig | ⚠ CLI accepts it, never validated | ✓ |
| **Cam-to-cam time offset** | ✗ only cam→IMU dt | ✓ |
| Cam-IMU time offset | ✓ (`kalibr_calibrate_imu_camera`) | ✓ (`td`, within 1 ms of Kalibr on TUM-VI) |
| Auto focal-length init | ⚠ NaN on ~8 datasets | ✓ |
| Small calibration tags (30 mm) | ✗ all `tum_mic` fail | ✓ |
| No-FOV-overlap multi-cam | ✗ requires chain | ✓ |

`✓` works, `⚠` partially / unknown / needs manual intervention, `✗` not supported.

## Capability summary

### What Kalibr can do
- **GS pinhole-radtan** single- and multi-cam — works reliably (EuRoC canonical).
- **GS pinhole-equi** stereo — works on ~60% of TUM-VI-style fisheye stereo.
- **omni-radtan** (Mei model) — works as a fallback when pinhole-equi diverges.
- **Heterogeneous distortion per camera** via `--models a b c d` — accepted by
  the CLI; validated only on uniform-family rigs so far.
- **RS single-camera** — via the separate `kalibr_calibrate_rs_cameras` tool
  (measured below: converges on 3 of 5 ELP bags, readout biased +49 %).
- **Cam-to-IMU temporal calibration** — `kalibr_calibrate_imu_camera`
  estimates per-cam `td` against an IMU clock.
- **AprilGrid (88 mm)** and **chessboard** boards.
- **Manual focal-length init** via stdin pipe when the vanishing-point
  heuristic returns NaN.

### What Kalibr cannot do
- **Multi-camera rolling shutter** — there is no `kalibr_calibrate_rs_cameras`
  multi-cam mode. All `mvis_elp` (RS stereo), `mvis_elp_t265` (RS + cross-model
  + dt), and `tum_rs_*` multi-cam runs fail or have to be approximated as GS.
- **Camera-to-camera time offset** — Kalibr only models cam→IMU dt. On rigs
  where two non-synced cameras both need a `td` against each other (e.g.
  `virig_flir_t265`, no IMU), there is no native mechanism. The IMU-chain
  workaround works only when an IMU is present (e.g. `mvis_elp_t265`) and
  introduces extrinsic error.
- DS / EUCM / FOV **are** supported: `kalibr_calibrate_cameras --help` lists `pinhole-fov`, `ds-none`, `eucm-none`. An earlier claim to the contrary came from a model map in the batch runner that only mapped radtan/equi; fixed 2026-04-30 before the sweep.
- **Small calibration targets** (30 mm AprilTags, e.g. `tum_mic`) — init
  diverges on every collection.
- **No-FOV-overlap rigs** — Kalibr needs the cameras to share field-of-view
  for chained extrinsic estimation.
- **Equidistant fisheye optimizer convergence** — diverges on ~40% of TUM-VI
  collections without a model swap.
- **Two-pass outlier rerun** — single-pass robust kernel only.
- **Auto focal-length init** on certain wide-FOV cameras — needs manual input.

## Per-family measured results (228-row sweep, 2026-04-30)

| Family | Runs | Kalibr OK | Kalibr FAIL | Notes / failure mode |
|---|:-:|:-:|:-:|---|
| `euroc_mav_april/chess` | 3 | 3 | 0 | canonical Kalibr dataset |
| `hilti_2022 / 2023_handheld` | 12 | 12 | 0 | clean equi stereo + monos |
| `mvis_t265` (radtan) | 5 | 5 | 0 | — |
| `mvis_t265_equidistant` | 5 | 5 | 0 | — |
| `mvis_t265_ds` | 5 | 5 | 0 | DS works |
| `mvis_t265_eucm` | 5 | 5 | 0 | EUCM works |
| `mvis_t265_fov` | 5 | 4 | 1 | FOV mostly works |
| `virig_t265` (equi stereo) | 5 | 5 | 0 | — |
| `tum_vi_standard_1024` | 8 | 4 | 4 | pinhole-equi diverges on ~½ |
| `tum_vi_standard_512` | 8 | 5 | 3 | same |
| `tum_vi_raw` (1024) | 8 | 4 | 4 | same |
| `tum_vi_1024_equidistant/ds/eucm` | 24 | 12 | 12 | 50 % across all 3 models |
| `tum_vi_1024_fov` | 8 | 2 | 6 | FOV worse than equi/DS/EUCM |
| `tum_fisheye_bm2820 / bm4018s118 / gopro` (×4 models) | 36 | 36 | 0 | **DS/EUCM/FOV/equi all OK** |
| `tum_fisheye_bf2m2020s23 / bf5m13720` (×4 models) | 24 | 0 | 24 | every model fails on these two lenses (lens-level issue, not model) |
| `tum_rgbd_bag` GS attempts | 6 | 3 | 3 | small-board RGB OK, IR FAIL on first try (legacy GS sweep, before RS routing) |
| `tum_rgbd_bag` RS-tool (run 2026-04-30) | 6 | **6** | 0 | 3/3 small RGB OK directly. 3/3 IR rows fail until pre-converted with a mono16 converter (mono16 → mono8 with per-frame min-max normalization), then 3/3 converge. RGB readout +13.9 % bias, stddev 3.7× worse. IR readouts are wildly inconsistent per-bag (−55 %, +61.5 %, −1.2 %) — the mean lands within 2.8 % of MVIS by accident; stddev is 2× worse than MVIS (47 % vs 24 %). |
| `tum_rs / tum_rs_raw` | 4 | 4 | 0 | TUM-VI is GS-mode at acquisition |
| `tum_mic` (30 mm tags) | 6 | 0 | 6 | small-tag init fails |
| `tum_event` (image dirs, no bag) | 4 | 0 | 4 | not bag-based, Kalibr can't ingest |
| `virig_flir` (radtan single) | 5 | 1 | 4 | focal-length init fails on most |
| `virig_flir_t265` (mixed, no IMU) | 5 | 1 | 4 | no cam-cam dt support |
| `mvis_elp_single` (RS single, GS tool) | 5 | 0 | 5 | GS optimizer diverges to NaN on RS data. Skipped by the GS runner from 2026-04-30; routed to the RS tool. |
| `mvis_elp_single` (RS tool, sweet-spot config) | 5 | **3** | 2 | `kalibr_calibrate_rs_cameras --bag-freq 10 --frame-rate 30`. 3/5 converge; 2/5 (bag2, bag5) crash with autodiff overflow even at the sweet spot — non-deterministic by bag. **Per-bag intrinsics within 1–2 % of MVIS; cx biased −3.6 % toward image center; RS readout mean 22.55 ms vs MVIS 15.11 ms (+49 % bias); RS readout stddev 12.12 % vs MVIS 1.83 % (10× worse repeatability).** Useful as a sanity check, not authoritative. |
| `mvis_elp` (RS stereo) | 5 | 0 | 5 | no multi-cam RS tool |
| `mvis_elp_t265` (mixed + RS + dt) | 5 | 0 | 5 | three-axis combo defeats Kalibr (omni-radtan fallback also fails) |
| `fisheye_dataset / gopro_* / penncosyvio_*` (chessboard) | 16 | 0 | 16 | all FAIL — likely topic naming / bag-freq / detector issue (follow-up) |

**Aggregate: Kalibr 135/228 OK (59 %); MVIS 228/228 OK.**
(124 GS OK out of 175 attempted + 11 RS OK out of 11. Pre-wrapper-fix snapshot
was 116/228 = 50.9 %; the +19 came from docker `-i` + manual-focal retry-prompt +
multi-cam `--approx-sync` + auto-mono16 + DS detection-bug fix.)
(Counts post-classifier: 175 runnable + 53 structurally skipped. The 112
GS failures within the 175-runnable subset break down below.)

## GS failure mode breakdown — 56 true failures after routing (2026-04-30)

After applying the classifier (skips RS / image-dir rows) and RS-tool
routing, **56 of 175 GS-attempted rows fail**. Definitive breakdown:

| Reason | n | Recoverable? | Root cause |
|---|---:|---|---|
| Ultra-wide fisheye optimizer divergence (Kalibr reports "DLT 6-point" or "Did not converge") | 41 | ❌ probed bf=4/8/15/25 + window slicing | TUM-VI cam3–6 (any model) + TUM-fisheye bf2m2020s23 / bf5m13720 (any model). Wide-FOV lens-edge basin where Kalibr's optimizer can't converge; MVIS succeeds via different init. |
| FOV model `areParametersValid` assertion | 8 | ❌ model can't fit | bf2m2020s23_fov, bf5m13720_fov, tum_vi_1024_fov. FOV projection inadequate for these lenses. |
| ~~`nan_intrinsics` (5 rows)~~ | ~~5~~ | ✅ **FIXED 2026-05-01** | **Root cause: docker `run` was missing `-i`, so the wrapper's piped focal-length value never reached Kalibr's `std::cin >> input_guess`.** Kalibr's vanishing-point auto-init then read uninitialized memory (`1.51219e-315`) → NaN. virig_flir bag2 was the only one where auto-init happened to succeed. Fix: add `-i` to docker invocation; pipe focal value × 8 (for retry prompts). Recovery: virig_flir 1/5 → 5/5 OK; virig_flir_t265 1/5 → 4/5 OK. |
| Kalibr `plotGraph` KeyError on omni-radtan fallback | 1 | ⚠ Kalibr internal bug | virig_flir_t265/bag4. After fallback retry, no overlap edges → `MulticamGraph.plotGraph()` crashes with `KeyError: 'Attribute does not exist'`. |
| FOV `focal_init_failed` (vanishing-point NaN, FOV ignores stdin pipe) | 2 | ⚠ needs Kalibr patch | bf2m2020s23_fov/bag2 + tum_vi_1024_fov/cam5. (FOV-specific stdin handling differs from pinhole; -i fix doesn't help here.) |

### Probes tried (validated negative)

| Approach | Test | Result |
|---|---|---|
| Higher bag-freq (8, 15, 25) | bf2m2020s23/bag1, tum_vi_standard_1024/cam1 | "Did not converge in maxIterations" at every freq |
| `--bag-from-to` window slicing | TUM-VI 1024 cam1 | Same divergence |
| Higher fx seed via stdin | virig_flir bag2 | bag2 already OK; doesn't help bag1/3/4 |

### Categorization superseded above (legacy from before classifier+routing)

The original 112-row breakdown below was based on stale results that
included rows the classifier now correctly routes elsewhere:
- 19 RS rows now routed to the RS tool (was `non-broadcastable shape (1,3)`).
- 3 mono16 IR rows now auto-converted (was `no_corners_extracted`).
- ~32 image-dir rows (tum_event, tum_mic, penncosyvio, fisheye_dataset,
  gopro_*, fisheye*_radtan) now classified `skip_not_a_bag`.
- 1 virig_flir bag2 was incorrectly marked failed by an over-eager `"nan"`
  substring check; actual camchain is valid (fx=443.69 px).

Total recovered by classifier + bypasses + sanity checks: **~50 rows**.

### Categorized initial 112-row table (legacy)

| Failure mode | n | Root cause | Recoverable? |
|---|---:|---|---|
| **OpenCV DLT 6-point** (`cvFindExtrinsicCameraParams2`) | 41 | `initializeIntrinsics` requires ≥ 6 board corners per frame; only 4–5 visible at default bag-freq=4 on `bf2m2020s23`/`bf5m13720` | ⚠ Higher bag-freq helps some; BF2M 220°-fisheye then hits "Optimization diverged… Do the models fit the lenses well?" — lens-edge |
| **GS tool ran on RS rows** (legacy results) | 19 | `non-broadcastable shape (1,3)` because GS optimizer can't produce an RS readout | ✅ Classifier now routes to the RS tool |
| **Aslam runtime errors** (FOV / pinhole init) | 12 | `assert(_w valid)` on FOV init; `assert(input_guess > 0)` on chained init | ⚠ Manual focal seeding via stdin pipe works for `pinhole-*`; FOV / chained models ignore it |
| **`tum_mic` 30 mm tags** | 6 | Below detector threshold | ❌ Board-physical limitation |
| **`tum_event` / image-dirs** | 4 | Bag-only Kalibr | ✅ Classifier skips |
| **`virig_flir` / `*_fov` focal init** | 4 | Vanishing-point heuristic returns NaN; FOV ignores stdin | ⚠ Needs Kalibr patch |
| **No corners (mono16 IR)** | 3 | Same loader limitation as RS — a mono16 converter already exists | ✅ Apply mono16 conversion in GS path too |
| **Diverged-to-NaN** (TUM-VI pinhole-equi) | 8 | Optimizer divergence on equidistant fisheye | ⚠ omni-radtan fallback in place; remainder deeply diverges |
| **Other / unknown** | 15 | Mixed | — |

**Recoverable today (no Kalibr code changes)**: ~35 rows
- 19 RS rows: classifier routes to the RS tool (already wired)
- 4 image-dir rows: classifier skips with `skip_not_a_bag` (already wired)
- 3 mono16 IR rows: applying `auto_convert_mono16` in GS path
- ~9 of the DLT-6-point rows convergeable with higher bag-freq

**Recoverable only with Kalibr patches**: ~16 rows (FOV / chained init seeding)

**Genuine Kalibr capability gaps**: ~61 rows
- BF2M2020S23 / BF5M13720 ultra-wide fisheye lenses (24 rows): models don't fit
- 30 mm AprilTags (6 rows): detector threshold
- Deep TUM-VI equi divergence (8 rows): optimizer instability

The 2026-04-24 snapshot's 50/93 = 54 % is statistically consistent with this
larger sweep. Net new evidence:
- DS/EUCM/FOV **do work** for 3 of 5 TUM-fisheye lens families (`bm2820`,
  `bm4018s118`, `gopro` → 36/36).
- Two TUM lens families (`bf2m2020s23`, `bf5m13720`) fail across **every**
  model — not a Kalibr capability gap, lens-level (worth a follow-up).
- Chessboard families need a board-loader sanity check.

**Structural capacity check**: with the `MODEL_MAP` and target-YAML fixes
applied 2026-04-30, **all 228 manifest rows are structurally runnable** in
Kalibr (model + board + topic resolved). The 112 failures are runtime
failures (divergence, focal-init, small tags, RS-multi, no-cam-cam-dt),
not "Kalibr can't ingest this row".

## How comparisons are reported

For runs both tools complete:
- Per-camera `fx, fy, cx, cy` and distortion coefficients side-by-side.
- ΔP95 vs MVIS's pass-2 result.
- Where ground truth exists (EuRoC, TUM-VI factory, T265 factory): %
  deviation from GT for each tool.
- For RS runs: readout-time estimate side-by-side (Kalibr only when
  `kalibr_calibrate_rs_cameras` succeeds).
- For cross-model rigs: extrinsic chain accuracy (Kalibr) vs joint solve
  (MVIS), reported as relative-rotation and baseline drift.

The numerical body of the comparison lives in `doc/comparison_kalibr.md`.

## Related docs
- `doc/comparison_kalibr.md` — the dated numerical snapshot.
- `doc/comparison_basalt.md` — equivalent doc for Basalt.
