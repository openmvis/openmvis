# BLAS threads: the `mvis` CLI pins one thread

*Code: `src/mvis/cli/blas_threads.py`, applied by `src/mvis/cli/__init__.py`;
tests `tests/cli/test_cli_blas_threads.py`.*

## What

The `mvis` command (console script or `python -m mvis.cli.main`, and therefore
every solve a batch runner spawns through it) sets

    OPENBLAS_NUM_THREADS=1  OMP_NUM_THREADS=1  MKL_NUM_THREADS=1  OPENBLAS_MAIN_FREE=1

before numpy, OpenCV or scipy are imported. `mvis.cli.__init__` runs as the first
statement of the CLI import chain (`mvis/__init__.py` imports nothing); OpenBLAS
reads the variables when its shared object is loaded, so nothing later can pin
them (that is also why this cannot be a flag parsed in `main()`: `mvis.cli.main`
imports numpy at module level). The outcome is logged once at INFO:
`BLAS threads: pinned to 1 (default; MVIS_BLAS_THREADS=<n> or 0 to change)`.

| environment | effect |
|---|---|
| nothing set | pinned to 1 |
| `MVIS_BLAS_THREADS=n` | pinned to n (applied even when the variables below are exported: it is the mvis-specific request) |
| `MVIS_BLAS_THREADS=0` or empty | not pinned, BLAS default (all cores) |
| `OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS`, `OMP_NUM_THREADS` or `MKL_NUM_THREADS` exported, `MVIS_BLAS_THREADS` unset | not pinned: a value the user set is never overridden. OpenBLAS honours `OMP_NUM_THREADS` too, so an exported OMP value counts as a thread choice even without `OPENBLAS_NUM_THREADS` |
| numpy / scipy / cv2 already imported when `mvis.cli` loads | not pinned, environment untouched (the pin could not take effect) |

`OPENBLAS_MAIN_FREE` is only `setdefault`-ed (it keeps OpenBLAS from changing the
main thread's CPU affinity); an invalid `MVIS_BLAS_THREADS` is reported in the log
line and ignored, never an error.

**Only the CLI pins.** `import mvis` as a library (notebooks, other programs,
`pytest`) leaves the environment alone: the thread policy of a host process is its
own. Tests that need one thread pin in-process through `tests/_blas_threads.py`
(ctypes into the loaded OpenBLAS builds; `GOLDEN_EXACT`).

## Why

The cam-IMU and cam-only solves are per-factor Python on 3×3 / 2×N matrices;
numpy's bundled OpenBLAS still wakes one worker per core on every matmul and the
workers spin. Measured on one VI-Rig cam-IMU solve (about 4 min) on a 32-core
host:

| BLAS threads | wall | user | sys | two identical runs |
|---|---|---|---|---|
| OpenBLAS default (32) | 333–336 s (under load) | ≈ 615 s | **≈ 590 s** (spinning) | differ: 6032 dump entries, ≤ 1e-13 (intr), `err_final` 1e-10 |
| 1 | same within the load noise (243–265 s idle) | ≈ wall | 2–4 s | bitwise identical |

So the pin costs no wall time, halves the CPU per solve (parallel batches), and
makes a solve reproducible run to run — which every bitwise comparison between two
solves relies on. gtsam itself links neither TBB nor OpenMP nor BLAS in a pip
install, so the C++ side is sequential either way.

An mvis process maps three OpenBLAS builds (numpy's `libopenblas64_`, opencv's
`libopenblasp`, scipy's `libscipy_openblas`); the environment variables pin all of
them.

## How it was checked

The same solve run through the console script with no thread variables set and
with `OPENBLAS_NUM_THREADS=1` exported: the `MVIS_DUMP_STATE` dumps compared
bitwise (`compare_states` in `mvis.calibration.state_dump`, exact mode), plus
`/usr/bin/time -v` wall / user / sys against the table above.
