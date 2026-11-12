# Testing MVIS

Four tiers, cheapest first. Run the smallest one that covers your change.

| Tier | Command | What it runs | Time | When |
|---|---|---|---|---|
| Unit | `scripts/test.sh unit` | everything except slow and integration tests | about 4 min | while coding |
| Solver | `scripts/test.sh solver` | five real solves: `imu` and `t_ga` end to end, one twin, the golden factor fit | about 1 min | with unit, in CI |
| Integration | `scripts/test.sh integration` | digital twins and a polled-camera simulation through the full solver | about 3 min | before committing a solver change |
| Full | `scripts/test.sh full` | all of `tests/` | about 20 min | before a merge |

`scripts/test.sh` pins BLAS to one thread and tests this checkout's `src/`, not the editable install. Extra arguments go to pytest: `scripts/test.sh unit -k imu`. Set `PYTHON` to choose the interpreter.

The recording-based regression suite (real recordings against baselines) and the digital-twin regression suite (twins of those recordings against known values) are run by the maintainers, and their results are published on [openmvis.com](https://openmvis.com).

## What CI runs

GitHub CI runs only the unit and solver tiers, on pull requests and on pushes to main. It does not run the slow or integration tests, so run `scripts/test.sh full` locally before a merge.

## Where tests go

`tests/<area>/test_<topic>.py`, one folder per area: `camera`, `detection`, `calibration`, `cam_imu`, `simulation`, `config`, `io`, `mvisdb`, `datasets`, `cli`, `recalib`, and `integration`.

- Shared helpers live in `tests/helpers/`. A test module does not import from another test module.
- `tests/_reference/` is a frozen copy of earlier code that the bitwise speed-up tests compare against. Do not refactor it.
- Library code belongs in `src/mvis`; tests do not import from `scripts/` except to test a script's own command line.
- Tests that need real recordings skip when the data is absent; point `MVIS_DATASETS` at your datasets to run them.

## Slow tests

`tests/slow_tests.txt` lists tests over 5 s, and `tests/conftest.py` marks them `slow`, so no test file needs a marker. The maintainers refresh the list from the junit XML of a full run (`scripts/test.sh full --junitxml=/tmp/t.xml`); a new slow test can also be added to the list by hand.

## Integration tests and digital twins

`tests/integration/test_digital_twins.py` solves two twins of real stereo cam-IMU recordings, using no images:

| Twin | Covers |
|---|---|
| `twin_gs` | global shutter, 470 Hz IMU, `t_ga` +5.3 ms |
| `twin_rs` | rolling shutter with negative readout, 200 Hz IMU, `t_ga` -7.4 ms |

A twin (`mvis.simulation.twin`) takes a real solve, perturbed with a fixed seed (`perturb`: focal up to 1 %, principal point 4 px, other intrinsics 5 %, extrinsic rotation 10 mrad and lever 5 mm per axis, td 1 ms, `t_ga` 0.5 ms), as the truth and keeps its frame times, corner ids, IMU stamps and solve settings. The no-noise checks are the precise ones, because any error left is bias. Limits sit just above measured errors, so a failure means something changed. They also check that LM iterated, because `ok` alone does not show that it solved.

The noise is Gaussian and the twins have no lens or timing model errors, so they test the estimator, not a real recording.

The perturbation keeps real device calibrations out of the shipped fixtures (`tests/data/twins/`), and a solve has to find values nobody gave it. Making a new fixture needs the recording; the tests need only the fixture.

`tests/integration/test_sensor_clock_sim.py` runs a simulated polled camera through the sensor-clock path: both passes, the sidecar written and read back, and a solve from the map.

## Golden test

`tests/cam_imu/test_golden_camimu_multi.py` compares the board-mode solver against a stored snapshot, within a measured noise floor. It covers the board path only. `GOLDEN_EXACT=1` asks for bitwise equality, but currently passes only some of the time, even with no code change: one process gets one of two results, up to 5.9e-7 apart in `err_final`. Do not use it as a bitwise proof until that is explained.
