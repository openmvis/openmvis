# TUM-VI raw (dataset-calib-imu1)

Simulator example `tum_vi_raw`: 9691 base-IMU poses over 48.4 s at 200 Hz, 2 camera(s), 1 IMU(s), apriltag board 6 x 6.

```python
from mvis.simulation.examples import load_example, simulate
ex = load_example("tum_vi_raw")
data = simulate(ex, seed=0)   # cams, t_imu, w_m, a_m, aux_imus
```

## Source

Made from the public TUM-VI calibration recording `dataset-calib-imu1` (raw 1024 x 1024, CC BY 4.0, https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset).

| Camera | Model | Image | Rate (Hz) | Shutter |
|---|---|---|---|---|
| cam0 — cam0 (global shutter) | equidistant | 1024 x 1024 | 20.0 | global |
| cam1 — cam1 (global shutter) | equidistant | 1024 x 1024 | 20.0 | global |

Base IMU (imu0): 199.36 Hz, model calyx.

## How it was made

Taken from the digital twin `tum_vi_raw` the maintainers' twin regression builds from the recording's real solve (its truth).
The twin's truth (cameras, IMU, board pose and shape, IMU-filled trajectory) is copied as it is. Camera and IMU rates are the medians of the recording's stamps; the simulator samples them uniformly. The noise is the twin's: the corner sigma and the IMU densities that reproduce the real residuals.

The truth is not a real device calibration: it is moved off the solve it came from by the fixed seeded perturbation of `mvis.simulation.twin.perturb` (decision D3; `truth.perturbation` in `setup.yaml` has the seed, the name it is mixed with and the half-widths: up to 1 % focal, 4 px principal point, 5 % of each distortion coefficient, 10 mrad rotation and 5 mm lever per axis, 1 ms td, 0.5 ms t_ga).

`trajectory.txt` samples the truth spline at 200 Hz, time shifted to start at 0; `load_example` fits a 5 ms B-spline through it. The maintainers regenerate it with their example export script, which is not part of the public tree.

## Licence

CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

If you use it, please cite Schubert et al., "The TUM VI Benchmark for Evaluating Visual-Inertial Odometry", IROS 2018.
