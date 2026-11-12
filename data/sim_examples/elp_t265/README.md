# ELP + T265 (imu_mixed_bag1), rolling shutter

Simulator example `elp_t265`: 15913 base-IMU poses over 79.6 s at 200 Hz, 2 camera(s), 2 IMU(s), apriltag board 6 x 6.

```python
from mvis.simulation.examples import load_example, simulate
ex = load_example("elp_t265")
data = simulate(ex, seed=0)   # cams, t_imu, w_m, a_m, aux_imus
```

## Source

Made from our VI-Rig recording `4imus_4cams_01` ([calibench-vi/datasets](https://huggingface.co/datasets/calibench-vi/datasets), gated, CC BY-NC 4.0).

| Camera | Model | Image | Rate (Hz) | Shutter |
|---|---|---|---|---|
| cam0 — T265 fisheye 1 (global shutter) | equidistant | 848 x 800 | 30.0 | global |
| cam1 — ELP left (rolling shutter) | equidistant | 640 x 480 | 29.62 | rolling (readout 16.6 ms) |

Base IMU (T265 IMU (base)): 200.14 Hz, model calyx; aux `xsens` 399.84 Hz.

## How it was made

Taken from the digital twin `mvis_elp_t265` the maintainers' twin regression builds from the recording's real solve (its truth).
The twin's truth (cameras, IMUs, board pose and shape, IMU-filled trajectory) is copied as it is. Camera and IMU rates are the medians of the recording's stamps; the simulator samples them uniformly. The noise is the twin's: the corner sigma and the IMU densities that reproduce the real residuals.

The truth is not a real device calibration: it is moved off the solve it came from by the fixed seeded perturbation of `mvis.simulation.twin.perturb` (decision D3; `truth.perturbation` in `setup.yaml` has the seed, the name it is mixed with and the half-widths: up to 1 % focal, 4 px principal point, 5 % of each distortion coefficient, 10 mrad rotation and 5 mm lever per axis, 1 ms td, 0.5 ms t_ga).

`trajectory.txt` samples the truth spline at 200 Hz, time shifted to start at 0; `load_example` fits a 5 ms B-spline through it. The maintainers regenerate it with their example export script, which is not part of the public tree.

## Licence

CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/); commercial use by request (licensing@openmvis.com).

If you use it, please cite Yang, Geneva, Huang, "Multi-visual-inertial system: Analysis, calibration, and estimation", IJRR 2024, doi:10.1177/02783649241245726.
