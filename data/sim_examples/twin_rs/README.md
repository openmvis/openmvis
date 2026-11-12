# Regression twin twin_rs (rolling shutter)

Simulator example `twin_rs`: 1329 base-IMU poses over 6.6 s at 200 Hz, 2 camera(s), 1 IMU(s), apriltag board 6 x 6.

```python
from mvis.simulation.examples import load_example, simulate
ex = load_example("twin_rs")
data = simulate(ex, seed=0)   # cams, t_imu, w_m, a_m, aux_imus
```

## Source

Made from the digital-twin regression fixture `tests/data/twins/twin_rs.npz` (rolling-shutter stereo fisheye, 200 Hz IMU, 7 s window).

| Camera | Model | Image | Rate (Hz) | Shutter |
|---|---|---|---|---|
| cam0 | equidistant | 1920 x 1080 | 29.94 | rolling (readout -26.3 ms) |
| cam1 | equidistant | 1920 x 1080 | 29.94 | rolling (readout -26.3 ms) |

Base IMU: 199.08 Hz, model calyx.

## How it was made

Exported from the shipped fixture `tests/data/twins/twin_rs.npz`.
The twin's truth (cameras, IMU, board pose and shape, IMU-filled trajectory) is copied as it is. Camera and IMU rates are the medians of the recording's stamps; the simulator samples them uniformly. The noise is the twin's: the corner sigma and the IMU densities that reproduce the real residuals.

The truth is not a real device calibration: it is moved off the solve it came from by the fixed seeded perturbation of `mvis.simulation.twin.perturb` (decision D3; `truth.perturbation` in `setup.yaml` has the seed, the name it is mixed with and the half-widths: up to 1 % focal, 4 px principal point, 5 % of each distortion coefficient, 10 mrad rotation and 5 mm lever per axis, 1 ms td, 0.5 ms t_ga).

`trajectory.txt` samples the truth spline at 200 Hz, time shifted to start at 0; `load_example` fits a 5 ms B-spline through it. The maintainers regenerate it with their example export script, which is not part of the public tree.

## Licence

GPL-3.0-only, as the regression fixture it comes from (the repository licence).
