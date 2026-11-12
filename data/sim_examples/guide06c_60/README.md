# Calibration motion guide06c_60 (AprilGrid, 60 s)

Simulator example `guide06c_60`: 12001 base-IMU poses over 60.0 s at 200 Hz, 1 camera(s), 1 IMU(s), apriltag board 6 x 6.

```python
from mvis.simulation.examples import load_example, simulate
ex = load_example("guide06c_60")
data = simulate(ex, seed=0)   # cams, t_imu, w_m, a_m, aux_imus
```

## Source

Made from the openmvis.com Guides calibration motion `guide06c_60` (a hand-held 60 s motion built from our real AprilGrid take 06) on the Looper Insight-9 cam0 + IMU twin.

| Camera | Model | Image | Rate (Hz) | Shutter |
|---|---|---|---|---|
| cam0 — Looper Insight-9 left camera | equidistant | 544 x 640 | 20.0 | global |

Base IMU (Looper Insight-9 IMU): 400.0 Hz, model calyx.

## How it was made

The motion is the calibration-motion study's designed cam0 path (board frame, 200 Hz), moved to the IMU with the perturbed camera-to-IMU pose, so the camera follows the designed path exactly. The board stands vertical (world z up). Segments: Hold still 0.0-3.0 s; 1 Left-right 3.0-19.7 s; 2 Up-down 19.7-31.8 s; 3 Back-forth + roll 31.8-46.8 s; 4 Loop + roll 46.8-57.0 s; Hold still 57.0-60.0 s.
Rig: cam0 and IMU of our Looper Insight-9 (rates, image size, model and the IMU noise the study used), with the study's injected truth before the perturbation: td 3.0 ms, t_ga 1.5 ms, calyx IMU scale 0.3-0.5 %, seed-0 biases, 0.5 px corners. The study also cut corners at 75 deg incidence, by motion blur at the exposure (`exposure_s`) and by a 5 % drop-out; `simulate` does not.

The truth is not a real device calibration: it is moved off the solve it came from by the fixed seeded perturbation of `mvis.simulation.twin.perturb` (decision D3; `truth.perturbation` in `setup.yaml` has the seed, the name it is mixed with and the half-widths: up to 1 % focal, 4 px principal point, 5 % of each distortion coefficient, 10 mrad rotation and 5 mm lever per axis, 1 ms td, 0.5 ms t_ga).

`trajectory.txt` samples the truth spline at 200 Hz, time shifted to start at 0; `load_example` fits a 5 ms B-spline through it. The maintainers regenerate it with their example export script, which is not part of the public tree.

## Licence

CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/); commercial use by request (licensing@openmvis.com).

The motion and rig are built on our Looper Insight-9 recordings, which carry the same licence.
