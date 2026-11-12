# Simulator examples

Real motions with a full rig setup, ready for the simulator (`mvis.simulation.examples`):

```python
from mvis.simulation.examples import load_example, simulate
ex = load_example("virig_4imus_3cams")
data = simulate(ex, seed=0)   # cams, t_imu, w_m, a_m, aux_imus for run_cam_imu_calibration_multi
```

Each folder has `trajectory.txt` (`t theta_ItoG(3) p_IinG(3)`, the base IMU in a gravity-aligned world at
200 Hz, `R_ItoG = Exp(theta)`), `setup.yaml` (cameras: model, intrinsics, image size, rate, shutter and
readout, camera-to-IMU pose and td, corner noise; IMUs: rate, calyx intrinsics, noise densities, t_ga,
biases, aux poses and td; board layout, pose and measured heights) and a `README.md` (source, licence,
how it was made). The truth is never a real device calibration: it is the real solve moved off by the
fixed seeded perturbation (`mvis.simulation.twin.perturb`, decision D3), recorded under
`truth.perturbation`.

| Example | Source | Cameras | IMUs | Licence |
|---|---|---|---|---|
| `virig_4imus_3cams` | VI-Rig 4 IMUs + 3 cameras, `april_01` twin | radtan + 2 fisheye, global | 4 | CC BY-NC 4.0 |
| `elp_t265` | VI-Rig ELP + T265, `imu_mixed_bag1` twin | fisheye global + ELP rolling | 2 | CC BY-NC 4.0 |
| `tum_vi_raw` | TUM-VI raw `dataset-calib-imu1` twin | 2 fisheye, global | 1 | CC BY 4.0 |
| `guide06c_60` | calibration motion `guide06c_60`, AprilGrid, Looper Insight-9 | 1 fisheye, global | 1 | CC BY-NC 4.0 |
| `guide60b` | calibration motion `guide60b`, chessboard, Looper Insight-9 | 1 fisheye, global | 1 | CC BY-NC 4.0 |
| `twin_gs` | regression fixture `tests/data/twins/twin_gs.npz` (in the package) | 2 fisheye, global | 1 | GPL-3.0-only |
| `twin_rs` | regression fixture `tests/data/twins/twin_rs.npz` | 2 fisheye, rolling | 1 | GPL-3.0-only |

The wheel ships `twin_gs` only (`src/mvis/simulation/data/examples/`); the rest are in a source checkout.
Licences are also in `REUSE.toml` and each `setup.yaml` header: the VI-Rig and Looper Insight-9 derived
examples carry the CC BY-NC 4.0 of the recordings they are built on (commercial use by request,
licensing@openmvis.com). The maintainers regenerate the examples from their digital twins, the
regression fixtures and the calibration-motion study with an export script that is not part of the
public tree; the examples themselves are the files to use.

The pose-only trajectories the tests use (`calib_3D_v0`, `calib_3D_v28`, `calib_3D_v28_1axis`) are in
`tests/data/sim_trajectories/`; `sine_3d` is in the package (`mvis.simulation.trajectories`).
