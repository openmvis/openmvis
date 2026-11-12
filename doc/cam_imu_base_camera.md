# Cam-IMU: choosing the base camera and trajectory rate

In the joint cam-IMU solve, the **base camera** (`cam_topics[0]` / the first camera
in the prior) defines the **trajectory clock**: pose states `X(k)` are placed at the
base camera's frames, decimated to `state_hz`. Every other sensor is calibrated
*relative to this trajectory*. So the base camera's quality and the trajectory
density directly set how well the **absolute** camera/IMU→base offsets resolve.

## Rule: the base camera must be the cleanest sensor

Pick the camera with the **global shutter, highest frame rate, and lowest latency**
as `cam0`. A poor base camera (rolling shutter, low rate, or transport latency)
makes the trajectory clock noisy and injects a *coherent global bias* into the
**absolute** lever arms and time offsets of every sensor.

### Evidence (mvis_4imus_4cams, 4 cams + 4 IMUs)

The ELP is USB rolling-shutter, 25 Hz, ~135 ms acquisition latency; the t265 is
global-shutter, 30 Hz, ~0 latency. Re-running with each as the base, and sweeping
`state_hz` 20 → 0 (every frame):

| base camera | 20→0 lever Δ | 20→0 td Δ | reproj (dense) |
|---|--:|--:|--:|
| ELP (RS, 25 Hz, 135 ms lag) | ~15 mm (not converged) | ~6 ms | 1.86 px |
| **t265 (global, 30 Hz, ~0 lag)** | **~6 mm** | **~0.1–0.8 ms** | **1.38 px** |

A clean base converges the time offset to sub-ms and roughly halves the lever
residual, at lower reprojection error.

## What is / isn't sensitive

- **Absolute** offsets (each sensor ↔ base IMU: lever *and* td) are weakly observed.
  They absorb the trajectory-discretization and base-camera-clock choice as a
  *coherent shift on all sensors at once*: ~44 mm / ~17 ms going 8→20 Hz, ~15 mm /
  ~6 ms going 20→0 with the ELP base; ~26 mm / ~4 ms just from switching the base
  camera. Report them with this caveat.
- **Relative** calibration (sensor↔sensor: stereo baselines, inter-sensor td) is
  **converged and robust** — stable to ~1 mm / ~0.1 ms across every state rate and
  base-camera choice.

## Recommended settings

- **Base camera:** the cleanest global-shutter / high-rate / low-latency camera.
- **`state_hz`:** keep the trajectory dense — 20 Hz is near-converged on a good base
  (td < 1 ms to every-frame); `0` (every base-cam frame) is the dense ideal.
- **Per-camera `factor_hz`** (`camera_input.cameras`): base camera tracks `state_hz`
  (omit `factor_hz`); aux cameras can use fewer vision factors (e.g. 8 Hz).
- **Per-camera shutter** (`camera_input.cameras[i].estimate_readout`): set per camera
  on mixed-shutter rigs (rolling cams `true`, global `false`).

The VI-Rig cam-IMU configs in `examples/configs/tga_mvis/` (multi-camera,
multi-IMU) follow all of the above.

## Note: why the ELP has "~135 ms acquisition latency" (Oct 2026)

The ELP's large, variable delay is a driver artefact, not the camera: usb_cam 0.3.6
publishes from a fixed 25 Hz `ros::Rate` loop while the sensor free-runs at
29.62 Hz, stamps after dequeue + MJPEG decode, and dequeues the oldest of four
buffers. The stamps are regular but each frame's capture is 97-134 ms earlier,
varying as a sawtooth over one sensor period. Use `vision_pose: imu` (the default:
each corner's pose is the anchor state propagated through the raw IMU samples to
its capture time) with `camera_input.cameras[i].retime_sensor_clock: true` for the
ELP (a first pass solves each frame's capture offset and moves its stamp onto the
sensor's own clock); a fixed driver makes it unnecessary.
