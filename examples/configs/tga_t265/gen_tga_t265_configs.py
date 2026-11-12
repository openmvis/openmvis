# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
# SPDX-License-Identifier: GPL-3.0-only
"""Write the t265-only cam-IMU configs of tga_t265, tga_t265_4cams and tga_t265_5imus.

  tga_t265/        cam_imu_t265_{01..10}.yaml       mvis_4imus_3cams april bags 01-10
                   cam_imu_t265_01_notga.yaml       bag 01 with t_ga off
                   cam_imu_t265_01_tdanchor.yaml    bag 01 with sigma_td_prior 0.5 ms
  tga_t265_4cams/  cam_imu_t265_{01..15}.yaml       mvis_4imus_4cams bags 01-15
                   cam_imu_t265_01_notga.yaml       bag 01 with t_ga off
  tga_t265_5imus/  cam_imu_t265_{01,03}_bag{01..07}.yaml  vi_sensor_rigs_5imus, per t265 unit

Each template is the literal file with {bb} for the bag number and {u} for the
t265 unit. The camera priors (t265_calib.yaml, t265_0{1,3}_calib.yaml) are not
written here. Run from anywhere:
    python examples/configs/tga_t265/gen_tga_t265_configs.py
"""
from pathlib import Path

SPDX = ("# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>\n"
        "# SPDX-License-Identifier: GPL-3.0-only\n")   # generated configs carry the licence

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent

T_3CAMS = r'''# t265-only cam-IMU calibration with gyro<->accel time offset.
# Rig: mvis_4imus_3cams — t265 fisheye stereo + the t265's own BMI055 IMU as BASE
# (no aux IMUs). The BMI055 samples gyro @200 Hz / accel @62.5 Hz and the driver
# interpolates accel onto gyro stamps — exactly the internal gyro/accel offset
# t_ga is meant to expose (fe2 stamps were measured +2.8 ms late vs fe1).
# Run: mvis cam-imu -c <this>
mode: cam-imu
input:
  format: ros1
  path: ${MVIS_DATASETS}/mvis_4imus_3cams/virig_4imus_3cams_april_{bb}.bag
  cam_topics:
    - /t265/fisheye1/image_raw     # T265-1 (global)  <- base
    - /t265/fisheye2/image_raw     # T265-2 (global)
  imu_topics:
    - /t265/imu                    # BMI055 (base; the sensor under test)
  window_mode: motion
  segment_length: 80.0
  cache_dir: /tmp/mvis_detcache

board:
  board_type: apriltag
  rows: 6
  cols: 6
  tag_size: 0.088
  tag_spacing: 0.3
  tag_family: t36h11

camera_input:
  # t265 intrinsics from the committed 3cams camera-only prior (ids 1,2 -> 0,1)
  path: examples/configs/tga_t265/t265_calib.yaml

solve:
  max_iterations: 40
  estimate_t_ga: true              # BMI055 internal gyro<->accel offset
  sigma_t_ga: 0.010                # weak 10 ms prior

estimate:
  camera:
    intrinsics: false
    tr: false                      # global shutter
    td: true
    extrinsics: true
  imu:
    intrinsics: dw_da
    model: calyx

output:
  path: results/tga_t265/bag{bb}/cam_imu.json
'''

T_4CAMS = r'''# t265-only cam-IMU calibration with gyro<->accel time offset.
# Rig: mvis_4imus_4cams — t265 fisheye stereo + the t265's own BMI055 IMU as BASE
# (no aux IMUs). The BMI055 samples gyro @200 Hz / accel @62.5 Hz and the driver
# interpolates accel onto gyro stamps — exactly the internal gyro/accel offset
# t_ga is meant to expose (fe2 stamps were measured +2.8 ms late vs fe1).
# Run: mvis cam-imu -c <this>
mode: cam-imu
input:
  format: ros1
  path: ${MVIS_DATASETS}/mvis_4imus_4cams/4imus_4cams_{bb}.bag
  cam_topics:
    - /t265/fisheye1/image_raw     # T265-1 (global)  <- base
    - /t265/fisheye2/image_raw     # T265-2 (global)
  imu_topics:
    - /t265/imu                    # BMI055 (base; the sensor under test)
  window_mode: motion
  segment_length: 80.0
  cache_dir: /tmp/mvis_detcache

board:
  board_type: apriltag
  rows: 6
  cols: 6
  tag_size: 0.088
  tag_spacing: 0.3
  tag_family: t36h11

camera_input:
  # t265 intrinsics from the committed 3cams camera-only prior (ids 1,2 -> 0,1)
  path: examples/configs/tga_t265_4cams/t265_calib.yaml

solve:
  max_iterations: 40
  estimate_t_ga: true              # BMI055 internal gyro<->accel offset
  sigma_t_ga: 0.010                # weak 10 ms prior

estimate:
  camera:
    intrinsics: false
    tr: false                      # global shutter
    td: true
    extrinsics: true
  imu:
    intrinsics: dw_da
    model: calyx

output:
  path: results/tga_t265_4cams/bag{bb}/cam_imu.json
'''

T_5IMUS = r'''# t265-unit-{u} cam-IMU calibration with gyro<->accel time offset.
# Rig: vi_sensor_rigs_5imus — this rig carries TWO t265 units (t265_01, t265_03),
# each with its own fisheye stereo + BMI055 IMU. Calibrated per unit: this config
# uses the t265_{u} stereo pair + /t265_{u}/imu as base. Studies per-unit t_ga.
mode: cam-imu
input:
  format: ros1
  path: ${MVIS_DATASETS}/vi_sensor_rigs_5imus/5imus_4cams_{bb}.bag
  cam_topics:
    - /t265_{u}/fisheye1/image_raw     # T265-{u}-L (global)  <- base
    - /t265_{u}/fisheye2/image_raw     # T265-{u}-R (global)
  imu_topics:
    - /t265_{u}/imu                    # BMI055 of unit {u} (base; the sensor under test)
  window_mode: motion
  segment_length: 80.0
  cache_dir: /tmp/mvis_detcache

board:
  board_type: apriltag
  rows: 6
  cols: 6
  tag_size: 0.088
  tag_spacing: 0.3
  tag_family: t36h11

camera_input:
  path: examples/configs/tga_t265_5imus/t265_{u}_calib.yaml

solve:
  max_iterations: 40
  estimate_t_ga: true
  sigma_t_ga: 0.010

estimate:
  camera:
    intrinsics: false
    tr: false
    td: true
    extrinsics: true
  imu:
    intrinsics: dw_da
    model: calyx

output:
  path: results/tga_t265_5imus/unit{u}_bag{bb}/cam_imu.json
'''


def _sub(text, **kw):
    for k, v in kw.items():
        text = text.replace("{" + k + "}", v)
    return text


def _notga(text, bag_dir):
    """Bag 01 with t_ga off; results go to <bag_dir>_notga."""
    text = text.replace("estimate_t_ga: true ", "estimate_t_ga: false ", 1)
    return text.replace(f"/{bag_dir}/cam_imu.json", f"/{bag_dir}_notga/cam_imu.json", 1)


def _tdanchor(text):
    """Bag 01 with the common-mode camera td anchored at its init seed (0.5 ms)."""
    text = text.replace(
        "  sigma_t_ga: 0.010                # weak 10 ms prior\n",
        "  sigma_t_ga: 0.010\n"
        "  sigma_td_prior: 0.0005              # anchor common-mode cam td at the init seed (0.5 ms)\n",
        1)
    return text.replace("/bag01/cam_imu.json", "/bag01_tdanchor/cam_imu.json", 1)


def configs():
    """Return {path relative to examples/configs: file text}."""
    out = {}
    for b in range(1, 11):
        out[f"tga_t265/cam_imu_t265_{b:02d}.yaml"] = _sub(T_3CAMS, bb=f"{b:02d}")
    bag01 = out["tga_t265/cam_imu_t265_01.yaml"]
    out["tga_t265/cam_imu_t265_01_notga.yaml"] = _notga(bag01, "bag01")
    out["tga_t265/cam_imu_t265_01_tdanchor.yaml"] = _tdanchor(bag01)
    for b in range(1, 16):
        out[f"tga_t265_4cams/cam_imu_t265_{b:02d}.yaml"] = _sub(T_4CAMS, bb=f"{b:02d}")
    out["tga_t265_4cams/cam_imu_t265_01_notga.yaml"] = _notga(
        out["tga_t265_4cams/cam_imu_t265_01.yaml"], "bag01")
    for u in ("01", "03"):
        for b in range(1, 8):
            out[f"tga_t265_5imus/cam_imu_t265_{u}_bag{b:02d}.yaml"] = _sub(
                T_5IMUS, u=u, bb=f"{b:02d}")
    return out


def _write(root):
    root = Path(root)
    pairs = {}
    for rel, text in configs().items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(SPDX + text)
        pairs[str(CONFIGS / rel)] = str(dst)
    return pairs


def regenerate(out_dir):
    """Write every generated config under ``out_dir``; returns {committed: regenerated}."""
    return _write(out_dir)


def main():
    pairs = _write(CONFIGS)
    print(f"wrote {len(pairs)} configs under {CONFIGS}")


if __name__ == "__main__":
    main()
