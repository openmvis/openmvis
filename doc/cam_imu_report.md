# The cam-IMU calibration report

`mvis cam-imu` writes a self-contained HTML report (`<output>_report.html`) and
an optional text report. The HTML uses the same shell as the camera-only report
(`analysis/report_shell.py`: branding header, table-of-contents nav, sectioned
cards, grade badges, dark-mode + print CSS) and adds **interactive 3D plots**.

## Sections
1. **Summary** — status, reprojection RMS, camera/IMU counts, trajectory states,
   LM iterations, with quality grade badges (reproj good ≤1.0 px / marginal ≤2.0;
   χ²/dof good ≤1.5 / marginal ≤3.0).
2. **3D Geometry** — two rotatable/zoomable plotly figures:
   - *Sensor rig* — body-frame extrinsics: base IMU triad at the origin, each
     camera as a frustum at its `T_C→I`, each aux IMU as a triad at its `T_IstoI`,
     lever arms drawn and mm-labeled.
   - *Trajectory & scene* — the recovered IMU/base-camera path in the world plus the
     calibration board (board mode) or the landmark cloud (feature mode).
3. **Cameras** — per-camera intrinsics, distortion, `T_C→I` lever/rotation with σ,
   td/tr, and the stereo baseline.
4. **Base IMU** — biases + drift, Dw/Da diagonals (+ `R_wtoI` for the calyx model).
5. **Aux IMUs** — per-aux `T_IstoI`, inter-IMU td, biases, intrinsics.
6. **vs reference** *(optional)* — lever / rotation / baseline deltas against
   published values (`reference:` in the run config), with grade badges.

## Programmatic use
```python
from mvis.analysis.cam_imu_report import write_cam_imu_report
write_cam_imu_report(result, html_path="rep.html", text_path="rep.txt",
                     reference=ref, board=board)   # board feeds the 3D scene
```
`board` (+ `landmarks` for feature mode) are optional — without them the report
still renders, and the rig 3D plot still draws (it needs only the per-camera
calibrations). All 3D data is read from `result.values`; no solver changes.
