# Stages 5 & 7 — results table and release report

## Results table (show after Stage 5, repeat in the report)
| Sensor | Quantity | Value | σ | Unit |
|---|---|---|---|---|
| camN | fx, fy, cx, cy | … | … | px |
| camN | distortion (model) | … | | |
| camN→base IMU | translation p_CinI | x, y, z | | mm |
| camN→base IMU | rotation log(R) | rx, ry, rz | | rad |
| camN | td (t_imu = t_cam + td) | … | | ms |
| IMU | Dw / Da diagonal | … | | — |
| IMU | t_ga | … | … | ms |
| aux IMU k | p_IsinI, log(R), td | … | | mm, rad, ms |
| all | reprojection median / P95 / RMS | … | | px |

## `07_release/calibration_report.md`
1. **Summary** — rig, dataset, final grade (GREEN/YELLOW/RED), verified or
   unverified, date, `pymvis` version (and git sha when a checkout was used).
2. **Sensor suite** — from `02_sensor_suite.yaml`, base sensors marked.
3. **Calibrated vs fixed parameters** — from `03_calibration_spec.yaml`.
4. **Models and settings** — from `04_configs/rationale.md`, including the
   preset (Fast / Full).
5. **Results** — the table above.
6. **Verification** — every check from `06_verification.json` with value and
   threshold; independent check result; reference comparison (CAD or Device
   wiki).
7. **Limitations** — every accepted WARN, every waiver from `run.md`.
8. **Conventions** — frames (T_C→I maps camera points into the IMU frame),
   rotation as log(R) rotation vector in rad, units, time-offset sign.
9. **Files** — list of files in `07_release/`.
10. **How to cite** — the block below (drop the Calibench-VI entry when no
    Calibench-VI data was used).

Footer: `Calibrated with MVIS (pymvis <version>[, <sha>]).`

## Citation block
```bibtex
@misc{openmvis,
  title  = {{OpenMVIS}: open-source multi-visual-inertial sensor calibration},
  author = {{OpenMVIS Contributors}},
  year   = {2026},
  doi    = {10.5281/zenodo.24239182},
  url    = {https://openmvis.com}
}
@article{yang2024mvis,
  title   = {Multi-visual-inertial system: Analysis, calibration, and estimation},
  author  = {Yang, Yulin and Geneva, Patrick and Huang, Guoquan},
  journal = {The International Journal of Robotics Research},
  year    = {2024},
  volume  = {43},
  number  = {13},
  pages   = {1995--2026},
  doi     = {10.1177/02783649241245726}
}
@misc{calibench2026,
  author       = {Yang, Yulin},
  title        = {Calibench-VI: the visual-inertial calibration benchmark},
  year         = {2026},
  howpublished = {Hugging Face Datasets},
  note         = {https://huggingface.co/datasets/calibench-vi/datasets}
}
```

## Files in `07_release/`
- MVIS cam-IMU result (`*_calib.yaml`) and summary JSON;
- Kalibr `*_camchain-imucam.yaml` (kb4 exports as pinhole-equi, exact below
  90°; the export warns when the fit used corners past 90°);
- camera-only result YAML;
- `06_verification.json` copy and `calibration_report.md`.

## Sharing the result
A public result goes through
[openmvis/calibench-submissions](https://github.com/openmvis/calibench-submissions):
results records and community recordings (CC BY-NC 4.0 by default) are
reviewed and then appear on the openmvis.com Device wiki and leaderboard.
