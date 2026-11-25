# Local gravity in camera + IMU solves

A camera + IMU solve holds the gravity magnitude |g| fixed. Gravity is most of what an
accelerometer measures, so the accelerometer scale the solve estimates absorbs the
ratio g_true / g_used. MVIS used 9.81 everywhere before 0.4.0. That reads the accelerometer
scale high wherever g is lower, and low wherever it is higher.

**Set it for every camera + IMU run.** If the config gives neither `gravity:` nor `site:`,
MVIS uses **9.81 m/s²**. The run log warns about it, `mvis check` names it, and the result
records `gravity_source: default`. Near the equator that reads the accelerometer scale up to
0.3 % high (see the table below).

## Setting it

Put the local value in the config, with the place as a comment:

```yaml
gravity: 9.8013   # m/s², Newark, Delaware
```

Take the value from the table below, or from the calculator on
[openmvis.com/reference/local-gravity/](https://openmvis.com/reference/local-gravity/).
`solve.gravity:` is the same key. Or give where the data was recorded as
`site: {latitude_deg: 39.68, altitude_m: 40, name: "Newark, Delaware"}`: MVIS turns latitude and
altitude into WGS84 normal gravity, within about 0.05 % of a measured value. You can give both;
they must agree (see below).

The report names the place (MVIS 0.4.1 and later). With `gravity:`, it takes the comment after
the value, without a leading unit, up to the first `;`: `# m/s², Newark, Delaware` gives
"Newark, Delaware", and `# m/s², Wuhan, China; local 9.7936` gives "Wuhan, China". With
`site:`, it uses the site name and coordinates instead.

- For `site:`, latitude and altitude, not a city name: city names repeat (Córdoba, Spain and Córdoba,
  Argentina), and a site outside any city still has coordinates from a phone or a map.
- Precision: anywhere in the right city is enough. One degree of latitude moves g by at most
  0.01 %; 100 m of altitude by 0.003 %. `altitude_m` can be left out (0) below a few hundred
  metres; it matters for high cities (Kunming −0.06 %, Lhasa −0.11 %).
- Both `gravity` and `site` given: they must agree to 0.01 m/s², else the config is rejected.
- A value outside 9.70–9.90 is rejected (a unit or typing mistake).
- Neither given: 9.81, so configs written before this key give the same results as before.
- `longitude_deg` and `name` are labels: g ignores them. The cam-IMU report (HTML and text) prints
  the gravity used and where it came from, e.g. `9.7879 m/s² — WGS84 normal gravity at Shenzhen, China
  (22.54° N, 114.06° E, 20 m); vs 9.81: -0.23 %`, or `9.8013 m/s² — set in config for Newark,
  Delaware; vs 9.81: -0.09 %` from the comment.
- Older published runs give `gravity: 9.81` plus `site:`: the site names the place and the runs
  keep reproducing at 9.81.

## Common recording places

City centres. Copy the g column into `gravity:` with the city as a comment, or the latitude
and altitude into `site:`. The last column is how far the accelerometer scale reads off if the
run keeps the 9.81 default.

| Place | latitude_deg | altitude_m | g [m/s²] | Accel-scale bias at 9.81 |
|---|---|---|---|---|
| Singapore | 1.35 | 15 | 9.7803 | +0.30 % |
| Bangalore | 12.97 | 920 | 9.7801 | +0.31 % |
| Mexico City | 19.43 | 2240 | 9.7791 | +0.32 % |
| Hong Kong | 22.32 | 10 | 9.7877 | +0.23 % |
| Shenzhen | 22.54 | 20 | 9.7879 | +0.23 % |
| São Paulo | −23.55 | 760 | 9.7862 | +0.24 % |
| Abu Dhabi | 24.45 | 5 | 9.7892 | +0.21 % |
| Kunming | 25.04 | 1890 | 9.7837 | +0.27 % |
| Lhasa | 29.65 | 3650 | 9.7817 | +0.29 % |
| Wuhan | 30.59 | 25 | 9.7936 | +0.17 % |
| Shanghai | 31.23 | 5 | 9.7942 | +0.16 % |
| Córdoba, Argentina | −31.42 | 400 | 9.7931 | +0.17 % |
| Sydney | −33.87 | 20 | 9.7963 | +0.14 % |
| Tokyo | 35.68 | 40 | 9.7978 | +0.12 % |
| Seoul | 37.57 | 40 | 9.7994 | +0.11 % |
| San Francisco | 37.77 | 15 | 9.7997 | +0.11 % |
| Washington, DC | 38.90 | 20 | 9.8007 | +0.10 % |
| Newark, Delaware | 39.68 | 40 | 9.8013 | +0.09 % |
| Denver | 39.74 | 1610 | 9.7965 | +0.14 % |
| Beijing | 39.90 | 45 | 9.8015 | +0.09 % |
| Boston | 42.36 | 10 | 9.8038 | +0.06 % |
| Zurich | 47.37 | 410 | 9.8071 | +0.03 % |
| Dübendorf (UZH-FPV) | 47.40 | 440 | 9.8070 | +0.03 % |
| Seattle | 47.61 | 50 | 9.8084 | +0.02 % |
| Munich | 48.14 | 520 | 9.8074 | +0.03 % |
| Paris | 48.86 | 35 | 9.8096 | +0.00 % |
| London | 51.51 | 15 | 9.8120 | −0.02 % |
| Equator / poles (sea level) | 0 / ±90 | 0 | 9.7803 / 9.8322 | +0.30 % / −0.23 % |

Southern latitudes are negative. Any other place: `python -c "from mvis.core.gravity import normal_gravity; print(normal_gravity(LAT, ALT))"`.

## Where it is recorded

The value used, and where it came from (`default`, `config` or `site`), is in
`cam_imu.json` (`gravity_mps2`, `gravity_source`, and `gravity_site` or `gravity_place`), the calibration YAML (`imus[0].gravity`),
the HTML report, the run database's world frame, and any digital twin captured from the run.
`mvis check` names it, with the place. With the default, the run log warns and says how to set it.

## Why not estimate it

|g| and the accelerometer scale can only be told apart when the motion's accelerations are a
sizeable part of g. Hand-held calibration takes reach a few tenths of a m/s², so a free |g| would
trade off against the scale. MVIS holds |g| at the given value instead.

## Checked on

- A digital twin (`tests/integration/test_digital_twins.py::test_local_gravity_removes_the_accel_scale_bias`)
  recorded at 9.7879: solved at 9.81 the accelerometer scale reads +0.11 to +0.20 % high on every axis;
  with `gravity: 9.7879` it matches the solve whose gravity was right all along.
- Camera-only runs have no IMU and are not affected.
