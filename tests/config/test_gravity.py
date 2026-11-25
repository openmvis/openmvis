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
"""Local gravity: the normal-gravity formula, resolving `gravity` / `site`, and the config keys."""
import pytest

from mvis.core.gravity import STANDARD_MVIS, normal_gravity, resolve


def test_normal_gravity_known_values():
    assert normal_gravity(0.0) == pytest.approx(9.7803, abs=1e-4)          # equator
    assert normal_gravity(90.0) == pytest.approx(9.8322, abs=1e-4)         # pole
    assert normal_gravity(45.0) == pytest.approx(9.8062, abs=1e-4)
    assert normal_gravity(22.54, 20) == pytest.approx(9.7879, abs=1e-4)    # Shenzhen
    assert normal_gravity(45.0) - normal_gravity(45.0, 100) == pytest.approx(3.086e-4)


@pytest.mark.parametrize("gravity, site, want", [
    (None, None, (STANDARD_MVIS, "default")),
    (9.7879, None, (9.7879, "config")),
    (None, {"latitude_deg": 22.54, "altitude_m": 20}, (pytest.approx(9.7879, abs=1e-4), "site")),
    (9.788, {"latitude_deg": 22.54, "name": "Shenzhen"}, (9.788, "config")),   # both, agreeing
])
def test_resolve(gravity, site, want):
    assert resolve(gravity, site) == want


@pytest.mark.parametrize("gravity, site, msg", [
    (981.0, None, "m/s\\^2"),
    (9.18, None, "outside"),
    (9.83, {"latitude_deg": 22.54}, "disagree"),
    (None, {"lat": 22.5}, "latitude_deg"),
    (None, {"latitude_deg": 95}, r"\[-90, 90\]"),
])
def test_resolve_rejects(gravity, site, msg):
    with pytest.raises(ValueError, match=msg):
        resolve(gravity, site)


def _cfg(tmp_path, extra=""):
    p = tmp_path / "c.yaml"
    p.write_text("mode: cam-imu\ninput: {format: euroc, path: /x}\n" + extra)
    return str(p)


def test_config_default_is_unchanged(tmp_path):
    from mvis.core.config_parser import parse_cam_imu_config
    c = parse_cam_imu_config(_cfg(tmp_path))
    assert (c.solve.gravity, c.solve.gravity_source) == (9.81, "default")


@pytest.mark.parametrize("extra, g, source", [
    ("gravity: 9.7879\n", 9.7879, "config"),
    ("solve: {gravity: 9.7879}\n", 9.7879, "config"),
    ("site: {latitude_deg: 22.54, altitude_m: 20, name: Shenzhen}\n", pytest.approx(9.7879, abs=1e-4), "site"),
])
def test_config_keys(tmp_path, caplog, extra, g, source):
    from mvis.core.config_parser import parse_cam_imu_config
    with caplog.at_level("WARNING", logger="mvis.core.config_parser"):
        c = parse_cam_imu_config(_cfg(tmp_path, extra))
    assert "unknown key" not in caplog.text
    assert (c.solve.gravity, c.solve.gravity_source) == (g, source)


def test_config_gravity_and_solve_gravity_must_agree(tmp_path):
    from mvis.core.config_parser import parse_cam_imu_config
    with pytest.raises(ValueError, match="disagree"):
        parse_cam_imu_config(_cfg(tmp_path, "gravity: 9.79\nsolve: {gravity: 9.81}\n"))



@pytest.mark.parametrize("extra, status, want", [
    ("", "pass", "|g| = 9.81 m/s^2"),
    ("site: {latitude_deg: 22.54}\n", "pass", "9.7879 m/s^2 (site)")])
def test_doctor_names_the_gravity(tmp_path, extra, status, want):
    from mvis.doctor.check_config import check_config
    f = {x.id: x for x in check_config(_cfg(tmp_path, extra))}["solve.gravity"]
    assert f.status == status and want in f.message, f.message


@pytest.mark.parametrize("extra, warned", [("", True), ("gravity: 9.7879\n", False)])
def test_run_log_warns_on_the_default(tmp_path, caplog, extra, warned):
    """A cam-IMU run without `gravity` / `site` says so in its log before it starts."""
    from mvis.cli.main import calibrate_cam_imu
    from mvis.core.config_parser import parse_cam_imu_config
    c = parse_cam_imu_config(_cfg(tmp_path, extra + "camera_input: {path: /nonexistent/prior.yaml}\n"))
    with caplog.at_level("WARNING", logger="mvis"), pytest.raises(OSError):
        calibrate_cam_imu(c)                       # stops at the missing prior, after the check
    assert ("|g| = 9.81" in caplog.text) is warned, caplog.text


def test_of_result_falls_back_to_the_default():
    from types import SimpleNamespace
    from mvis.core.gravity import of_result
    assert of_result(SimpleNamespace(diagnostics={"gravity": 9.7879})) == 9.7879
    assert of_result(SimpleNamespace(diagnostics={})) == STANDARD_MVIS          # results from before 0.4.0
    assert of_result(SimpleNamespace()) == STANDARD_MVIS                        # stand-ins without diagnostics


def test_site_longitude_is_display_only():
    site = {"latitude_deg": 22.54, "altitude_m": 20}
    assert resolve(None, {**site, "longitude_deg": 114.06}) == resolve(None, site)
    with pytest.raises(ValueError, match=r"\[-180, 180\]"):
        resolve(None, {**site, "longitude_deg": 214.0})


def test_config_keeps_the_site_for_the_report(tmp_path):
    from mvis.core.config_parser import parse_cam_imu_config
    c = parse_cam_imu_config(_cfg(tmp_path, "gravity: 9.81\nsite: {latitude_deg: 39.68, longitude_deg: -75.75, "
                                            "altitude_m: 30, name: 'Newark, DE'}\n"))
    assert (c.solve.gravity, c.solve.gravity_source) == (9.81, "config")       # g unchanged, site for display
    assert c.solve.gravity_site["name"] == "Newark, DE"
    assert c.solve.gravity_site["g_site"] == pytest.approx(9.8013, abs=1e-4)
    assert parse_cam_imu_config(_cfg(tmp_path)).solve.gravity_site is None


def test_describe_names_the_source():
    from mvis.core.gravity import describe, site_record
    sz = site_record({"latitude_deg": 22.54, "longitude_deg": 114.06, "altitude_m": 20, "name": "Shenzhen, China"})
    assert describe(sz["g_site"], "site", sz) == ("9.7879 m/s² — WGS84 normal gravity at Shenzhen, China "
                                                   "(22.54° N, 114.06° E, 20 m); vs 9.81: -0.23 %")
    nw = site_record({"latitude_deg": 39.68, "longitude_deg": -75.75, "altitude_m": 30, "name": "Newark, DE"})
    assert describe(9.81, "config", nw) == ("9.8100 m/s² — set in config; Newark, DE (39.68° N, 75.75° W, 30 m) "
                                            "gives 9.8013 (-0.09 % vs 9.81)")
    assert describe(9.7879, "config") == "9.7879 m/s² — set in config; vs 9.81: -0.23 %"
    assert describe(9.81, "default") == "9.8100 m/s² — MVIS default, no site given"
    assert describe(9.7879, "site") == "9.7879 m/s² — WGS84 normal gravity at the config site; vs 9.81: -0.23 %"
    assert describe(9.7931, "site", {"latitude_deg": -31.42}) == ("9.7931 m/s² — WGS84 normal gravity at "
                                                                  "site (31.42° S); vs 9.81: -0.17 %")


@pytest.mark.parametrize("extra, place", [
    ("gravity: 9.8013   # m/s², Newark, Delaware\n", "Newark, Delaware"),
    ("gravity: 9.8013   # m/s^2, Newark, Delaware\n", "Newark, Delaware"),
    ("gravity: 9.8013 #m/s2,Newark, Delaware  \n", "Newark, Delaware"),
    ("gravity: 9.8013   # Newark, Delaware\n", "Newark, Delaware"),
    ("gravity: 9.8013   # m/s²\n", None),                        # unit only: no place
    ("gravity: 9.8013   #\n", None),
    ("gravity: 9.81   # m/s², Wuhan, China; local 9.7936, solved at 9.81\n", "Wuhan, China"),  # ends at ';'
    ("gravity: 9.8013   # m/s², Newark, DE (Univ. of Delaware); local 9.8013\n",
     "Newark, DE (Univ. of Delaware)"),                          # brackets stay
    ("gravity: 9.8013   # m/s²; local 9.8013\n", None),             # nothing before ';'
    ("gravity: 9.8013\n", None),                                 # no comment
    ("solve:\n  max_iterations: 5\n  gravity: 9.8013   # m/s², Newark, Delaware\n", "Newark, Delaware"),
    ("solve:\n  gravity: 9.8013\nsite_note: 1\n", None),
    ("solve: {gravity: 9.8013}   # m/s², Newark, Delaware\n", None),   # flow style: not read
    ("# gravity: 9.81   # m/s², Shenzhen\ngravity: 9.8013   # m/s², Newark, Delaware\n", "Newark, Delaware"),
    ("gravity: 9.8013\nfoo:\n  gravity: 1   # m/s², Nowhere\n", None),     # not solve's gravity:
    ("gravity: 9.81   # under 0.1 %: kept\nsite: {latitude_deg: 48.26, altitude_m: 482}\n", None),  # site wins
])
def test_config_keeps_the_place_from_the_gravity_comment(tmp_path, extra, place):
    from mvis.core.config_parser import parse_cam_imu_config
    c = parse_cam_imu_config(_cfg(tmp_path, extra))
    assert c.solve.gravity_place == place


def test_config_from_a_dict_has_no_place():
    """No file text, no comment: a config built from a dict names no place."""
    from mvis.core.config import CamImuRunConfig
    from mvis.core.config_parser import _parse_dataclass, _resolve_gravity
    data = {"input": {"format": "euroc", "path": "/x"}, "gravity": 9.8013}
    c = _parse_dataclass(data, CamImuRunConfig, "cam_imu")
    _resolve_gravity(c, data)
    assert (c.solve.gravity, c.solve.gravity_source, c.solve.gravity_place) == (9.8013, "config", None)


def test_describe_names_the_place_from_the_comment():
    from mvis.core.gravity import describe, site_record
    assert describe(9.8013, "config", None, "Newark, Delaware") == (
        "9.8013 m/s² — set in config for Newark, Delaware; vs 9.81: -0.09 %")
    nw = site_record({"latitude_deg": 39.68, "altitude_m": 40, "name": "Newark, DE"})
    assert describe(9.8013, "config", nw, "Somewhere else").startswith(
        "9.8013 m/s² — set in config; Newark, DE (39.68° N, 40 m) gives")        # the site wins
    assert describe(9.81, "default", None, "Newark") == "9.8100 m/s² — MVIS default, no site given"
