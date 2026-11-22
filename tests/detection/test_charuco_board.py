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
"""ChArUco board (`board_type: charuco`): OpenCV's corner layout, subpixel corners in
rendered images, partial views, config wiring, the detection store, and a camera
calibration from rendered images."""
import copy
import pickle

import cv2
import numpy as np
import pytest

from mvis.core.config import BoardConfig, CalibrationConfig
from mvis.core.config_validator import validate_config
from mvis.core.types import FrameObservations
from mvis.detection.board_factory import create_board
from mvis.detection.charuco_board import CharucoBoard
from mvis.mvisdb.bridge import board_from_geometry
from mvis.mvisdb.mvisdatabase import BoardType

if not hasattr(cv2.aruco, "CharucoDetector"):
    pytest.skip("OpenCV < 4.7 has no CharucoDetector", allow_module_level=True)

ROWS, COLS, SQ, MK = 6, 8, 0.04, 0.03        # squares, metres
PPM = 4000.0                                 # rendered canvas pixels per metre


def _cv_board(legacy=False, rows=ROWS, cols=COLS):
    b = cv2.aruco.CharucoBoard((cols, rows), SQ, MK,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
    b.setLegacyPattern(legacy)
    return b


def _canvas(legacy=False, rows=ROWS, cols=COLS):
    """The board at PPM with no margin: board point (x, y) m sits at pixel centre
    coordinate x * PPM - 0.5 (pixel edges at integer multiples of PPM * SQ)."""
    size = (int(round(cols * SQ * PPM)), int(round(rows * SQ * PPM)))
    return _cv_board(legacy, rows, cols).generateImage(size, marginSize=0)


_A = np.array([[PPM, 0, -0.5], [0, PPM, -0.5], [0, 0, 1.0]])   # board metres -> canvas px
_K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
_IMG = (640, 480)


_SS = 4
_UP = np.array([[_SS, 0, (_SS - 1) / 2], [0, _SS, (_SS - 1) / 2], [0, 0, 1.0]])  # image -> 4x


def _render(rvec, tvec, canvas=None):
    """Pinhole image of the board at pose (rvec, tvec) (board in camera), rendered 4x
    and area-downsampled, and the true pixel of every inner corner (OpenCV board frame:
    first inner corner at (SQ, SQ))."""
    canvas = _canvas() if canvas is None else canvas
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    H = _K @ np.column_stack([R[:, 0], R[:, 1], tvec])          # board plane -> image
    big = cv2.warpPerspective(canvas, _UP @ H @ np.linalg.inv(_A),
                              (_IMG[0] * _SS, _IMG[1] * _SS), flags=cv2.INTER_LINEAR,
                              borderValue=255)
    img = cv2.resize(big, _IMG, interpolation=cv2.INTER_AREA)
    pts = _cv_board().getChessboardCorners().astype(float)
    uv, _ = cv2.projectPoints(pts, np.asarray(rvec, float), np.asarray(tvec, float),
                              _K, np.zeros(5))
    return img, uv.reshape(-1, 2)


def test_points_follow_opencv_layout():
    board = CharucoBoard(ROWS, COLS, SQ, MK)
    cv_pts = _cv_board().getChessboardCorners().astype(float)
    assert board.num_points() == len(cv_pts) == (ROWS - 1) * (COLS - 1)
    ours = np.array([board.point_3d(i) for i in range(board.num_points())])
    np.testing.assert_allclose(ours, cv_pts - cv_pts[0], atol=1e-12)


def test_frontal_board_corners_subpixel():
    """Every corner of a frontal board, with ids in order, within 0.05 px of the truth."""
    board = CharucoBoard(ROWS, COLS, SQ, MK)
    small = cv2.resize(_canvas(), None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    obs = board.detect(small)
    assert [o.point_id for o in obs] == list(range(board.num_points()))
    truth = (_cv_board().getChessboardCorners()[:, :2] * PPM * 0.1) - 0.5
    err = np.array([o.pixel for o in obs]) - truth
    assert np.abs(err).max() < 0.05


def _pose(rng, z=(0.35, 0.6)):
    rvec = rng.uniform(-0.45, 0.45, 3)
    R, _ = cv2.Rodrigues(rvec)
    centre = np.array([COLS * SQ / 2, ROWS * SQ / 2, 0.0])
    t = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.09, 0.09),
                  rng.uniform(*z)]) - R @ centre
    return rvec, t


def test_tilted_views_corners_within_a_tenth_px():
    """Perspective views: every corner within 0.15 px of the truth. OpenCV's own
    corners are about 0.5 px off (down and right); the re-refinement removes that."""
    board, canvas, rng = CharucoBoard(ROWS, COLS, SQ, MK), _canvas(), np.random.RandomState(1)
    raw_bias, n = [], 0
    for _ in range(8):
        img, truth = _render(*_pose(rng), canvas)
        obs = board.detect(img)
        if not obs:
            continue
        n += 1
        err = np.array([o.pixel - truth[o.point_id] for o in obs])
        assert np.abs(err).max() < 0.15
        raw_c, raw_ids = board._detect_raw(img)
        raw_bias.append((raw_c - truth[raw_ids]).mean(axis=0))
    assert n >= 6
    # documents why _refine exists; if OpenCV fixes it, the refinement is still right
    assert np.all(np.mean(raw_bias, axis=0) > 0.3)


def test_partial_view_keeps_ids():
    """Half the board out of view: the visible corners come with their own ids."""
    board = CharucoBoard(ROWS, COLS, SQ, MK)
    small = cv2.resize(_canvas(), None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    left = small[:, : small.shape[1] // 2].copy()
    obs = board.detect(left)
    ids = {o.point_id for o in obs}
    assert 6 <= len(ids) < board.num_points()
    assert all(i % board.cols < COLS // 2 for i in ids)           # left half only
    truth = (_cv_board().getChessboardCorners()[:, :2] * PPM * 0.1) - 0.5
    for o in obs:
        assert np.abs(o.pixel - truth[o.point_id]).max() < 0.05


def test_one_row_of_corners_is_no_pose():
    board = CharucoBoard(ROWS, COLS, SQ, MK)
    board._detect_raw = lambda gray: (np.random.rand(7, 2) * 100, np.arange(7))   # row 0
    assert board.detect(np.zeros((100, 100), np.uint8)) == []


def test_legacy_pattern_board():
    """A board printed with OpenCV < 4.6 (even row count) needs legacy_pattern."""
    small = cv2.resize(_canvas(legacy=True), None, fx=0.1, fy=0.1,
                       interpolation=cv2.INTER_AREA)
    legacy = CharucoBoard(ROWS, COLS, SQ, MK, legacy_pattern=True)
    assert len(legacy.detect(small)) == legacy.num_points()
    assert len(CharucoBoard(ROWS, COLS, SQ, MK).detect(small)) < legacy.num_points()


def test_copies_and_pickles_rebuild_the_detector():
    board = CharucoBoard(ROWS, COLS, SQ, MK)
    small = cv2.resize(_canvas(), None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    n = len(board.detect(small))
    for other in (copy.deepcopy(board), pickle.loads(pickle.dumps(board))):
        assert len(other.detect(small)) == n


def test_config_builds_the_board():
    cfg = BoardConfig(board_type="charuco", rows=ROWS, cols=COLS, square_size=SQ,
                      marker_size=MK, aruco_dict="DICT_6X6_250")
    board = create_board(cfg)
    assert isinstance(board, CharucoBoard)
    assert (board.rows, board.cols) == (ROWS - 1, COLS - 1)


@pytest.mark.parametrize("kw, msg", [
    (dict(marker_size=0.05), "marker_size"),
    (dict(rows=2), "at least 3"),
    (dict(aruco_dict="DICT_9X9_1"), "Unknown ArUco dictionary"),
])
def test_validator_rejects_bad_charuco(kw, msg):
    b = dict(board_type="charuco", rows=ROWS, cols=COLS, square_size=SQ, marker_size=MK)
    b.update(kw)
    errors = validate_config(CalibrationConfig(board=BoardConfig(**b)))
    assert any(msg in e for e in errors), errors
    with pytest.raises(ValueError):
        CharucoBoard(b["rows"], b["cols"], SQ, b["marker_size"],
                     b.get("aruco_dict", "DICT_6X6_250"))


def test_store_types_it_as_chessboard_and_saves(tmp_path):
    """mvisdb has no 'charuco' board_type (schema CHECK): ChArUco is stored as chessboard
    rows, one point per corner id, and a save + load round trip works."""
    from mvis.mvisdb.bridge import db_from_frame_observations
    from mvis.mvisdb.manager import DatabaseManager
    board = CharucoBoard(ROWS, COLS, SQ, MK)
    db_board = board_from_geometry(board)
    assert db_board.board_type is BoardType.CHESSBOARD
    assert db_board.corners_per_tag == 1
    assert len(db_board.pts) == (ROWS - 1) * (COLS - 1)
    img, _ = _render([0.1, -0.1, 0.0], [-0.12, -0.08, 0.45])
    obs = board.detect(img)
    assert obs
    frames = {0: [FrameObservations(camera_id=0, timestamp=0.0, observations=obs,
                                    image_size=_IMG)]}
    db = db_from_frame_observations(frames, board, image_sizes={0: _IMG})
    m = DatabaseManager()
    m.save(db, tmp_path / "c.mvisdb", attach_run=False)
    back = m.load(tmp_path / "c.mvisdb")
    ids = [d.point_id for d in back.camera(0).frames[0].board_detections[0].detections]
    assert ids == [o.point_id for o in obs]


def test_fingerprint_follows_the_detector_settings():
    """The detection cache is keyed on board_fingerprint: every detector setting is in
    it, and a ChArUco board never shares a key with the chessboard of the same corners."""
    from mvis.detection.chessboard import Chessboard
    from mvis.io.capture_key import board_fingerprint
    base = board_fingerprint(CharucoBoard(ROWS, COLS, SQ, MK))
    assert base != board_fingerprint(Chessboard(ROWS - 1, COLS - 1, SQ))
    assert base != board_fingerprint(CharucoBoard(ROWS, COLS, SQ, MK, refine=False))
    assert base != board_fingerprint(CharucoBoard(ROWS, COLS, SQ, MK, legacy_pattern=True))
    assert base != board_fingerprint(CharucoBoard(ROWS, COLS, SQ, MK, "DICT_5X5_250"))
    for knob in ("_min_corners", "_upscale_below", "_subpix_iters", "_subpix_eps",
                 "_detector_rev"):
        b = CharucoBoard(ROWS, COLS, SQ, MK)
        setattr(b, knob, getattr(b, knob) * 2)
        assert board_fingerprint(b) != base, knob


def test_refine_off_keeps_opencv_corners():
    img, _ = _render([0.15, -0.2, 0.05], [-0.1, -0.07, 0.4])
    raw_c, raw_ids = CharucoBoard(ROWS, COLS, SQ, MK)._detect_raw(img)
    obs = CharucoBoard(ROWS, COLS, SQ, MK, refine=False).detect(img)
    by_id = dict(zip(raw_ids.tolist(), raw_c))
    for o in obs:
        np.testing.assert_array_equal(o.pixel, by_id[o.point_id])


def test_swapped_rows_cols_warn_once(caplog):
    """rows and cols swapped: markers decode, no corner comes out, and one warning
    says what to check."""
    img, _ = _render([0.0, 0.0, 0.0], [-0.16, -0.12, 0.5])
    board = CharucoBoard(COLS, ROWS, SQ, MK)
    with caplog.at_level("WARNING", logger="mvis.detection.charuco_board"):
        assert board.detect(img) == []
        board.detect(img)
    hits = [r for r in caplog.records if "no corner found" in r.getMessage()]
    assert len(hits) == 1, caplog.text


_HUB_BOARD = """board:
  board_type: charuco
  rows: 9
  cols: 14
  square_size: 0.04
  marker_size: 0.03
  dictionary: DICT_5X5_250
"""


def _cam_imu_yaml(tmp_path, board=_HUB_BOARD, extra=""):
    p = tmp_path / "cam_imu.yaml"
    p.write_text("mode: cam-imu\n" + board + extra)
    return str(p)


def test_dictionary_alias_and_all_corners(tmp_path, caplog):
    """The published ALG configs say `dictionary:` (OpenCV's name): it maps to
    aruco_dict without an unknown-key warning, and the board has 104 corners."""
    from mvis.core.config_parser import parse_cam_imu_config
    with caplog.at_level("WARNING", logger="mvis.core.config_parser"):
        cfg = parse_cam_imu_config(_cam_imu_yaml(tmp_path))
    assert "unknown key" not in caplog.text
    assert cfg.board.aruco_dict == "DICT_5X5_250"
    assert create_board(cfg.board).num_points() == 104
    assert cfg.solve.corner_subsample == "all"


@pytest.mark.parametrize("board, msg", [
    (_HUB_BOARD.replace("  dictionary: DICT_5X5_250\n", ""), "needs its dictionary"),
    (_HUB_BOARD + "  aruco_dict: DICT_6X6_250\n", "disagree"),
])
def test_dictionary_missing_or_conflicting_fails(tmp_path, board, msg):
    from mvis.core.config_parser import parse_cam_imu_config
    with pytest.raises(ValueError, match=msg):
        parse_cam_imu_config(_cam_imu_yaml(tmp_path, board))


def test_corner_subsample_is_all_for_charuco(tmp_path, caplog):
    """ChArUco ids are chessboard corners, not tag * 4 + corner: a two_anti request
    (camera-only or cam-IMU) becomes 'all' with a warning."""
    from mvis.core.config_parser import parse_cam_imu_config, parse_cam_run_config
    with caplog.at_level("WARNING", logger="mvis.core.config_parser"):
        ci = parse_cam_imu_config(_cam_imu_yaml(
            tmp_path, extra="solve:\n  corner_subsample: two_anti\n"))
    assert ci.solve.corner_subsample == "all"
    assert "no meaning on a ChArUco board" in caplog.text
    p = tmp_path / "cam.yaml"
    p.write_text(_HUB_BOARD + "input:\n  cameras: []\n")
    assert parse_cam_run_config(str(p)).corner_subsample == "all"   # default two_anti


def test_doctor_checks_the_charuco_block(tmp_path):
    from mvis.doctor import check_config as cc
    good = {f.id: f.status for f in cc.check_config(_cam_imu_yaml(tmp_path))}
    assert good["board.charuco_layout"] == "pass"
    assert "config.unknown_key" not in good
    assert "board.aruco_dict" not in good
    small = _HUB_BOARD.replace("DICT_5X5_250", "DICT_5X5_50")
    bad = {f.id: f.status for f in cc.check_config(_cam_imu_yaml(tmp_path, small))}
    assert bad["board.aruco_dict"] == "fail"           # 63 markers > 50


def test_calibrates_a_camera_from_rendered_views(caplog):
    """Rendered views at known poses, some cut by the image edge: the camera-only
    pipeline recovers fx = fy = 500, cx = 320, cy = 240."""
    from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
    board = CharucoBoard(ROWS, COLS, SQ, MK)
    canvas = _canvas()
    rng = np.random.RandomState(3)
    frames, partial = [], 0
    while len(frames) < 14:
        img, _ = _render(*_pose(rng, z=(0.3, 0.55)), canvas)
        obs = board.detect(img)
        if not obs:
            continue
        partial += len(obs) < board.num_points()
        frames.append(FrameObservations(camera_id=0, timestamp=0.1 * len(frames),
                                        observations=obs, image_size=_IMG))
    assert partial > 0                                   # partial views were used
    pc = PipelineConfig(max_iterations=10, max_iterations_single=10,
                        rerun_after_outlier_rejection=False,
                        distortion_models={0: "radtan"}, verbose=False)
    pc.corner_subsample = "two_anti"                      # ignored: nothing to thin
    with caplog.at_level("INFO", logger="mvis.calibration.pipeline_cam"):
        r = CalibrationPipeline(board, pc).run({0: frames}, {0: _IMG})
    assert "no meaning on a ChArUco board" in caplog.text
    assert "Corner subsample (" not in caplog.text          # no corner was thinned
    fx, fy, cx, cy = np.asarray(r.cameras[0].intrinsics, float)[:4]
    np.testing.assert_allclose([fx, fy, cx, cy], [500, 500, 320, 240], atol=0.5)
    from mvis.io.output_writer import OutputWriter
    b = OutputWriter.to_full_dict(r, board=board)["board"]       # enough to rebuild the board
    assert (b["squares_rows"], b["squares_cols"], b["marker_size"], b["aruco_dict"]) == \
        (ROWS, COLS, MK, "DICT_6X6_250")


def test_doctor_fails_charuco_on_opencv_without_charuco_detector(tmp_path, monkeypatch):
    from mvis.doctor import check_config as cc
    cfg = tmp_path / "c.yaml"
    cfg.write_text("board: {board_type: charuco}\n")
    assert {"aruco", "charuco"} <= cc.config_needs(str(cfg))
    ok = {f.id: f.status for f in cc._check_cv2(True, True)}
    assert ok["env.cv2_charuco"] == "pass"
    monkeypatch.delattr(cv2.aruco, "CharucoDetector")
    old = {f.id: f.status for f in cc._check_cv2(True, True)}
    assert old["env.cv2_charuco"] == "fail"


def test_example_config_parses():
    from pathlib import Path
    from mvis.core.config_parser import parse_cam_run_config
    p = Path(__file__).parents[2] / "examples/configs/example_charuco_cam.yaml"
    cfg = parse_cam_run_config(str(p))
    board = create_board(cfg.board)
    assert isinstance(board, CharucoBoard) and board.num_points() == 104
    assert cfg.corner_subsample == "all"
