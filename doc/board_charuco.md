# ChArUco boards

A ChArUco board is a chessboard with an ArUco marker in each white square
(`board_type: charuco`). The calibration points are the chessboard's inner corners, as on
a plain chessboard: same point ids, one point per id. The markers name the corners, so a
board that is partly out of view or hidden is still used.

```yaml
board:
  board_type: charuco
  rows: 9                  # SQUARES, as board generators print them (8 x 13 inner corners)
  cols: 14
  square_size: 0.04        # metres
  marker_size: 0.03        # metres, smaller than square_size
  aruco_dict: DICT_5X5_250 # required; `dictionary:` (OpenCV's name) is read the same way
  charuco_legacy: false    # true for boards printed with OpenCV < 4.6 (even row counts)
  charuco_refine: true     # cornerSubPix on each corner (see below)
```

- **rows and cols count squares.** For a chessboard they count inner corners. A ChArUco
  board of 9 x 14 squares has 8 x 13 = 104 corners.
- **The dictionary is required.** MVIS stops with an error if it is missing, or if
  `dictionary` and `aruco_dict` name two different ones: a board read with the wrong
  dictionary finds no corners.
- **Point ids follow OpenCV**: id = r * (cols - 1) + c for inner corner (r, c). The board
  frame starts at the first inner corner, as for a chessboard (OpenCV starts at the board's
  outer corner, one square away).
- **Corner refinement.** OpenCV's `CharucoDetector` returns corners about 0.49 px down and
  right of the true corner. On rendered boards: raw OpenCV corners are off by
  (+0.487, +0.487) px on average; after MVIS's `cornerSubPix` pass they are within
  0.001 px on average (0.047 px RMS), the same as MVIS's plain chessboard detector (0.040 px
  RMS). The half-window is never wider than the white gap to the nearest marker,
  (square - marker) / 2. Set `charuco_refine: false` to keep OpenCV's corners.
- **All corners are used.** `corner_subsample` thins AprilGrid tags (four corners per tag);
  on a ChArUco board it has nothing to thin, so MVIS uses `all` and warns if another mode
  was asked for.
- Frames with fewer than six corners, or corners all on one board row or column, are
  dropped (no usable pose). With fewer than 85 % of the corners found, a 2x upscaled pass is
  tried and kept when it finds more.
- If markers decode but no corner comes out, MVIS warns once: rows and cols swapped or
  given as inner corners, the wrong dictionary or the wrong `charuco_legacy`.
- `mvis doctor` checks the dictionary (known, with enough markers for the board), the
  sizes (metres, marker inside the square) and that OpenCV has `CharucoDetector` (>= 4.7).
- Kalibr and Basalt have no ChArUco target, so comparison rows skip these boards.
- Results files store ChArUco detections as chessboard rows (one point per corner id).
