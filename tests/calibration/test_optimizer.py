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
"""Tests for GTSAM optimizer."""

import logging

import numpy as np
import cv2
import gtsam
import pytest

from mvis.detection.chessboard import Chessboard
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.core.types import Observation, FrameObservations


def _generate_synthetic_data(
    board: Chessboard,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    num_frames: int = 20,
    image_size: tuple[int, int] = (640, 480),
    noise_sigma: float = 0.5,
    seed: int = 42,
):
    """Generate synthetic data for optimizer testing."""
    rng = np.random.RandomState(seed)
    board_pts = board.all_points_3d().astype(np.float32)
    frames = []
    gt_rvecs = []
    gt_tvecs = []

    attempts = 0
    while len(frames) < num_frames and attempts < num_frames * 5:
        attempts += 1
        rvec = rng.uniform(-0.4, 0.4, 3).astype(np.float64)
        tvec = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(0.5, 0.9),
        ], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(board_pts, rvec, tvec, K, dist_coeffs)
        img_pts = img_pts.reshape(-1, 2)

        w, h = image_size
        valid = (
            (img_pts[:, 0] > 20)
            & (img_pts[:, 0] < w - 20)
            & (img_pts[:, 1] > 20)
            & (img_pts[:, 1] < h - 20)
        )
        if not np.all(valid):
            continue

        img_pts += rng.randn(*img_pts.shape) * noise_sigma

        observations = [
            Observation(point_id=idx, pixel=img_pts[idx], row=int(img_pts[idx, 1]))
            for idx in range(len(img_pts))
        ]

        frames.append(
            FrameObservations(
                camera_id=0,
                timestamp=float(len(frames)),
                observations=observations,
                image_size=image_size,
            )
        )
        gt_rvecs.append(rvec)
        gt_tvecs.append(tvec)

    return frames, gt_rvecs, gt_tvecs


class TestOptimizerSingleCamera:
    @pytest.fixture
    def setup(self):
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K_gt = np.array([
            [500.0, 0, 320.0],
            [0, 500.0, 240.0],
            [0, 0, 1],
        ])
        dist_gt = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        return board, K_gt, dist_gt

    def test_single_camera_optimization(self, setup):
        """Test that optimizer reduces reprojection error."""
        board, K_gt, dist_gt = setup
        frames, _, _ = _generate_synthetic_data(
            board, K_gt, dist_gt, num_frames=20, noise_sigma=0.5
        )

        # Initialize
        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(
            frames, (640, 480), "radtan"
        )
        board_poses = initializer.init_board_poses(frames, intrinsics, dist, "radtan")

        # Optimize
        opts = OptimizerOptions(max_iterations=100, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)
        result = optimizer.solve_single_camera(
            frames, intrinsics, dist, board_poses,
            distortion_model="radtan",
            image_size=(640, 480),
        )

        # Check we got a result
        assert len(result.cameras) == 1
        assert len(result.reprojection_errors) == 1
        err = result.reprojection_errors[0]
        assert err.num_observations > 0

        # RMS should be reasonable (< 2 pixels for 0.5 px noise)
        assert result.overall_rms < 2.0

        # Percentiles should be monotonic: median <= p90 <= p95 <= max
        assert err.median <= err.p90 + 1e-9
        assert err.p90 <= err.p95 + 1e-9
        assert err.p95 <= err.max + 1e-9

        # board_pose_times parallel to board_poses
        assert len(result.board_pose_times) == len(result.board_poses)

    def test_optimizer_with_zero_distortion(self, setup):
        """Test optimization with zero distortion (simpler case)."""
        board, _, _ = setup
        K_gt = np.array([
            [500.0, 0, 320.0],
            [0, 500.0, 240.0],
            [0, 0, 1],
        ])
        dist_gt = np.zeros(5)

        frames, _, _ = _generate_synthetic_data(
            board, K_gt, dist_gt, num_frames=15, noise_sigma=0.3
        )

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(
            frames, (640, 480), "radtan"
        )
        board_poses = initializer.init_board_poses(frames, intrinsics, dist, "radtan")

        opts = OptimizerOptions(max_iterations=50)
        optimizer = Optimizer(board, opts)
        result = optimizer.solve_single_camera(
            frames, intrinsics, dist, board_poses,
            distortion_model="radtan",
            image_size=(640, 480),
        )

        assert result.overall_rms < 1.5

    def test_optimizer_preserves_board_pose_count(self, setup):
        """Test that optimizer returns correct number of board poses."""
        board, K_gt, dist_gt = setup
        frames, _, _ = _generate_synthetic_data(
            board, K_gt, dist_gt, num_frames=10, noise_sigma=0.3
        )

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(
            frames, (640, 480), "radtan"
        )
        board_poses = initializer.init_board_poses(frames, intrinsics, dist, "radtan")

        opts = OptimizerOptions(max_iterations=50)
        optimizer = Optimizer(board, opts)
        result = optimizer.solve_single_camera(
            frames, intrinsics, dist, board_poses, "radtan", (640, 480)
        )

        n_valid = sum(1 for p in board_poses if p is not None)
        assert len(result.board_poses) == n_valid

    @pytest.mark.parametrize("pose_count_delta", [-1, 1])
    def test_rejects_frame_pose_length_mismatch(self, setup, pose_count_delta):
        """Every frame list must have a one-to-one pose timeline.

        Silently truncating with ``zip(frames, poses)`` can pair a
        downsampled frame with a pose from a different timestamp, while still
        producing a deceptively small reprojection error.
        """
        board, K_gt, dist_gt = setup
        frames, _, _ = _generate_synthetic_data(
            board, K_gt, dist_gt, num_frames=8, noise_sigma=0.3)
        intrinsics = np.array([
            K_gt[0, 0], K_gt[1, 1], K_gt[0, 2], K_gt[1, 2]])
        dist = dist_gt[:4].copy()
        poses = Initializer(board).init_board_poses(
            frames, intrinsics, dist, "radtan")
        assert len(poses) == len(frames)

        if pose_count_delta < 0:
            mismatched_poses = poses[:pose_count_delta]
        else:
            mismatched_poses = poses + [poses[-1]] * pose_count_delta

        optimizer = Optimizer(board, OptimizerOptions(max_iterations=1))
        with pytest.raises(ValueError):
            optimizer.solve_single_camera(
                frames, intrinsics, dist, mismatched_poses,
                "radtan", (640, 480), camera_id=0)

    def test_valid_frame_indices_match_input_frame_indices(self, setup):
        """Result pose metadata maps each B variable to its input frame.

        ``valid_frame_indices`` uses indices local to the exact frame list
        passed to this solve.  It is parallel to all three board-pose result
        arrays and lets a caller compose that local mapping with an upstream
        subsampling map.
        """
        board, K_gt, dist_gt = setup
        frames, _, _ = _generate_synthetic_data(
            board, K_gt, dist_gt, num_frames=8, noise_sigma=0.3)
        intrinsics = np.array([
            K_gt[0, 0], K_gt[1, 1], K_gt[0, 2], K_gt[1, 2]])
        dist = dist_gt[:4].copy()
        poses = Initializer(board).init_board_poses(
            frames, intrinsics, dist, "radtan")
        assert all(pose is not None for pose in poses)

        poses[2] = None
        poses[5] = None
        expected_indices = [0, 1, 3, 4, 6, 7]

        optimizer = Optimizer(board, OptimizerOptions(max_iterations=1))
        result = optimizer.solve_single_camera(
            frames, intrinsics, dist, poses,
            "radtan", (640, 480), camera_id=0)

        assert result.valid_frame_indices == expected_indices
        assert len(result.valid_frame_indices) == len(result.board_poses)
        assert len(result.valid_frame_indices) == len(result.board_poses_gtsam)
        assert len(result.valid_frame_indices) == len(result.board_pose_times)
        assert result.board_pose_times == pytest.approx(
            [frames[i].timestamp for i in expected_indices])


    def test_valid_frame_indices_skip_low_point_frames_with_pose(self, setup):
        """A frame with a pose but too few points gets no B variable.

        A positional rebuild from non-None init poses would count it and
        shift every later pose by one; valid_frame_indices must not.
        """
        board, K_gt, dist_gt = setup
        frames, _, _ = _generate_synthetic_data(
            board, K_gt, dist_gt, num_frames=8, noise_sigma=0.3)
        intrinsics = np.array([
            K_gt[0, 0], K_gt[1, 1], K_gt[0, 2], K_gt[1, 2]])
        dist = dist_gt[:4].copy()
        poses = Initializer(board).init_board_poses(
            frames, intrinsics, dist, "radtan")
        assert all(pose is not None for pose in poses)

        low = frames[4]
        keep = board.min_points_for_pose() - 1
        frames[4] = FrameObservations(
            camera_id=low.camera_id, timestamp=low.timestamp,
            observations=list(low.observations[:keep]),
            image_size=low.image_size)
        assert frames[4].num_points < board.min_points_for_pose()
        expected_indices = [0, 1, 2, 3, 5, 6, 7]

        optimizer = Optimizer(board, OptimizerOptions(max_iterations=1))
        result = optimizer.solve_single_camera(
            frames, intrinsics, dist, poses,
            "radtan", (640, 480), camera_id=0)

        assert result.valid_frame_indices == expected_indices
        assert len(result.board_poses_gtsam) == len(expected_indices)
        assert result.board_pose_times == pytest.approx(
            [frames[i].timestamp for i in expected_indices])


class TestTimeOffsetPoseBracketing:
    """Test that the optimizer applies dt_init when selecting bracketing poses.

    When a non-ref camera has a large time offset (|dt| > pose spacing),
    the optimizer must shift the lookup time by dt_init to select the
    correct pose pair. Otherwise interpolation lambda exceeds [0,1],
    causing extrapolation and diverged poses.
    """

    @pytest.fixture
    def two_camera_data(self):
        """Synthetic two-camera data with a known time offset.

        Both cameras see the same board moving along a smooth trajectory.
        ref camera (cam0): frames at t=0.0, 0.1, 0.2, ..., 2.9  (30 frames, 10Hz)
        non-ref camera (cam1): frames at t=0.25, 0.35, ..., 2.65 (25 frames, 10Hz)
        True time offset: dt = -0.12s (cam1 timestamps are 0.12s ahead of ref)

        Both cameras use the SAME board poses (co-located cameras with
        identity extrinsic) so that the time offset is the only difference.
        """
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        board_pts = board.all_points_3d().astype(np.float32)

        rng = np.random.RandomState(42)
        image_size = (640, 480)
        dt_true = -0.12  # true time offset

        # Generate a smooth trajectory of board poses
        def make_pose(t):
            """Smooth trajectory: gentle rotation + translation over time."""
            rvec = np.array([0.1 * np.sin(t), 0.15 * np.cos(t), 0.05 * t])
            tvec = np.array([0.05 * np.sin(t * 0.5), 0.03 * np.cos(t * 0.7), 0.6])
            return rvec.astype(np.float64), tvec.astype(np.float64)

        def make_frames(cam_id, timestamps):
            frames = []
            poses = []
            for t in timestamps:
                rvec, tvec = make_pose(t)
                img_pts, _ = cv2.projectPoints(board_pts, rvec, tvec, K, dist)
                img_pts = img_pts.reshape(-1, 2)
                w, h = image_size
                valid = ((img_pts[:, 0] > 10) & (img_pts[:, 0] < w - 10)
                         & (img_pts[:, 1] > 10) & (img_pts[:, 1] < h - 10))
                if not np.all(valid):
                    frames.append(FrameObservations(
                        camera_id=cam_id, timestamp=t,
                        observations=[], image_size=image_size))
                    poses.append(None)
                    continue
                img_pts += rng.randn(*img_pts.shape) * 0.3
                observations = [
                    Observation(point_id=idx, pixel=img_pts[idx],
                                row=int(img_pts[idx, 1]))
                    for idx in range(len(img_pts))
                ]
                R, _ = cv2.Rodrigues(rvec)
                T_cam_board = np.eye(4)
                T_cam_board[:3, :3] = R
                T_cam_board[:3, 3] = tvec
                # Board poses use T_board_cam convention (inverse of PnP)
                pose = gtsam.Pose3(np.linalg.inv(T_cam_board))
                frames.append(FrameObservations(
                    camera_id=cam_id, timestamp=t,
                    observations=observations, image_size=image_size))
                poses.append(pose)
            return frames, poses

        # Ref: 30 frames covering t=0..2.9
        # Non-ref: 25 frames in the middle, well within ref range after dt shift
        ref_times = [i * 0.1 for i in range(30)]
        nonref_times = [0.25 + i * 0.1 for i in range(25)]

        ref_frames, ref_poses = make_frames(0, ref_times)
        nonref_frames, nonref_poses = make_frames(1, nonref_times)

        T_ext = gtsam.Pose3()

        return {
            "board": board,
            "ref_frames": ref_frames,
            "ref_poses": ref_poses,
            "nonref_frames": nonref_frames,
            "nonref_poses": nonref_poses,
            "K": np.array([500.0, 500.0, 320.0, 240.0]),
            "dist": dist[:4],
            "T_ext": T_ext,
            "dt_true": dt_true,
            "image_size": image_size,
        }

    def test_bracketing_with_large_dt(self, two_camera_data):
        """With |dt| > pose spacing, optimizer should still converge.

        Without the dt_init fix, lambda would exceed [0,1] causing
        extrapolation. With the fix, bracketing uses t+dt_init so
        lambda stays in [0,1].
        """
        d = two_camera_data
        board = d["board"]

        all_frames = {0: d["ref_frames"], 1: d["nonref_frames"]}
        all_intrinsics = {0: d["K"].copy(), 1: d["K"].copy()}
        all_dist_coeffs = {0: d["dist"].copy(), 1: d["dist"].copy()}
        all_board_poses = {0: d["ref_poses"], 1: d["nonref_poses"]}
        distortion_models = {0: "radtan", 1: "radtan"}
        image_sizes = {0: d["image_size"], 1: d["image_size"]}
        extrinsics = {1: d["T_ext"]}

        opts = OptimizerOptions(
            max_iterations=50, pixel_sigma=1.0,
            optimize_time_offset=True,
        )
        optimizer = Optimizer(board, opts)

        # With dt_init = dt_true, the bracketing should be correct
        result = optimizer.solve_multi_camera(
            all_frames, all_intrinsics, all_dist_coeffs,
            all_board_poses, extrinsics, distortion_models,
            reference_camera=0, image_sizes=image_sizes,
            time_offsets_init={1: d["dt_true"]},
        )

        assert result is not None
        assert len(result.cameras) == 2

        # The optimizer should produce a valid result with calibrated
        # intrinsics and reprojection errors. With dt-aware bracketing,
        # the optimization should converge (no numerical explosion).
        # Note: some board poses may be removed by the z-depth sanitizer,
        # which is the correct behavior for this challenging case.

        # Key check: reprojection errors should be reasonable
        for err in result.reprojection_errors:
            assert err.median < 10.0, \
                f"cam{err.camera_id} median={err.median:.1f}px too high"
            # Max error should not explode (no numerical divergence)
            assert err.max < 1000.0, \
                f"cam{err.camera_id} max={err.max:.1f}px — numerical divergence"

    def test_bracketing_without_dt_init_zero(self, two_camera_data):
        """With dt_init=0 (no time offset), behavior should be unchanged."""
        d = two_camera_data
        board = d["board"]

        all_frames = {0: d["ref_frames"], 1: d["nonref_frames"]}
        all_intrinsics = {0: d["K"].copy(), 1: d["K"].copy()}
        all_dist_coeffs = {0: d["dist"].copy(), 1: d["dist"].copy()}
        all_board_poses = {0: d["ref_poses"], 1: d["nonref_poses"]}
        distortion_models = {0: "radtan", 1: "radtan"}
        image_sizes = {0: d["image_size"], 1: d["image_size"]}
        extrinsics = {1: d["T_ext"]}

        opts = OptimizerOptions(
            max_iterations=50, pixel_sigma=1.0,
            optimize_time_offset=True,
        )
        optimizer = Optimizer(board, opts)

        # dt_init=0 — no shift, same as old behavior
        result = optimizer.solve_multi_camera(
            all_frames, all_intrinsics, all_dist_coeffs,
            all_board_poses, extrinsics, distortion_models,
            reference_camera=0, image_sizes=image_sizes,
            time_offsets_init={1: 0.0},
        )

        assert result is not None
        assert len(result.cameras) == 2
        for err in result.reprojection_errors:
            assert err.p95 < 10.0

    def test_single_camera_unaffected(self, two_camera_data):
        """Single-camera optimization should be completely unaffected."""
        d = two_camera_data
        board = d["board"]

        opts = OptimizerOptions(max_iterations=50, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)

        valid_frames = [f for f, p in zip(d["ref_frames"], d["ref_poses"])
                        if p is not None and f.observations]
        valid_poses = [p for p in d["ref_poses"] if p is not None]
        # Trim to match
        n = min(len(valid_frames), len(valid_poses))
        valid_frames = valid_frames[:n]
        valid_poses = valid_poses[:n]

        result = optimizer.solve_single_camera(
            d["ref_frames"], d["K"].copy(), d["dist"].copy(),
            d["ref_poses"], "radtan", d["image_size"],
        )

        assert result is not None
        assert len(result.cameras) == 1

    def test_stage4_downweights_the_matching_timestamp(
        self, two_camera_data, caplog,
    ):
        """Filtering/reindexing must not change which sample is downweighted."""
        d = two_camera_data
        target_frame = d["nonref_frames"][7]
        target_frame.observations = target_frame.observations[:7]
        assert len(target_frame.observations) == 7
        # Simulate a Stage-4 list that removed an earlier Stage-3 sample: the
        # target moves from local index 7 to local index 6, but its timestamp
        # identity remains unchanged.
        nonref_frames = list(d["nonref_frames"])
        nonref_poses = list(d["nonref_poses"])
        del nonref_frames[2]
        del nonref_poses[2]
        assert nonref_frames[6] is target_frame

        optimizer = Optimizer(
            d["board"],
            OptimizerOptions(max_iterations=1, pixel_sigma=1.0),
        )
        with caplog.at_level(logging.INFO, logger="mvis.calibration.optimizer"):
            optimizer.solve_multi_camera(
                {0: d["ref_frames"], 1: nonref_frames},
                {0: d["K"].copy(), 1: d["K"].copy()},
                {0: d["dist"].copy(), 1: d["dist"].copy()},
                {0: d["ref_poses"], 1: nonref_poses},
                {1: d["T_ext"]},
                {0: "radtan", 1: "radtan"},
                reference_camera=0,
                image_sizes={0: d["image_size"], 1: d["image_size"]},
                outlier_frame_timestamps={1: {target_frame.timestamp}},
            )

        assert "7 downweighted" in caplog.text

    def test_legacy_outlier_indices_are_converted_at_the_boundary(
        self, two_camera_data, caplog,
    ):
        """The public legacy index API remains explicit and compatible."""
        d = two_camera_data
        target_idx = 7
        target_frame = d["nonref_frames"][target_idx]
        target_frame.observations = target_frame.observations[:9]

        optimizer = Optimizer(
            d["board"],
            OptimizerOptions(max_iterations=1, pixel_sigma=1.0),
        )
        with caplog.at_level(logging.INFO, logger="mvis.calibration.optimizer"):
            optimizer.solve_multi_camera(
                {0: d["ref_frames"], 1: d["nonref_frames"]},
                {0: d["K"].copy(), 1: d["K"].copy()},
                {0: d["dist"].copy(), 1: d["dist"].copy()},
                {0: d["ref_poses"], 1: d["nonref_poses"]},
                {1: d["T_ext"]},
                {0: "radtan", 1: "radtan"},
                reference_camera=0,
                image_sizes={0: d["image_size"], 1: d["image_size"]},
                outlier_frames={1: {target_idx}},
            )

        assert "9 downweighted" in caplog.text

    def test_rejects_mixed_outlier_identity_apis(self, two_camera_data):
        """Callers must not ambiguously provide both indices and timestamps."""
        d = two_camera_data
        optimizer = Optimizer(
            d["board"],
            OptimizerOptions(max_iterations=1, pixel_sigma=1.0),
        )

        with pytest.raises(ValueError, match="either legacy outlier_frames"):
            optimizer.solve_multi_camera(
                {0: d["ref_frames"], 1: d["nonref_frames"]},
                {0: d["K"].copy(), 1: d["K"].copy()},
                {0: d["dist"].copy(), 1: d["dist"].copy()},
                {0: d["ref_poses"], 1: d["nonref_poses"]},
                {1: d["T_ext"]},
                {0: "radtan", 1: "radtan"},
                outlier_frames={},
                outlier_frame_timestamps={},
            )


class TestDivergedPoseRemoval:
    """Test that diverged board poses are properly handled.

    The optimizer replaces diverged poses (norm > threshold) with identity
    matrices to preserve B-variable indexing. The pipeline removes diverged
    frames before Stage 5 rebuild so the factor graph is reconstructed
    without them.
    """

    def test_list_length_invariant(self):
        """board_poses, board_poses_gtsam, board_pose_times must all be
        the same length (== bp_count) regardless of diverged poses."""
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        board_pts = board.all_points_3d().astype(np.float32)
        rng = np.random.RandomState(42)
        image_size = (640, 480)

        frames, gt_rvecs, gt_tvecs = _generate_synthetic_data(
            board, K, dist, num_frames=15, noise_sigma=0.3)

        initializer = Initializer(board)
        intrinsics, d = initializer.init_intrinsics_opencv(
            frames, image_size, "radtan")
        board_poses = initializer.init_board_poses(frames, intrinsics, d, "radtan")

        opts = OptimizerOptions(max_iterations=50, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)
        result = optimizer.solve_single_camera(
            frames, intrinsics, d, board_poses, "radtan", image_size)

        assert result is not None
        assert len(result.board_poses) == len(result.board_poses_gtsam)
        assert len(result.board_poses) == len(result.board_pose_times)

    def test_diverged_stats_populated(self):
        """diverged_pose_stats should always be populated."""
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])

        frames, _, _ = _generate_synthetic_data(
            board, K, dist, num_frames=10, noise_sigma=0.3)

        initializer = Initializer(board)
        intrinsics, d = initializer.init_intrinsics_opencv(
            frames, (640, 480), "radtan")
        board_poses = initializer.init_board_poses(
            frames, intrinsics, d, "radtan")

        opts = OptimizerOptions(max_iterations=50, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)
        result = optimizer.solve_single_camera(
            frames, intrinsics, d, board_poses, "radtan", (640, 480))

        assert result.diverged_pose_stats is not None
        assert "total" in result.diverged_pose_stats
        assert "removed" in result.diverged_pose_stats
        assert "kept" in result.diverged_pose_stats
        assert result.diverged_pose_stats["total"] == \
            result.diverged_pose_stats["removed"] + \
            result.diverged_pose_stats["kept"]

    def test_identity_poses_not_in_valid_poses(self):
        """When filtering for 3D plotting, identity poses (from sanitizer)
        should be excluded."""
        # Simulate a result with some identity poses
        poses = [
            np.array([[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.6], [0, 0, 0, 1]]),
            np.eye(4),  # diverged — replaced with identity
            np.array([[1, 0, 0, -0.1], [0, 1, 0, 0.1], [0, 0, 1, 0.8], [0, 0, 0, 1]]),
            np.eye(4),  # diverged
        ]

        # The standard filter used before 3D plotting
        valid_poses = [p for p in poses
                       if p is not None and not np.allclose(p, np.eye(4))]
        assert len(valid_poses) == 2

    def test_timeline_indexing_with_identity_poses(self):
        """Downstream timeline indexing must work even when some
        board_poses are identity (diverged). The B-variable indexing
        must be preserved: position i = B-variable i."""
        from mvis.calibration.factors.camera_projection import find_bracketing_poses

        # Simulate 10 poses with 2 diverged (identity)
        n = 10
        times = [float(i) * 0.1 for i in range(n)]
        poses = []
        for i in range(n):
            if i in (3, 7):  # diverged
                poses.append(np.eye(4))
            else:
                T = np.eye(4)
                T[2, 3] = 0.6  # board at 0.6m
                poses.append(T)

        # Build timeline — include ALL poses (identity included)
        # This is what generate_report.py and outlier_rejection.py do
        bp_timeline_t = np.array(times)
        bp_timeline_idx = list(range(n))

        # Bracket lookup should return valid indices
        bracket = find_bracketing_poses(0.15, bp_timeline_t, bp_timeline_idx)
        assert bracket is not None
        bp0_idx, bp1_idx, t0, t1 = bracket
        assert 0 <= bp0_idx < n
        assert 0 <= bp1_idx < n
        # The poses at these indices exist (may be identity but won't crash)
        assert poses[bp0_idx] is not None
        assert poses[bp1_idx] is not None


class TestStage5WarmStart:
    """Test that Stage 5 warm-starts board poses from Stage 4 results."""

    def test_warm_start_pose_lookup_by_timestamp(self):
        """Stage 4 poses can be looked up by rounded timestamp."""
        # Simulate Stage 4 result with board_poses and board_pose_times
        times = [1.0, 1.1, 1.2, 1.3, 1.4]
        poses = [
            np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0.6], [0, 0, 0, 1]]),
            np.eye(4),  # identity — should be skipped
            np.array([[1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 1, 0.7], [0, 0, 0, 1]]),
            None,  # None — should be skipped
            np.array([[1, 0, 0, 0.05], [0, 1, 0, 0], [0, 0, 1, 0.8], [0, 0, 0, 1]]),
        ]

        # Build lookup (same logic as pipeline)
        s4_pose_by_time = {}
        for bp, t in zip(poses, times):
            if bp is not None and not np.allclose(bp, np.eye(4)):
                s4_pose_by_time[round(t, 6)] = bp

        # Should have 3 entries (skipped identity + None)
        assert len(s4_pose_by_time) == 3
        assert round(1.0, 6) in s4_pose_by_time
        assert round(1.1, 6) not in s4_pose_by_time  # identity
        assert round(1.2, 6) in s4_pose_by_time
        assert round(1.3, 6) not in s4_pose_by_time  # None
        assert round(1.4, 6) in s4_pose_by_time

    def test_pnp_rejected_not_overridden(self):
        """Frames rejected by PnP gate (pose=None) must stay rejected,
        even if a Stage 4 warm-start pose exists."""
        s4_pose_by_time = {
            round(1.0, 6): gtsam.Pose3(),
            round(1.1, 6): gtsam.Pose3(),
        }

        # Simulate new_poses from quality gate: frame at t=1.0 rejected
        pnp_poses = [None, gtsam.Pose3()]  # first rejected, second ok
        frame_times = [1.0, 1.1]

        # Apply warm-start logic
        for fi, pose in enumerate(pnp_poses):
            if pose is None:
                continue  # PnP rejected — keep rejected
            t_key = round(frame_times[fi], 6)
            s5 = s4_pose_by_time.get(t_key)
            if s5 is not None:
                pnp_poses[fi] = s5

        # First frame should stay None (PnP rejected)
        assert pnp_poses[0] is None
        # Second frame should be warm-started
        assert pnp_poses[1] is not None

    def test_stage4_result_has_pose_times(self):
        """Multi-camera solve should produce board_pose_times parallel
        to board_poses, enabling timestamp-based warm-start lookup."""
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K_gt = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist_gt = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])

        frames, _, _ = _generate_synthetic_data(
            board, K_gt, dist_gt, num_frames=15, noise_sigma=0.3)

        initializer = Initializer(board)
        intrinsics, dist = initializer.init_intrinsics_opencv(
            frames, (640, 480), "radtan")
        board_poses = initializer.init_board_poses(
            frames, intrinsics, dist, "radtan")

        opts = OptimizerOptions(max_iterations=50, pixel_sigma=1.0)
        optimizer = Optimizer(board, opts)
        result = optimizer.solve_single_camera(
            frames, intrinsics, dist, board_poses, "radtan", (640, 480))

        # board_pose_times must be parallel to board_poses
        assert len(result.board_pose_times) == len(result.board_poses)
        # All times should be positive (valid timestamps)
        for t in result.board_pose_times:
            assert t >= 0.0


class TestPnPFullCorners:
    """Test that PnP uses full-corner frames when pnp_frames is provided."""

    def test_pnp_frames_used_when_provided(self):
        """apply_quality_gates should use pnp_frames for PnP init."""
        from mvis.detection.quality_gate import (
            QualityGateConfig, apply_quality_gates,
        )
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        board_pts = board.all_points_3d().astype(np.float32)
        rng = np.random.RandomState(42)

        # Generate frames with all corners
        full_frames = []
        for i in range(10):
            rvec = rng.uniform(-0.3, 0.3, 3).astype(np.float64)
            tvec = np.array([rng.uniform(-0.05, 0.05),
                             rng.uniform(-0.05, 0.05), 0.6])
            img_pts, _ = cv2.projectPoints(board_pts, rvec, tvec, K, dist)
            img_pts = img_pts.reshape(-1, 2)
            if not np.all((img_pts[:, 0] > 10) & (img_pts[:, 0] < 630)
                          & (img_pts[:, 1] > 10) & (img_pts[:, 1] < 470)):
                continue
            obs = [Observation(point_id=j, pixel=img_pts[j],
                               row=int(img_pts[j, 1]))
                   for j in range(len(img_pts))]
            full_frames.append(FrameObservations(
                camera_id=0, timestamp=float(i),
                observations=obs, image_size=(640, 480)))

        # Create subsampled frames (top_left = point_id % 4 == 0)
        sub_frames = []
        for f in full_frames:
            sub_obs = [o for o in f.observations if o.point_id % 4 == 0]
            sub_frames.append(FrameObservations(
                camera_id=0, timestamp=f.timestamp,
                observations=sub_obs, image_size=(640, 480)))

        intrinsics = {0: np.array([500.0, 500.0, 320.0, 240.0])}
        dist_coeffs = {0: dist[:4]}
        dm = {0: "radtan"}

        # Run WITH pnp_frames (full corners)
        cfg = QualityGateConfig(
            min_tags_per_frame=0,
            reinit_board_poses=True,
        )
        _, poses_full, _ = apply_quality_gates(
            {0: sub_frames}, board, intrinsics, dist_coeffs, dm, cfg,
            pnp_frames={0: full_frames},
        )

        # Run WITHOUT pnp_frames (subsampled only)
        _, poses_sub, _ = apply_quality_gates(
            {0: sub_frames}, board, intrinsics, dist_coeffs, dm, cfg,
        )

        # Both should produce poses for the same frames
        assert len(poses_full[0]) == len(poses_sub[0])

        # Both should have valid (non-None) poses for same frames
        n_valid_full = sum(1 for p in poses_full[0] if p is not None)
        n_valid_sub = sum(1 for p in poses_sub[0] if p is not None)
        assert n_valid_full >= n_valid_sub  # full corners can only help

    def test_pnp_frames_none_preserves_behavior(self):
        """When pnp_frames=None, behavior is unchanged (uses all_frames)."""
        from mvis.detection.quality_gate import (
            QualityGateConfig, apply_quality_gates,
        )
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])

        frames, _, _ = _generate_synthetic_data(
            board, K, dist, num_frames=8, noise_sigma=0.3)

        intrinsics = {0: np.array([500.0, 500.0, 320.0, 240.0])}
        dist_coeffs = {0: dist[:4]}
        dm = {0: "radtan"}

        cfg = QualityGateConfig(
            min_tags_per_frame=0,
            reinit_board_poses=True,
        )

        # Run with pnp_frames=None (default)
        _, poses_default, _ = apply_quality_gates(
            {0: frames}, board, intrinsics, dist_coeffs, dm, cfg,
            pnp_frames=None,
        )

        # Should produce valid poses
        n_valid = sum(1 for p in poses_default[0] if p is not None)
        assert n_valid > 0

    def test_timestamp_matching(self):
        """pnp_frames must match filtered frames by timestamp."""
        # If a frame is removed by quality gate (min_tags), the
        # corresponding pnp_frame should also be skipped
        from mvis.detection.quality_gate import (
            QualityGateConfig, apply_quality_gates,
        )
        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        board_pts = board.all_points_3d().astype(np.float32)
        rng = np.random.RandomState(42)

        full_frames = []
        sub_frames = []
        for i in range(10):
            rvec = rng.uniform(-0.3, 0.3, 3).astype(np.float64)
            tvec = np.array([rng.uniform(-0.05, 0.05),
                             rng.uniform(-0.05, 0.05), 0.6])
            img_pts, _ = cv2.projectPoints(board_pts, rvec, tvec, K, dist)
            img_pts = img_pts.reshape(-1, 2)
            if not np.all((img_pts[:, 0] > 10) & (img_pts[:, 0] < 630)
                          & (img_pts[:, 1] > 10) & (img_pts[:, 1] < 470)):
                continue
            all_obs = [Observation(point_id=j, pixel=img_pts[j],
                                   row=int(img_pts[j, 1]))
                       for j in range(len(img_pts))]
            sub_obs = [o for o in all_obs if o.point_id % 4 == 0]

            full_frames.append(FrameObservations(
                camera_id=0, timestamp=float(i),
                observations=all_obs, image_size=(640, 480)))
            sub_frames.append(FrameObservations(
                camera_id=0, timestamp=float(i),
                observations=sub_obs, image_size=(640, 480)))

        intrinsics = {0: np.array([500.0, 500.0, 320.0, 240.0])}
        dist_coeffs = {0: dist[:4]}
        dm = {0: "radtan"}

        # min_tags_per_frame=10 will drop frames with few subsampled obs
        cfg = QualityGateConfig(
            min_tags_per_frame=3,
            reinit_board_poses=True,
        )
        filtered, poses, _ = apply_quality_gates(
            {0: sub_frames}, board, intrinsics, dist_coeffs, dm, cfg,
            pnp_frames={0: full_frames},
        )

        # Poses length should match filtered frames
        assert len(poses[0]) == len(filtered[0])


class TestAdaptiveSubsampleAndDualFilter:
    """Test adaptive corner subsampling and dual quality filter."""

    def test_upgrade_excellent_frame(self):
        """Frame with 100% tags → all corners."""
        from mvis.detection.corner_subsample import upgrade_high_quality_frames
        # 36/36 tags = 100% (excellent)
        sub_obs = [Observation(point_id=i*4, pixel=np.array([i*10.0, 0.0]))
                   for i in range(36)]
        full_obs = [Observation(point_id=i*4+j, pixel=np.array([i*10.0+j, j*1.0]))
                    for i in range(36) for j in range(4)]
        sub_f = FrameObservations(camera_id=0, timestamp=1.0,
                                   observations=sub_obs, image_size=(640, 480))
        full_f = FrameObservations(camera_id=0, timestamp=1.0,
                                    observations=full_obs, image_size=(640, 480))
        result = upgrade_high_quality_frames(
            {0: [sub_f]}, {0: [full_f]}, "top_left", num_tags=36)
        # 36/36 = 100% → all corners (144 obs)
        assert len(result[0][0].observations) == 144

    def test_upgrade_good_frame(self):
        """Frame with 95%+ tags → two_diag."""
        from mvis.detection.corner_subsample import upgrade_high_quality_frames
        # 35/36 tags = 97% (good)
        sub_obs = [Observation(point_id=i*4, pixel=np.array([i*10.0, 0.0]))
                   for i in range(35)]
        full_obs = [Observation(point_id=i*4+j, pixel=np.array([i*10.0+j, j*1.0]))
                    for i in range(35) for j in range(4)]
        sub_f = FrameObservations(camera_id=0, timestamp=1.0,
                                   observations=sub_obs, image_size=(640, 480))
        full_f = FrameObservations(camera_id=0, timestamp=1.0,
                                    observations=full_obs, image_size=(640, 480))
        result = upgrade_high_quality_frames(
            {0: [sub_f]}, {0: [full_f]}, "top_left", num_tags=36)
        # 35/36 = 97%: 95% <= 97% < 100% → two_diag (70 obs)
        assert len(result[0][0].observations) == 70

    def test_no_upgrade_normal_frame(self):
        """Frame with < 95% tags → stays top_left."""
        from mvis.detection.corner_subsample import upgrade_high_quality_frames
        # 20/36 tags = 56%
        sub_obs = [Observation(point_id=i*4, pixel=np.array([i*10.0, 0.0]))
                   for i in range(20)]
        full_obs = [Observation(point_id=i*4+j, pixel=np.array([i*10.0+j, j*1.0]))
                    for i in range(20) for j in range(4)]
        sub_f = FrameObservations(camera_id=0, timestamp=1.0,
                                   observations=sub_obs, image_size=(640, 480))
        full_f = FrameObservations(camera_id=0, timestamp=1.0,
                                    observations=full_obs, image_size=(640, 480))
        result = upgrade_high_quality_frames(
            {0: [sub_f]}, {0: [full_f]}, "top_left", num_tags=36)
        # 20/36 = 56% < 85% → stays top_left (20 obs)
        assert len(result[0][0].observations) == 20

    def test_upgrade_mode_all(self):
        """mode='all' → no change."""
        from mvis.detection.corner_subsample import upgrade_high_quality_frames
        obs = [Observation(point_id=i, pixel=np.array([0.0, 0.0])) for i in range(10)]
        f = FrameObservations(camera_id=0, timestamp=1.0,
                               observations=obs, image_size=(640, 480))
        result = upgrade_high_quality_frames({0: [f]}, {0: [f]}, "all", num_tags=36)
        assert result[0][0] is f

    def test_dual_filter_6tags_6pts(self):
        """6 tags × 1 corner = 6 pts → KEPT (both pass)."""
        from mvis.detection.frame_quality import filter_frames_dual
        obs = [Observation(point_id=i*4, pixel=np.array([0.0, 0.0]))
               for i in range(6)]
        f = FrameObservations(camera_id=0, timestamp=1.0,
                               observations=obs, image_size=(640, 480))
        result = filter_frames_dual([f], min_tag_count=4, min_point_count=6)
        assert len(result) == 1

    def test_dual_filter_5tags_20pts(self):
        """5 tags × 4 corners = 20 pts → KEPT (both pass)."""
        from mvis.detection.frame_quality import filter_frames_dual
        obs = [Observation(point_id=i*4+j, pixel=np.array([0.0, 0.0]))
               for i in range(5) for j in range(4)]
        f = FrameObservations(camera_id=0, timestamp=1.0,
                               observations=obs, image_size=(640, 480))
        result = filter_frames_dual([f], min_tag_count=4, min_point_count=6)
        assert len(result) == 1

    def test_dual_filter_5tags_5pts_dropped(self):
        """5 tags × 1 corner = 5 pts → DROP (pts < 10)."""
        from mvis.detection.frame_quality import filter_frames_dual
        obs = [Observation(point_id=i*4, pixel=np.array([0.0, 0.0]))
               for i in range(5)]
        f = FrameObservations(camera_id=0, timestamp=1.0,
                               observations=obs, image_size=(640, 480))
        result = filter_frames_dual([f], min_tag_count=4, min_point_count=10)
        assert len(result) == 0  # 5 pts < 10

    def test_dual_filter_2tags_8pts_dropped(self):
        """2 tags × 4 corners = 8 pts → DROP (tags < 4)."""
        from mvis.detection.frame_quality import filter_frames_dual
        obs = [Observation(point_id=i*4+j, pixel=np.array([0.0, 0.0]))
               for i in range(2) for j in range(4)]
        f = FrameObservations(camera_id=0, timestamp=1.0,
                               observations=obs, image_size=(640, 480))
        result = filter_frames_dual([f], min_tag_count=4, min_point_count=6)
        assert len(result) == 0  # 2 tags < 4

    def test_dual_filter_4tags_topleft_dropped(self):
        """4 tags × 1 corner = 4 pts → DROP (pts < 10)."""
        from mvis.detection.frame_quality import filter_frames_dual
        obs = [Observation(point_id=i*4, pixel=np.array([0.0, 0.0]))
               for i in range(4)]
        f = FrameObservations(camera_id=0, timestamp=1.0,
                               observations=obs, image_size=(640, 480))
        result = filter_frames_dual([f], min_tag_count=4, min_point_count=10)
        assert len(result) == 0  # 4 pts < 10

    def test_dual_filter_4tags_all_kept(self):
        """4 tags × 4 corners = 16 pts → KEPT (both pass)."""
        from mvis.detection.frame_quality import filter_frames_dual
        obs = [Observation(point_id=i*4+j, pixel=np.array([0.0, 0.0]))
               for i in range(4) for j in range(4)]
        f = FrameObservations(camera_id=0, timestamp=1.0,
                               observations=obs, image_size=(640, 480))
        result = filter_frames_dual([f], min_tag_count=4, min_point_count=10)
        assert len(result) == 1  # 4 tags >=4, 16 pts >= 10
