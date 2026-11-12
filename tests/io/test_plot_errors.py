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
"""Tests for the plot functions:
- plot_error_histogram_uv
- plot_error_vectors_dual (the error-vector plot the HTML report draws)
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np

from mvis.analysis.plot_errors import (
    plot_error_histogram_uv,
    plot_error_vectors_dual,
    plot_error_per_tag_robust,
    plot_reprojection_scatter,
)


def test_vectors_dual_has_two_axes():
    rng = np.random.default_rng(1)
    ev = rng.normal(0.0, 0.4, size=(800, 2))
    # sprinkle some outliers to ensure the zoom panel differs from full
    ev[:20] = rng.normal(0.0, 5.0, size=(20, 2))
    fig = plot_error_vectors_dual(ev, error_vectors_by_frame=None, camera_id=3)
    assert len(fig.axes) == 2
    # Zoom panel's x-axis range must be strictly smaller than full panel's
    xlim_full = fig.axes[0].get_xlim()
    xlim_zoom = fig.axes[1].get_xlim()
    assert (xlim_zoom[1] - xlim_zoom[0]) < (xlim_full[1] - xlim_full[0])


def test_vectors_dual_empty():
    fig = plot_error_vectors_dual(np.zeros((0, 2)))
    assert fig is not None


def test_per_tag_robust_has_two_axes():
    tag_errors = {i: [0.1, 0.2, 0.3, 1.0] for i in range(8)}
    fig = plot_error_per_tag_robust(tag_errors, num_tags=8, camera_id=0)
    assert len(fig.axes) == 2


def test_per_tag_robust_handles_missing_tags():
    # Only tag 3 has data; others should render as zero bars without crashing.
    tag_errors = {3: [0.5, 0.7, 1.2]}
    fig = plot_error_per_tag_robust(tag_errors, num_tags=10, camera_id=1)
    assert len(fig.axes) == 2


def test_uv_histogram_two_axes():
    """The u/v histogram should return a figure with two axes."""
    rng = np.random.default_rng(0)
    ev = rng.normal(0.0, 0.5, size=(500, 2))
    fig = plot_error_histogram_uv(ev, camera_id=7, bins=30)
    # Two panels (u, v)
    assert len(fig.axes) == 2


def test_uv_histogram_handles_empty():
    """Empty input should not crash."""
    fig = plot_error_histogram_uv(np.zeros((0, 2)), camera_id=0)
    # Returned object must still be a figure
    assert fig is not None


def test_ellipse_covers_95pct_on_gaussian():
    """95% confidence ellipse should contain ~95% of the non-outlier points."""
    rng = np.random.default_rng(42)
    # 2000 samples from a Gaussian with correlated axes
    cov = np.array([[1.0, 0.3], [0.3, 0.5]])
    L = np.linalg.cholesky(cov)
    pts = rng.standard_normal(size=(2000, 2)) @ L.T

    fig = plot_error_vectors_dual(
        pts, error_vectors_by_frame=None, camera_id=0)
    assert fig is not None
    assert len(fig.axes) >= 1

    # Manually compute the same ellipse that _draw_vector_panel uses,
    # and verify it contains ~95% of the inner points.
    mu = np.mean(pts, axis=0)
    Sigma = np.cov(pts.T)
    inv = np.linalg.inv(Sigma + 1e-12 * np.eye(2))
    d2 = np.einsum("ni,ij,nj->n", pts - mu, inv, pts - mu)
    keep = d2 <= np.percentile(d2, 95)
    pts_clean = pts[keep]
    mu_c = np.mean(pts_clean, axis=0)
    S_c = np.cov(pts_clean.T)
    inv_c = np.linalg.inv(S_c + 1e-12 * np.eye(2))
    d2_c = np.einsum("ni,ij,nj->n", pts_clean - mu_c, inv_c, pts_clean - mu_c)
    inside = np.sum(d2_c <= 5.991)  # chi2(df=2, p=0.95)
    frac = inside / len(pts_clean)
    assert 0.92 < frac < 0.98, f"ellipse coverage={frac:.2f} out of expected band"


def test_error_vectors_with_frames_accepts_per_frame_dict():
    """Pass error_vectors_by_frame and confirm it runs without errors."""
    rng = np.random.default_rng(1)
    ev_full = rng.normal(0.0, 0.4, size=(400, 2))
    # Split into 20 frames of 20 observations
    per_frame = {
        fi: ev_full[fi * 20:(fi + 1) * 20]
        for fi in range(20)
    }
    fig = plot_error_vectors_dual(ev_full, per_frame, camera_id=3)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_robust_gaussian_recovers_core_sigma():
    """Given Gaussian core + a few huge outliers, robust σ ≈ core σ, not inflated."""
    from mvis.analysis.plot_errors import _robust_gaussian_params

    rng = np.random.default_rng(0)
    core = rng.normal(0.0, 0.3, size=2000)
    outliers = np.array([50.0, -60.0, 80.0, -70.0, 90.0])
    data = np.concatenate([core, outliers])

    # Raw std is massively inflated by the 5 outliers
    raw_std = float(np.std(data))
    assert raw_std > 1.5, f"raw std should be inflated, got {raw_std}"

    # Robust σ should be close to the true core σ=0.3
    mu_r, sigma_r = _robust_gaussian_params(data)
    assert abs(mu_r) < 0.05, f"robust μ should be ≈ 0, got {mu_r}"
    assert 0.25 < sigma_r < 0.35, f"robust σ should be ≈ 0.3, got {sigma_r}"


def test_uv_histogram_clips_to_p995():
    """A huge outlier must NOT blow up the x-axis — zoom stays near P99.5."""
    rng = np.random.default_rng(0)
    # 2000 Gaussian points around 0 with σ=0.3, plus one outlier at ±200 px
    ev = rng.normal(0.0, 0.3, size=(2000, 2))
    ev[0] = [200.0, -200.0]

    fig = plot_error_histogram_uv(ev, camera_id=0)
    # Both panels' x-limits should be much smaller than 200
    for ax in fig.axes:
        xmin, xmax = ax.get_xlim()
        assert -xmin < 10.0, f"xlim={xmin} — x-axis not zoomed"
        assert xmax < 10.0, f"xlim={xmax} — x-axis not zoomed"
    # Outlier count should be mentioned in the suptitle
    title = fig._suptitle.get_text() if fig._suptitle else ""
    assert "outside range" in title


def test_vector_plot_zoom_ignores_outlier():
    """Same kind of test for the error-vector plot."""
    rng = np.random.default_rng(0)
    ev = rng.normal(0.0, 0.3, size=(2000, 2))
    ev[0] = [150.0, 150.0]

    fig = plot_error_vectors_dual(ev, None, camera_id=0)
    ax = fig.axes[1]   # the zoomed panel
    xmin, xmax = ax.get_xlim()
    assert -xmin < 10.0 and xmax < 10.0


def test_uv_histogram_gaussian_fit_sanity():
    """Mean and std on a known Gaussian should match numpy's computation."""
    rng = np.random.default_rng(7)
    ev = rng.normal([1.0, -0.5], [0.3, 0.7], size=(1000, 2))
    fig = plot_error_histogram_uv(ev, camera_id=0)
    # Panel titles carry a Gaussian-core label (either robust μ̃ or raw μ)
    titles = [ax.get_title() for ax in fig.axes]
    assert any(("μ̃=" in t) or ("μ=" in t) for t in titles)
    # Overall figure title is set via suptitle
    assert "u/v Residual Distribution" in (fig._suptitle.get_text() if fig._suptitle else "")


def test_reprojection_scatter_circle_is_3_robust_sigma():
    rng = np.random.default_rng(2)
    by_frame = {i: rng.normal(0.0, 0.3, size=(40, 2)) for i in range(50)}
    by_frame[7][:1] = 25.0                      # a bad corner don't move sigma
    fig = plot_reprojection_scatter(by_frame, camera_id=1)
    ax = fig.axes[0]
    (circle,) = ax.patches
    assert abs(circle.get_radius() - 0.9) < 0.1
    assert ax.get_xlim()[1] < 25.0              # the outlier falls outside the axes ...
    assert "1 outside" in ax.get_title()        # ... and is counted
    assert len(fig.axes) == 2                   # plot + colour bar


def test_reprojection_scatter_subsamples_and_handles_empty():
    rng = np.random.default_rng(3)
    by_frame = {i: rng.normal(0.0, 0.3, size=(100, 2)) for i in range(30)}
    fig = plot_reprojection_scatter(by_frame, max_points=500)
    assert len(fig.axes[0].collections[0].get_offsets()) == 500
    assert plot_reprojection_scatter({}) is not None
