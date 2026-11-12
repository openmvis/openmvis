#!/usr/bin/env bash
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
# =============================================================================
# MVIS Environment Setup Script
#
# Creates a conda environment 'mvis' with all required dependencies
# and installs the MVIS package in development mode.
#
# Usage:
#   bash scripts/setup_env.sh
#
# After running:
#   conda activate mvis
#   mvis check --env --smoke      # installation verification
#   mvis run -c <config.yaml>     # config needs `mode: cam` or `mode: cam-imu`
#   scripts/test.sh unit
# =============================================================================

set -e

ENV_NAME="mvis"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " MVIS Environment Setup"
echo "============================================"
echo ""
echo "Repo:        $REPO_DIR"
echo "Env name:    $ENV_NAME"
echo "Python:      $PYTHON_VERSION"
echo ""

# --- Check conda is available ---
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# --- Remove existing env if requested ---
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Reusing existing environment."
    fi
fi

# --- Create conda environment ---
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# --- Activate environment ---
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# --- Install MVIS in development mode, with the extras ---
# pyproject.toml is the single dependency list (numpy<2 for the GTSAM 4.2
# wheels, OpenCV <4.12 for numpy<2, gtsam>=4.2,<4.3); the extras are AprilTag
# boards, ArUco boards (opencv-contrib), ROS1 bags, the rerun viewer and pytest.
echo ""
echo "Installing MVIS with extras (pip install -e \".[apriltag,aruco,ros1,viz,dev]\")..."
pip install --upgrade pip
cd "$REPO_DIR"
pip install -e ".[apriltag,aruco,ros1,viz,dev]"

# ffmpeg: MP4 encoder for the calibration/sim videos. matplotlib's FFMpegWriter
# needs the `ffmpeg` binary on PATH; without it the renderer silently falls back
# to GIF (Pillow). Installed from conda-forge.
conda install -n "$ENV_NAME" -c conda-forge ffmpeg -y \
    || echo "WARNING: ffmpeg install failed (videos -> GIF fallback instead of MP4)"

# --- Verify installation ---
echo ""
echo "============================================"
echo " Verifying installation..."
echo "============================================"
echo ""

# Installation verification (Stage 0): versions, extras, CLI, and a short synthetic solve.
mvis check --env --smoke || echo "WARNING: mvis check --env --smoke reported warnings/failures (see above)"

# --- ffmpeg (optional, for MP4 calibration/sim videos) ---
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg:    $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}') (MP4 video OK)"
else
    echo "ffmpeg:    NOT found — videos will use GIF fallback (run: conda install -c conda-forge ffmpeg)"
fi

# --- Run the unit tier (the full suite takes about 20 min: scripts/test.sh full) ---
echo ""
echo "Running the unit tests..."
cd "$REPO_DIR"
scripts/test.sh unit || echo "WARNING: unit tests reported failures (see above)"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "To use MVIS:"
echo "  conda activate $ENV_NAME"
echo "  mvis run -c examples/configs/example_config.yaml"
echo ""
echo "To run tests (see doc/testing.md):"
echo "  conda activate $ENV_NAME"
echo "  scripts/test.sh unit        # fast tests"
echo "  scripts/test.sh full        # everything, before a merge"
echo ""
