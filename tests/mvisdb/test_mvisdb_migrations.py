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
"""Migrations runner tests (I1): forward-only, one transaction, injectable."""
from __future__ import annotations

import sqlite3

import pytest

from mvis.mvisdb.manager import (
    SCHEMA_VERSION, DatabaseManager, SchemaTooNewError,
)
from mvis.mvisdb.migrations import apply_migrations, current_version
from tests.helpers.mvisdb_fixtures import make_db


class _M2:
    VERSION = SCHEMA_VERSION + 1

    @staticmethod
    def up(conn):
        conn.execute("ALTER TABLE run ADD COLUMN note TEXT")


class _M3Bad:
    VERSION = SCHEMA_VERSION + 2

    @staticmethod
    def up(conn):
        raise RuntimeError("boom")


def _fresh(tmp_path):
    path = tmp_path / "bag.mvisdb"
    DatabaseManager().save(make_db(), path)
    return path


def test_apply_forward_and_bump(tmp_path):
    path = _fresh(tmp_path)
    conn = sqlite3.connect(path)
    assert current_version(conn) == SCHEMA_VERSION  # fresh files start current
    assert apply_migrations(conn, [_M2]) == SCHEMA_VERSION + 1
    conn.execute("SELECT note FROM run")            # column exists
    assert apply_migrations(conn, [_M2]) == SCHEMA_VERSION + 1  # idempotent
    conn.close()


def test_failed_migration_rolls_back(tmp_path):
    path = _fresh(tmp_path)
    conn = sqlite3.connect(path)
    with pytest.raises(RuntimeError, match="boom"):
        apply_migrations(conn, [_M2, _M3Bad])
    assert current_version(conn) == SCHEMA_VERSION  # ALL of it rolled back
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("SELECT note FROM run")
    conn.close()


def test_too_new_file_refused(tmp_path):
    path = _fresh(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_meta SET schema_version = 99")
    conn.commit(); conn.close()
    with pytest.raises(SchemaTooNewError):
        DatabaseManager().load(path)
