"""
Root conftest.py.

Having this file at the repo root is what makes `import cribsandladders`
and `import game_params` work from anything under tests/, since pytest
adds this file's directory to sys.path when it collects it. The explicit
sys.path insert below is a belt-and-suspenders backstop for that -- it
shouldn't be needed, but costs nothing if it isn't.

Fixtures here are intentionally generic (seeded RNG, temp dir/db) so later
phases (Board/Optimizer work) can reuse them instead of re-inventing temp
fixtures per test module.
"""
import random
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def seeded_rng():
    """A random.Random instance seeded for deterministic test runs.

    Use this instead of the global `random` module wherever code accepts
    (or is refactored to accept) an injectable RNG. Fixed seed means the
    same fixture always produces the same sequence.
    """
    return random.Random(1337)


@pytest.fixture
def temp_db_path(tmp_path):
    """Path to a throwaway sqlite db file, not yet created."""
    return tmp_path / "test.db"


@pytest.fixture
def temp_sqlite_conn(temp_db_path):
    """An open connection to a throwaway sqlite db, closed on teardown."""
    conn = sqlite3.connect(temp_db_path)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Scratch directory for tests that need to write files (DXF/SVG/XML)."""
    d = tmp_path / "output"
    d.mkdir()
    return d
