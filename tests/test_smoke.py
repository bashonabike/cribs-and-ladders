"""
Phase 0 smoke test.

Purpose: prove the pytest harness itself works (discovery, config,
fixtures, sys.path setup) without depending on anything not guaranteed to
be installed. Enums.py has zero imports beyond stdlib `enum`, so it's the
one module in this codebase guaranteed to import cleanly everywhere.

This is deliberately not testing game behavior -- that starts in Phase 2
once ScoreHand/Deck/CribbageGame are decoupled from game_params.
"""
import pytest

from Enums import Event, InstanceEventType, OrthoLineTraceType


def test_enums_import_and_have_expected_members():
    assert Event.NONE.value == 0
    assert Event.CHUTE.value == 1
    assert Event.LADDER.value == 2
    assert OrthoLineTraceType.START.value == 0
    assert InstanceEventType.CHUTEANDLADDER.value == 2


def test_seeded_rng_fixture_is_deterministic(seeded_rng):
    first = seeded_rng.random()
    again = random_from_fresh_seed()
    assert first == again


def random_from_fresh_seed():
    import random
    return random.Random(1337).random()


def test_temp_sqlite_conn_fixture_works(temp_sqlite_conn):
    temp_sqlite_conn.execute("CREATE TABLE t (x INTEGER)")
    temp_sqlite_conn.execute("INSERT INTO t VALUES (1)")
    temp_sqlite_conn.commit()
    row = temp_sqlite_conn.execute("SELECT x FROM t").fetchone()
    assert row == (1,)


def test_temp_output_dir_fixture_is_writable(temp_output_dir):
    f = temp_output_dir / "example.txt"
    f.write_text("ok")
    assert f.read_text() == "ok"


@pytest.mark.unit
def test_marker_registration_does_not_warn():
    # If this collects without a "PytestUnknownMarkWarning" (and --strict-markers
    # would turn that into an error), markers are registered correctly.
    assert True
