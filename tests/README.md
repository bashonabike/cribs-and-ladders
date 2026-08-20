# tests/

Replaces the old `test/` directory (deleted -- it imported a `cribbage`
package that hasn't existed since the rename to `cribsandladders`, and
didn't run).

## Running

```
pip install -r requirements-dev.txt
pytest
```

`pytest -m unit` runs only tests that don't need the heavy deps (scipy,
ezdxf, shapely, matplotlib, the compiled `scoretree`/`markovgame`
extensions). `pytest -m integration` is the rest.

## Layout (fills in per phase, see `tdd-refactor-assessment.md`)

- `test_smoke.py` -- Phase 0. Proves the harness works. Not game logic.
- `test_config.py` -- Phase 1. Covers `cribsandladders.config.GameConfig`
  (derivation logic, per-instance independence) and the `game_params.py`
  backward-compat shim (same values as before, no import-time DB access).
- `test_deck.py`, `test_score_hand.py`, `test_cribbage_game.py`,
  `test_eventsetbuilder.py` -- pre-existing tests found scattered inside
  `cribsandladders/` (not under the old `test/` dir, so Phase 0 missed
  them) and relocated here so pytest actually discovers them. Not all of
  them pass:
  - `test_deck.py` -- passes as-is (zero heavy deps).
  - `test_score_hand.py` -- Phase 1 additions verifying config injection.
  - `test_cribbage_game.py` -- fixed (all 8 tests pass; split into 8 from
    the original 7 since one crammed two unrelated scenarios into a
    single test). Root causes were: `Card(suit, rank)` argument order
    swapped from the real `Card(rank, suit)` constructor; two tests
    compared a `Card` object directly against a bare int; the "illegal
    move" test tried to trigger `IllegalMoveException` via an empty
    hand, which actually short-circuits into an unrelated crash instead
    (see the docstring on `test_illegal_move_card_not_in_hand`); and the
    mock `Track`/`Player` fixtures were missing several attributes
    (`num`, `efflength`, `wins`) real code reads unconditionally. Fixed
    by using a real `Track()` instance instead of `MagicMock(spec=Track)`
    (Track has no-arg construction and no side effects, so this is more
    robust than patching mock attributes one crash at a time) and by
    redesigning the illegal-move test into two that actually exercise
    the intended branches.
  - `test_eventsetbuilder.py` -- needs numpy + pandas (via
    `cribsandladders.Board`) to import; not yet updated for config
    injection since `EventSetBuilder` is Phase 3/4 scope.
- `test_player.py`, `test_crib_squad.py` -- still to write, Phase 2.
- `test_board.py`, `test_base_layout.py`, `test_board_setter.py` -- Phase 3.
- `test_optimizer.py`, `test_possible_events.py`, `test_evaluator.py`,
  `test_stats.py` -- Phase 4.
- `test_integration_*.py` -- Phase 5.

## Fixtures

Defined in the root `conftest.py` so they're available everywhere:

- `seeded_rng` -- a `random.Random(1337)` instance for deterministic tests.
- `temp_db_path` / `temp_sqlite_conn` -- throwaway sqlite db per test.
- `temp_output_dir` -- scratch dir for tests that write files (DXF/SVG/XML).
