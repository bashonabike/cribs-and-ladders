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
- `test_deck.py`, `test_score_hand.py`, `test_cribbage_game.py`,
  `test_player.py`, `test_crib_squad.py` -- Phase 2, once `game_params`
  is decoupled into an injectable config.
- `test_board.py`, `test_base_layout.py`, `test_board_setter.py` -- Phase 3.
- `test_optimizer.py`, `test_event_set_builder.py`,
  `test_possible_events.py`, `test_evaluator.py`, `test_stats.py` -- Phase 4.
- `test_integration_*.py` -- Phase 5.

## Fixtures

Defined in the root `conftest.py` so they're available everywhere:

- `seeded_rng` -- a `random.Random(1337)` instance for deterministic tests.
- `temp_db_path` / `temp_sqlite_conn` -- throwaway sqlite db per test.
- `temp_output_dir` -- scratch dir for tests that write files (DXF/SVG/XML).
