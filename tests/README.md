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
    injection since `EventSetBuilder` is Phase 4 scope.
- `test_player.py`, `test_crib_squad.py` -- Phase 2, done.
- `test_board.py`, `test_base_layout.py`, `test_board_setter.py`,
  `test_dxf_writer.py` -- Phase 3, done. What each covers:
  - `test_base_layout.py` -- `_ccw`/`_intersect`/`check_intersections`/
    `build_interception_test_vector_set` (already pure) plus
    `svgParserHoles`/`svgParserVectors`, exercised via `io.StringIO`
    SVG strings instead of fixture files on disk -- `xml.dom.minidom.parse`
    accepts any file-like object, so this needed no source changes, just
    tests. `setTrackHolesets` now takes an injected `config: GameConfig`
    (replacing `game_params`); both findmode branches are covered.
  - `test_board.py` -- `Track`'s in-memory methods (`setEffLandingForHoles`,
    `setEventImpedance`, `getHoleByCoords`/`getHoleByNum`, the `getXAsDF`
    helpers) were already pure, just untested. One test
    (`test_set_eff_landing_for_holes_redirects_through_ladder_but_not_chute`)
    is a characterization test flagging a likely pre-existing bug: the
    chute branch of `setEffLandingForHoles` redirects to `chute.start`
    instead of `chute.end` (the ladder branch correctly uses `.end`),
    which makes chute landings a no-op. Not fixed here -- game-logic
    bugs are out of scope for Phase 3 (testability/extraction) -- but
    pinned down so a future fix has a safety net. `Board` now takes an
    injected `config: GameConfig` too, and `setBoardAfterSetter`'s
    `import cribsandladders.PossibleEvents` was moved from module scope
    into the findmode branch (it pulls in matplotlib); two tests assert
    that import no longer happens when findmode is False, and that
    `import cribsandladders.Board` itself works with matplotlib blocked.
  - `test_board_setter.py` -- `setBoardFromDb`'s non-findmode branch
    (turn already-fetched Board/Track/Chute/Ladder DataFrames into
    domain objects) was pulled out into a new pure function,
    `hydrate_tracks_from_dataframes`, so it's unit-testable with small
    hand-built DataFrames instead of a real db. `setBoardFromDb` itself
    is covered end-to-end by `@pytest.mark.integration` tests against a
    temp sqlite db (both the findmode stub-track-generation branch,
    which does real INSERT/DELETE and so wasn't a good extraction
    candidate, and the non-findmode branch). Also fixed: the db path
    was hardcoded to `'Boards/AllBoards.db'` in two places (one of which
    ignored the `boardDBName` "constant" it was named after); both now
    resolve from `config.db_path`.
  - `test_dxf_writer.py` -- DXFWriter.py did `import ezdxf` at module
    scope, which meant even its already-pure coordinate/vector math
    (`convert_mm_to_in`, `compute_offset_curve`, etc.) required ezdxf
    installed just to import the module under test. That math moved to
    a new `cribsandladders/dxf_geometry.py` with no ezdxf dependency
    (DXFWriter imports it back unchanged), so those are now plain unit
    tests. `buildDXFFile` also gained an `output_dir` parameter
    (default `"Boards"`, preserving old behavior) replacing a hardcoded
    `"Boards/" + board.boardName` path, so the one
    `@pytest.mark.integration` test can point it at `temp_output_dir`
    and read the result back with `ezdxf.readfile` to assert on layer
    names and entity counts -- a structural snapshot at the I/O
    boundary rather than a byte-for-byte one, since the filename embeds
    a timestamp and layer colors are randomized. Uses
    `pytest.importorskip("ezdxf")` so it skips cleanly if ezdxf isn't
    installed rather than failing collection.
- `test_optimizer.py`, `test_possible_events.py`, `test_evaluator.py`,
  `test_stats.py` -- Phase 4.
- `test_integration_*.py` -- Phase 5.

## Fixtures

Defined in the root `conftest.py` so they're available everywhere:

- `seeded_rng` -- a `random.Random(1337)` instance for deterministic tests.
- `temp_db_path` / `temp_sqlite_conn` -- throwaway sqlite db per test.
- `temp_output_dir` -- scratch dir for tests that write files (DXF/SVG/XML).
