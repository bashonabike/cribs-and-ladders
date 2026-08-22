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
    `cribsandladders.Board`) to import. Updated in Phase 4 for config
    injection -- see the Phase 4 section below.
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
  `test_stats.py`, `test_eventsetbuilder.py` -- Phase 4 (Optimizer/
  board-design subsystem), done except for one explicitly deferred piece.
  `PossibleEvents`, `Stats`, `Evaluator`, `Optimizer`, and
  `EventSetBuilder` all moved off `game_params` onto an injected
  `config: GameConfig` (same pattern as earlier phases), and each had its
  heavy/optional imports (`scipy`, `lightgbm`, two `sklearn` submodules,
  `markovgame`) moved from module scope into the one method that actually
  uses them, so the rest of each module imports and is testable without
  those packages installed. What each file covers:
  - `test_possible_events.py` -- the geometry helpers (`ccw`, `intersect`,
    `orientation`, `doIntersect`, bounding-box/angle math) were already
    pure, just untested. `hydrate_candidate_events_from_dataframe` is a
    new pure method pulled out of `tryRetrieveCache`'s DB-cache-hit path
    (same idea as Phase 3's `hydrate_tracks_from_dataframes`), tested
    with small hand-built DataFrames instead of a real cache db. Also
    fixed a hardcoded `'Boards/AllBoards.db'` path to use
    `config.db_path`.
  - `test_stats.py` -- `build_insert_stat_stub` is a new pure function
    pulled out of the raw-SQL-building logic in `insertStatsRecord`.
    `calc_metrics`/`insertStatsRecord`/`print_metrics`/`print_temp_maps`
    now read from injected config instead of `game_params`; the sqlite
    path is `config.db_path` instead of a literal. Two pre-existing bugs
    were found and documented (with a TODO in the source and a
    characterization test expecting the `AttributeError`, not fixed --
    game-logic/plotting bugs are out of scope for this phase): `print_metrics`
    references `self.soexcites_pegs`/`self.lengths_in_rounds`, which
    aren't set anywhere on `Stats`; `print_temp_maps` calls
    `.groupby('track').to_list('track')`, which isn't a real
    `DataFrameGroupBy` method.
  - `test_evaluator.py` -- `Evaluator.__init__` does no I/O, so it's
    constructed directly with small fake `EventSetBuilder`/`Board`/`Track`
    stand-ins rather than needing `object.__new__`. Covers config-injected
    scoring in `detMetrics(onlyGameBoardStats=True)` (orthos/multis/cancels
    targets, negative-cancels clamping, early-termination against
    `config.finishlinelength`) and the full `CurveOptimizer` class
    (`find_optimal_scale_analytical`, `apply_scaling`,
    `least_squares_difference`, zero-denominator handling). One
    `@pytest.mark.integration` test uses `pytest.importorskip("scipy")`
    for `find_optimal_scale`, the one method that still needs scipy
    (moved from module scope into that method only).
  - `test_optimizer.py` -- `Optimizer.__init__` opens a real sqlite
    connection and runs queries, so most methods are tested via
    `object.__new__(Optimizer)` with just the attributes each method
    reads set by hand (same pattern as `test_possible_events.py`). Covers
    `setParamFromBounds`, `detWeighedScoring`, `getFminStarterParams`,
    `getFminBounds`, `setupFminParamsList`, and `runIncrIteration`
    (verified against a hand-computed expected value, that it scales
    proportionally with `config.changebaseincrperiter`, and that
    out-of-bounds results clamp instead of crashing). One
    `@pytest.mark.integration` test builds a temp sqlite db with the
    `OptimizerParamPairings`/`BoardTrackHints` schema and exercises
    `__init__`/`retrievePairingsSettings` end-to-end. The db path is now
    `config.optimizer_db_path` instead of a hardcoded
    `'etc/Optimizer.db'` literal.
  - `test_eventsetbuilder.py` -- rewritten for config injection.
    `retrieveOrGenerateBenchmarkMoves` is mocked out in `setUp` (the real
    repo's `etc/Optimizer.db` hit a sandbox-specific disk I/O error
    unrelated to this migration) so `EventSetBuilder` can be constructed
    without a live db. Nine hardcoded `'etc/Optimizer.db'` literals
    scattered across `EventSetBuilder`/`ParamSet`/`OrthoLineTrace` were
    all routed through `config.optimizer_db_path`, with a regression test
    (`test_monte_carlo_reads_from_configured_db_path_not_hardcoded_one`)
    pinning that down. `test_try_event_set` is skipped with an explicit
    reason (`track1.candidateEvents` is `None` in the shared fixture, a
    pre-existing gap flagged by the original author's own TODO, unrelated
    to this migration) rather than papered over.
  - **EventSetBuilder decomposition (follow-up to the above)**:
    `EventSetBuilder.py` was 2669 lines and defined four classes
    (`EventSetBuilder`, `OrthoPath`, `OrthoLineTrace`, `ParamSet`) in one
    file. It's now 1863 lines and one class. What moved out, each into
    its own module with its own test file:
    - `cribsandladders/ortho_path.py` (`OrthoPath`) and
      `cribsandladders/ortho_line_trace.py` (`OrthoLineTrace`) -- neither
      ever depended on anything off `EventSetBuilder` itself (`OrthoPath`
      is a plain data holder; `OrthoLineTrace` only needs a duck-typed
      `possibleEvents`), so they moved verbatim. Covered by
      `test_ortho_path_and_line_trace.py`, including two tests asserting
      `EventSetBuilder.py` still re-exports the identical class objects.
    - `cribsandladders/param_set.py` (`ParamSet`) -- the Monte-Carlo/
      midpoint/fmin parameter-search class, also moved verbatim (only
      ever needed `board`/`tracks`). `test_eventsetbuilder.py`'s
      `TestParamSet` (already rewritten for config injection, see above)
      needed no changes since the import
      (`from cribsandladders.EventSetBuilder import ParamSet`) still
      resolves via the re-export.
    - `cribsandladders/event_curve_math.py` -- pure functions pulled out
      of a dozen `EventSetBuilder` methods (`actualizeCurve`,
      `discretizeCurve`, `normalizeCurveMagnitude`,
      `integrateAndNormalizeCurve`, `getNormalizedIdealCurve`,
      `getPointsInProximity`, `tryGetDispAllowance`,
      `searchOrderedListForVal`, `indexStartOfEachHoleInCands`,
      `recalcTrackCompletionPcts`, `orthoBoundingBox`,
      `boundingBoxPlusVector`), plus a new
      `build_track_dict_from_benchmark_moves_df` pulled out of
      `retrieveOrGenerateBenchmarkMoves`'s DataFrame-to-dict hydration
      (same idea as Phase 3/4's other `hydrate_*` extractions). Covered
      by `test_event_curve_math.py` (19 tests, zero mocking needed --
      every function here is a plain function of its arguments). Found
      and pinned down one more pre-existing quirk along the way:
      `discretize_curve(..., accumulate=True)`'s bucket-boundary check
      (`curveIdx < i * discFactor`) makes the accumulation lag by one
      bucket (bucket 0 always comes out `0.0` regardless of the curve's
      actual values) -- not fixed, just characterized. Also discovered
      `indexStartOfEachHoleInCands` has zero call sites anywhere in the
      repo (confirmed by grep) and its own body implies
      `trackEventOverview` must be a `dict` keyed by both int and the
      string `'candeventstartlookup'` it adds -- documented in the test
      rather than guessed at silently.
    - `cribsandladders/event_set_plotter.py` -- the refactor plan's "thin
      plotting adapter tests can no-op." `EventSetBuilder.__init__` now
      takes an optional `plotter` (defaults to a real `EventSetPlotter`);
      `plotBoard`/`testPlotVectorsOnHoles`/`plot_coordinates_and_vectors`
      delegate to it. `test_event_set_plotter.py` discovered that
      `EventSetPlotter`'s real methods end in `plt.waitforbuttonpress()`,
      which blocks forever even under matplotlib's non-interactive `Agg`
      backend (confirmed by hand: `timeout 5 python3 -c "...plt.show();
      plt.waitforbuttonpress()"` hits the timeout) -- so those tests
      patch `event_set_plotter.plt` with a `MagicMock` rather than
      calling real matplotlib, and a separate `NoOpEventSetPlotter` is
      what tests should actually inject into `EventSetBuilder` when they
      don't care about plotting at all (see
      `test_eventsetbuilder.py::test_accepts_injected_plotter`).
    - **What's still one class, on purpose**: the actual event-search/
      placement algorithm (`scoreEventsForHole` alone is ~470 lines;
      `tryEventSet`, `tryGetEventForHole`, `runPartialTrackEffLengthHoles`,
      `updateVectorsTest`, `buildPartialSetIntoTrack`,
      `getEffectorsForDisps`) stayed in `EventSetBuilder` itself. These
      methods share mutable state across a single build attempt (the
      `allVectorsTest`/`baseVectorsTest` collision sets, in-place
      `trackEventsOverview` dicts, `t.eventSetBuild` mutation) in ways
      that don't decompose into independently-callable pure units
      without redesigning the algorithm's control flow, not just moving
      code -- a deeper task than "extract the already-separable pieces,"
      consistent with why this was flagged as the biggest lift in the
      original refactor plan.
    - Every moved method kept a thin delegating wrapper on
      `EventSetBuilder` (e.g. `actualizeCurve` now just calls
      `event_curve_math.actualize_curve(...)`), so no existing call
      site -- production code (`cribbage_main.py`) or the rest of the
      test suite -- needed to change.
- `test_integration_gameplay.py`, `test_integration_board_optimizer.py` --
  Phase 5 (integration tests), done. Both are `@pytest.mark.integration`
  via a module-level `pytestmark`.
  - `test_integration_gameplay.py` -- full agent-vs-agent playthroughs:
    a real `Board`/`CribSquad`/`CribbageGame` played to completion with
    a seeded `random.Random` injected through the `rng=` params Phase 2
    already added. Two AI seams still needed standing in for, same
    reasoning as Phase 2's unit tests: `Player.pegging_move()`'s default
    `move_selector` calls the compiled `scoretree` extension (Tier 4,
    "outside Python unit-test reach" per the refactor plan), so tests
    inject `_first_legal_card_selector`, a deterministic pure-Python
    stand-in that always plays a legal card. `Player.discard_crib()`'s
    `expected_hand_value()` needs the huge precomputed `rankLookupTable`
    pandas artifact (out of scope, see `test_player.py`), so it's
    patched with `_approximate_hand_value`, which still calls the real
    `ScoreHand.score_hand()` (just against the first discarded card
    standing in for the cut card) rather than returning a mocked
    constant -- `discard_crib()` is still choosing between genuinely
    different real hand scores. Neither stand-in changes what's under
    test: the engine's dealing/discarding/pegging/hand-scoring/board-
    movement/win-detection orchestration, not either AI's decision
    quality. Covers 3-player and 2-player games, a board with no
    chutes/ladders at all, and two determinism checks (same seed -> byte
    -for-byte identical move log; different seeds -> not all identical,
    confirming the rng injection is actually wired through
    `Deck.shuffle()`).
  - `test_integration_board_optimizer.py` -- chains two things Phase
    3/4's own integration tests only exercised separately: a `Board`
    built by the real `setBoardFromDb()` I/O path against a temp
    `AllBoards.db`, then a real `Optimizer` constructed against that
    board and a *second* temp db (`Optimizer.db`), both resolved through
    `GameConfig(data_root=tmp_path)` rather than the historical
    hardcoded `'Boards/AllBoards.db'` / `'etc/Optimizer.db'` literals.
    Runs one full parameter-adjustment cycle end to end
    (`runIncrIteration` -> `setBestIterParams` -> `setupFminParamsList`
    -> `getFminStarterParams`/`getFminBounds`) and checks the numbers
    are internally consistent (hand-computed expected shift, same
    formula as `test_optimizer.py`), plus that an out-of-bounds result
    clamps instead of writing a value outside the seeded
    `BoardTrackHints` bounds. Note: `setBoardFromDb`'s non-findmode
    branch doesn't populate `Track.Track_ID` (only findmode does), so
    this keys Optimizer params by `track.num` -- documented in the file
    since it's the kind of thing a real call site would trip over too.

## Fixtures

Defined in the root `conftest.py` so they're available everywhere:

- `seeded_rng` -- a `random.Random(1337)` instance for deterministic tests.
- `temp_db_path` / `temp_sqlite_conn` -- throwaway sqlite db per test.
- `temp_output_dir` -- scratch dir for tests that write files (DXF/SVG/XML).
