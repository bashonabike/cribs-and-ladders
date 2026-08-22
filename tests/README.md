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
- `test_stats_metrics.py`, `test_evaluator_metrics.py` -- Refactor Mk II
  ([[Refactor Mk ii]] in the Obsidian vault), Phase 6, done. `Stats.
  calc_metrics()` (137 lines) and `Evaluator.detMetrics()` (193 lines)
  were both linear "compute one named statistic, write it to `self.*`
  or append it to `self.results`, repeat" pipelines with close to zero
  direct test coverage of any individual statistic -- Phase 4
  characterized two pre-existing `Stats` bugs rather than fixing them
  specifically because the method was too big to safely touch. Each
  named block in both methods is now its own function taking exactly
  the DataFrame(s)/inputs it needs instead of reading `self.board`/
  `self.moves` directly, same pattern as the existing `hydrate_tracks_
  from_dataframes`/`hydrate_candidate_events_from_dataframe`/`build_
  insert_stat_stub` extractions. Both `calc_metrics()` and `detMetrics()`
  are now thin orchestrators -- pure code motion, no logic changed, and
  every pre-existing black-box test in `test_stats.py`/`test_evaluator.py`
  still passes unchanged.
  - `cribsandladders/stats_metrics.py` -- `prep_joined_moves_df` (the
    DataFrame-building/joining preamble) plus `calc_soexcites_and_
    repeats`, `calc_avg_length_in_rounds`, `calc_events_by_track`,
    `calc_move_histograms`, `calc_lookforward_events_by_track`. All
    pure functions of already-built DataFrames, tested in
    `test_stats_metrics.py` against small scenarios built from real
    `Move`/`Track`/`Chute`/`Ladder` objects (the same way
    `test_stats.py` already does) rather than hand-built raw
    DataFrames -- `prep_joined_moves_df`'s output shape depends on
    which columns exist at all, which isn't worth hand-guessing when
    building it from real domain objects is one function call. Found
    (and characterized, not fixed) one more pre-existing quirk along
    the way: `eventsin1bytrack`/`eventsin2bytrack` are built via
    `zip(laddersinNbytrack, chutesinNbytrack)`, which silently
    truncates to the *shorter* list whenever only ladders or only
    chutes have any per-track hits at all -- so the by-track list can
    under-report even though the independently-computed scalar total
    (`eventsin1`/`eventsin2`) stays correct. See
    `test_calc_lookforward_events_by_track_counts_ladder_starting_
    one_space_ahead`'s comment.
  - `cribsandladders/evaluator_metrics.py` -- `structure_scalar_stats`,
    `early_termination_stats`, `balance_stats`, `gamelength_stat`,
    `twohits_stat`, `soexcite_stat`, `repeats_stat`,
    `events_hit_skew_stat`, `event_length_distribution_stats`. All pure
    functions -- no `self`, no `EventSetBuilder` needed -- tested
    directly in `test_evaluator_metrics.py` with small fakes. Five
    blocks (`eventSpacingHist_curvefit`, the onlyGameBoardStats-only
    `trackEventLengthDistribution_curvefit_T*`, and the three
    `*OverTime_curvefit` results) still need `Evaluator.
    processActualHistCurve`/`Evaluator.discreteRegression`, which
    delegate to `self.eventSetBuilder`'s curve-math methods -- those
    became named private methods on `Evaluator`
    (`_event_spacing_histogram_result` etc.) instead of moving to
    `evaluator_metrics.py`, since doing that properly would mean either
    passing the whole `EventSetBuilder` through as a parameter or
    duplicating its curve-math delegation. Flagged in both modules'
    docstrings as a reasonable follow-up: `Evaluator` could call
    `cribsandladders.event_curve_math` directly instead of proxying
    through `self.eventSetBuilder`, since that's exactly what
    `EventSetBuilder`'s own delegating wrappers already do.
- `test_possible_events_build_set.py` -- Refactor Mk II ([[Refactor Mk
  ii]] in the Obsidian vault), Phase 7, done. `PossibleEvents.buildSet()`
  was ~300 lines: setup, a per-hole-pair double loop with a two-way
  `if`/`elif` branch (each branch itself dozens of lines of real
  geometry -- direct-route/multi-track-loop-closure search on one side,
  orthogonal/loopy-sidestep search on the other), and a DB-persistence
  tail, all inline in one method with effectively zero test coverage
  (`test_possible_events.py`'s own module docstring calls it out as "not
  a good target for unit tests in one pass"). Confirmed by reading (no
  shared local variables between the `if` and `elif` branch bodies) that
  the three pieces could be split with pure code motion:
  - `_try_direct_or_multi_track_event(self, board, t, h_a, h_b)` -- the
    `if not checkAngleForOrtho(...)` branch body.
  - `_try_orthogonal_sidestep_event(self, t, h_a, h_b)` -- the
    `elif checkAngleForOrtho(...)` branch body.
  - `_persist_candidate_events_to_db(self, sqlConn)` -- the DB-persistence
    tail (dedupe, delete, rebuild the insert DataFrame, `executemany`).
  - `buildSet` itself is now a thin coordinator: the setup preamble, the
    per-hole-pair loop deciding which branch method to call, then a call
    to the persistence method.

  Because none of the three pieces had any existing tests to lean on,
  this file is a golden/characterization test built *before* the split:
  a small but real `Board` (one zigzagging 9-hole track -- a straight
  line turned out to make every hole pair uniformly accept or reject in
  the ortho-sidestep branch, which is a weak regression signal) is run
  through the pre-refactor `buildSet()` once, the output hand-verified
  to be a genuine mix (21 accepted / 15 rejected pairs out of 36), and
  that exact result hardcoded as the expected value -- for both the
  resulting `CandidateEvent`s and the rows written to an in-memory
  `TempCandidateEvents` table (schema reverse-engineered via `PRAGMA
  table_info`/`sqlite_master.sql` against the real `etc/Temp.db`, since
  no `CREATE TABLE` for it exists anywhere in the repo). A second test
  reruns the same fixture with a tighter
  `maxloopyorthoeventdisplacementincrements` budget and checks fewer
  candidates are accepted, confirming the accept/reject decision
  actually responds to config rather than being a fixed artifact of
  the fixture's geometry. Both tests passed unchanged after the split.

  Also moved `matplotlib.pyplot` from a module-level import to a lazy
  import inside `testPlotVectorsOnHoles` (its only user) -- the module's
  real geometry code uses `matplotlib.path`, not `pyplot`, so nothing
  else in the file needs it at import time.
- `test_eventsetbuilder.py::test_build_track_state_computes_expected_fields_for_one_track`
  -- Refactor Mk II ([[Refactor Mk ii]] in the Obsidian vault), Phase 8
  step 1, done. `EventSetBuilder.tryEventSet`'s setup preamble (~115
  lines: build one working-state dict per active track, retrieve/
  normalize the energy and length-distribution curves, compute each
  track's candidate-event specs/energy potential/length-distribution
  curves/spacing histogram) had no branching back into the placement
  loop that follows it, making it the safest possible first extraction
  out of this 1798-line class -- pulled out verbatim into
  `_build_track_state(self, params) -> list[dict]`, with `tryEventSet`
  now just calling it and continuing with the placement loop unchanged.
  (Needed a `return trackEventsOverview` added at the end, since the
  original code never returned it -- it just kept reading/mutating the
  same local for the rest of the method; that's the one line of this
  extraction that isn't literally a cut-and-paste, since splitting a
  method always needs *some* seam.)

  Nothing exercised this preamble for real before: `test_try_event_set`
  (the pre-existing test targeting `tryEventSet` as a whole) is skipped
  because its fixture's `track1.candidateEvents` is `None`, and
  separately, `EventSetBuilder`'s default curve file paths
  (`Boards/MicroBoard1/CURVES/*.svg`) don't exist anywhere in this repo
  (see the comment on `config.eventenergyfile`) -- so even a fixture
  with real candidate events would have hit a `FileNotFoundError`. The
  new test works around both: real minimal SVG curve files (same
  points `test_event_curve_math.py`'s own scaling test uses) written to
  a temp `data_root`, and a real `Track` with hand-built candidate-event
  mocks (only `.startHole.num`/`.endHole.num`/`.length`/`.canBeLadder`/
  `.isShared` are read). `runPartialTrackEffLengthHoles` (the one call
  in this preamble needing a real Optimizer db and/or the compiled
  `markovgame` extension for its Monte-Carlo simulation) is mocked to a
  fixed return value -- same pattern `test_optimize_setup` already uses
  to mock out `tryEventSet`/`buildSetIntoEvents` rather than stand up
  their real dependencies. With only one track in play, its average
  candidate energy potential trivially equals the "overall" average, so
  the energy-skew adjustment is exactly zero and `optevents`/
  `optfirstchute`/`candeventspecs` ordering are all asserted against
  exact, curve-shape-independent values; the few fields that do depend
  on the curve files' actual shape (`trackenergycurve`,
  `trackenergyintegral`, `lengthdistidealcurve`, `lengthovertimeideal`)
  are only checked for being non-empty lists, not exact values.
- Refactor Mk II Phase 8 step 2, done: `trackEventsOverview`'s ~50-key
  plain dict (built as a `dict(...)` literal, threaded through
  `tryEventSet`'s placement loop and `scoreEventsForHole`/
  `tryGetEventForHole` via `t['key']` string access) is now a real
  `TrackBuildState` dataclass (`EventSetBuilder.py`, defined just above
  the `EventSetBuilder` class). Purely structural -- every field's
  value/semantics are unchanged, this just makes the schema explicit
  and typo-proof, and per the Mk II plan, sets up unit-testing
  individual scoring rules directly (via `object.__new__
  (TrackBuildState)` + hand-set attributes, the same pattern
  `test_optimizer.py`/`test_possible_events.py` already use for partial
  fixtures) without needing a full placement run.

  Converting every real call site's `t['key']`/`t_sub['key']`/
  `t_match['key']` to `t.key` (`_build_track_state`,
  `scoreEventsForHole`, `tryGetEventForHole`, the placement loop, and
  `event_curve_math.recalc_track_completion_pcts`) was mechanical but
  needed care in a few places a plain identifier-based search-and-
  replace would've silently mishandled: `trackEventsOverview[i]['key']`
  (list-index-then-dict-key, not a bare `t['key']` the same regex would
  catch) had two call sites that needed fixing separately; and several
  *other* dicts in the same file happen to reuse key names that also
  appear on `TrackBuildState` (`prevEffLengths`/`effLengths` entries
  also have a `track_id`/`efflength` key, `self.eventNodesByTrack`
  entries also have a `tracknum` key, and the `fitness`/
  `idealEventWithFitness` scoring-result dict returned by
  `scoreEventsForHole` also has a `lasteventtop` key) -- those are
  genuinely different, smaller dict shapes and were deliberately left
  as plain dicts, not converted. `candeventspecs` (a `TrackBuildState`
  field) similarly still holds a list of plain per-candidate dicts
  (`eventtop`/`eventbase`/`length`/`canbeladder`/`isshared`/`event`) --
  only the outer per-track structure was in scope for this step.

  Existing test coverage (`test_eventsetbuilder.py`'s 16 tests,
  `test_event_curve_math.py`'s 19 tests, full offline suite) all pass
  unchanged; `test_recalc_track_completion_pcts_averages_over_viable_
  tracks` and the new `_build_track_state` test both updated to build
  `TrackBuildState` instances instead of dicts.
- Refactor Mk II Phase 8 step 3, done: `scoreEventsForHole`'s two-hit
  detection was four ~25-line near-duplicate blocks (ladder-instance
  forward/backward, chute-instance forward/backward), each scanning a
  "same event type as the one being placed" position list and an
  "opposite type" one, with subtly different net-length formulas and
  guard flags per block. Collapsed into one shared
  `_scan_two_hits_for_direction(...)` method (defined just above
  `scoreEventsForHole`), called four times with the position list/dict/
  match-key/guard-flag/net-length-formula specific to each block passed
  in explicitly -- pure code motion, every formula/flag verified by hand
  against the original block it replaces (see the method's own
  docstring for the full mapping).

  Collapsing surfaced exactly the asymmetry the Mk II plan called out:
  in every one of the four original blocks, a same-dir-twohits
  rejection (`config.onlysamedirtwohits`, plus a "matched item's length
  is within 3 of this event's length" check) only ever applied to
  matches against the *opposite* event type from the one being placed
  (chute matches when placing a ladder, ladder matches when placing a
  chute) -- same-type matches were never guarded, in both directions.
  Preserved verbatim, not fixed, since it's unclear from the code alone
  whether that's intentional (the guard is really about mixed-type
  two-hits) or a copy-paste gap -- documented as a TODO on
  `_scan_two_hits_for_direction`'s docstring, with a passing
  characterization test
  (`test_scan_two_hits_for_direction_guards_only_the_flagged_side`) that
  exercises the exact guarded/unguarded split. Two more
  `_scan_two_hits_for_direction`-specific tests cover the ordinary
  strict/loose hit counting and the "count happens before the length
  check can still reject it" ordering. All exercise the extracted
  method directly with small hand-built lists/dicts (needs only
  `self.config.onlysamedirtwohits` and the already-pure
  `searchOrderedListForVal`) rather than the full
  `scoreEventsForHole`/`tryEventSet` machinery, which needs a much
  larger fixture (see the Phase 8 step 1 section above) -- this was the
  practical way to get a real, passing characterization test for this
  specific behavior without that larger undertaking.
- Refactor Mk II Phase 8 step 4, done (last step of Phase 8 -- see
  [[Refactor Mk ii]] in the Obsidian vault): `scoreEventsForHole`'s
  per-instance-type scoring body (the CHUTEONLY/LADDERONLY/
  CHUTEANDLADDER loop: balance scoring, energy-buffer scoring, two-hit
  detection, cancel impedance, end-of-track weighting, length-histogram
  scoring, length-over-time scoring -- everything after the candidate-
  gating/cursor-walking code, which stays in `scoreEventsForHole`
  unchanged) is now `_score_candidate_instance(t, hole, candEventSpecs,
  instType, ..., params, explicitEvent, explicitChute, explicitLadder)
  -> dict | None`, called once per instType from a now much shorter
  loop in `scoreEventsForHole`. Pure code motion: every `continue` in
  the original loop body became a `return None` (caller: `if fitness is
  not None: eventFitnesses.append(fitness)`), and the original
  `eventFitnesses.append(dict(...))` became `return dict(...)`.

  Locked in with a golden/characterization test built *before* the
  extraction (same approach Phase 7 used for `PossibleEvents.buildSet`,
  and Phase 8 step 1 for `_build_track_state`):
  `test_score_events_for_hole_returns_expected_fitness_for_explicit_
  ladder_event` calls the real (pre-refactor) `scoreEventsForHole`
  through its existing `explicitEvent` seam -- which bypasses the
  candidate-cursor/`candeventspecs` machinery entirely, going straight
  to the per-instType scoring body this step touches -- with
  config/state values chosen so every intermediate branch collapses to
  a known constant (`curEstLengthDiscr == 0`, `eventPosRelMidpoints ==
  0`, `lenDistDisp == 0`, `scoreMod == 1.0`, all verified by hand-tracing
  the method against these exact inputs), leaving a score (1.1) anyone
  can re-derive by hand rather than "whatever the code happens to
  produce". `runPartialTrackEffLengthHoles` (needs a real Optimizer db
  and/or the compiled `markovgame` extension) is mocked to a fixed
  value, same pattern the Phase 8 step 1 test already uses. Passed
  unchanged against the post-extraction code.

  This is the last of Phase 8's four steps (setup preamble ->
  `TrackBuildState` -> two-hit dedup -> scoring-body extraction);
  `EventSetBuilder.py`'s largest method (`scoreEventsForHole`, 473
  lines before Phase 8) is now three much smaller, independently named
  pieces. Phase 9 (the `EventSetBuilder` Category A collision-tracking
  redesign -- `allVectorsTest`/`baseVectorsTest`/`tryEventSet`'s main
  placement loop) remains separately scoped future work per the Mk II
  plan, not started here.

## Fixtures

Defined in the root `conftest.py` so they're available everywhere:

- `seeded_rng` -- a `random.Random(1337)` instance for deterministic tests.
- `temp_db_path` / `temp_sqlite_conn` -- throwaway sqlite db per test.
- `temp_output_dir` -- scratch dir for tests that write files (DXF/SVG/XML).
