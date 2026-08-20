# Cribs and Ladders — TDD Refactor Assessment

Date: 2026-08-19

## What's actually in this repo

Two bundled subsystems, 7,752 lines of first-party Python (excluding the two vendored pybind11 copies under `MarkovBind/` and `ScoreTreeTry2/`, which are third-party and out of scope):

- **Cribbage engine** — `Deck`, `ScoreHand`, `Player`, `CribbageGame`, `CribSquad`. Rules, scoring, and AI agents. `Player` calls out to a compiled C++ pybind11 extension (`scoretree`) for move search.
- **Board design system ("ladders")** — `Board`, `BaseLayout`, `BoardSetter`, `PossibleEvents`, `EventSetBuilder`, `Optimizer`, `Evaluator`, `Stats`, `DXFWriter`. Procedural board layout, DXF/SVG export, and a Monte Carlo event-placement optimizer, all backed by SQLite. `EventSetBuilder` calls a second C++ extension (`markovgame`).

| File | LOC |
|---|---|
| EventSetBuilder.py | 2184 |
| PossibleEvents.py | 1156 |
| Evaluator.py | 522 |
| Stats.py | 511 |
| popRankLookupTable.py | 438 |
| DXFWriter.py | 381 |
| Optimizer.py | 376 |
| CribbageGame.py | 340 |
| ScoreHand.py | 350 |
| game_params.py | 328 |
| cribbage_main.py | 287 |
| Board.py | 243 |
| BaseLayout.py | 143 |
| BoardSetter.py | 130 |
| Player.py | 118 |
| Deck.py | 107 |
| cribbage_scenarios.py | 54 |
| CribSquad.py | 53 |
| Enums.py | 23 |

## Current test state: effectively zero

`test/` has 5 files (571 lines) and none of them run:

- All import `from cribbage import ...` / `from cribbage.X import *`. The package is `cribsandladders`, not `cribbage` — it was renamed at some point and the tests were never updated.
- `test/agentTest.py` references an undefined `Agents` module and calls `discard_crib`/agent constructors that don't match current signatures.
- Root `test.py` imports a nonexistent `testing` module.
- No `pytest`/`unittest` config, no `requirements.txt` or `pyproject.toml`, no CI.

I confirmed this by trying to import the package in a clean environment: even `cribsandladders.ScoreHand` — the most "pure logic" module in the codebase — fails to import because it pulls in `game_params`, which does `from scipy.stats import norm` at module load time. `scipy`, `ezdxf`, `shapely`, `mystic`, and the two compiled extensions (`scoretree`, `markovgame`) aren't installed in this sandbox; they presumably are on your machine, but the point stands structurally: there is no lightweight import path into any of this code today.

## Structural blockers to TDD

1. **`game_params.py` is a global config module, not injectable state.** 328 lines of module-level constants — including hardcoded absolute Windows paths like `C:\\Users\\Dell 5290\\Documents\\...` — read directly throughout the codebase as `gp.numplayers`, `gp.flushmods`, etc. Even `ScoreHand.score_hand` branches on `gp.numplayers` instead of taking it as a parameter. Nearly every module transitively imports `game_params`, and importing it eagerly loads `scipy`. There's no way to run two tests with different configs in the same process without monkeypatching globals.

2. **Heavy/native dependencies with no pure-logic seam.** `scipy`, `ezdxf`, `shapely`, `seaborn`, `matplotlib`, `pandas`, `mystic`, plus two custom compiled pybind11 extensions (`scoretree` for card search, `markovgame` for event building) are required just to import large parts of the package. `Player.py` and `EventSetBuilder.py` cannot be imported without the compiled extensions present.

3. **SQLite and filesystem calls are inline in business logic.** 10 files (`Optimizer`, `EventSetBuilder`, `PossibleEvents`, `Stats`, `Evaluator`, `BoardSetter`, `DXFWriter`, `game_params`, `popRankLookupTable`, `cribbage_main`) call `sqlite3.connect(...)` or do file I/O directly inside the same methods that contain the algorithm. There's no repository/DAO boundary to substitute a temp DB or fake.

4. **`matplotlib.pyplot` calls are interleaved with computation** in `DXFWriter`, `Evaluator`, `EventSetBuilder`, `PossibleEvents`, and `Stats` — plotting and math live in the same functions.

5. **Unseeded, non-injectable randomness.** `random`/`np.random` is used directly (not passed in) in ~14 modules, including `Deck.shuffle`, `CribSquad`, `Optimizer`, `EventSetBuilder`. Tests on anything that touches these will be flaky unless the RNG is seeded or injected.

6. **God objects.** `EventSetBuilder` is a single 2184-line class. `PossibleEvents` is 1156 lines. `CribbageGame`, `Evaluator`, `Stats`, `Optimizer` each mix state-machine logic, scoring, and I/O in one class. These need to be decomposed before individual behaviors are unit-testable in isolation — that decomposition *is* most of the "refactor" work you're asking about.

7. **The AI search logic is opaque to Python tests.** `Player.getCardToPlay` hands off directly to the compiled `scoretree` extension. That logic can only be integration-tested (build the extension, run it) unless you define a narrow interface so a Python reference implementation can stand in for unit tests.

## Testability tiers

**Tier 1 — cheap to test once decoupled from `game_params`:** `Deck.py`, `ScoreHand.py`, `Enums.py`, the free functions in `CribbageGame.py` (`min_card`, `can_peg`). This is the highest-value, lowest-effort target.

**Tier 2 — moderate, needs seams for I/O/randomness but logic is tractable:** `Player.py` (excluding the `scoretree` call), `CribbageGame.py`, `CribSquad.py`, `Board.py`/`BaseLayout.py` (geometry), `BoardSetter.py`.

**Tier 3 — heavy, needs real refactor before tests are meaningful:** `Optimizer`, `EventSetBuilder`, `PossibleEvents`, `Evaluator`, `Stats`, `DXFWriter` — DB, filesystem, plotting, and optimization math are all interleaved.

**Tier 4 — outside Python unit-test reach:** the two C++ extensions. Test these via a small integration suite that requires them built, not via unit tests.

## Proposed phased plan

**Phase 0 — Harness (~0.5–1 day).** Add `pytest` + `pytest-cov`, a `requirements-dev.txt`, `pyproject.toml`/`pytest.ini` for discovery, fix the `cribbage` → `cribsandladders` import mismatch, delete or rewrite the currently-broken `test/` files, add a `conftest.py` with a seeded-RNG fixture and a temp-dir/temp-db fixture.

**Phase 1 — Decouple `game_params` (~2–4 days).** Turn it into a `GameConfig` object that's constructed and passed explicitly (default instance for production call sites, override in tests), and pull the hardcoded machine-specific paths out into config/env. This is mechanical but touches nearly every file, and it's the single change that unlocks testing everything downstream — do it early.

**Phase 2 — TDD the cribbage engine (~1.5–3 weeks).** `Deck`, `ScoreHand`, `CribbageGame`, `CribSquad`, `Player`. This is the actual playable game and the highest-value target. Since there's no existing spec, start with characterization tests that pin down current behavior, then refactor under that safety net. Inject a seeded `Random` instance instead of using the global `random` module. For `Player`, define a narrow interface around the `scoretree` call so a Python fake can substitute in unit tests, and cover the real extension with a handful of integration tests.

**Phase 3 — Board/geometry (~1–1.5 weeks).** Extract pure layout/geometry math out of `Board`, `BaseLayout`, `BoardSetter`, `DXFWriter` from the XML/DXF-writing side effects. Unit-test the geometry; snapshot-test the XML/DXF output at the I/O boundary.

**Phase 4 — Optimizer/board-design subsystem (~3–5 weeks, the biggest lift).** `Optimizer`, `EventSetBuilder`, `PossibleEvents`, `Evaluator`, `Stats`. Requires introducing a persistence seam (repository pattern around the `sqlite3` calls), a thin plotting adapter tests can no-op, and breaking `EventSetBuilder`'s 2184-line class into smaller, independently testable units. This is where the bulk of "full refactor" effort concentrates, largely because of `EventSetBuilder` and `PossibleEvents` alone.

**Phase 5 — Integration tests (~3–5 days).** Full agent-vs-agent playthroughs with seeded RNG asserting final score/state; a board-build + optimizer round trip against a temp SQLite DB and temp directory instead of the hardcoded paths.

## Rough effort total

- Engine only (Phases 0–2, partial 5): **~3–4.5 weeks** solo, if you want to scope this down to "the actual game has real tests" first.
- Full scope including the board/optimizer subsystem (all phases): **~8–13 weeks** solo, given the size of `EventSetBuilder`/`PossibleEvents` and the DB/plotting decoupling work in Phase 4.

These are solo-developer estimates assuming familiarity with the codebase (you), working in focused blocks — not calendar time with interruptions.

## Recommended sequencing

Start with the engine (Phase 0–2). It's self-contained, highest value, and doesn't require decomposing `EventSetBuilder`. Treat the two C++ extensions as black boxes covered by a small integration suite rather than something to unit-test from Python. Only take on Phase 4 (the optimizer subsystem) once the `game_params` decoupling from Phase 1 has proven out on the simpler engine code — the same pattern will need to repeat across a much larger surface area there.

## Immediate next steps

1. Decide scope: engine-only first, or full project.
2. Phase 0 harness + fix/delete the broken `test/` directory.
3. Spike the `game_params` → `GameConfig` refactor on `ScoreHand.py` alone to validate the pattern before rolling it out further.
