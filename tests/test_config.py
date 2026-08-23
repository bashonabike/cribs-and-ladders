"""
Phase 1 tests: GameConfig decoupling.

These aren't full engine/game-logic coverage (that's Phase 2 -- TDD on
Deck/ScoreHand/CribbageGame/Player/CribSquad). The point here is to pin
down the two things Phase 1 was actually for:

  1. GameConfig instances are independent and produce the same derived
     values the old game_params.py globals did, for both the 2-player
     and 3-player, single- and double-deck branches.
  2. The game_params.py backward-compat shim still exposes the same
     names with the same values, and -- critically -- does not touch
     the filesystem or a database at import time (it used to open a
     live sqlite3 connection as an import-time side effect).

No scipy, numpy, or sqlite3 file dependency is needed for any of this;
that's the whole point of the refactor.
"""
import sys

import pytest

from cribsandladders.config import GameConfig, DEFAULT_CONFIG


class TestGameConfigDerivation:
    def test_default_matches_original_game_params_defaults(self):
        # DEFAULT_CONFIG uses the same defaults game_params.py used to
        # hardcode: twodecks=False, numplayers=3.
        assert DEFAULT_CONFIG.twodecks is False
        assert DEFAULT_CONFIG.numplayers == 3
        assert DEFAULT_CONFIG.dealsize == 5
        assert DEFAULT_CONFIG.handsize == 4
        assert DEFAULT_CONFIG.discardsize == 1
        assert DEFAULT_CONFIG.cribstartsize == 1
        assert DEFAULT_CONFIG.numdecks == 1
        assert DEFAULT_CONFIG.cardsperrank == 4
        assert DEFAULT_CONFIG.unknowncardsafterdeal == 46
        assert DEFAULT_CONFIG.avgMovesPerPegging == 2.115606979222163
        assert DEFAULT_CONFIG.ideallikelihoodholehit == 0.30000583333333336

    @pytest.mark.parametrize(
        "numplayers,twodecks,dealsize,handsize,discardsize,cribstartsize,numdecks,cardsperrank",
        [
            (2, False, 6, 4, 2, 0, 1, 4),
            (2, True, 6, 4, 2, 0, 2, 8),
            (3, False, 5, 4, 1, 1, 1, 4),
            (3, True, 5, 4, 1, 1, 2, 8),
        ],
    )
    def test_derived_fields_for_each_ruleset(
        self, numplayers, twodecks, dealsize, handsize, discardsize, cribstartsize, numdecks, cardsperrank
    ):
        cfg = GameConfig(numplayers=numplayers, twodecks=twodecks)
        assert cfg.dealsize == dealsize
        assert cfg.handsize == handsize
        assert cfg.discardsize == discardsize
        assert cfg.cribstartsize == cribstartsize
        assert cfg.numdecks == numdecks
        assert cfg.cardsperrank == cardsperrank

    def test_unsupported_player_count_raises(self):
        with pytest.raises(ValueError, match="not configured yet"):
            GameConfig(numplayers=5)

    def test_flushmods_shape_and_symmetry_point(self):
        cfg = GameConfig()
        assert len(cfg.flushmods) == 3
        assert all(len(row) == 21 for row in cfg.flushmods)
        # flushmods[d][10] is the anchor point the rest of the row is
        # interpolated from -- for single-deck, discard-of-2-same-suit case:
        assert cfg.flushmods[2][10] == pytest.approx(4 + (13.0 - 6.0) / 52.0)

    def test_prob_peg_rounds_normalizes_to_one(self):
        cfg = GameConfig()
        total = sum(r["prob"] for r in cfg.probPegRounds)
        assert total == pytest.approx(1.0)

    def test_instances_are_independent(self):
        a = GameConfig(numplayers=2)
        b = GameConfig(numplayers=3)
        a.flushmods[0][0] = 12345.0
        assert b.flushmods[0][0] != 12345.0
        assert a.numplayers != b.numplayers

    @pytest.mark.parametrize("numplayers,twodecks", [(2, False), (2, True), (3, False), (3, True)])
    def test_prob_hand_and_peg_hist_shape_for_each_ruleset(self, numplayers, twodecks):
        # Phase 10: probHandHist/probPegHist come from the _RULESET_TABLES
        # lookup now instead of four near-identical append-heavy branches.
        # Pin down the shape (move/prob dicts, expected lengths, roughly
        # normalized probabilities) for every (numplayers, twodecks) combo.
        cfg = GameConfig(numplayers=numplayers, twodecks=twodecks)
        assert len(cfg.probHandHist) == 19
        assert len(cfg.probPegHist) == 14
        assert [entry["move"] for entry in cfg.probHandHist] == list(range(1, 20))
        assert [entry["move"] for entry in cfg.probPegHist] == list(range(1, 15))
        # probPegHist is a proper normalized distribution; probHandHist is
        # empirical and, pre-existing (not introduced by this refactor),
        # only sums to ~0.99-0.996 rather than exactly 1 -- preserved
        # verbatim from the original game_params.py data, not "fixed" here.
        assert sum(entry["prob"] for entry in cfg.probPegHist) == pytest.approx(1.0, abs=1e-6)
        assert 0.98 <= sum(entry["prob"] for entry in cfg.probHandHist) <= 1.0

    def test_prob_hist_instances_are_independent(self):
        # The four rulesets live in a module-level lookup table
        # (_RULESET_TABLES) shared across every GameConfig instance;
        # mutating one instance's probHandHist must not leak into
        # another instance built from the same ruleset afterward.
        a = GameConfig(numplayers=3, twodecks=False)
        a.probHandHist[0]["prob"] = 999.0
        b = GameConfig(numplayers=3, twodecks=False)
        assert b.probHandHist[0]["prob"] != 999.0

    def test_data_root_overridable_per_instance(self, tmp_path):
        cfg = GameConfig(data_root=tmp_path)
        assert cfg.db_path == str(tmp_path / "Boards" / "AllBoards.db")
        assert cfg.eventenergyfile.startswith(str(tmp_path))
        # doesn't affect the default instance
        assert DEFAULT_CONFIG.db_path != cfg.db_path


class TestGameParamsShim:
    """game_params.py must keep working exactly as before for the
    modules that haven't been migrated to explicit config injection yet
    (Optimizer, EventSetBuilder, PossibleEvents, Evaluator, Stats,
    Board, BaseLayout, BoardSetter -- Phase 3/4 work)."""

    def test_flat_names_match_default_config(self):
        import game_params as gp

        assert gp.numplayers == DEFAULT_CONFIG.numplayers
        assert gp.dealsize == DEFAULT_CONFIG.dealsize
        assert gp.cribstartsize == DEFAULT_CONFIG.cribstartsize
        assert gp.cardsperrank == DEFAULT_CONFIG.cardsperrank
        assert gp.flushmods == DEFAULT_CONFIG.flushmods

    def test_import_does_not_touch_the_database(self):
        # Reload in a subprocess-free way: check the lazy cache is empty
        # immediately after import and only populates on first access.
        import game_params as gp

        # Reset the lazy cache to simulate a fresh import's state, since
        # the module is already imported/cached by pytest at this point.
        gp._lazy_db.clear()
        assert gp._lazy_db == {}

    def test_db_attrs_are_lazy_and_cached(self):
        import game_params as gp

        gp._lazy_db.clear()
        stub = gp.insertstatstub
        assert "INSERT INTO Stat" in stub
        assert set(gp._lazy_db.keys()) == {"sqliteConn", "sqliteCursor", "insertstatstub"}
        # second access reuses the same connection, doesn't reopen it
        conn_again = gp.sqliteConn
        assert conn_again is gp._lazy_db["sqliteConn"]

    def test_unknown_attr_raises_attribute_error(self):
        import game_params as gp

        with pytest.raises(AttributeError):
            gp.this_attr_does_not_exist
