"""
Tests for cribsandladders.CribSquad.

Unblocked by the Phase 2 change to Player.py that made the scoretree
import lazy: CribSquad constructs real Player instances internally, and
Player used to import the compiled scoretree extension at module scope,
so importing CribSquad (even just to test track/risk assignment logic
that has nothing to do with pegging search) used to require the
extension to be built. It no longer does.
"""
import random
import unittest

from cribsandladders.CribSquad import CribSquad
from cribsandladders.config import GameConfig


def fake_move_selector(*args, **kwargs):
    return 0


class FakeTrack:
    """Minimal stand-in for cribsandladders.Board.Track -- CribSquad only
    ever reads .num off whatever's in the tracks list it's given."""
    def __init__(self, num):
        self.num = num


class TestCribSquadTrackAssignment(unittest.TestCase):
    def test_no_tracks_assigns_zero_to_everyone(self):
        squad = CribSquad(None, [], config=GameConfig(numplayers=3), move_selector=fake_move_selector)
        self.assertEqual(squad.tracksUsed, [0, 0, 0])

    def test_single_track_assigns_zero_to_everyone(self):
        squad = CribSquad(None, [FakeTrack(7)], config=GameConfig(numplayers=3), move_selector=fake_move_selector)
        self.assertEqual(squad.tracksUsed, [0, 0, 0])

    def test_enough_tracks_assigns_by_track_num(self):
        tracks = [FakeTrack(10), FakeTrack(20), FakeTrack(30)]
        squad = CribSquad(None, tracks, config=GameConfig(numplayers=3), move_selector=fake_move_selector)
        self.assertEqual(squad.tracksUsed, [10, 20, 30])
        self.assertEqual([p.tracknum for p in squad.players], [10, 20, 30])

    def test_explicit_tracks_used_of_correct_length_is_kept_as_is(self):
        squad = CribSquad(None, [], tracksUsed=[5, 6, 7], config=GameConfig(numplayers=3),
                           move_selector=fake_move_selector)
        self.assertEqual(squad.tracksUsed, [5, 6, 7])

    def test_explicit_tracks_used_of_wrong_length_is_replaced(self):
        # tracksUsed has 2 entries but numplayers=3 -> gets recomputed
        squad = CribSquad(None, [], tracksUsed=[5, 6], config=GameConfig(numplayers=3),
                           move_selector=fake_move_selector)
        self.assertEqual(squad.tracksUsed, [0, 0, 0])


class TestCribSquadPlayerCreation(unittest.TestCase):
    def test_creates_one_player_per_numplayers(self):
        squad = CribSquad(None, [], config=GameConfig(numplayers=3), move_selector=fake_move_selector)
        self.assertEqual(len(squad.players), 3)
        self.assertEqual([p.num for p in squad.players], [1, 2, 3])

    def test_two_player_config_creates_two_players(self):
        squad = CribSquad(None, [], config=GameConfig(numplayers=2), move_selector=fake_move_selector)
        self.assertEqual(len(squad.players), 2)

    def test_homo_risk_gives_everyone_the_same_risk(self):
        squad = CribSquad(None, [], homoRisk=True, config=GameConfig(numplayers=3),
                           move_selector=fake_move_selector)
        self.assertEqual([p.risk for p in squad.players], [11, 11, 11])

    def test_injected_move_selector_is_passed_to_players(self):
        squad = CribSquad(None, [], config=GameConfig(numplayers=2), move_selector=fake_move_selector)
        for p in squad.players:
            self.assertIs(p.move_selector, fake_move_selector)


class TestCribSquadRngInjection(unittest.TestCase):
    def test_seeded_rng_gives_deterministic_risks(self):
        squad_a = CribSquad(None, [], config=GameConfig(numplayers=3), rng=random.Random(7),
                             move_selector=fake_move_selector)
        squad_b = CribSquad(None, [], config=GameConfig(numplayers=3), rng=random.Random(7),
                             move_selector=fake_move_selector)
        self.assertEqual([p.risk for p in squad_a.players], [p.risk for p in squad_b.players])

    def test_risks_are_within_expected_bounds(self):
        squad = CribSquad(None, [], config=GameConfig(numplayers=3), rng=random.Random(123),
                           move_selector=fake_move_selector)
        for p in squad.players:
            self.assertGreaterEqual(p.risk, 1)
            self.assertLessEqual(p.risk, 21)

    def test_reset_risks_uses_injected_rng_deterministically(self):
        squad_a = CribSquad(None, [], config=GameConfig(numplayers=3), rng=random.Random(99),
                             move_selector=fake_move_selector)
        squad_b = CribSquad(None, [], config=GameConfig(numplayers=3), rng=random.Random(99),
                             move_selector=fake_move_selector)
        squad_a.resetRisks()
        squad_b.resetRisks()
        self.assertEqual([p.risk for p in squad_a.players], [p.risk for p in squad_b.players])

    def test_reset_risks_respects_homo_risk(self):
        squad = CribSquad(None, [], homoRisk=True, config=GameConfig(numplayers=3),
                           move_selector=fake_move_selector)
        squad.players[0].risk = 999
        squad.resetRisks()
        self.assertEqual([p.risk for p in squad.players], [11, 11, 11])


class TestCribSquadResetsAndLookups(unittest.TestCase):
    def setUp(self):
        self.squad = CribSquad(None, [], config=GameConfig(numplayers=3), move_selector=fake_move_selector)

    def test_reset_can_play_sets_everyone_true(self):
        for p in self.squad.players:
            p.canPlay = False
        self.squad.resetCanPlay()
        self.assertTrue(all(p.canPlay for p in self.squad.players))

    def test_reset_wins_zeroes_everyone(self):
        for p in self.squad.players:
            p.wins = 5
        self.squad.resetWins()
        self.assertTrue(all(p.wins == 0 for p in self.squad.players))

    def test_reset_scores_zeroes_everyone(self):
        for p in self.squad.players:
            p.score = 42
        self.squad.resetScores()
        self.assertTrue(all(p.score == 0 for p in self.squad.players))

    def test_get_player_by_num(self):
        p2 = self.squad.getPlayerByNum(2)
        self.assertEqual(p2.num, 2)

    def test_get_player_by_num_missing_returns_none(self):
        self.assertIsNone(self.squad.getPlayerByNum(999))

    def test_get_next_pegging_player_skips_cannot_play(self):
        # players numbered 1,2,3; starting from 1, player 2 can't play
        self.squad.getPlayerByNum(2).canPlay = False
        nxt = self.squad.getNextPeggingPlayer(1)
        self.assertEqual(nxt.num, 3)

    def test_get_next_pegging_player_returns_none_if_nobody_else_can_play(self):
        for p in self.squad.players:
            if p.num != 1:
                p.canPlay = False
        self.assertIsNone(self.squad.getNextPeggingPlayer(1))

    def test_done_pegging_true_when_all_hands_empty(self):
        for p in self.squad.players:
            p.pegginghand = []
        self.assertTrue(self.squad.donePegging())

    def test_done_pegging_false_when_any_hand_nonempty(self):
        for p in self.squad.players:
            p.pegginghand = []
        self.squad.players[0].pegginghand = ["not empty"]
        self.assertFalse(self.squad.donePegging())


if __name__ == "__main__":
    unittest.main()
