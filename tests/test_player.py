"""
Tests for cribsandladders.Player.

pegging_move() tests use an injected fake move_selector (see
Player.__init__ / Phase 2's lazy-scoretree change) instead of the real
compiled scoretree extension. discard_crib() tests patch
expected_hand_value() directly rather than building a real pandas
rankLookupTable -- that table is a large precomputed artifact belonging
to popRankLookupTable.py, out of scope here; discard_crib()'s own job is
just "call expected_hand_value() for every candidate 4-hand and keep the
one with the max value", which is what these tests characterize.
"""
import unittest
import unittest.mock as mock

from cribsandladders.Deck import Card, SPADES, HEARTS, CLUBS, DIAMONDS
from cribsandladders.Player import Player, PossibleHand
from cribsandladders.config import GameConfig


def fake_move_selector(*args, **kwargs):
    return 0


class FakeDeck:
    """Minimal stand-in for cribsandladders.Deck.Deck -- deal_hand() only
    ever calls .drawCards(count) on whatever it's given."""
    def __init__(self, cards):
        self.cards = cards

    def drawCards(self, count):
        drawn, self.cards = self.cards[:count], self.cards[count:]
        return drawn


class TestPlayerDealHand(unittest.TestCase):
    def test_deal_hand_draws_requested_count(self):
        player = Player(risk=11, num=1, rankLookupTable=None, move_selector=fake_move_selector)
        deck = FakeDeck([Card(r, SPADES) for r in range(1, 8)])
        player.deal_hand(deck, 5)
        self.assertEqual(len(player.hand), 5)
        self.assertEqual(len(deck.cards), 2)


class TestGetPossible4Hands(unittest.TestCase):
    def test_three_player_config_discards_singles(self):
        # dealsize=5 for the default (3-player) config -> "else" branch:
        # one candidate per card removed, each discarding a single card.
        player = Player(risk=11, num=1, rankLookupTable=None, config=GameConfig(numplayers=3),
                         move_selector=fake_move_selector)
        hand = [Card(2, SPADES), Card(4, HEARTS), Card(6, CLUBS), Card(8, DIAMONDS), Card(10, SPADES)]
        possible = player.get_possible_4_hands(hand)

        self.assertEqual(len(possible), 5)
        for ph in possible:
            self.assertIsInstance(ph, PossibleHand)
            self.assertEqual(len(ph.hand), 4)
            self.assertEqual(len(ph.discard), 1)
            # the kept hand + discard should reconstruct the original 5 cards
            self.assertEqual(sorted(ph.hand + ph.discard), sorted(hand))

    def test_two_player_config_discards_pairs(self):
        # dealsize=6 for a 2-player config -> "if" branch: candidates
        # discard 2 cards at a time (every combination of 2).
        player = Player(risk=11, num=1, rankLookupTable=None, config=GameConfig(numplayers=2),
                         move_selector=fake_move_selector)
        hand = [Card(r, SPADES) for r in (2, 4, 6, 8, 10, 12)]
        possible = player.get_possible_4_hands(hand)

        for ph in possible:
            self.assertEqual(len(ph.hand), 4)
            self.assertEqual(len(ph.discard), 2)
            self.assertEqual(sorted(ph.hand + ph.discard), sorted(hand))
        # C(6,2) = 15 distinct discard pairs, all cards have distinct
        # ranks so no duplicate (hand, discard) combinations collapse.
        self.assertEqual(len(possible), 15)


class TestDiscardCrib(unittest.TestCase):
    def test_discards_the_highest_value_option(self):
        player = Player(risk=11, num=1, rankLookupTable="fake-table", config=GameConfig(numplayers=3),
                         move_selector=fake_move_selector)
        player.hand = [Card(2, SPADES), Card(9, SPADES), Card(5, HEARTS), Card(3, CLUBS), Card(13, DIAMONDS)]

        # Stand in for the real (pandas-backed) hand-value lookup: value
        # the discard purely by its rank, so the King (13) should always
        # "win" and get discarded.
        with mock.patch(
            "cribsandladders.Player.expected_hand_value",
            side_effect=lambda hand, discard, rankTable, risk, is_dealer, config: discard[0].rank,
        ):
            final_discard = player.discard_crib(is_dealer=False)

        self.assertEqual(final_discard, [Card(13, DIAMONDS)])
        self.assertEqual(
            sorted(player.hand),
            sorted([Card(2, SPADES), Card(9, SPADES), Card(5, HEARTS), Card(3, CLUBS)]),
        )
        self.assertEqual(sorted(player.pegginghand), sorted(player.hand))
        # pegginghand must be an independent copy, not aliased to hand
        self.assertIsNot(player.pegginghand, player.hand)

    def test_discards_the_lowest_value_option_when_inverted(self):
        player = Player(risk=11, num=1, rankLookupTable="fake-table", config=GameConfig(numplayers=3),
                         move_selector=fake_move_selector)
        player.hand = [Card(2, SPADES), Card(9, SPADES), Card(5, HEARTS), Card(3, CLUBS), Card(13, DIAMONDS)]

        with mock.patch(
            "cribsandladders.Player.expected_hand_value",
            side_effect=lambda hand, discard, rankTable, risk, is_dealer, config: -discard[0].rank,
        ):
            final_discard = player.discard_crib(is_dealer=False)

        # lowest rank (2) now maximizes -rank, so it should get discarded
        self.assertEqual(final_discard, [Card(2, SPADES)])


class TestPeggingMove(unittest.TestCase):
    def setUp(self):
        self.player = Player(risk=11, num=1, rankLookupTable=None, config=GameConfig(numplayers=3))
        self.player.pegginghand = [Card(5, SPADES), Card(7, HEARTS)]

    def test_uses_injected_move_selector_and_unmuxes_result(self):
        chosen = self.player.pegginghand[0]
        self.player.move_selector = mock.MagicMock(return_value=chosen.muxed)

        picked_muxed, soexcite = self.player.pegging_move(
            sequence=[], current_sum=0, effLandingForHoles=[], nextPlayerEffLandingForHoles=[],
            nextPlayerCardsInHand=0, nextPlayerCurPos=0,
        )

        self.assertEqual(picked_muxed, chosen.muxed)
        self.assertFalse(soexcite)
        self.player.move_selector.assert_called_once()

    def test_soexcite_flag_set_when_result_is_1000_or_more(self):
        chosen = self.player.pegginghand[0]
        self.player.move_selector = mock.MagicMock(return_value=1000 + chosen.muxed)

        picked_muxed, soexcite = self.player.pegging_move(
            sequence=[], current_sum=0, effLandingForHoles=[], nextPlayerEffLandingForHoles=[],
            nextPlayerCardsInHand=0, nextPlayerCurPos=0,
        )

        self.assertEqual(picked_muxed, chosen.muxed)
        self.assertTrue(soexcite)

    def test_move_selector_receives_configured_numdecks(self):
        cfg = GameConfig(twodecks=True)
        player = Player(risk=11, num=1, rankLookupTable=None, config=cfg)
        player.pegginghand = [Card(5, SPADES)]
        player.move_selector = mock.MagicMock(return_value=Card(5, SPADES).muxed)

        player.pegging_move(
            sequence=[], current_sum=0, effLandingForHoles=[], nextPlayerEffLandingForHoles=[],
            nextPlayerCardsInHand=0, nextPlayerCurPos=0,
        )

        called_numdecks = player.move_selector.call_args[0][-1]
        self.assertEqual(called_numdecks, cfg.numdecks)
        self.assertEqual(cfg.numdecks, 2)


if __name__ == "__main__":
    unittest.main()
