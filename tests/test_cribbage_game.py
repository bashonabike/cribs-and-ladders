# python -m pytest tests/test_cribbage_game.py -v
#
# Relocated from cribsandladders/test_cribbage_game.py during Phase 0/1
# harness work -- pytest's testpaths is scoped to tests/, and this file
# belongs alongside the rest of the suite, not inside the package. No
# other changes; still needs pandas (via cribsandladders.Board) to
# import, which is a Phase 3 concern, not this file's.

import unittest
from unittest.mock import MagicMock, patch
from cribsandladders.CribbageGame import min_card, min_card_val, can_peg, CribbageGame, IllegalMoveException
from cribsandladders.Deck import Card#,  Rank
from cribsandladders.Board import Board, Track
import game_params as gp

# Constants for card
SPADES = 0
HEARTS = 1
CLUBS = 2
DIAMONDS = 3

#Hokey but that's what the ai gen'd
ACE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9
TEN = 10
JACK = 11
QUEEN = 12
KING = 13

CARD_RANK = {
    11: "J",
    12: "Q",
    13: "K",
    1: "A",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10"
}

CARD_SUIT = {
    SPADES: "♠",
    HEARTS: "♥",
    CLUBS: "♣",
    DIAMONDS: "♦"
}

class TestCribbageGameHelpers(unittest.TestCase):
    """
    NOTE: this class previously constructed cards as Card(SUIT, RANK)
    throughout, but the real Card constructor (cribsandladders/Deck.py)
    is Card(rank, suit) -- so every card here was silently built with
    its rank and suit swapped. That's also why test_min_card compared
    the result directly against a bare rank int (ACE) instead of a Card
    -- min_card returns a Card, not an int. Fixed below: arguments are
    now Card(RANK, SUIT) to match the real signature, and test_min_card
    compares against the expected Card object.
    """

    def test_min_card(self):
        """Test finding the card with the minimum pegging value."""
        cards = [
            Card(ACE, HEARTS),      # Value 1
            Card(THREE, DIAMONDS),  # Value 3
            Card(FIVE, CLUBS),      # Value 5
            Card(TWO, SPADES)       # Value 2
        ]
        self.assertEqual(min_card(cards), Card(ACE, HEARTS))

    def test_min_card_val(self):
        """Test getting the minimum pegging value from a hand."""
        cards = [
            Card(TEN, HEARTS),     # Value 10
            Card(FOUR, DIAMONDS),  # Value 4
            Card(JACK, CLUBS),     # Value 10
            Card(TWO, SPADES)      # Value 2
        ]
        self.assertEqual(min_card_val(cards), 2)

    def test_can_peg(self):
        """Test if a card can be played without exceeding 31."""
        cards = [
            Card(TEN, HEARTS),     # Value 10
            Card(FOUR, DIAMONDS),  # Value 4
            Card(JACK, CLUBS)      # Value 10
        ]
        self.assertTrue(can_peg(cards, 20))   # Can play 10 (30) or 4 (24)
        self.assertFalse(can_peg(cards, 30))  # No cards can be played


class TestCribbageGame(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the board and squad objects
        self.mock_board = MagicMock(spec=Board)
        # A real Track() instead of MagicMock(spec=Track): Track sets all
        # its attributes in __init__ rather than at class level, so
        # spec=Track only catches typos in attribute *names* -- it
        # doesn't give a mock any of Track's real default values, and
        # every attribute the test doesn't explicitly stub raises
        # AttributeError the first time real code (score_points,
        # checkChuteOrLadderForPos, Stats.Move) reads it. Track() takes
        # no constructor args and has no side effects, so a real
        # instance is both simpler and more robust here than patching
        # MagicMock attributes one crash at a time.
        self.mock_track = Track()
        # efflength defaults to 0 on a fresh Track, which would make any
        # positive pegging/hand score an immediate (false) win. Set it
        # high enough that score_points()'s win check never trips
        # unless a test means it to (test_score_points_win_condition
        # overrides this itself).
        self.mock_track.efflength = 1000
        self.mock_board.getTrackByNum.return_value = self.mock_track

        # Mock the squad and player objects
        self.mock_squad = MagicMock()
        self.mock_player = MagicMock()
        self.mock_player.num = 1
        self.mock_player.score = 0
        self.mock_player.pegginghand = []
        self.mock_player.canPlay = True
        # Without a real starting int, `player.wins += 1` in score_points()
        # calls MagicMock's auto-mocked __iadd__ and leaves self.mock_player.wins
        # as a Mock instead of an int, which fails any assertEqual(..., 1).
        self.mock_player.wins = 0
        self.mock_squad.getPlayerByNum.return_value = self.mock_player
        self.mock_squad.players = [self.mock_player]
        self.mock_squad.resetCanPlay.return_value = None
        self.mock_squad.getNextPeggingPlayer.return_value = None
        self.mock_squad.donePegging.return_value = True

        # Create the game instance
        self.game = CribbageGame(self.mock_board, self.mock_squad, trial=1)
        self.game.verbose = False  # Disable print statements during tests

    def test_initialization(self):
        """Test that the game initializes correctly."""
        self.assertEqual(self.game.round, 0)
        self.assertEqual(self.game.moveNum, 0)
        self.assertIsInstance(self.game.moves, list)
        self.assertEqual(len(self.game.moves), 0)

    def test_score_points_win_condition(self):
        """Test scoring points that trigger a win."""
        # Set up track length to 100 for testing
        self.mock_track.efflength = 100

        # Test winning score
        self.mock_player.score = 99
        result = self.game.score_points(2, "Test win", self.mock_player, False)
        self.assertTrue(result)  # Should win with 101 points
        self.assertEqual(self.mock_player.wins, 1)

        # Test non-winning score
        self.mock_player.score = 0
        result = self.game.score_points(50, "Test non-win", self.mock_player, False)
        self.assertFalse(result)  # Should not win with 50 points
        self.assertEqual(self.mock_player.wins, 1)  # Win count should not increase

    #TODO: fix this!!
    # @patch('cribsandladders.CribbageGame.Deck')
    # def test_play_game_flow(self, mock_deck):
    #     """Test the main game flow with mocked deck and player interactions."""
    #     # Set up mock deck and cards
    #     mock_card = MagicMock()
    #DOTHIS??MagicMock(spec=Card)
    #     mock_card.
    #     mock_card .= FIVE
    #     mock_deck.return_value.drawCard.return_value = mock_card
    #
    #     # Mock player's pegging hand and behavior
    #     self.mock_player.pegginghand = [mock_card]
    #     self.mock_player.pegging_move.return_value = (mock_card.muxed, False)
    #
    #     # Mock score_hand to not end the game
    #     with patch.object(self.game, 'score_hand', return_value=False):
    #         # Mock pegging to not end the game
    #         with patch.object(self.game, 'pegging', return_value=False):
    #             # Mock run_round to end the game after first round
    #             with patch.object(self.game, 'run_round', side_effect=[True]):
    #                 moves = self.game.play_game()
    #                 self.assertIsInstance(moves, list)
    #                 self.assertTrue(any(hasattr(move, 'winningMove') for move in moves))

    def test_illegal_move_card_not_in_hand(self):
        """
        pegging() should raise IllegalMoveException when the agent's
        pegging_move() returns a card that isn't actually in hand.

        (Original test tried to hit this via an *empty* pegginghand, but
        that path never reaches this check at all: can_peg() on an empty
        hand just returns False immediately, so the player gets marked
        canPlay=False and skipped. With only one mocked player in the
        squad, that cascades into the "everyone's stuck" branch, which
        calls score_points() with lastPlayedPlayer still None -- and
        None has no .tracknum, so it crashed with an unrelated
        AttributeError instead of ever raising IllegalMoveException.)
        """
        card = Card(TEN, HEARTS)
        self.mock_player.pegginghand = [card]
        # pegging_move "chooses" a muxed value that doesn't match the
        # one real card in hand.
        self.mock_player.pegging_move.return_value = (card.muxed + 1, False)

        with self.assertRaises(IllegalMoveException):
            self.game.pegging()

    def test_illegal_move_exceeds_31(self):
        """
        pegging() should raise IllegalMoveException if the chosen card
        would push the total over 31, even when a different card in
        hand would have been legal. can_peg() only checks whether the
        *minimum-value* card fits -- it doesn't validate whichever card
        pegging_move() actually returns.

        (The original test patched score_pegging() to raise the
        exception directly, which doesn't exercise this validation at
        all -- it only checks that pegging() propagates whatever
        score_pegging() raises. Reproducing the real branch needs the
        pegging total built up over a few plays first, since no single
        card is worth more than 10 and the total starts at 0.)
        """
        card1 = Card(TEN, HEARTS)    # play 1: total 0 -> 10
        card2 = Card(TEN, SPADES)    # play 2: total 10 -> 20
        card3 = Card(THREE, CLUBS)   # play 3: total 20 -> 23
        card4 = Card(TWO, DIAMONDS)  # remains; legal (2 + 23 = 25)
        card5 = Card(TEN, CLUBS)     # remains; illegal if chosen (10 + 23 = 33)

        self.mock_player.pegginghand = [card1, card2, card3, card4, card5]
        self.mock_player.pegging_move.side_effect = [
            (card1.muxed, False),
            (card2.muxed, False),
            (card3.muxed, False),
            (card5.muxed, False),  # picks the busting card over the legal one
        ]

        with self.assertRaises(IllegalMoveException):
            self.game.pegging()

    def test_score_pegging_awards_a_point_for_a_single_card_go(self):
        """
        CHARACTERIZATION TEST, not a spec of intended behavior -- this
        pins down a quirk (very likely a bug) in score_pegging() so any
        future fix is a deliberate, visible diff instead of an
        accidental one.

        score_pegging()'s reverse-scan loop that builds up runBuild
        (`for c in range(1, len(seq)): ...`) never executes when only
        one card has been played in the current Go, since
        range(1, 1) is empty. runBuild is left as [card.rank] (length
        1), and the run check `runMax - runMin == len(runBuild) - 1`
        trivially holds (0 == 0), so pegScore gets +1 for a "run" of a
        single card. Standard cribbage requires >=3 cards for a run --
        the very first card played in any Go should score 0 unless it
        happens to hit 15 (impossible on the first card alone) or you
        count "1 for last card", which is separate logic entirely.

        Net effect: every single first play of a Go currently scores 1
        point it shouldn't.
        """
        card = Card(TEN, HEARTS)
        result = self.game.score_pegging([card], total=10, player=self.mock_player, soexcite=False)

        self.assertFalse(result)
        self.assertEqual(self.mock_player.score, 1)

    def test_check_chute_or_ladder(self):
        """Test the chute and ladder checking logic."""
        # Mock the track's chutes and ladders
        self.mock_track.eventsListChute = [5, 15, 25]
        self.mock_track.chutes = [
            MagicMock(start=5, end=2),   # Chute from 5 to 2 (move back 3)
            MagicMock(start=15, end=10), # Chute from 15 to 10 (move back 5)
            MagicMock(start=25, end=20)  # Chute from 25 to 20 (move back 5)
        ]

        self.mock_track.eventsListLadder = [8, 18, 28]
        self.mock_track.ladders = [
            MagicMock(start=8, end=12),   # Ladder from 8 to 12 (move forward 4)
            MagicMock(start=18, end=22),  # Ladder from 18 to 22 (move forward 4)
            MagicMock(start=28, end=32)   # Ladder from 28 to 32 (move forward 4)
        ]

        # Test no event
        pos, event = self.game.checkChuteOrLadderForPos(self.mock_track, 3, 0)
        self.assertEqual(pos, 3)
        self.assertEqual(event.name, 'NONE')

        # Test chute
        pos, event = self.game.checkChuteOrLadderForPos(self.mock_track, 5, 0)
        self.assertEqual(pos, 2)  # Should move from 5 to 2
        self.assertEqual(event.name, 'CHUTE')

        # Test ladder
        pos, event = self.game.checkChuteOrLadderForPos(self.mock_track, 8, 0)
        self.assertEqual(pos, 12)  # Should move from 8 to 12
        self.assertEqual(event.name, 'LADDER')


if __name__ == '__main__':
    unittest.main()
