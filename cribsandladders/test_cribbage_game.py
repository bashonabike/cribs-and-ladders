# python -m unittest cribsandladders/test_cribbage_game.py -v

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

CARD_= {
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

CARD_= {
    SPADES: "♠",
    HEARTS: "♥",
    CLUBS: "♣",
    DIAMONDS: "♦"
}

class TestCribbageGameHelpers(unittest.TestCase):
    def test_min_card(self):
        """Test finding the card with the minimum pegging value."""
        cards = [
            Card(HEARTS, ACE),    # Value 1
            Card(DIAMONDS, THREE), # Value 3
            Card(CLUBS, FIVE),     # Value 5
            Card(SPADES, TWO)      # Value 2
        ]
        self.assertEqual(min_card(cards), ACE)
        
    def test_min_card_val(self):
        """Test getting the minimum pegging value from a hand."""
        cards = [
            Card(HEARTS, TEN),     # Value 10
            Card(DIAMONDS, FOUR),   # Value 4
            Card(CLUBS, JACK),      # Value 10
            Card(SPADES, TWO)       # Value 2
        ]
        self.assertEqual(min_card_val(cards), 2)
        
    def test_can_peg(self):
        """Test if a card can be played without exceeding 31."""
        cards = [
            Card(HEARTS, TEN),     # Value 10
            Card(DIAMONDS, FOUR),   # Value 4
            Card(CLUBS, JACK)       # Value 10
        ]
        self.assertTrue(can_peg(cards, 20))   # Can play 10 (30) or 4 (24)
        self.assertFalse(can_peg(cards, 30))  # No cards can be played


class TestCribbageGame(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the board and squad objects
        self.mock_board = MagicMock(spec=Board)
        self.mock_track = MagicMock(spec=Track)
        self.mock_track.effLandingForHoles = []
        self.mock_board.getTrackByNum.return_value = self.mock_track
        
        # Mock the squad and player objects
        self.mock_squad = MagicMock()
        self.mock_player = MagicMock()
        self.mock_player.num = 1
        self.mock_player.score = 0
        self.mock_player.pegginghand = []
        self.mock_player.canPlay = True
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

    def test_illegal_move_exception(self):
        """Test that illegal moves raise the appropriate exception."""
        # Test playing a card not in hand
        self.mock_player.pegginghand = []
        with self.assertRaises(IllegalMoveException):
            self.game.pegging()
            
        # Test playing a card that would exceed 31
        card = Card(HEARTS, TEN)
        self.mock_player.pegginghand = [card]
        with patch.object(self.game, 'score_pegging', side_effect=IllegalMoveException("Test")):
            with self.assertRaises(IllegalMoveException):
                self.game.pegging()

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
