import unittest
import unittest.mock as mock
from cribsandladders.Deck import Deck, Card
import itertools
import copy as cp

# Constants for card suits
SPADES = 0
HEARTS = 1
CLUBS = 2
DIAMONDS = 3

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

class TestDeck(unittest.TestCase):
    def setUp(self):
        self.deck = Deck()

        self.rd_patch = mock.patch('random.random', return_value=0.5)
        self.mock_rand = self.rd_patch.start()
    def tearDown(self):
        self.rd_patch.stop()

    def test_initialization(self):
        self.assertIsInstance(self.deck.cards, list)

    def test_shuffle(self):
        old_order = [(c.rank, c.suit) for c in self.deck.cards]
        self.deck.shuffle()
        #NOTE: extreme 1/52! unlikely same order
        new_order = [(c.rank, c.suit) for c in self.deck.cards]

        self.assertNotEqual(old_order, new_order)

    def test_draw_card(self):
        #given
        top_card = self.deck.cards[-1]

        #when
        drawn_card = self.deck.drawCard()

        #then
        self.assertEqual(top_card, drawn_card)





if __name__ == "__main__":
    unittest.main()