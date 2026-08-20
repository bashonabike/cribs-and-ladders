# Originally relocated from cribsandladders/test_deck.py during Phase
# 0/1 harness work; expanded during Phase 2 (TDD the engine) with
# fuller Deck/Card coverage and a determinism test for the injected-RNG
# shuffle() added in Phase 2.

import random
import unittest
import unittest.mock as mock
from cribsandladders.Deck import Deck, Card, card_to_string, peg_val, SPADES, HEARTS, CLUBS, DIAMONDS
import itertools
import copy as cp

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

    def test_initialization_has_52_unique_cards(self):
        self.assertEqual(len(self.deck.cards), 52)
        seen = set((c.rank, c.suit) for c in self.deck.cards)
        self.assertEqual(len(seen), 52)
        for suit in (SPADES, HEARTS, CLUBS, DIAMONDS):
            for rank in range(1, 14):
                self.assertIn((rank, suit), seen)

    def test_shuffle(self):
        old_order = [(c.rank, c.suit) for c in self.deck.cards]
        self.deck.shuffle()
        #NOTE: extreme 1/52! unlikely same order
        new_order = [(c.rank, c.suit) for c in self.deck.cards]

        self.assertNotEqual(old_order, new_order)

    def test_shuffle_with_seeded_rng_is_deterministic(self):
        deck_a = Deck()
        deck_b = Deck()
        deck_a.shuffle(random.Random(42))
        deck_b.shuffle(random.Random(42))
        order_a = [(c.rank, c.suit) for c in deck_a.cards]
        order_b = [(c.rank, c.suit) for c in deck_b.cards]
        self.assertEqual(order_a, order_b)

    def test_shuffle_with_different_seeds_differs(self):
        deck_a = Deck()
        deck_b = Deck()
        deck_a.shuffle(random.Random(1))
        deck_b.shuffle(random.Random(2))
        order_a = [(c.rank, c.suit) for c in deck_a.cards]
        order_b = [(c.rank, c.suit) for c in deck_b.cards]
        self.assertNotEqual(order_a, order_b)

    def test_draw_card(self):
        #given
        top_card = self.deck.cards[-1]

        #when
        drawn_card = self.deck.drawCard()

        #then
        self.assertEqual(top_card, drawn_card)

    def test_draw_card_removes_from_deck(self):
        self.assertEqual(len(self.deck.cards), 52)
        self.deck.drawCard()
        self.assertEqual(len(self.deck.cards), 51)

    def test_draw_card_raises_when_empty(self):
        empty = Deck()
        empty.cards = []
        with self.assertRaises(Exception):
            empty.drawCard()

    def test_draw_cards_returns_requested_count(self):
        cards = self.deck.drawCards(5)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(self.deck.cards), 47)

    def test_draw_cards_raises_when_not_enough_left(self):
        with self.assertRaises(Exception):
            self.deck.drawCards(53)


class TestCard(unittest.TestCase):
    def test_equality_by_rank_and_suit(self):
        self.assertEqual(Card(5, SPADES), Card(5, SPADES))

    def test_inequality_different_rank(self):
        self.assertNotEqual(Card(5, SPADES), Card(6, SPADES))

    def test_inequality_different_suit(self):
        self.assertNotEqual(Card(5, SPADES), Card(5, HEARTS))

    def test_lt_by_rank(self):
        self.assertTrue(Card(4, SPADES) < Card(5, SPADES))
        self.assertFalse(Card(5, SPADES) < Card(4, SPADES))

    def test_lt_by_suit_when_rank_ties(self):
        self.assertTrue(Card(5, SPADES) < Card(5, HEARTS))

    def test_gt_by_rank(self):
        self.assertTrue(Card(6, SPADES) > Card(5, SPADES))

    def test_hash_allows_use_in_sets(self):
        s = {Card(5, SPADES), Card(5, SPADES), Card(6, SPADES)}
        self.assertEqual(len(s), 2)

    def test_muxed_is_unique_per_rank_suit_combo(self):
        seen = set()
        for suit in (SPADES, HEARTS, CLUBS, DIAMONDS):
            for rank in range(1, 14):
                m = Card(rank, suit).muxed
                self.assertNotIn(m, seen)
                seen.add(m)


class TestCardToString(unittest.TestCase):
    def test_known_cards(self):
        self.assertEqual(card_to_string(Card(1, SPADES)), "A♠")
        self.assertEqual(card_to_string(Card(11, HEARTS)), "J♥")
        self.assertEqual(card_to_string(Card(13, DIAMONDS)), "K♦")
        self.assertEqual(card_to_string(Card(10, CLUBS)), "10♣")


class TestPegVal(unittest.TestCase):
    def test_number_cards_peg_at_face_value(self):
        for rank in range(1, 11):
            self.assertEqual(peg_val(Card(rank, SPADES)), rank)

    def test_face_cards_peg_at_ten(self):
        for rank in (11, 12, 13):
            self.assertEqual(peg_val(Card(rank, SPADES)), 10)


if __name__ == "__main__":
    unittest.main()
