import random
from enum import Enum


SPADES = 0
HEARTS = 1
CLUBS = 2
DIAMONDS = 3


CARD_RANK = {
    11:"J",
    12:"Q",
    13:"K",
    1:"A",
    2:"2",
    3:"3",
    4:"4",
    5:"5",
    6:"6",
    7:"7",
    8:"8",
    9:"9",
    10:"10"
}

CARD_SUIT = {
    SPADES:"♠",
    HEARTS:"♥",
    CLUBS:"♣",
    DIAMONDS:"♦"
}


def card_to_string(card):
    if card.rank in CARD_RANK and card.suit in CARD_SUIT:
        # return CARD_VALUE_STRINGS[card.rank] +" of "+SUIT_STRINGS[card.suit]
        return CARD_RANK[card.rank]+CARD_SUIT[card.suit]
    else:
        raise Exception("Invalid card: "+card)


def peg_val(card):
    return 10 if card.rank>10 else card.rank


class Deck:

    def __init__(self):
        self.cards = []
        for suit in range(4):
            for i in range(1, 14):
                self.cards.append(Card(i, suit))

    def shuffle(self):
        """
        shuffles the deck
        :return: None
        """
        random.shuffle(self.cards)

    def drawCard(self):
        """
        Draws a single card if possible. Throws an Exception if not
        :return: a card
        """
        if len(self.cards) < 1:
            raise Exception("Deck is out of cards")
        return self.cards.pop()

    def drawCards(self, count):
        """
        Draws count cards if possible. Throws and Exception if not
        :param count: the number of cards to draw
        :return: a list of cards
        """
        if len(self.cards) < count:
            raise Exception("Deck is out of cards")
        return [self.cards.pop() for i in range(count)]

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.muxed = 100*suit + rank

    #Allows objects to be removed from copy of hand
    def __hash__(self):
        return hash((self.rank, self.suit))

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    # def __str__(self):
    #     return str(vars(self))

    def __lt__(self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.suit < other.suit)

    def __gt__(self, other):
        return self.rank > other.rank or (self.rank == other.rank and self.suit > other.suit)

if __name__ == "__main__":
    for i in range(10):
        deck=Deck()
        deck.shuffle()
        print(deck.cards)
