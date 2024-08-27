import random
from enum import Enum
import Card as cd


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

class DeckWithoutSpecCards:

    def __init__(self, numdecks, cardsOmit):
        self.cards = []
        cardsOmit.sort(key = lambda c: (c.suit, c.rank))
        cardsOmit_idx = 0
        for suit in range(4):
            for rank in range(1, 14):
                for d in range(0, numdecks):
                    tentCard = cd.Card(rank, suit)
                    if cardsOmit_idx < len(cardsOmit) and cardsOmit[cardsOmit_idx] == tentCard:
                        cardsOmit_idx += 1
                    else:
                        self.cards.append(tentCard)

    def getDeckAsList(self):
        return self.cards

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

