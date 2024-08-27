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
class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

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
