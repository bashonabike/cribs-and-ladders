import random

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


def card_to_string(card):
    """
    Converts a Card object into a human-readable string (e.g., 'A♠').

    Args:
        card (Card): The card object to convert.
    Returns:
        str: The rank symbol and suit icon.
    """
    if card.rank in CARD_RANK and card.suit in CARD_SUIT:
        return CARD_RANK[card.rank] + CARD_SUIT[card.suit]
    else:
        raise Exception("Invalid card: " + str(card))


def peg_val(card):
    """
    Returns the value of a card for cribbage pegging.
    Face cards (J, Q, K) are worth 10; others are worth their rank.
    """
    return 10 if card.rank > 10 else card.rank


class Deck:
    """
    Represents a standard 52-card deck.
    """

    def __init__(self):
        """Initializes a full deck of 52 cards."""
        self.cards = []
        for suit in range(4):
            for i in range(1, 14):
                self.cards.append(Card(i, suit))

    def shuffle(self):
        """
        Shuffles the deck in place.
        """
        random.shuffle(self.cards)

    def drawCard(self):
        """
        Draws a single card from the top of the deck.

        Returns:
            Card: The drawn card.
        Raises:
            Exception: If the deck is empty.
        """
        if len(self.cards) < 1:
            raise Exception("Deck is out of cards")
        return self.cards.pop()

    def drawCards(self, count):
        """
        Draws a specified number of cards from the deck.

        Args:
            count (int): Number of cards to draw.
        Returns:
            list[Card]: A list containing the drawn cards.
        Raises:
            Exception: If the deck has fewer cards than requested.
        """
        if len(self.cards) < count:
            raise Exception("Deck is out of cards")
        return [self.cards.pop() for i in range(count)]


class Card:
    """
    Represents an individual playing card with a rank and suit.
    """

    def __init__(self, rank, suit):
        """
        Initializes a card.

        Args:
            rank (int): 1 (Ace) through 13 (King).
            suit (int): 0 (Spades) through 3 (Diamonds).
        """
        self.rank = rank
        self.suit = suit
        # 'muxed' provides a unique integer representation for sorting/indexing
        self.muxed = 100 * suit + rank

    def __hash__(self):
        """Allows Card objects to be used in sets or as dictionary keys."""
        return hash((self.rank, self.suit))

    def __eq__(self, other):
        """Checks equality between two cards."""
        return self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other):
        """Less-than comparison for sorting (by rank, then suit)."""
        return self.rank < other.rank or (self.rank == other.rank and self.suit < other.suit)

    def __gt__(self, other):
        """Greater-than comparison for sorting."""
        return self.rank > other.rank or (self.rank == other.rank and self.suit > other.suit)


if __name__ == "__main__":
    for i in range(10):
        deck = Deck()
        deck.shuffle()
        # Print string representation of cards for verification
        print([card_to_string(c) for c in deck.cards])