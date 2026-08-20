from cribsandladders.ScoreHand import expected_hand_value
from cribsandladders.config import GameConfig, DEFAULT_CONFIG
from copy import deepcopy


def _scoretree_move_selector(handMuxed, nextPlayerCardsInHand, seqMuxed, effLandingForHoles,
                              nextPlayerEffLandingForHoles, current_sum, score, nextPlayerCurPos, numdecks):
    """
    Default pegging move selector: delegates to the compiled scoretree
    C++ extension. Imported lazily (only when this function actually
    runs) rather than at module scope, so that importing Player.py --
    and anything that imports Player.py, like CribSquad.py -- doesn't
    require scoretree to be built. Only calling pegging_move() with the
    default selector does.

    A different, injectable move_selector (see Player.__init__) can
    stand in for this in tests, without needing the extension at all.
    """
    import scoretree as stcpp
    return stcpp.getCardToPlay(handMuxed, nextPlayerCardsInHand, seqMuxed, effLandingForHoles,
                               nextPlayerEffLandingForHoles, current_sum, score, nextPlayerCurPos, numdecks)


class Player():

    def __init__(self, risk, num, rankLookupTable, tracknum=-1, config: GameConfig = DEFAULT_CONFIG,
                 move_selector=None):
        """
        Initializes a player with a risk value, player number, and rank lookup table.

        :param risk: the risk value for the player
        :param num: the player number
        :param rankLookupTable: the rank lookup table for the player
        :param tracknum: the track number for the player, defaults to -1
        :param config: game configuration (defaults to the module-level DEFAULT_CONFIG)
        :param move_selector: callable implementing the pegging move search
            (same signature as scoretree.getCardToPlay). Defaults to the
            real scoretree extension, imported lazily on first use. Tests
            can inject a fake here instead of needing the compiled
            extension built.
        """
        self.tracknum = tracknum
        self.num = num
        self.score = 0
        self.risk = risk
        self.hand = None
        self.pegginghand = None
        self.canPlay = True
        self.wins = 0
        self.rankLookupTable = rankLookupTable
        self.config = config
        self.move_selector = move_selector or _scoretree_move_selector

    def deal_hand(self, deck, count):
        """
        Deals a specified number of cards from the deck to the player's hand.

        :param deck: the deck to draw from
        :param count: the number of cards to draw
        :return: None
        """
        self.hand = deck.drawCards(count)

    def discard_crib(self, is_dealer):
        """

        :param hand:6 card hand dealt to player
        :param is_dealer: if the player is the dealer
        :return:
        """

        # creates a list of expected values for each 4 card hand
        four_card_hands = self.get_possible_4_hands(self.hand)
        # num_four_card_hands =len(four_card_hands)
        for hand in four_card_hands:
            value = expected_hand_value(hand.hand, hand.discard, self.rankLookupTable, self.risk, is_dealer, self.config)
            # Blend augmented and actual hand values as per risk tolerance
            # effvalue = abs(self.risk)*aug_value + (1.0 - abs(self.risk))*value
            # NOTE order them before performing lookup!
            # Maybe pull bulk table into Panda, build comb index, sort, use that as lookup
            # Cmpr speed may be faster just do db query
            # if dealer, add the 2 values, else subtract the two valuse (once both blended)
            # pre-cache for both calcs too, use to pop tree
            # maybe use hash table!
            # While at it factor in value of flush w/ 5 cards if cut card is same suit
            # factor into the risk calc
            hand.value = value

        # gets list of cards to discard corresponding to max value
        final_discard = max(four_card_hands, key=lambda x: x.value).discard
        for card in final_discard:
            self.hand.remove(card)
        self.pegginghand = deepcopy(self.hand)
        return final_discard

    def get_possible_4_hands(self, hand):
        """
        Gets all possible 4 card hands from a dealt hand. Hand size
        varies with self.config.dealsize (5 for the default 3-player
        config, 6 for 2-player) -- this is NOT always a 6 card hand.

        When dealsize > 5 (2-player rules: discard 2 to the crib), each
        candidate 4-hand discards a 2-card combination. Otherwise
        (3-player rules: discard 1 to the crib), each candidate
        discards a single card.

        For each 4 card hand, creates a PossibleHand object with the 4 card hand and the discarded card(s) not in it.
        Uses a set to avoid duplicates and only adds the PossibleHand object if the set does not already contain the hand.
        Returns a list of PossibleHand objects.

        :param hand: the dealt hand (size = self.config.dealsize) to generate possible 4 card hands from
        :return: a list of PossibleHand objects
        """
        possible_4 = []
        parallel_hand_set = set()
        for i in range(len(hand)):
            first_card = hand[i]
            if self.config.dealsize > 5:
                for j in range(i + 1, len(hand)):
                    second_card = hand[j]
                    copyhand = deepcopy(hand)
                    tempdiscard = [first_card, second_card]
                    copyhand.remove(second_card)
                    copyhand.remove(first_card)
                    copyhand.sort()
                    copyhand_t = tuple(copyhand)
                    tempdiscard.sort()
                    if copyhand_t not in parallel_hand_set:
                        parallel_hand_set.add(copyhand_t)
                        possible_4.append(PossibleHand(copyhand, tempdiscard))
            else:
                copyhand = deepcopy(hand)
                copyhand.remove(first_card)
                copyhand.sort()
                copyhand_t = tuple(copyhand)
                if copyhand_t not in parallel_hand_set:
                    parallel_hand_set.add(copyhand_t)
                    possible_4.append(PossibleHand(copyhand, [first_card]))

        return possible_4

    def pegging_move(self, sequence, current_sum, effLandingForHoles, nextPlayerEffLandingForHoles,
                     nextPlayerCardsInHand, nextPlayerCurPos):
        """
        Chooses a card to play during pegging
        :param sequence: the current sequence
        :param current_sum: the current sum on the table
        :param effLandingForHoles: the current pos of the player's peg
        :param nextPlayerEffLandingForHoles: the current pos of the player's peg
        :param nextPlayerCardsInHand: the current pos of the player's peg
        :param nextPlayerCurPos: the current pos of the player's peg
        :return: a single Card
        """
        handMuxed = [c.muxed for c in self.pegginghand]
        seqMuxed = [c.muxed for c in sequence]
        resultMuxed = self.move_selector(handMuxed, nextPlayerCardsInHand, seqMuxed, effLandingForHoles,
                                          nextPlayerEffLandingForHoles, current_sum,
                                          self.score, nextPlayerCurPos, self.config.numdecks)
        soexcite = resultMuxed >= 1000
        cardToPlayMuxed = resultMuxed % 1000
        if cardToPlayMuxed == 0:
            sdfsd = ""
        return cardToPlayMuxed, soexcite


class PossibleHand:
    def __init__(self, hand, discard):
        """
        Initializes a PossibleHand object.

        :param hand: a list of 4 cards representing a possible hand
        :param discard: a list of 2 cards representing the discard from the hand
        :return: None
        """
        self.hand = hand
        self.discard = discard
        self.value = 0
