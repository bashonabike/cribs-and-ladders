import random
import heapq
import math

from cribsandladders.CribbageGame import can_peg
from cribsandladders.Deck import card_to_string, peg_val
from cribsandladders.ScoreHand import expected_hand_value, crib_cards_value
from cribsandladders.Deck import Deck, card_to_string, peg_val, Card
from copy import deepcopy
import cribsandladders.ScoreHand as scorer
import game_params as gp
from itertools import combinations
import time as tm
import scoretree as stcpp


class Player():

    def __init__(self, risk, num,rankLookupTable, tracknum = -1):
        self.tracknum = tracknum
        self.num = num
        self.score = 0
        self.risk=risk
        self.hand = None
        self.pegginghand = None
        self.canPlay = True
        self.wins = 0
        self.rankLookupTable = rankLookupTable

    def deal_hand(self, deck, count):
        self.hand = deck.drawCards(count)
    def discard_crib(self, is_dealer):
        """

        :param hand:6 card hand dealt to player
        :param is_dealer: if the player is the dealer
        :return:
        """

        # creates a list of expected values for each 4 card hand
        four_card_hands=self.get_possible_4_hands(self.hand)
        # num_four_card_hands =len(four_card_hands)
        for hand in four_card_hands:
            value = expected_hand_value(hand.hand, hand.discard,self.rankLookupTable, self.risk, is_dealer)
            #Blend augmented and actual hand values as per risk tolerance
            # effvalue = abs(self.risk)*aug_value + (1.0 - abs(self.risk))*value
            #NOTE order them before performing lookup!
            #Maybe pull bulk table into Panda, build comb index, sort, use that as lookup
            #Cmpr speed may be faster just do db query
            #if dealer, add the 2 values, else subtract the two valuse (once both blended)
            #pre-cache for both calcs too, use to pop tree
            #maybe use hash table!
            #While at it factor in value of flush w/ 5 cards if cut card is same suit
            #factor into the risk calc
            hand.value = value

        # gets list of cards to discard corresponding to max value
        final_discard = max(four_card_hands, key=lambda x: x.value).discard
        for card in final_discard:
            self.hand.remove(card)
        self.pegginghand = deepcopy(self.hand)
        return final_discard

    def get_possible_4_hands(self, hand):
        possible_4 = []
        parallel_hand_set = set()
        for i in range(len(hand)):
            first_card = hand[i]
            if gp.dealsize > 5:
                for j in range(i+1, len(hand)):
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
        resultMuxed = stcpp.getCardToPlay(handMuxed, nextPlayerCardsInHand, seqMuxed, effLandingForHoles, nextPlayerEffLandingForHoles, current_sum,
                                            self.score, nextPlayerCurPos, gp.numdecks)
        soexcite = resultMuxed >= 1000
        cardToPlayMuxed = resultMuxed%1000
        if cardToPlayMuxed == 0:
            sdfsd=""
        return cardToPlayMuxed, soexcite

class PossibleHand:
    def __init__(self, hand, discard):
        self.hand = hand
        self.discard = discard
        self.value = 0