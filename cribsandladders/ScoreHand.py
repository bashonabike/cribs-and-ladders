import heapq
import math
from itertools import combinations
import cribsandladders.Deck as dk
from copy import deepcopy
import game_params as gp



def score_hand(hand4cards, cutcard, is_crib=False):
    """
    Returns the total point value of a 4 card hand with the given cut card
    :param hand4cards: the 4 cards in the player's hand
    :param cutcard: cut card
    :param is_crib: if the hand being scored is the crib
    :return: integer point value of the hand
    """


    total_points = 0
    total_points += right_jack(hand4cards,cutcard)
    total_points += flush(hand4cards, cutcard, is_crib)

    sorted5cards=sort_cards(hand4cards,cutcard)

    total_points += two_card_fifteens(sorted5cards)
    total_points += three_card_fifteens(sorted5cards)
    total_points += four_card_fifteens(sorted5cards)
    total_points += five_card_fifteens(sorted5cards)
    total_points += runs(sorted5cards)
    total_points += pairs(sorted5cards)

    return total_points


def sort_cards(hand4cards,cutcard):
    """
    puts the hand of 4 cards and the cut card into one sorted hand
    :param hand4cards: 4 cards in the player's hand
    :param cutcard: cut card
    :return: sorted five card hand
    """
    hand_queue = []

    for c in hand4cards:
        heapq.heappush(hand_queue, c)
    heapq.heappush(hand_queue,cutcard)
    sorted5cards = heapq.nsmallest(5, hand_queue)
    return sorted5cards


def right_jack(hand4cards, cutcard):
    """
    Returns the point value from right jacks in the given hand
    :param hand4cards: the 4 cards in the player's hand
    :param cutcard: cut card
    :return: 1 point if the hand contains the right jack, 0 otherwise
    """
    points = 0
    # right jack
    for card in hand4cards:
        if card.rank == 11 and cutcard.suit == card.suit:  # if card in hand is a Jack and its suit matches the cut card
            points += 1
    return points


def flush(hand4cards, cutcard, is_crib):
    """
    Returns the point value from flushes in the given hand
    :param hand4cards: the 4 cards in the player's hand
    :param cutcard: cut card
    :return: points from flushes
    """
    points=0
    # flushes
    if hand4cards[0].suit == hand4cards[1].suit == hand4cards[2].suit == hand4cards[3].suit:
        points += 4
        if hand4cards[0].suit == cutcard.suit:
            points += 1
    #Crib can only be scored as flush if 5 card flush
    if is_crib:
        if points==4:
            points=0

    return points



def two_card_fifteens(sorted5cards):
    """
    Returns the point value of pairs of cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from two card 15's
    """
    points=0
    index_combinations2 = combinations([0,1,2,3,4], 2)
    for combination in list(index_combinations2):
        card1 = sorted5cards[combination[0]]
        value1=dk.peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2=dk.peg_val(card2)
        if value1 + value2 == 15:
            points += 2
    return points

def three_card_fifteens(sorted5cards):
    """
    Returns the point value of 3 cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from three card 15's
    """
    points=0
    index_combinations3 = combinations([0, 1, 2, 3, 4], 3)
    for combination in list(index_combinations3):
        card1 = sorted5cards[combination[0]]
        value1 = dk.peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2 = dk.peg_val(card2)
        card3 = sorted5cards[combination[2]]
        value3 = dk.peg_val(card3)
        if value1 + value2 + value3 == 15:
            points += 2
    return points

def four_card_fifteens(sorted5cards):
    """
    Returns the point value of 4 cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from four card 15's
    """
    points=0
    index_combinations4 = combinations([0, 1, 2, 3, 4], 4)
    for combination in list(index_combinations4):
        card1 = sorted5cards[combination[0]]
        value1 = dk.peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2 = dk.peg_val(card2)
        card3 = sorted5cards[combination[2]]
        value3 = dk.peg_val(card3)
        card4 = sorted5cards[combination[3]]
        value4=dk.peg_val(card4)
        if value1 + value2 + value3 + value4 == 15:
            points += 2
    return points

def five_card_fifteens(sorted5cards):
    """
    Returns the point value of 5 cards that sum to 15
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from five card 15's
    """
    points=0
    sum=0
    for i in range(5):
        card=sorted5cards[i]
        sum+=dk.peg_val(card)
    if sum==15:
        points+=2
    return points


def runs(sorted5cards):
    """
    Returns the point value from runs
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from runs
    """
    points=0
    for start_index in range(3):
        next_index=start_index+1
        consecutive_cards_count = 1
        duplicates_count = 0
        while next_index<5:
            if sorted5cards[start_index].rank == sorted5cards[next_index].rank:
                duplicates_count += 1
            elif sorted5cards[start_index].rank == sorted5cards[next_index].rank - 1:
                consecutive_cards_count += 1
            else:
                break
            start_index = next_index
            next_index += 1
        multiplier = 1
        if duplicates_count > 0:
            multiplier = duplicates_count*2
        if consecutive_cards_count >= 3:
            points += multiplier * consecutive_cards_count
            break
    return points

def pairs(sorted5cards):
    """
    Returns the point value from pairs (includes 3 of a kind and 4 of a kind)
    Allowing for 5 of a kind if using 2 decks
    :param sorted5cards: sorted list of 4 cards in the player's hand and the cut card
    :return: points from pairs
    """
    points=0
    start_card_index = 0
    while start_card_index < 4:
        index = start_card_index + 1
        for i in range(index, 5):
            if sorted5cards[start_card_index].rank == sorted5cards[i].rank:
                points += 2
        start_card_index += 1
    return points



def card_counts_list(pothand, potdiscard):
    """
    :param pothand: a list of 4 cards the player is keeping
    :param potdiscard: a list of the 2 cards the player is planning to discard
    :return:list of how many of each value are still in the 46 cards in the deck
    """
    card_counts=[]
    #NOTE: omitting suit since likely has minimal influence on discard decisions (only affects nibs & bolstering an
    #existing 4 card flush
    for i in range(13):
        #Allow for 8 cards in 2 deck play
        card_counts.append(gp.cardsperrank)
    for card in pothand:
        value=card.rank
        card_counts[value-1] -= 1
    if not isinstance(potdiscard, dk.Card):
        for card in potdiscard:
            value=card.rank
            card_counts[value-1] -= 1
    else:
        value = potdiscard.rank
        card_counts[value - 1] -= 1
    return card_counts


def flush_adder(hand, discard, risk):
    #NOTE: for performance, we are using additive products
    if hand[0].suit == hand[1].suit == hand[2].suit == hand[3].suit:
        #Factor in likelihood cut is also same suit
        if discard is dk.Card:
            if discard.suit == hand[0].suit:
                return gp.flushmods[1][risk-1]
            else:
                return gp.flushmods[0][risk-1]
        elif discard[0].suit == hand[0].suit:
            if len(discard) > 1 and discard[1].suit == hand[0].suit:
                return gp.flushmods[2][risk-1]
            else:
                return gp.flushmods[1][risk-1]
        else:
            return gp.flushmods[0][risk-1]

    return 0.0




def expected_hand_value(pothand, potdiscard, rankLookupTable, risk, hascrib):
    """
      Returns the expected point value of a hand (taking into account all possible cut cards)
      :param pothand: a list of 4 cards the player is keeping
      :param potdiscard: a list of the 1 or 2 cards the player is planning to discard
      :param risk: value between 1 and 21
      :return: expected point value for the 4 card hand
      """

    # card_counts=card_counts_list(pothand, potdiscard)
    # base_expected_value,aug_expected_value = 0.0, 0.0

    #Order tuples by rank, then suit
    pothandsorted = sorted(pothand, key = lambda c: (c.rank, c.suit))
    if not isinstance(potdiscard,  dk.Card):
        potdiscardsorted = sorted(potdiscard, key = lambda c: (c.rank, c.suit))
    else:
        potdiscardsorted = [potdiscard]

    #initiate sql conn & gen hashes
    handhash = int("{:02d}{:02d}{:02d}{:02d}".format(pothandsorted[0].rank, pothandsorted[1].rank,
                                                     pothandsorted[2].rank, pothandsorted[3].rank))
    if gp.numplayers == 2:
        discardhash = int("{:02d}{:02d}".format(potdiscardsorted[0].rank, potdiscardsorted[1].rank))
    else:
        discardhash = int("{:02d}".format(potdiscardsorted[0].rank))

    #Retrieve modulation from dataframe
    aug_expected_value = rankLookupTable.loc[(handhash, discardhash, 1 if hascrib else 0), "ValR{}".format(risk)]
    aug_expected_value += flush_adder(pothandsorted, potdiscardsorted, risk)

    handcounter, discardcounter = 0, 0
    drawpicks = 0


    #
    #
    # for suit in range (1,5):
    #     for rank in range (1,14):
    #         drawpicks = gp.numdecks
    #         if (handcounter < gp.handsize and (pothandsorted[handcounter].rank, pothandsorted[handcounter].suit) ==
    #                 (rank, suit)):
    #             handcounter += 1
    #             drawpicks -= 1
    #         if (discardcounter < gp.discardsize and drawpicks > 0 and
    #                 (potdiscardsorted[discardcounter].rank, potdiscardsorted[discardcounter].suit) ==
    #                 (rank, suit)):
    #             discardcounter += 1
    #             drawpicks -= 1
    #
    #         if drawpicks > 0:
    #             hand_value=score_hand(pothand, dk.Card(rank, suit), False)
    #             #gets the score of the hand for each possible cut card
    #             #incorporating risk
    #             if risk < 0.0:
    #                 aug_hand_value=math.sqrt(hand_value)
    #             else:
    #                 aug_hand_value=hand_value*hand_value
    #
    #             probability = float(drawpicks)/float(gp.unknowncardsafterdeal)               #calculates the probability of drawing that cut card
    #             base_expected_value += (hand_value*probability)
    #             #multiplies the calculated score by the probability of drawing that cut card, adds to total expected value
    #             aug_expected_value += (aug_hand_value*probability)


    return aug_expected_value


def crib_cards_value(discard, yourCrib):
    value=0
    if len(discard) == 2:
        card1_value=discard[0].rank
        card2_value=discard[1].rank
        if card1_value==card2_value:
            value+=2
        if card1_value+card2_value==15:
            value+=2
        if card1_value==5:
            value+=1
        if card2_value==5:
            value+=1
        if card1_value-card2_value==1 or card2_value-card1_value==1:
            value+=1
    elif len(discard) == 1:
        card1_value=discard[0].rank
        if card1_value==5:
            value+=1
    else:
        raise Exception("Game not configured for {} discards".format(len(discard)))

    if yourCrib:
        return value
    else:
        return -1*value

