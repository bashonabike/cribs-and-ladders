import cribsandladders.Deck as dk
import math
import cribsandladders.ScoreHand as sc
import itertools as it
from copy import deepcopy
import sqlite3 as sql
import cribsandladders.Player as pl
import heapq
import numpy
from io import StringIO

numdecks = 2
players = 2

dealsize = 5 if players == 3 else 6
cardsperrank = 4*numdecks
handsize = 4
discardsize = 1 if players == 3 else 2
unknowncardsafterdeal = 52*numdecks - dealsize

#TODO: re pop the 2 deck 2 player play
def peg_val(card):
    return 10 if card>10 else card
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
        if card == 11:  # if card in hand is a Jack and its suit matches the cut card
            #APPROXIMATE!! actual value modulated by suit makeup of hand, but only slightly
            points += 0.25
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
    index_combinations2 = it.combinations([0,1,2,3,4], 2)
    for combination in list(index_combinations2):
        card1 = sorted5cards[combination[0]]
        value1=peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2=peg_val(card2)
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
    index_combinations3 = it.combinations([0, 1, 2, 3, 4], 3)
    for combination in list(index_combinations3):
        card1 = sorted5cards[combination[0]]
        value1 = peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2 = peg_val(card2)
        card3 = sorted5cards[combination[2]]
        value3 = peg_val(card3)
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
    index_combinations4 = it.combinations([0, 1, 2, 3, 4], 4)
    for combination in list(index_combinations4):
        card1 = sorted5cards[combination[0]]
        value1 = peg_val(card1)
        card2 = sorted5cards[combination[1]]
        value2 = peg_val(card2)
        card3 = sorted5cards[combination[2]]
        value3 = peg_val(card3)
        card4 = sorted5cards[combination[3]]
        value4=peg_val(card4)
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
        sum+=peg_val(card)
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
            if sorted5cards[start_index] == sorted5cards[next_index]:
                duplicates_count += 1
            elif sorted5cards[start_index] == sorted5cards[next_index] - 1:
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
            if sorted5cards[start_card_index] == sorted5cards[i]:
                points += 2
        start_card_index += 1
    return points
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
    # total_points += flush(hand4cards, cutcard, is_crib)

    sorted5cards= sort_cards(hand4cards,cutcard)

    total_points += two_card_fifteens(sorted5cards)
    total_points += three_card_fifteens(sorted5cards)
    total_points += four_card_fifteens(sorted5cards)
    total_points += five_card_fifteens(sorted5cards)
    total_points += runs(sorted5cards)
    total_points += pairs(sorted5cards)

    return total_points
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
        card_counts.append(cardsperrank)
    for card in pothand:
        value=card
        card_counts[value-1] -= 1
    if not isinstance(potdiscard, int):
        for card in potdiscard:
            value=card
            card_counts[value-1] -= 1
    else:
        value = potdiscard
        card_counts[value - 1] -= 1
    return card_counts

def expected_hand_value(pothand, potdiscard, mycrib):
    """
      Returns the expected point value of a hand (taking into account all possible cut cards)
      :param pothand: a list of 4 cards the player is keeping
      :param potdiscard: a list of the 2 cards the player is planning to discard
      :param risk: -1 for risk averse, 0 for risk neutral, 1 for risk loving
      :return: expected point value for the 4 card hand
      """

    card_counts=card_counts_list(pothand, potdiscard)
    base_expected_value,sqrt_expected_val, sqrd_expected_val = 0.0, 0.0,0.0

    #Order tuples by suit, then rank
    pothandsorted = sorted(pothand)
    potdiscardsorted = sorted(potdiscard)

    drawpicks = [4*numdecks]*13
    for rank in range(1, 14):
        if pothandsorted.count(rank) > 0:
            drawpicks[rank-1] -= pothandsorted.count(rank)
        if (drawpicks[rank-1] > 0 and
                potdiscardsorted.count(rank) ):
            drawpicks[rank-1] -= potdiscardsorted.count(rank)

    cribranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    rawcribdeals = it.combinations_with_replacement(ranks, 3 if players == 3 else 2)
    cribdeals = set()
    for cribdeal in rawcribdeals:
        criblegal = True
        for cribcardrank in deal:
            if cribdeal.count(cribcardrank) > drawpicks[cribcardrank-1]:
                criblegal = False
                break
        if criblegal:
            [cribdeals.add(i) for i in it.permutations(cribdeal)]
            # tempcrib = sorted(cribdeal)
            # tempcrib.append(drawpicks[tempcrib[0] - 1])
            # tempcrib.append(drawpicks[tempcrib[1] - 1])
            # cribdeals.add(tuple(tempcrib))
            # if tempcrib[0] != tempcrib[1]:
            #     #Also add reversed pick order
            #     tempcrib2 = sorted(cribdeal, reverse=True)
            #     tempcrib2.append(drawpicks[tempcrib2[0] - 1])
            #     tempcrib2.append(drawpicks[tempcrib2[1] - 1])
            #     cribdeals.add(tuple(tempcrib2))


    for rank in range (1,14):
        if drawpicks[rank-1] > 0:
            hand_value=score_hand(pothand, rank, False)
            crib_hand_value = 0.0
            probsum = 0.0
            for partcrib in cribdeals:
                partcrib_l = list(partcrib)
                if players == 2:
                    cribhandval = float(score_hand([potdiscardsorted[0],potdiscardsorted[1], partcrib_l[0], partcrib_l[1]], rank, False))
                else:
                    cribhandval = float(score_hand([potdiscardsorted[0], partcrib_l[0], partcrib_l[1], partcrib_l[2]], rank, False))
                curprob = 1.0
                picksleft = []
                for i in range(0,len(partcrib_l)):
                    picksleft.append((drawpicks[partcrib_l[i] - 1]))

                # if partcrib_l[0] == rank and partcrib_l[2] > 0:
                #     partcrib_l[2] -= 1
                # if partcrib_l[1] == rank and partcrib_l[3] > 0:
                #     partcrib_l[3] -= 1
                # if partcrib_l[1] == partcrib_l[0] and partcrib_l[3] > 0:
                #     partcrib_l[3] -= 1
                #
                # cribhandval *= partcrib_l[2]/(52*2-float(len(pothand)+len(potdiscard) + 1))
                # cribhandval *= partcrib_l[3]/(52*2-float(len(pothand)+len(potdiscard) + 1))
                # curprob *= partcrib_l[2]/(52*2-float(len(pothand)+len(potdiscard) + 1))
                # curprob *= partcrib_l[3]/(52*2-float(len(pothand)+len(potdiscard) + 1))
                rankcount,nilit  = 0,  False
                for i in range(0,len(picksleft)):
                    if partcrib_l[i] == rank:
                        if picksleft[i] - rankcount == 0:
                            nilit = True
                            break
                        else:
                            rankcount += 1
                            picksleft[i] -= rankcount
                if nilit:
                    cribhandval = 0
                    curprob = 0
                else:
                    for pick in picksleft:
                        cribhandval *= pick / (52 * numdecks - float(len(pothand) + len(potdiscard) + 1))
                        curprob *= pick / (52 * numdecks - float(len(pothand) + len(potdiscard) + 1))

                probsum += curprob
                crib_hand_value +=cribhandval
            if mycrib:
                #incorporating risk
                sqrt_val=math.sqrt(abs(hand_value))*numpy.sign(hand_value) + math.sqrt(abs(crib_hand_value))*numpy.sign(hand_value)
                sqrd_val=hand_value*hand_value*numpy.sign(hand_value) + crib_hand_value*crib_hand_value*numpy.sign(hand_value)
                hand_value += crib_hand_value
            else:
                #incorporating risk
                sqrt_val=math.sqrt(abs(hand_value))*numpy.sign(hand_value) - math.sqrt(abs(crib_hand_value))*numpy.sign(hand_value)
                sqrd_val=hand_value*hand_value*numpy.sign(hand_value) - crib_hand_value*crib_hand_value*numpy.sign(hand_value)
                hand_value -= crib_hand_value

                #gets the score of the hand for each possible cut card

            probability = float(drawpicks[rank-1])/(52*numdecks-float(len(pothand)+len(potdiscard)))               #calculates the probability of drawing that cut card
            base_expected_value += (hand_value*probability)               #calculates the probability of drawing that cut card
            sqrt_expected_val += (sqrt_val*probability)               #calculates the probability of drawing that cut card
            sqrd_expected_val += (sqrd_val*probability)

            #multiplies the calculated score by the probability of drawing that cut card, adds to total expected value

    return base_expected_value, sqrt_expected_val, sqrd_expected_val

def get_possible_4_hands(hand):
    possible_4 = []
    possible_discard = []
    for i in range(len(hand)):
        first_card = hand[i]
        if dealsize > 5:
            for j in range(i+1, len(hand)):
                second_card = hand[j]
                copyhand = deepcopy(hand)
                tempdiscard = [first_card, second_card]
                copyhand.remove(second_card)
                copyhand.remove(first_card)
                copyhand.sort()
                tempdiscard.sort()
                if not copyhand in possible_4:
                    possible_4.append(copyhand)
                    possible_discard.append(tempdiscard)
        else:
            copyhand = deepcopy(hand)
            copyhand.remove(first_card)
            copyhand.sort()
            if not copyhand in possible_4:
                possible_4.append(copyhand)
                possible_discard.append([first_card])

    return possible_4, possible_discard

#TODO: flush modulating table w/ approx rank based vals, if hand all same suit

sqliteConn = sql.connect('etc/Lookups.db')
sqliteCursor = sqliteConn.cursor()
ranks = [1,2,3,4,5,6,7,8,9,10,11,12,13]
rawdeals = it.combinations_with_replacement(ranks, dealsize)
deals = set()
for deal in rawdeals:
    legal = True
    for cardrank in deal:
        if deal.count(cardrank) > 4*numdecks:
            legal = False
            break
    if legal:
        deals.add(tuple(sorted(deal)))

for cr in range(0,2):
    dealcounter = 0
    sqliteCursor.execute('BEGIN TRANSACTION')
    for deal in deals:
        posshand, possdiscard = get_possible_4_hands(list(deal))
        for iterhand, iterdiscard in zip(posshand, possdiscard):
            temp1 = list(iterhand)
            temp2 = list(iterdiscard)
            base_expected_value, sqrt_val, sqrd_val = expected_hand_value(list(iterhand), list(iterdiscard), False if cr == 0 else True)
            iterhand.sort()
            iterdiscard.sort()
            handhash = int("{:02d}{:02d}{:02d}{:02d}".format(iterhand[0],iterhand[1],iterhand[2],iterhand[3]))
            if discardsize == 1:
                discardhash= int("{:02d}".format(iterdiscard[0]))
            else:
                discardhash= int("{:02d}{:02d}".format(iterdiscard[0],iterdiscard[1]))

            query_sb = StringIO()
            query_sb.write("INSERT INTO HandDiscards{}Player{}DeckRank VALUES({},{},{},"
                           .format("Three" if players == 3 else "Two", "Two" if numdecks == 2 else "One",
                                   handhash, discardhash, cr))
            for r in range(1,11):
                query_sb.write("{},".format((float(r)/10.0)*base_expected_value + (1.0-(float(r)/10.0))*sqrt_val))
            query_sb.write("{},".format(base_expected_value))
            for r in range(1,11):
                query_sb.write("{}".format((1.0-(float(r)/10.0))*base_expected_value + (float(r)/10.0)*sqrd_val))
                if r < 10: query_sb.write(",")
            query_sb.write(")")

            query = query_sb.getvalue()
            query_sb.close()
            sqliteCursor.execute(query)

        dealcounter +=1
        if dealcounter > 100:
            sqliteCursor.execute('END TRANSACTION')
            sqliteConn.commit()
            sqliteCursor.execute('BEGIN TRANSACTION')
            dealcounter = 0

    sqliteCursor.execute('END TRANSACTION')
    sqliteConn.commit()
