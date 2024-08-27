import copy as cp
import itertools as it
import numpy as np

import random

def getCardToPlay(hand_i, nextPlayerNumCardsLeftInHand_i, sequence_i, effLandingForHoles_i, effLandingForNextPlayerHoles_i,
             current_sum_i, current_pos_i, nextPlayerCurrPos_i, numdecks_i):
    effLandingForHoles = effLandingForHoles_i
    effLandingForNextPlayerHoles = effLandingForNextPlayerHoles_i
    current_pos = current_pos_i
    nextPlayerCurrPos = nextPlayerCurrPos_i
    current_sum = current_sum_i
    soexcite_peg = False
    numdecks = numdecks_i
    nextPlayerNumCardsLeftInHand = nextPlayerNumCardsLeftInHand_i
    hand = hand_i
    sequence = sequence_i

    legal_moves = []
    some_events, some_no_events = False, False

    # Level 1: Consider all cards in hand that are legal moves
    level_1 = []
    for card in hand:
        if peg_val(card) + current_sum <= 31:
            legal_moves.append(card)
            score, hasEvent = scoreCalc(hand, sequence, current_sum, card,
                                        current_pos, nextPlayerCurrPos, effLandingForHoles, effLandingForNextPlayerHoles)
            newNode = {
                'card': card,
                'utility': score,
                'hasEvent': hasEvent,
                'sumFromPlay': current_sum + card['rank'],
                'children': []
            }
            level_1.append(newNode)
            if hasEvent:
                some_events = True
            else:
                some_no_events = True

    if some_events and some_no_events:
        soexcite_peg = True

    # Get remaining eligible cards
    deckWithoutCards = buildDeckWithoutSpecCards(numdecks, hand + sequence)

    # Level 2: For each level 1 move, generate probability nodes with legal moves
    for n in level_1:
        card_deck = cp.deepcopy(deckWithoutCards)
        if n['card'] in card_deck:
            card_deck.remove(n['card'])
        for card in card_deck:
            if n['sumFromPlay'] + card['rank'] > 31 and card in card_deck:
                card_deck.remove(card)

        level_2 = []
        for card in card_deck:
            newChanceNode = {
                'card': card,
                'probability': nextPlayerNumCardsLeftInHand / len(card_deck),
                'children': []
            }
            level_2.append(newChanceNode)
        n['children'] = level_2

    # Level 3: For each probability node, add opponent's possible moves
    for n in level_1:
        scoreDistribution = []
        for m in n['children']:
            testHand = cp.deepcopy(hand)
            testHand.remove(n['card'])
            testSequence = cp.deepcopy(sequence)
            testSequence.append(n['card'])
            score, hasEvent = scoreCalc(testHand, testSequence, current_sum + n['sumFromPlay'], m['card'],
                                         current_pos, nextPlayerCurrPos, effLandingForHoles,
                                        effLandingForNextPlayerHoles,

                                        True)
            scoreDistribution.append(score)

            newNode = {
                'card': m['card'],
                'utility': score,
                'hasEvent': hasEvent
            }
            m['children'].append(newNode)

        # Pad distribution with zeros for illegal plays
        scoreDistribution.extend([0] * (len(deckWithoutCards) - len(n['children'])))
        scoreDistribution.sort()
        n['likelyOpponentScoreLine'] = np.percentile(scoreDistribution,
                                                     int(100 * ((nextPlayerNumCardsLeftInHand - 1) / nextPlayerNumCardsLeftInHand)))

    # Determine likely opponent score and set utility
    for n in level_1:
        curOpponentLikelyScore = 0
        for m in n['children']:
            child = m['children'][0]
            if child['utility'] >= n['likelyOpponentScoreLine']:
                m['utility'] = child['utility']
                curOpponentLikelyScore += m['utility'] * m['probability']
        n['likelyOpponentScore'] = curOpponentLikelyScore
        n['netPointsForPlay'] = n['utility'] - n['likelyOpponentScore']

    return recommendCard(level_1)

def recommendCard(level_1):
    if len(level_1) == 0:
        return None

    curMax = -1000
    bestCard = None

    for n in level_1:
        if n['netPointsForPlay'] > curMax:
            curMax = n['netPointsForPlay']
            bestCard = n['card']

    return bestCard

def peg_val(card):
    return 10 if card['rank'] > 10 else card['rank']

def scoreCalc(hand, sequence, current_sum, card, current_pos, nextPlayerCurrPos, effLandingForHoles, effLandingForNextPlayerHoles, nextPlayer=False):
    baseScore = 0

    curRunBuild = [card['rank']]
    ofAKindBuild = [card]
    validRunBuilds, validOfAKindBuilds = [], []
    runMin, runMax = card['rank'], card['rank']

    for idx in range(len(sequence) - 1, -1, -1):
        if len(curRunBuild) > 0 and sequence[idx]['rank'] not in curRunBuild:
            runMin = min(runMin, sequence[idx]['rank'])
            runMax = max(runMax, sequence[idx]['rank'])
            curRunBuild.append(sequence[idx]['rank'])
            if len(curRunBuild) > 1 and runMax - runMin == len(curRunBuild) - 1:
                validRunBuilds.append(len(curRunBuild))
        else:
            curRunBuild = []

        if len(ofAKindBuild) > 0 and ofAKindBuild[0]['rank'] == sequence[idx]['rank']:
            ofAKindBuild.append(sequence[idx])
            if len(ofAKindBuild) > 1:
                validOfAKindBuilds.append(len(ofAKindBuild))
        else:
            ofAKindBuild = []

    if len(validRunBuilds) > 0:
        baseScore += max(validRunBuilds)
    if len(validOfAKindBuilds) > 0:
        baseScore += max(validOfAKindBuilds) * (max(validOfAKindBuilds) - 1)

    if peg_val(card) + current_sum in (15, 31):
        baseScore += 2

    if baseScore == 0:
        return 0, False

    baseHole = current_pos + baseScore if not nextPlayer else nextPlayerCurrPos + baseScore
    landHole = effLandingForHoles[baseHole - 1] if not nextPlayer else effLandingForNextPlayerHoles[baseHole - 1]

    return (landHole - (current_pos if not nextPlayer else nextPlayerCurrPos)), landHole != baseHole

def buildDeckWithoutSpecCards(numdecks, cardsOmit):
    """
    Builds a deck with the specified number of decks, omitting certain cards.
    :param numdecks: The number of decks to include.
    :param cardsOmit: A list of cards (as dictionaries) to omit from the deck.
    """
    cards = []
    cardsOmit.sort(key=lambda c: (c['suit'], c['rank']))
    cardsOmit_idx = 0

    for suit in range(4):
        for rank in range(1, 14):
            for _ in range(numdecks):
                tentCard = {'rank': rank, 'suit': suit}
                if cardsOmit_idx < len(cardsOmit) and cardsOmit[cardsOmit_idx] == tentCard:
                    cardsOmit_idx += 1
                else:
                    cards.append(tentCard)

    return cards
