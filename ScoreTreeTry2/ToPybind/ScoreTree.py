import Node as nd
import ChanceNode as chc
import  DeckWithoutSpecCards as dk
import copy as cp
import Card as cd
import itertools as it
import numpy as np

class scoreTree:


    # Initializes an expectimax tree of a specified depth
    def __init__(self, hand, nextPlayerNumCardsLeftInHand, sequence, effLandingForHoles, effLandingForNextPlayerHoles,
                 current_sum, current_pos, nextPlayerCurrPos, numdecks):
        # Initialize tree
        # print("hand:", hand)
        # print("sequence:", sequence)
        # print("current sum:", current_sum)
        self.root = nd.Node(None, 0, False)
        self.effLandingForHoles = effLandingForHoles
        self.effLandingForNextPlayerHoles = effLandingForNextPlayerHoles
        self.current_pos = current_pos
        self.nextPlayerCurrPos = nextPlayerCurrPos
        self.current_sum = current_sum
        self.soexcite_peg = False
        self.numdecks = numdecks
        self.nextPlayerNumCardsLeftInHand = nextPlayerNumCardsLeftInHand

        #TODO: add in predictive support, eg: if total goes to 21 what are odds opponent will get to 31, etc

        # Consider all cards in hand that are legal moves
        legal_moves = []
        some_events, some_no_events, is_event = False, False, False
        for card in hand:
            if self.peg_val(card) + current_sum <= 31:
                legal_moves.append(card)
                score, hasEvent = self.scoreCalc(hand, sequence, current_sum, card)
                newNode = nd.Node(card, score, hasEvent)
                if newNode.hasEvent:
                    some_events = True
                else:
                    some_no_events = True
                newNode.sumFromPlay = current_sum + card.rank
                self.root.addChild(newNode)
        if some_events and some_no_events:
            self.soexcite_peg = True

        # Get remaining eligible cards
        deckWithoutCards = dk.DeckWithoutSpecCards(numdecks, hand + sequence).getDeckAsList()

        # For each level 2 node, put in prob nodes with legal moves
        for cnode in self.root.children:
            # Get rid of illegal or used cards
            card_deck = cp.deepcopy(deckWithoutCards)
            if cnode.getCard() in card_deck:
                card_deck.remove(cnode.getCard())
            for card in card_deck:
                if cnode.getSumFromPlay() + card.rank > 31 and card in card_deck:
                    card_deck.remove(card)

            # Make new nodes in children of current node using prob nodes
            for card in card_deck:
                newChanceNode = chc.ChanceNode(card, self.nextPlayerNumCardsLeftInHand / len(card_deck))
                cnode.addChild(newChanceNode)

        # Go through all prob nodes and add regular nodes for opponent
        for n in self.root.getChildren():
            scoreDistribution = []
            for m in n.getChildren():
                testHand = cp.deepcopy(hand)
                testHand.remove(n.getCard())
                testSequence = cp.deepcopy(sequence)
                testSequence.append(n.getCard())
                score, hasEvent = self.scoreCalc(testHand, testSequence, current_sum + n.getSumFromPlay(), m.getCard(), True)
                scoreDistribution.append(score)
                newNode = nd.Node(m.getCard(), score, hasEvent)
                m.addChild(newNode)
            #pad distribution with 0's for all illegal plays
            scoreDistribution.extend([0]*(len(deckWithoutCards) - len(n.getChildren())))
            scoreDistribution.sort()
            #Opponent will play the best card they have, their ability to choose deps on cards left
            #Determine percentile based on this
            n.likelyOpponentScoreLine = np.percentile(scoreDistribution,
                                                      int(100*((nextPlayerNumCardsLeftInHand - 1)/nextPlayerNumCardsLeftInHand)))

        #Repeat process, determine likely opponent score, set utility only if above requisite score line
        for n in self.root.getChildren():
            curOpponentLikelyScore = 0
            for m in n.getChildren():
                #Assume only one child
                child = m.getChildren()[0]
                if child.utility >= n.likelyOpponentScoreLine:
                    m.utility = child.utility
                    curOpponentLikelyScore += m.utility*m.probability
            n.likelyOpponentScore = curOpponentLikelyScore



    # Searches the expectimax tree and recommends the card to be played
    # 0 for  no risk, 1 for medium risk, 2 for high risk
    def recommendCard(self, risk):
        if len(self.root.getChildren()) == 0:
            return None

        curMax = -1000
        bestNode = None

        for cnode in self.root.getChildren():
            if cnode.utility - cnode.likelyOpponentScore > curMax:
                curMax =cnode.utility - cnode.likelyOpponentScore
                bestNode = cnode

        return bestNode.getCard()




    # Tells if a move is illegal
    def isIllegal(self, hand, sequence, current_sum, card):
        pass

    # Returns updated parameters for playing a card
    def makeMoveUpdate(self, hand, sequence, current_sum, card):
        pass

    def peg_val(self, card):
        return 10 if card.rank > 10 else card.rank

    # Get a score for a move
    def scoreCalc(self, hand, sequence, current_sum, card, nextPlayer=False):
        baseScore = 0

        #Reverse thru sequence, checking for score adders
        curRunBuild = [card.rank]
        validRunBuilds, validOfAKindBuilds = [], []
        runMin, runMax = card.rank, card.rank
        ofAKindBuild = [card]
        runClosed, kindClosed = False, False

        for c in range(0, len(sequence)):
            idx = len(sequence) - 1 - c

            if len(curRunBuild) > 0 and not sequence[idx].rank in curRunBuild:
                if sequence[idx].rank < runMin: runMin = sequence[idx].rank
                if sequence[idx].rank > runMax: runMax = sequence[idx].rank
                curRunBuild.append(sequence[idx].rank)
                if len(curRunBuild) > 1 and runMax - runMin == len(curRunBuild) - 1:
                    validRunBuilds.append(len(curRunBuild))
            else:
                curRunBuild = []

            if len(ofAKindBuild) > 0 and ofAKindBuild[0].rank == sequence[idx].rank:
                ofAKindBuild.append(sequence[idx])
                if len(ofAKindBuild) > 1: validOfAKindBuilds.append(len(ofAKindBuild))
            else:
                ofAKindBuild = []

        #If run build is sequence, ordered or not, the diff of max and min should be the size minus 1
        #Eg: "7, 9, 8" min is 7, max is 9, diff is 2 which is len - 1
        if len(validRunBuilds) > 0:
            baseScore += max(validRunBuilds)
        if len(validOfAKindBuilds) > 0:
            baseScore += max(validOfAKindBuilds)*(max(validOfAKindBuilds)-1)

        # Get sum to 15 or 31
        if self.peg_val(card) + current_sum in (15, 31):
            baseScore += 2

        if baseScore == 0: return 0, False

        #Check for event
        if not nextPlayer:
            baseHole = self.current_pos + baseScore
            landHole = self.effLandingForHoles[baseHole-1]
            pos = self.current_pos
        else:
            baseHole = self.nextPlayerCurrPos + baseScore
            landHole = self.effLandingForNextPlayerHoles[baseHole-1]
            pos = self.current_pos

        #If eff landing diff from base score, event
        if landHole != baseHole:
            return landHole - pos, True
        else:
            return baseHole - pos, False

