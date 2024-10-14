from cribsandladders.Deck import Deck, card_to_string, peg_val
from cribsandladders.Stats import Move
from copy import deepcopy
#from cribsandladders.ScoreHand as sh import score_hand
import cribsandladders.ScoreHand as sh

from cribsandladders.CribSquad import CribSquad
from cribsandladders.Board import Board, Track, Chute, Ladder
import game_params as gp
from collections import Counter
import bisect
import random
import Enums as en
import cribsandladders.Deck as dk
import time as tm
import itertools as it


PAIR_SCORES = {2: ("Pair", 2), 3: ("3 of a kind", 6), 4: ("Four of a kind", 12)}

def min_card(hand):
    """
    :returns a card with the minimum pegging value
    """
    c = hand[0]
    for i in range(1, len(hand)):
        if peg_val(hand[i]) < peg_val(c):
            c = hand[i]
    return c


def min_card_val(hand):
    """
    :returns the pegging value of the minimum card
    """
    return min([peg_val(c) for c in hand])


def can_peg(hand, total):
    """
    :returns true if a hand has a card that can be played
    :param hand: the hand to evaluate
    :param total: the total score on the table
    """
    if len(hand) == 0:
        return False

    return min_card_val(hand) + total <= 31


class CribbageGame:
    """
    A game of cribsandladders
    The game is run by calling play_game()
    The game can only be run once.
    To disable most of the printout, set game.verbose to false
    """
    def __init__(self, board, squad, trial, threadNum=-1):
        self.board = board
        self.squad = squad
        self.firstDeal = random.randint(1, gp.numplayers)
        self.currentDealer = self.firstDeal
        self.verbose = False
        self.threadNum = threadNum
        self.trial = trial
        self.moveNum = 0
        self.round = 0
        self.moves = []

    def play_game(self):
        """
        The main gameplay function
        To get the scores post game, check game.a_score and game.b_score
        """
        self.currentDealer = random.randint(1, gp.numplayers)
        while not self.run_round():
            #NOTE: we put the +1 ouside the mod since players start at 1
            self.currentDealer = (self.currentDealer) % gp.numplayers + 1

        if len(self.moves) > 0:
            self.moves[len(self.moves) - 1].winningMove = True
        return self.moves


    def run_round(self):
        """
        Runs a round of cribsandladders between player[] and player[]
        :param a_is_dealer: if a is the dealer and gets the crib
        :return: true if the game was won, false if otherwise
        """

        # setup
        self.round += 1
        deck = Deck()
        deck.shuffle()
        crib = []

        if gp.cribstartsize > 0:
            crib.extend(deck.drawCards(gp.cribstartsize))

        for player in self.squad.players:
            player.deal_hand(deck, gp.dealsize)
            crib.extend(player.discard_crib(player.num == self.currentDealer))

        # cut card
        cut_card = deck.drawCard()
        if self.verbose:
            print("The cut card is the", card_to_string(cut_card))
        if cut_card.rank == 11:  # if the a jack is turned
            self.score_points(2, "His heels", self.squad.getPlayerByNum(self.currentDealer), True)

        # run pegging
        if self.pegging():
            # if the game was won during pegging, return true
            return True

        self.print_scores()

        if self.verbose:
            print("The cut card is the", card_to_string(cut_card))

        # Score hands starting with person to the L of the dealer
        for i in range(gp.numplayers):
            #+1 outside of modsince players start at 1 not 0
            curScorer = (self.currentDealer + i) % gp.numplayers + 1
            if self.score_hand(self.squad.getPlayerByNum(curScorer).hand, cut_card, self.squad.getPlayerByNum(curScorer),
                            False):
                return True

        # score the crib
        if self.score_hand(crib, cut_card,self.squad.getPlayerByNum(self.currentDealer), True):
            return True

        self.print_scores()
        self.squad.resetCanPlay()

        return False


    def calc_metrics(self):
        #TODO replace me
        #Determine repeated chutes & ladders
        events_by_player = ([group for group in zip(self.player, self.chutehit, self.ladderhit)
                             if group != ('A', (0, 0), (0, 0)) and group != ('B', (0, 0), (0, 0))])
        event_counts = Counter(events_by_player)
        self.eventRepeats = len(events_by_player) - len(event_counts)



    def print_scores(self):
        if self.verbose:
            for p in self.squad.players:
                print("Player 1 score: {} ".format(p.score))

            print("\n")

    def score_points(self, amount, reason, player, pegMove, soexcite = False):
        """
        scores points for a player
        :param amount: the amount of points scored
        :param reason: the reason for the points as a string. Prints in the format of (amount) for (reason)
        :param is_a: if the player who scored points was a
        :param player: player getting scored
        """
        if amount == 0:
            return False

        curTrack = self.board.getTrackByNum(player.tracknum)
        newPos, event = self.checkChuteOrLadderForPos(curTrack, amount, player.score)
        self.moveNum += 1
        #event: 0 is non, 1 is chute, 2 is ladder
        self.moves.append(Move(self.threadNum, self.trial, curTrack, self.moveNum, self.round, player.num, player.score,
                               amount, reason, event, newPos, soexcite, pegMove))
        player.score = newPos
        if player.score > curTrack.efflength:
            player.wins += 1
            return True

        return False

    def pegging(self):
        """
        :param hand_a: the hand of player a. Must be a copy/mutable
        :param hand_b: the hand of player b. Must be a copy/mutable
        :param is_a: a starts the pegging
        :returns true if the game was won
        """
        #
        # next_player = hand_b if a_goes_first else hand_a
        total = 0
        seq = []
        currentPlayerNum = self.currentDealer
        lastPlayedPlayer = None
        attempts = 0

        while True:
            cannotPlayCounter = 0
            attempts += 1

            while True:
                attempts += 1
                if attempts > 1000:
                    raise Exception("Max attempts")
                player = self.squad.getPlayerByNum(currentPlayerNum)
                if player.canPlay and can_peg(player.pegginghand, total):
                    break

                player.canPlay = False
                #NOTE: we put the +1 ouside the mod since players start at 1
                currentPlayerNum = (currentPlayerNum) % gp.numplayers +  1
                cannotPlayCounter += 1

                if cannotPlayCounter >= gp.numplayers:
                    #NOTE: we score 31 as 2, so no extra point
                    if total != 31 and self.score_points(1, "Last card", lastPlayedPlayer, True):
                        return True
                    if self.squad.donePegging():
                        return False

                    total = 0
                    seq = []
                    self.squad.resetCanPlay()
                    cannotPlayCounter = 0

            # the current player can play
            curTrack = self.board.getTrackByNum(player.tracknum)
            nextPlayer = self.squad.getNextPeggingPlayer(player.num)
            nextPlayerEffHoles_l = []
            nextPlayerCardsInHand, nextPlayerCurrPos = -1, -1
            if nextPlayer is not None:
                nextPlayerEffHoles_l = self.board.getTrackByNum(nextPlayer.tracknum).effLandingForHoles
                nextPlayerCardsInHand = len(nextPlayer.pegginghand)
                nextPlayerCurrPos = nextPlayer.score
            (pickMuxed, soexcite) = player.pegging_move(deepcopy(seq), total, curTrack.effLandingForHoles,
                                                   nextPlayerEffHoles_l, nextPlayerCardsInHand, nextPlayerCurrPos)
            pick = None
            for card in player.pegginghand:
                if card.muxed == pickMuxed:
                    pick = card
                    break

            # a card should be played
            if pick is None:
                raise IllegalMoveException("Must play a card if able to. data:" + str(
                    (deepcopy(player.pegginghand), deepcopy(seq), total)) + "   player " + str(player.num))
            if pick not in player.pegginghand:
                raise IllegalMoveException("Must play a card from your hand")
            if peg_val(pick) + total > 31:
                raise IllegalMoveException("Cannot play a card resulting in a sum over 31")
            seq.append(pick)
            player.pegginghand.remove(pick)
            total += peg_val(pick)
            if self.score_pegging(seq, total, player, soexcite):
                return True

            lastPlayedPlayer = self.squad.getPlayerByNum(currentPlayerNum)
            #NOTE: we put the +1 ouside the mod since players start at 1
            currentPlayerNum = (currentPlayerNum)%gp.numplayers + 1

    def score_pegging(self, seq, total, player, soexcite):
        """
        Scores a single play in pegging
        :param seq:
        :param total:
        :param is_a:
        :return:
        """
        pegScore = 0
        card = seq[-1]

        # Reverse thru sequence, checking for score adders
        runBuild = [card.rank]
        runMin, runMax = card.rank, card.rank
        ofAKindBuild = [card]
        for c in range(1, len(seq)):
            idx = len(seq) - 1 - c
            if len(runBuild) > 0 and not seq[idx].rank in runBuild:
                if seq[idx].rank < runMin: runMin = seq[idx].rank
                if seq[idx].rank > runMax: runMax = seq[idx].rank
                runBuild.append(seq[idx].rank)
            else:
                runBuild = []

            if len(ofAKindBuild) > 0 and ofAKindBuild[0].rank == seq[idx].rank:
                ofAKindBuild.append(seq[idx])
            else:
                ofAKindBuild = []

        # If run build is seq, ordered or not, the diff of max and min should be the size minus 1
        # Eg: "7, 9, 8" min is 7, max is 9, diff is 2 which is len - 1
        if len(runBuild) > 0 and runMax - runMin == len(runBuild) - 1:
            pegScore += len(runBuild)
        if len(ofAKindBuild) > 1:
            pegScore += 2 * len([p for p in it.combinations(ofAKindBuild, 2)])

        # Get sum to 15
        if peg_val(card) + total == 15:
            pegScore += 2

        # Get sum to 31 (NOTE we lump final move point in here)
        if peg_val(card) + total == 31:
            pegScore += 2

        if pegScore > 0:
            return self.score_points(pegScore, "", player, True, soexcite)

        return False

    def pegging_round(self, hand_a, hand_b, a_goes_first):
        pass

    def score_hand(self, hand4cards, cutcard, player, is_crib = False):
        """
        scores points from a given hand
        :param hand4cards: the hand's cards
        :param cutcard: the cut card
        :return:
        """
        return self.score_points(sh.score_hand(hand4cards, cutcard, is_crib), "Their cards", player, False)

    def checkChuteOrLadderForPos(self, track, prospScore, currPos):
        if prospScore == 0:
            return currPos + prospScore, en.Event.NONE

        chute_index = bisect.bisect_left(track.eventsListChute, (currPos + prospScore))
        if chute_index != len(track.eventsListChute) and track.eventsListChute[chute_index] == currPos + prospScore:
            return (currPos + prospScore +
                    (track.chutes[chute_index].end - track.chutes[chute_index].start), en.Event.CHUTE)

        ladder_index = bisect.bisect_left(track.eventsListLadder, (currPos + prospScore))
        if ladder_index != len(track.eventsListLadder) and track.eventsListLadder[ladder_index] == currPos + prospScore:
            return (currPos + prospScore +
                    (track.ladders[ladder_index].end - track.ladders[ladder_index].start), en.Event.LADDER)

        return currPos + prospScore, en.Event.NONE



class IllegalMoveException(Exception):
    pass
