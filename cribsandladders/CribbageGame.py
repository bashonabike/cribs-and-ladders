from cribsandladders.Deck import Deck, card_to_string, peg_val
from cribsandladders.Stats import Move
from copy import deepcopy
import cribsandladders.ScoreHand as sh

from cribsandladders.Board import Board, Track, Chute, Ladder
import game_params as gp
from collections import Counter
import bisect
import random
import Enums as en
import itertools as it

PAIR_SCORES = {2: ("Pair", 2), 3: ("3 of a kind", 6), 4: ("Four of a kind", 12)}


def min_card(hand):
    """
    Finds the card in the hand with the lowest pegging value.

    Returns:
        Card: The card with the minimum peg_val.
    """
    c = hand[0]
    for i in range(1, len(hand)):
        if peg_val(hand[i]) < peg_val(c):
            c = hand[i]
    return c


def min_card_val(hand):
    """
    Gets the lowest pegging value available in a hand.

    Returns:
        int: The minimum pegging value.
    """
    return min([peg_val(c) for c in hand])


def can_peg(hand, total):
    """
    Checks if any card in the hand can be played without exceeding 31.

    Args:
        hand (list): Current cards in player's hand.
        total (int): Current running pegging total.

    Returns:
        bool: True if a play is possible, False otherwise.
    """
    if len(hand) == 0:
        return False

    return min_card_val(hand) + total <= 31


class CribbageGame:
    """
    Manages the lifecycle of a single Cribbage game session, including
    dealing, pegging, scoring hands, and board movement.
    """

    def __init__(self, board, squad, trial, threadNum=-1):
        """Initializes game state and players."""
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
        The main gameplay loop. Rotates dealers and runs rounds until a winner is found.

        Returns:
            list: A list of Move objects recorded during the game.
        """
        self.currentDealer = random.randint(1, gp.numplayers)
        while not self.run_round():
            # NOTE: we put the +1 ouside the mod since players start at 1
            self.currentDealer = (self.currentDealer) % gp.numplayers + 1

        if len(self.moves) > 0:
            self.moves[len(self.moves) - 1].winningMove = True
        return self.moves

    def run_round(self):
        """
        Executes a full round of Cribbage: Dealing, Discarding, Pegging, and Counting.

        Returns:
            bool: True if a player reached the winning position during the round.
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
            # +1 outside of mod since players start at 1 not 0
            curScorer = (self.currentDealer + i) % gp.numplayers + 1
            if self.score_hand(self.squad.getPlayerByNum(curScorer).hand, cut_card,
                               self.squad.getPlayerByNum(curScorer),
                               False):
                return True

        # score the crib
        if self.score_hand(crib, cut_card, self.squad.getPlayerByNum(self.currentDealer), True):
            return True

        self.print_scores()
        self.squad.resetCanPlay()

        return False

    def calc_metrics(self):
        """
        Calculates post-game metrics such as frequency of landing on chutes or ladders.
        """
        # TODO replace me
        # Determine repeated chutes & ladders
        events_by_player = ([group for group in zip(self.player, self.chutehit, self.ladderhit)
                             if group != ('A', (0, 0), (0, 0)) and group != ('B', (0, 0), (0, 0))])
        event_counts = Counter(events_by_player)
        self.eventRepeats = len(events_by_player) - len(event_counts)

    def print_scores(self):
        """Prints current player scores to console if verbose mode is enabled."""
        if self.verbose:
            for p in self.squad.players:
                print("Player 1 score: {} ".format(p.score))
            print("\n")

    def score_points(self, amount, reason, player, pegMove, soexcite=False):
        """
        Updates a player's score and checks for board events (chutes/ladders).

        Args:
            amount (int): Points earned.
            reason (str): Label for the points (e.g., "15-2").
            player (Player): The player receiving points.
            pegMove (bool): Whether points were scored during pegging.

        Returns:
            bool: True if the player has won the game.
        """
        if amount == 0:
            return False

        curTrack = self.board.getTrackByNum(player.tracknum)
        newPos, event = self.checkChuteOrLadderForPos(curTrack, amount, player.score)
        self.moveNum += 1
        # event: 0 is non, 1 is chute, 2 is ladder
        self.moves.append(Move(self.threadNum, self.trial, curTrack, self.moveNum, self.round, player.num, player.score,
                               amount, reason, event, newPos, soexcite, pegMove))
        player.score = newPos
        if player.score > curTrack.efflength:
            player.wins += 1
            return True

        return False

    def pegging(self):
        """
        Manages the pegging phase where players play cards sequentially to reach 31.

        Returns:
            bool: True if a player wins during this phase.
        """
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
                # NOTE: we put the +1 ouside the mod since players start at 1
                currentPlayerNum = (currentPlayerNum) % gp.numplayers + 1
                cannotPlayCounter += 1

                if cannotPlayCounter >= gp.numplayers:
                    # NOTE: we score 31 as 2, so no extra point
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

            # validate move
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
            # NOTE: we put the +1 ouside the mod since players start at 1
            currentPlayerNum = (currentPlayerNum) % gp.numplayers + 1

    def score_pegging(self, seq, total, player, soexcite):
        """
        Evaluates the current pegging sequence for runs, pairs, and totals (15/31).

        Args:
            seq (list): The sequence of cards played in the current "Go".
            total (int): Running total of card values.
            player (Player): Player who just played.

        Returns:
            bool: Result of score_points (True if player wins).
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
        """Placeholder for specific pegging round logic if needed."""
        pass

    def score_hand(self, hand4cards, cutcard, player, is_crib=False):
        """
        Calculates and applies score for a player's static hand or crib.
        """
        return self.score_points(sh.score_hand(hand4cards, cutcard, is_crib), "Their cards", player, False)

    def checkChuteOrLadderForPos(self, track, prospScore, currPos):
        """
        Checks if landing on a specific square triggers a chute (slide back)
        or a ladder (climb forward).

        Args:
            track (Track): The track the player is on.
            prospScore (int): The points they are about to move.
            currPos (int): Current position on board.

        Returns:
            tuple: (new_position, event_type)
        """
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
    """Raised when a player attempts an invalid Cribbage move."""
    pass