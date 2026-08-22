"""
Phase 5 (integration tests) -- full agent-vs-agent Cribbage playthroughs.

CribbageGame/CribSquad/Player wire together dealing, discarding, pegging,
hand scoring, and chute/ladder board movement. Phase 2 covered each piece
in isolation; nothing before this ran the whole loop end to end. That's
what this file does: build a real Board/CribSquad/CribbageGame and call
play_game() until someone wins, using a seeded RNG for the shuffle/
dealer-rotation randomness CribbageGame/CribSquad already take as an
injectable dependency (their `rng=` params).

Two pieces of "AI" still need standing in for, same as Phase 2's unit
tests and for the same reasons (see their docstrings there):

- Player.pegging_move() defaults to the compiled `scoretree` extension.
  That's Tier 4 in tdd-refactor-assessment.md -- "outside Python
  unit-test reach", meant to be covered by a small integration suite
  that requires the extension built, not by these tests.
  `move_selector` is Player's injectable seam for this;
  `_first_legal_card_selector` below is a deterministic pure-Python
  stand-in that always plays a legal card when one exists.

- Player.discard_crib() calls expected_hand_value(), which needs a huge
  precomputed `rankLookupTable` -- a pandas artifact built by the
  separate popRankLookupTable.py script, out of scope (see
  tests/test_player.py's module docstring for the same call).
  `_approximate_hand_value` below patches expected_hand_value with a
  real (not mocked-constant) call into ScoreHand.score_hand, so
  discard_crib() still picks between genuinely different real hands by a
  real scoring function -- just without the expected-value-over-all-
  possible-cut-cards precomputation.

Neither stand-in changes what's under test here: this file exercises the
*engine's* orchestration (dealing/discarding/pegging/hand-scoring/board-
movement/win-detection), not the quality of either AI's decisions.
"""
import random
import unittest.mock as mock

import pytest

import cribsandladders.ScoreHand as sh
from cribsandladders.Board import Board, Chute, Ladder, Track
from cribsandladders.CribbageGame import CribbageGame
from cribsandladders.CribSquad import CribSquad
from cribsandladders.config import GameConfig

pytestmark = pytest.mark.integration


def _first_legal_card_selector(handMuxed, nextPlayerCardsInHand, seqMuxed, effLandingForHoles,
                                nextPlayerEffLandingForHoles, current_sum, score, nextPlayerCurPos, numdecks):
    """Deterministic stand-in for the compiled scoretree extension:
    always plays the first card in hand (in `pegginghand` order) that
    doesn't bust 31. CribbageGame.can_peg() already guarantees at least
    one such card exists whenever pegging_move() gets called, so no
    fallback is needed.

    handMuxed entries are `100*suit + rank` (see Deck.Card.__init__), so
    `muxed % 100` recovers rank without needing a real Card object.
    """
    for muxed in handMuxed:
        rank = muxed % 100
        peg_val = min(rank, 10)
        if peg_val + current_sum <= 31:
            return muxed
    raise AssertionError("no legal card in hand -- can_peg() should have prevented this call")


def _approximate_hand_value(pothand, potdiscard, rankLookupTable, risk, hascrib, config):
    """Deterministic stand-in for expected_hand_value(): scores `pothand`
    treating the first discarded card as if it were the cut card. See
    module docstring."""
    discard_list = potdiscard if isinstance(potdiscard, list) else [potdiscard]
    return sh.score_hand(pothand, discard_list[0], is_crib=False)


def _make_track(num, efflength, chute=None, ladder=None):
    track = Track()
    track.num = num
    track.length = efflength
    track.efflength = efflength
    if chute is not None:
        start, end = chute
        track.setChutes([Chute(start, end, num)])
        track.setEventChutes([start])
    if ladder is not None:
        start, end = ladder
        track.setLadders([Ladder(start, end, num)])
        track.setEventLadders([start])
    return track


def _play_full_game(numplayers, seed, efflength=30, with_events=True):
    """Builds a board/squad/game and plays it to completion.

    Returns (board, squad, moves).
    """
    config = GameConfig(numplayers=numplayers, findmode=False)
    tracks = [
        _make_track(
            i + 1, efflength,
            chute=(12, 6) if with_events else None,
            ladder=(8, 16) if with_events else None,
        )
        for i in range(numplayers)
    ]

    board = Board(config=config)
    board.tracks = tracks

    rng = random.Random(seed)
    squad = CribSquad(
        rankLookupTable=None, tracks=tracks, config=config, rng=rng,
        move_selector=_first_legal_card_selector,
    )

    with mock.patch("cribsandladders.Player.expected_hand_value", side_effect=_approximate_hand_value):
        game = CribbageGame(board, squad, trial=1, config=config, rng=rng)
        moves = game.play_game()

    return board, squad, moves


def _move_fingerprint(moves):
    """A comparable, order-preserving summary of a game's moves -- used
    to check that two runs with the same seed produced not just the same
    final score, but the same game. (Move's real attribute names -- see
    cribsandladders/Stats.py -- are `player`/`currpos`/`pegMove`/
    `winningMove`; there's no raw `.event`, just the derived `.hasEvent`
    bool plus `.chuteamt`/`.ladderamt`.)"""
    return [
        (m.player, m.currpos, m.hasEvent, m.chuteamt, m.ladderamt, m.pegMove, m.winningMove)
        for m in moves
    ]


class TestThreePlayerGame:
    def test_reaches_a_single_deterministic_winner(self):
        board, squad, moves = _play_full_game(numplayers=3, seed=2024)

        assert len(moves) > 0
        assert moves[-1].winningMove is True

        winners = [p for p in squad.players if p.wins > 0]
        assert len(winners) == 1, "exactly one player should have won"
        winner = winners[0]
        assert winner.wins == 1

        winner_track = board.getTrackByNum(winner.tracknum)
        assert winner.score > winner_track.efflength

        # No player's final recorded position in the move log should
        # exceed a full lap-and-a-bit past the finish -- catches runaway
        # scoring bugs (e.g. double-applying an event) that would still
        # technically produce "a winner".
        for m in moves:
            assert m.currpos <= winner_track.efflength + 40

    def test_every_move_is_attributed_to_a_real_player_on_a_real_track(self):
        _, squad, moves = _play_full_game(numplayers=3, seed=99)
        player_nums = {p.num for p in squad.players}
        for m in moves:
            assert m.player in player_nums


class TestTwoPlayerGame:
    def test_reaches_a_single_deterministic_winner(self):
        board, squad, moves = _play_full_game(numplayers=2, seed=7)

        assert len(moves) > 0
        assert moves[-1].winningMove is True

        winners = [p for p in squad.players if p.wins > 0]
        assert len(winners) == 1
        winner_track = board.getTrackByNum(winners[0].tracknum)
        assert winners[0].score > winner_track.efflength


class TestDeterminism:
    def test_same_seed_and_strategies_produce_identical_games(self):
        _, squad_a, moves_a = _play_full_game(numplayers=3, seed=555)
        _, squad_b, moves_b = _play_full_game(numplayers=3, seed=555)

        assert _move_fingerprint(moves_a) == _move_fingerprint(moves_b)
        assert [p.score for p in squad_a.players] == [p.score for p in squad_b.players]
        assert [p.wins for p in squad_a.players] == [p.wins for p in squad_b.players]

    def test_different_seeds_can_produce_different_games(self):
        # Not a strict requirement of any single seed, but if every seed
        # we try collapses to the exact same move sequence, that's a
        # strong signal the rng injection isn't actually wired through
        # play_game()/run_round()'s Deck.shuffle() calls.
        fingerprints = {
            tuple(_move_fingerprint(_play_full_game(numplayers=3, seed=s)[2]))
            for s in (1, 2, 3, 4, 5)
        }
        assert len(fingerprints) > 1


class TestGameWithoutBoardEvents:
    def test_still_terminates_and_produces_a_winner(self):
        # Sanity check that a board with no chutes/ladders at all (pure
        # racing) doesn't break score_points()/checkChuteOrLadderForPos()
        # -- eventsListChute/eventsListLadder empty rather than populated.
        board, squad, moves = _play_full_game(numplayers=3, seed=42, with_events=False)
        assert moves[-1].winningMove is True
        assert sum(p.wins for p in squad.players) == 1
