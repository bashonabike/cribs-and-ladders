"""
Phase 1 tests for cribsandladders.ScoreHand: confirm the module imports
with zero heavy dependencies (no game_params, no scipy) now that it
takes an explicit `config` parameter instead of reading module globals,
and that injecting a different GameConfig actually changes behavior.

Full behavioral coverage of scoring rules (runs, pairs, flushes, 15s,
right jack, ...) is Phase 2 work, done as part of TDD-refactoring the
engine. The one behavioral check here (test_max_hand_scores_29) is a
sanity check that the refactor so far hasn't broken anything, not a
substitute for that coverage.
"""
import sys

from cribsandladders.config import GameConfig
from cribsandladders.Deck import Card
import cribsandladders.ScoreHand as sh


def test_importing_score_hand_does_not_pull_in_game_params_or_scipy():
    assert "game_params" not in sys.modules
    assert not any(name.startswith("scipy") for name in sys.modules)


def test_max_hand_scores_29():
    # Four 5s plus the jack of the cut card's suit ("his nibs") is the
    # canonical maximum-value cribbage hand: 29 points.
    hand = [Card(5, 0), Card(5, 1), Card(11, 2), Card(5, 3)]
    cut = Card(5, 2)
    assert sh.score_hand(hand, cut) == 29


def test_card_counts_list_respects_injected_config():
    hand = [Card(1, 0), Card(2, 0), Card(3, 0)]

    one_deck_counts = sh.card_counts_list(hand, [], config=GameConfig(twodecks=False))
    two_deck_counts = sh.card_counts_list(hand, [], config=GameConfig(twodecks=True))

    # rank 1 was used once in the kept hand; one-deck config starts at 4
    # cards per rank, two-deck config starts at 8.
    assert one_deck_counts[0] == 4 - 1
    assert two_deck_counts[0] == 8 - 1


def test_card_counts_list_default_config_matches_default_game_params():
    from cribsandladders.config import DEFAULT_CONFIG

    hand = [Card(1, 0)]
    counts = sh.card_counts_list(hand, [])
    assert counts[0] == DEFAULT_CONFIG.cardsperrank - 1


def test_flush_adder_uses_injected_flushmods():
    hand = [Card(2, 0), Card(4, 0), Card(6, 0), Card(8, 0)]  # all clubs
    discard = [Card(3, 1), Card(5, 1)]  # different suit

    cfg = GameConfig()
    expected = cfg.flushmods[0][10]  # risk=11 -> index 10
    assert sh.flush_adder(hand, discard, risk=11, config=cfg) == expected
