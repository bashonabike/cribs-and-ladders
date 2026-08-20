"""
Characterization + config-injection tests for cribsandladders.ScoreHand.

The expected values below for score_hand()/sort_cards()/right_jack()/
flush()/the fifteens helpers/runs()/pairs()/crib_cards_value() were
verified two ways: (1) run against the real implementation in this repo
and (2) cross-checked by hand against standard cribbage scoring rules
(see inline comments on the trickier fixtures -- the "double run"
hand in particular, which is easy to get wrong by hand). Where a
comment says "verified against official rules", that's a stronger claim
than "this is what the code currently returns" -- it means both agree.

This is Phase 2 (TDD the engine) work: with these in place, ScoreHand
can be refactored with a real safety net instead of by inspection.
"""
import sys

from cribsandladders.config import GameConfig, DEFAULT_CONFIG
from cribsandladders.Deck import Card, SPADES, HEARTS, CLUBS, DIAMONDS
import cribsandladders.ScoreHand as sh


# ---------------------------------------------------------------------
# import hygiene (Phase 1 goal: ScoreHand must not need game_params/scipy)
# ---------------------------------------------------------------------

def test_importing_score_hand_does_not_pull_in_game_params_or_scipy():
    assert "game_params" not in sys.modules
    assert not any(name.startswith("scipy") for name in sys.modules)


# ---------------------------------------------------------------------
# right_jack
# ---------------------------------------------------------------------

def test_right_jack_scores_when_suit_matches_cut():
    hand = [Card(10, SPADES), Card(11, DIAMONDS), Card(1, HEARTS), Card(3, SPADES)]
    cut = Card(5, DIAMONDS)
    assert sh.right_jack(hand, cut) == 1


def test_right_jack_zero_when_suit_does_not_match_cut():
    hand = [Card(10, SPADES), Card(11, DIAMONDS), Card(1, HEARTS), Card(3, SPADES)]
    cut = Card(5, HEARTS)
    assert sh.right_jack(hand, cut) == 0


def test_right_jack_zero_when_no_jack_in_hand():
    hand = [Card(10, SPADES), Card(4, DIAMONDS), Card(1, HEARTS), Card(3, SPADES)]
    cut = Card(5, DIAMONDS)
    assert sh.right_jack(hand, cut) == 0


# ---------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------

def test_flush_four_same_suit_cut_different():
    hand = [Card(2, SPADES), Card(4, SPADES), Card(6, SPADES), Card(8, SPADES)]
    cut = Card(9, HEARTS)
    assert sh.flush(hand, cut, is_crib=False) == 4


def test_flush_five_same_suit_including_cut():
    hand = [Card(2, SPADES), Card(4, SPADES), Card(6, SPADES), Card(8, SPADES)]
    cut = Card(9, SPADES)
    assert sh.flush(hand, cut, is_crib=False) == 5


def test_flush_zero_when_hand_not_all_same_suit():
    hand = [Card(2, SPADES), Card(4, SPADES), Card(6, HEARTS), Card(8, SPADES)]
    cut = Card(9, SPADES)
    assert sh.flush(hand, cut, is_crib=False) == 0


def test_crib_flush_of_four_is_voided():
    # A crib can only score a flush if all 5 cards (including cut) match;
    # a 4-card flush that doesn't extend to the cut scores 0 in the crib,
    # even though the same hand would score 4 as a normal hand.
    hand = [Card(2, SPADES), Card(4, SPADES), Card(6, SPADES), Card(8, SPADES)]
    cut = Card(9, HEARTS)
    assert sh.flush(hand, cut, is_crib=True) == 0


def test_crib_flush_of_five_still_counts():
    hand = [Card(2, SPADES), Card(4, SPADES), Card(6, SPADES), Card(8, SPADES)]
    cut = Card(9, SPADES)
    assert sh.flush(hand, cut, is_crib=True) == 5


# ---------------------------------------------------------------------
# fifteens (via sort_cards -> the four *_card_fifteens helpers)
# ---------------------------------------------------------------------

def test_two_card_fifteens_counts_each_qualifying_pair():
    # values [2,4,6,8,9]: only 6+9=15
    hand = [Card(2, SPADES), Card(4, SPADES), Card(6, SPADES), Card(8, SPADES)]
    cut = Card(9, HEARTS)
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.two_card_fifteens(sorted5) == 2


def test_three_card_fifteens_counts_each_qualifying_triple():
    # values [2,4,6,8,9]: only 2+4+9=15
    hand = [Card(2, SPADES), Card(4, SPADES), Card(6, SPADES), Card(8, SPADES)]
    cut = Card(9, HEARTS)
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.three_card_fifteens(sorted5) == 2


def test_fifteens_use_peg_value_not_rank_for_face_cards():
    # J/Q/K all peg as 10. 5 + 10(J) = 15.
    hand = [Card(5, SPADES), Card(2, HEARTS), Card(3, CLUBS), Card(4, DIAMONDS)]
    cut = Card(11, SPADES)  # Jack, peg_val 10
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.two_card_fifteens(sorted5) == 2


# ---------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------

def test_simple_run_of_three():
    hand = [Card(7, SPADES), Card(8, HEARTS), Card(9, CLUBS), Card(2, DIAMONDS)]
    cut = Card(13, SPADES)  # King, unrelated
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.runs(sorted5) == 3


def test_no_run_when_not_consecutive():
    hand = [Card(2, SPADES), Card(4, HEARTS), Card(7, CLUBS), Card(9, DIAMONDS)]
    cut = Card(13, SPADES)
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.runs(sorted5) == 0


def test_double_double_run_scores_four_distinct_paths():
    """
    Hand [4,5,5,6,6] (a "double-double run"): with two 5s and two 6s,
    there are 2x2=4 distinct ways to pick a 4-5-6 run, each independently
    counted -- this is a genuinely easy-to-get-wrong cribbage rule.
    Verified by hand against the official rule (4 runs x 3 cards = 12),
    matching what the code returns.
    """
    hand = [Card(4, SPADES), Card(5, SPADES), Card(6, SPADES), Card(6, CLUBS)]
    cut = Card(5, CLUBS)
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.runs(sorted5) == 12


# ---------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------

def test_pairs_counts_each_two_card_combination():
    # four of a kind = C(4,2) = 6 combinations x 2 points = 12
    hand = [Card(5, SPADES), Card(5, HEARTS), Card(5, CLUBS), Card(5, DIAMONDS)]
    cut = Card(2, SPADES)
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.pairs(sorted5) == 12


def test_pairs_zero_when_no_matching_ranks():
    hand = [Card(2, SPADES), Card(4, HEARTS), Card(7, CLUBS), Card(9, DIAMONDS)]
    cut = Card(13, SPADES)
    sorted5 = sh.sort_cards(hand, cut)
    assert sh.pairs(sorted5) == 0


# ---------------------------------------------------------------------
# score_hand (full aggregate)
# ---------------------------------------------------------------------

def test_max_hand_scores_29():
    # Four 5s plus the jack of the cut card's suit ("his nibs") is the
    # canonical maximum-value cribbage hand: 29 points.
    hand = [Card(5, 0), Card(5, 1), Card(11, 2), Card(5, 3)]
    cut = Card(5, 2)
    assert sh.score_hand(hand, cut) == 29


def test_double_double_run_total_score():
    # runs(12) + pairs(4, two pairs) + three_card_fifteens(8, four
    # 4-5-6 combinations at peg value 15) = 24. Cross-checked against
    # official cribbage scoring for this exact hand shape.
    hand = [Card(4, SPADES), Card(5, SPADES), Card(6, SPADES), Card(6, CLUBS)]
    cut = Card(5, CLUBS)
    assert sh.score_hand(hand, cut) == 24


def test_nothing_hand_scores_zero_or_only_incidental_fifteens():
    hand = [Card(2, SPADES), Card(4, HEARTS), Card(7, CLUBS), Card(9, DIAMONDS)]
    cut = Card(13, SPADES)
    # 4+9=13, 2+13(peg 10)=12, 7+... no pair sums to 15 individually,
    # but 2+4+9=15 (three-card) -> 2 points, nothing else.
    assert sh.score_hand(hand, cut) == 2


def test_score_hand_is_zero_for_a_true_bust_hand():
    hand = [Card(2, SPADES), Card(4, HEARTS), Card(7, CLUBS), Card(10, DIAMONDS)]
    cut = Card(13, SPADES)
    assert sh.score_hand(hand, cut) == 0


# ---------------------------------------------------------------------
# card_counts_list / flush_adder / expected_hand_value (config injection)
# ---------------------------------------------------------------------

def test_card_counts_list_respects_injected_config():
    hand = [Card(1, 0), Card(2, 0), Card(3, 0)]

    one_deck_counts = sh.card_counts_list(hand, [], config=GameConfig(twodecks=False))
    two_deck_counts = sh.card_counts_list(hand, [], config=GameConfig(twodecks=True))

    # rank 1 was used once in the kept hand; one-deck config starts at 4
    # cards per rank, two-deck config starts at 8.
    assert one_deck_counts[0] == 4 - 1
    assert two_deck_counts[0] == 8 - 1


def test_card_counts_list_default_config_matches_default_game_params():
    hand = [Card(1, 0)]
    counts = sh.card_counts_list(hand, [])
    assert counts[0] == DEFAULT_CONFIG.cardsperrank - 1


def test_flush_adder_uses_injected_flushmods():
    hand = [Card(2, 0), Card(4, 0), Card(6, 0), Card(8, 0)]  # all clubs
    discard = [Card(3, 1), Card(5, 1)]  # different suit

    cfg = GameConfig()
    expected = cfg.flushmods[0][10]  # risk=11 -> index 10
    assert sh.flush_adder(hand, discard, risk=11, config=cfg) == expected


# ---------------------------------------------------------------------
# crib_cards_value
# ---------------------------------------------------------------------

def test_crib_cards_value_pair_scores_two_plus_five_bonus():
    # pair of 5s: +2 for the pair, +1 +1 for each card being a 5 = 4
    assert sh.crib_cards_value([Card(5, SPADES), Card(5, HEARTS)], yourCrib=True) == 4


def test_crib_cards_value_negated_for_opponents_crib():
    assert sh.crib_cards_value([Card(5, SPADES), Card(5, HEARTS)], yourCrib=False) == -4


def test_crib_cards_value_sum_to_fifteen():
    assert sh.crib_cards_value([Card(9, SPADES), Card(6, HEARTS)], yourCrib=True) == 2


def test_crib_cards_value_adjacent_ranks():
    assert sh.crib_cards_value([Card(3, SPADES), Card(4, HEARTS)], yourCrib=True) == 1


def test_crib_cards_value_single_five_discard():
    assert sh.crib_cards_value([Card(5, SPADES)], yourCrib=True) == 1


def test_crib_cards_value_unrelated_cards_score_zero():
    assert sh.crib_cards_value([Card(2, SPADES), Card(9, HEARTS)], yourCrib=True) == 0
