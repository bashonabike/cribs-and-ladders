// main.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

using namespace std;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

struct Card {
    int rank;
    int suit;

    bool operator==(const Card& other) const {
        return rank == other.rank && suit == other.suit;
    }
};

struct Node {
    Card card;
    int utility;
    bool hasEvent;
    int sumFromPlay;
    std::vector<Node> children;
    int cardsinhand;
	int cardsindeck;
    int likelyOpponentScoreLine;
    double likelyOpponentScore;
    double netPointsForPlay;
};

int peg_val(const Card& card) {
    return card.rank > 10 ? 10 : card.rank;
}

std::pair<int, bool> scoreCalc(const std::vector<Card>& sequence, int current_sum,
    const Card& card, int current_pos, int nextPlayerCurrPos, const std::vector<int>& effLandingForHoles,
    const std::vector<int>& effLandingForNextPlayerHoles, bool nextPlayer = false) {
    int baseScore = 0;

    std::vector<int> curRunBuild = { card.rank };
    std::vector<Card> ofAKindBuild = { card };
    std::vector<int> validRunBuilds, validOfAKindBuilds;
    int runMin = card.rank, runMax = card.rank;

    for (int idx = sequence.size() - 1; idx >= 0; --idx) {
        if (!curRunBuild.empty() && std::find(curRunBuild.begin(), curRunBuild.end(), sequence[idx].rank) == curRunBuild.end()) {
            runMin = std::min(runMin, sequence[idx].rank);
            runMax = std::max(runMax, sequence[idx].rank);
            curRunBuild.push_back(sequence[idx].rank);
            if (curRunBuild.size() > 1 && runMax - runMin == static_cast<int>(curRunBuild.size()) - 1) {
                validRunBuilds.push_back(curRunBuild.size());
            }
        }
        else {
            curRunBuild.clear();
        }

        if (!ofAKindBuild.empty() && ofAKindBuild[0].rank == sequence[idx].rank) {
            ofAKindBuild.push_back(sequence[idx]);
            if (ofAKindBuild.size() > 1) {
                validOfAKindBuilds.push_back(ofAKindBuild.size());
            }
        }
        else {
            ofAKindBuild.clear();
        }
    }

    if (!validRunBuilds.empty()) {
        baseScore += *std::max_element(validRunBuilds.begin(), validRunBuilds.end());
    }
    if (!validOfAKindBuilds.empty()) {
        baseScore += *std::max_element(validOfAKindBuilds.begin(), validOfAKindBuilds.end()) *
            (*std::max_element(validOfAKindBuilds.begin(), validOfAKindBuilds.end()) - 1);
    }

    if (peg_val(card) + current_sum == 15 || peg_val(card) + current_sum == 31) {
        baseScore += 2;
    }

    if (baseScore == 0) return { 0, false };

    int baseHole = nextPlayer ? nextPlayerCurrPos + baseScore : current_pos + baseScore;
	int landHole = baseHole;
	if (effLandingForNextPlayerHoles.size() > 0) {
		landHole = nextPlayer ? effLandingForNextPlayerHoles[baseHole - 1] : effLandingForHoles[baseHole - 1];
	}

    return { (landHole - (nextPlayer ? nextPlayerCurrPos : current_pos)), landHole != baseHole };
}

std::vector<Card> buildDeckWithoutSpecCards(int numdecks, const std::vector<Card>& cardsOmit) {
    std::vector<Card> cards;
    std::vector<Card> sortedCardsOmit = cardsOmit;
    std::sort(sortedCardsOmit.begin(), sortedCardsOmit.end(), [](const Card& a, const Card& b) {
        return std::tie(a.suit, a.rank) < std::tie(b.suit, b.rank);
        });

    size_t cardsOmit_idx = 0;
    for (int suit = 0; suit < 4; ++suit) {
        for (int rank = 1; rank <= 13; ++rank) {
            for (int d = 0; d < numdecks; ++d) {
                Card tentCard = { rank, suit };
                if (cardsOmit_idx < sortedCardsOmit.size() && sortedCardsOmit[cardsOmit_idx] == tentCard) {
                    ++cardsOmit_idx;
                }
                else {
                    cards.push_back(tentCard);
                }
            }
        }
    }

    return cards;
}

Card recommendCard(const std::vector<Node>& level_1) {
    if (level_1.empty()) return { 0, 0 };

    double curMax = -1000.0;
	bool started = false;
    Card bestCard = { 0, 0 };

    for (const auto& n : level_1) {
        if (n.netPointsForPlay > curMax || !started) {
            curMax = n.netPointsForPlay;
            bestCard = n.card;
			started = true;
        }
    }

    return bestCard;
}

int getCardToPlay(std::vector<int>& handMuxed, int nextPlayerNumCardsLeftInHand, std::vector<int>& sequenceMuxed,
    std::vector<int>& effLandingForHoles, std::vector<int>& effLandingForNextPlayerHoles,
    int current_sum, int current_pos, int nextPlayerCurrPos, int numdecks) {
		
		
    std::vector<Card> hand;
    std::vector<Card> sequence;
	
	for (int cardMuxed: handMuxed) {
		Card handCard = {cardMuxed%100, cardMuxed/100};
		hand.push_back(handCard);
	}
	
	for (int cardMuxed: sequenceMuxed) {
		Card seqCard = {cardMuxed%100, cardMuxed/100};
		sequence.push_back(seqCard);
	}
		
    bool soexcite_peg = false;
    std::vector<Node> level_1;
    bool some_events = false, some_no_events = false;

    for (const auto& card : hand) {
        if (peg_val(card) + current_sum <= 31) {
            int score;
            bool hasEvent;
            std::tie(score, hasEvent) = scoreCalc(sequence, current_sum, card,
                current_pos, nextPlayerCurrPos, effLandingForHoles, effLandingForNextPlayerHoles);
            Node newNode = { card, score, hasEvent, current_sum + card.rank };
			newNode.netPointsForPlay = (double)score;
            level_1.push_back(newNode);
            if (hasEvent) some_events = true;
            else some_no_events = true;
        }
    }
    if (some_events && some_no_events) soexcite_peg = true;
	
	if (nextPlayerCurrPos > -1 && nextPlayerNumCardsLeftInHand > 0 && effLandingForNextPlayerHoles.size() > 0) {

		std::vector<Card> cardsPlayed(hand);
		cardsPlayed.insert(cardsPlayed.end(), sequence.begin(), sequence.end());
		std::vector<Card> deckWithoutCards = buildDeckWithoutSpecCards(numdecks, cardsPlayed);
		int cardsAvailForNextOpponent = deckWithoutCards.size();
		
		for (auto& n : level_1) {
			std::vector<int> scoreDistribution;
			for (const auto& card : deckWithoutCards) {
				if (n.sumFromPlay + card.rank <= 31) {
					int score;
					bool hasEvent;
					std::vector<Card> testSequence(sequence);
					testSequence.push_back(n.card);
					std::tie(score, hasEvent) = scoreCalc(testSequence, current_sum + n.sumFromPlay, card,
						current_pos, nextPlayerCurrPos, effLandingForHoles, effLandingForNextPlayerHoles, true);
					scoreDistribution.push_back(score);
				}
			}
			scoreDistribution.resize(deckWithoutCards.size(), 0.0);
			std::sort(scoreDistribution.begin(), scoreDistribution.end());
			int scoreLineIdx = static_cast<int>(((scoreDistribution.size() - 1) *
				(nextPlayerNumCardsLeftInHand - 1)) / nextPlayerNumCardsLeftInHand);
			int curOpponentLikelyScoreUnmod = 0;
			for (int idx = scoreLineIdx; idx < scoreDistribution.size(); idx++) {
				curOpponentLikelyScoreUnmod += scoreDistribution[idx] * nextPlayerNumCardsLeftInHand;
			}
			n.likelyOpponentScore = (double)curOpponentLikelyScoreUnmod/(double)cardsAvailForNextOpponent;
			n.netPointsForPlay = (double)n.utility - n.likelyOpponentScore;
		}
		

	/*	for (auto& n : level_1) {
			std::vector<int> scoreDistribution;
			for (auto& m : n.children) {
				std::vector<Card> testHand = hand;
				testHand.erase(std::remove(testHand.begin(), testHand.end(), n.card), testHand.end());
				std::vector<Card> testSequence = sequence;
				testSequence.push_back(n.card);
				int score;
				bool hasEvent;
				std::tie(score, hasEvent) = scoreCalc(testSequence, current_sum + n.sumFromPlay, m.card,
					current_pos, nextPlayerCurrPos, effLandingForHoles, effLandingForNextPlayerHoles, true);
				scoreDistribution.push_back(score);

				Node newNode = { m.card, score, hasEvent };
				m.children.push_back(newNode);
			}

			scoreDistribution.resize(deckWithoutCards.size(), 0.0);
			std::sort(scoreDistribution.begin(), scoreDistribution.end());
			int scoreLineIdx = static_cast<int>(((scoreDistribution.size() - 1) *
				(nextPlayerNumCardsLeftInHand - 1)) / nextPlayerNumCardsLeftInHand);
			int curOpponentLikelyScoreUnmod = 0;
			for (int idx = scoreLineIdx; idx < scoreDistribution.size(); idx++) {
				curOpponentLikelyScoreUnmod += scoreDistribution[idx] * nextPlayerNumCardsLeftInHand;
			}
			n.likelyOpponentScore = (double)curOpponentLikelyScoreUnmod/(double)cardsAvailForNextOpponent;
			n.netPointsForPlay = (double)n.utility - n.likelyOpponentScore;*/
//			n.likelyOpponentScoreLine = scoreDistribution[static_cast<int>((scoreDistribution.size() - 1) *
//				(nextPlayerNumCardsLeftInHand - 1) / nextPlayerNumCardsLeftInHand)];
		//}

/*		for (auto& n : level_1) {
			int curOpponentLikelyScoreUnmod = 0;
			for (auto& m : n.children) {
				const Node& child = m.children[0];
				if (child.utility >= n.likelyOpponentScoreLine) {
					m.utility = child.utility;
					curOpponentLikelyScoreUnmod += m.utility * nextPlayerNumCardsLeftInHand;
				}
			}
			n.likelyOpponentScore = (double)curOpponentLikelyScoreUnmod/(double)cardsAvailForNextOpponent;
			n.netPointsForPlay = (double)n.utility - n.likelyOpponentScore;
		}*/
	}
    
    auto bestCard = recommendCard(level_1);
    int muxedCard = 100 * bestCard.suit + bestCard.rank;
	if (soexcite_peg) {muxedCard += 1000;}
    return muxedCard;
}

namespace py = pybind11;

PYBIND11_MODULE(scoretree, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scoretree

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("getCardToPlay", getCardToPlay, R"pbdoc(
        Get card to play
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}