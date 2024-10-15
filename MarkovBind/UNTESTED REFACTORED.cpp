#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <map>

using namespace std;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

double runPartialTrackEffLengthHoles(
    int trackActualLength,
    int probminimodeliters,
    const std::vector<std::vector<int>>& curReadSeqs,
    const std::vector<int>& effHoleMap,
    int numplayers,
    double ideallikelihoodholehit) {

    // Simulation
    int movesAllTrials = 0;
    int iters = probminimodeliters;
    int effHoleMapSize = effHoleMap.size();  // Precompute size
    double eventlessCtrlPartialMoves = ideallikelihoodholehit * trackActualLength;  // Precompute

    for (int trial = 0; trial < iters; ++trial) {
        int moveCounter = 0;
        int curPos = 0;
        int countLoops = 0;
        const std::vector<int>& curReadSeq = curReadSeqs[trial];  // Reference to avoid copying

        while (curPos <= trackActualLength) {
            int curMove = curReadSeq[moveCounter];
            if (moveCounter == 0) {
                countLoops++;
            }
            moveCounter = (moveCounter + 1) % curReadSeq.size();

            // Move player
            if (curPos + curMove > effHoleMapSize) {
                curPos += curMove;
            }
            else {
                curPos = effHoleMap[curPos + curMove];
            }

            movesAllTrials++;

            // Check for infinite loops
            if (countLoops > 10) {
                return 9999999;
            }

            if (curPos > trackActualLength) break;
        }
    }

    // Forecast length of the game
    double actualPartialMoves = static_cast<double>(movesAllTrials) / iters;
    double shiftPct = actualPartialMoves / eventlessCtrlPartialMoves;
    double forecastedTrackEffLengthHoles = trackActualLength * shiftPct;

    return forecastedTrackEffLengthHoles;
}

namespace py = pybind11;

PYBIND11_MODULE(markovgame, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: markovgame

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("runPartialTrackEffLengthHoles", runPartialTrackEffLengthHoles, R"pbdoc(
        Run markov chain model to get effective length thus far
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
