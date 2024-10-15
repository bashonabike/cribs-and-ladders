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
    int moveCounter = 0;

    for (int trial = 0; trial < iters; ++trial) {
        // Trial Gameplay setup
        moveCounter = 0;
        // int dealer = rand() % numplayers + 1;
        int curPos = 0;
        int countLoops = 0;
        // std::vector<int> curReadSeq = curReadSeqs[trial];

        while (curPos <= trackActualLength) {
            // Get current move
            // int curMove = curReadSeq[moveCounter];
            int curMove = curReadSeqs[trial][moveCounter];
            if (moveCounter == 0) {
                countLoops++;
            }
            // moveCounter = (moveCounter + 1) % curReadSeq.size();
            moveCounter = (moveCounter + 1) % curReadSeqs[trial].size();

            // Move player
            if (curPos + curMove > effHoleMap.size()) {
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
    double eventlessCtrlPartialMoves = ideallikelihoodholehit * trackActualLength;
    double shiftPct = actualPartialMoves / eventlessCtrlPartialMoves;
    double forecastedTrackEffLengthHoles = trackActualLength * shiftPct;

    return forecastedTrackEffLengthHoles;
}

//int main() {
//    srand(static_cast<unsigned int>(time(0))); // Seed for random generator
//
//    // Sample data for testing
//    int track_id = 1;
//    std::vector<Event> partialEventSet = {
//        {1, 10, true, false},
//        {5, 3, false, true}
//    };
//    int trackActualLength = 50;
//    int probminimodeliters = 100;
//    std::unordered_map<int, std::vector<std::vector<int>>> track_dict = {
//        {1, {{3, 5, 2}, {4, 1, 6}}}
//    };
//    int numplayers = 4;
//    double ideallikelihoodholehit = 0.5;
//
//    double result = runPartialTrackEffLengthHoles(track_id, partialEventSet, trackActualLength, probminimodeliters, track_dict, numplayers, ideallikelihoodholehit);
//
//    std::cout << "Forecasted Track Effective Length: " << result << std::endl;
//
//    return 0;
//}



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