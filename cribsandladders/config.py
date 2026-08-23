"""
GameConfig: injectable configuration for cribbage rules + board-design
parameters.

This replaces the old pattern in game_params.py of module-level globals
read directly throughout the codebase (`gp.numplayers`, `gp.flushmods`,
...). That module also opened a live sqlite3 connection and ran a query
against it as an *import-time side effect*, and hardcoded an absolute
path to a specific machine's user profile. Together those meant that
importing almost anything in this package -- because almost everything
transitively imports game_params -- required a working sqlite db at a
path relative to the process's current working directory, plus scipy,
before a single line of game logic could be tested.

GameConfig fixes that: it's a plain dataclass you construct explicitly.
Each instance is independent (safe to build a different one per test),
constructing one does no I/O, and computing its derived fields needs
only the standard library -- no scipy, no numpy, no sqlite3.

`game_params.py` at the repo root still exists and still exposes the
exact same module-level names it always did, sourced from a single
DEFAULT_CONFIG instance defined here, so call sites that haven't been
migrated yet (`import game_params as gp; gp.numplayers`) keep working
unchanged. Modules that need per-call/per-test configurability --
currently ScoreHand, CribbageGame, Player, CribSquad -- take an explicit
`config: GameConfig` parameter (or store `self.config` on the instance)
that defaults to DEFAULT_CONFIG, so nothing about their public API
changes for existing callers.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


def _default_data_root() -> Path:
    """Base directory for board-design file paths (curve SVGs, the
    AllBoards sqlite db, etc). Override with the CRIBS_AND_LADDERS_DATA_DIR
    env var; otherwise defaults to the repo root, which is portable across
    machines -- unlike the old hardcoded 'C:\\Users\\Dell 5290\\...' path.
    """
    env = os.environ.get("CRIBS_AND_LADDERS_DATA_DIR")
    if env:
        return Path(env)
    # this file lives at <repo_root>/cribsandladders/config.py
    return Path(__file__).resolve().parent.parent


def _normal_pdf(x: float, mean: float, std: float) -> float:
    """Gaussian PDF. Stdlib stand-in for scipy.stats.norm.pdf, so that
    building a GameConfig doesn't pull in scipy just for this one
    derived table."""
    return (1.0 / (std * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * ((x - mean) / std) ** 2)


# Per-(numplayers, twodecks) ruleset data: deal/hand/discard/crib sizes,
# the empirically-measured pegging-move and hand-move probability
# histograms, and the two scalars derived from them
# (avgMovesPerPegging, ideallikelihoodholehit). This used to be four
# near-identical if/elif branches in _compute_derived, each repeating
# the same `self.probHandHist.append(dict(move=..., prob=...))` /
# `self.probPegHist.append(dict(move=..., prob=...))` pattern ~18
# times -- i.e. it was data shaped like code. Phase 10 of the Mk II
# refactor plan moves it into an actual lookup table instead; the
# values themselves are unchanged from the original game_params.py.
_RULESET_TABLES = {
    (2, False): dict(
        dealsize=6, handsize=4, discardsize=2, cribstartsize=0,
        avgMovesPerPegging=2.335944235958582,
        ideallikelihoodholehit=0.28325666666666666,
        probHandHist=[
            (1, 0.00236094059873454),
            (2, 0.117287236325861),
            (3, 0.0236480395607792),
            (4, 0.150709569965402),
            (5, 0.0431622867641377),
            (6, 0.154135080143202),
            (7, 0.0618566436868448),
            (8, 0.155100919479048),
            (9, 0.0525760008241829),
            (10, 0.0703216888881258),
            (11, 0.0108302784192859),
            (12, 0.0828175035843371),
            (13, 0.00565337957915161),
            (14, 0.0259059572970235),
            (15, 0.00135217507018433),
            (16, 0.0242747619742615),
            (17, 0.00438276427510538),
            (18, 0.00387194258192464),
            (19, 0.000321946445281982),
        ],
        probPegHist=[
            (1, 0.749741528628781),
            (2, 0.203735862399241),
            (3, 0.0304929082616849),
            (4, 0.00654570355683461),
            (5, 0.00252652968056967),
            (6, 0.00659717404634134),
            (7, 0.0000604218789861656),
            (8, 0.00000447569473971597),
            (9, 0.0000201406263287219),
            (10, 0),
            (11, 0),
            (12, 0.000228260431725515),
            (13, 0),
            (14, 0.0000469947947670177),
        ],
    ),
    (2, True): dict(
        dealsize=6, handsize=4, discardsize=2, cribstartsize=0,
        avgMovesPerPegging=2.342926086134056,
        ideallikelihoodholehit=0.28256333333333333,
        probHandHist=[
            (1, 0.002381436036184),
            (2, 0.116997229581671),
            (3, 0.0234297000946524),
            (4, 0.150510215107208),
            (5, 0.0434147458865123),
            (6, 0.15384681877315),
            (7, 0.0629805552073924),
            (8, 0.154326563600766),
            (9, 0.0524564231781582),
            (10, 0.0707602010606251),
            (11, 0.0112502323088692),
            (12, 0.0825333984518505),
            (13, 0.00588227667013869),
            (14, 0.0257506277742001),
            (15, 0.00132686182052357),
            (16, 0.0249078328067666),
            (17, 0.0037644841878698),
            (18, 0.00421829686264171),
            (19, 0.000280931655811179),
        ],
        probPegHist=[
            (1, 0.806566806974083),
            (2, 0.146543923919091),
            (3, 0.0294897498609172),
            (4, 0.00809514312406185),
            (5, 0.00307976529123411),
            (6, 0.00593070003254012),
            (7, 0.0000356891683374096),
            (8, 0.00000839745137350814),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0.000199439470120818),
            (13, 0),
            (14, 0.0000503847082410488),
        ],
    ),
    (3, False): dict(
        dealsize=5, handsize=4, discardsize=1, cribstartsize=1,
        avgMovesPerPegging=2.115606979222163,
        ideallikelihoodholehit=0.30000583333333336,
        probHandHist=[
            (1, 0.00360317408125911),
            (2, 0.156059908591721),
            (3, 0.0302776225079188),
            (4, 0.188904963886058),
            (5, 0.0494251361807999),
            (6, 0.159432917940793),
            (7, 0.0671670009535396),
            (8, 0.140123740944114),
            (9, 0.0463754534793235),
            (10, 0.0537023641206063),
            (11, 0.00823660934468813),
            (12, 0.0539407490217999),
            (13, 0.00376483740505705),
            (14, 0.0161224914784248),
            (15, 0.000843937351351944),
            (16, 0.0131166496783174),
            (17, 0.0022386260261511),
            (18, 0.00197832067427306),
            (19, 0.000139742873113471),
        ],
        probPegHist=[
            (1, 0.730293050282142),
            (2, 0.226196219924063),
            (3, 0.0274142910085097),
            (4, 0.0069140567640704),
            (5, 0.00363883003643025),
            (6, 0.00523309069804843),
            (7, 0.0000699237132288673),
            (8, 0.00000279694852915469),
            (9, 0.00000279694852915469),
            (10, 0),
            (11, 0),
            (12, 0.000173410808807591),
            (13, 0),
            (14, 0.0000615328676414032),
        ],
    ),
    (3, True): dict(
        dealsize=5, handsize=4, discardsize=1, cribstartsize=1,
        avgMovesPerPegging=2.1183971225589944,
        ideallikelihoodholehit=0.2989738888888889,
        probHandHist=[
            (1, 0.00351686998571272),
            (2, 0.155440158259149),
            (3, 0.0310830860534125),
            (4, 0.187973953181668),
            (5, 0.0492718980107704),
            (6, 0.159152104626882),
            (7, 0.0673865259918672),
            (8, 0.140270908891087),
            (9, 0.0456533685020332),
            (10, 0.0546680953950984),
            (11, 0.00841850752829981),
            (12, 0.0542642048576767),
            (13, 0.00372843169579075),
            (14, 0.0157462358500934),
            (15, 0.000813276184196065),
            (16, 0.013353115727003),
            (17, 0.00224749972524453),
            (18, 0.00214858775689636),
            (19, 0.000173095944609298),
        ],
        probPegHist=[
            (1, 0.727272982511308),
            (2, 0.22676339868547),
            (3, 0.0287304203294466),
            (4, 0.00777150429706912),
            (5, 0.00443043127918175),
            (6, 0.00472242421519879),
            (7, 0.0000926516046977171),
            (8, 0.00000421143657716896),
            (9, 0.00000140381219238965),
            (10, 0),
            (11, 0),
            (12, 0.000172668899663927),
            (13, 0),
            (14, 0.0000379029291945206),
        ],
    ),
}


@dataclass
class GameConfig:
    # ---- user-settable trial params (mirrors old game_params.py) ----
    twodecks: bool = False
    numplayers: int = 3
    # Set to None for all tracks, otherwise 1-indexed list of track ints
    tracksused: Optional[List[int]] = None
    # NOTE: realistically at least 100 trials are needed for any semblance of accuracy
    numtrials: int = 1000
    nummaxthreads: int = 2
    batchnum: int = 1
    boardname: str = "Micro Board 7"
    # Set if seeking optimal events layout
    findmode: bool = True

    # ---- candidate set gen params ----
    minanglefromtracktangent: int = 30  # degrees
    maxloopyorthoeventdisplacementincrements: int = 12
    maxladderlength: int = 20
    eventminspacing: int = 5  # mm
    maxeventlineext: int = 100
    mincrowvectordistcancel: int = 12
    whenstartworryingaboutcancels: int = 12
    probminimodeliters: int = 500
    allowabletwohits: int = 3
    onlysamedirtwohits: bool = False
    maxtwohitnetgainloss: int = 25
    randomfeatheringamount: int = 9
    maxefflengthdisp: int = 24
    goodscorecutoffperc: float = 0.5
    gamelengthtightness: int = 5
    idealcancelspct: float = 0.75
    finishlinelength: int = 15

    # ---- optimizer bounding params ----
    maxeventsetfinesseiters: int = 10
    maxeventsettrials: int = 100
    maxitersconvergeoneventtrialset: int = 400
    maxitertrynewbuild: int = 400
    maxitertrackstalled: int = 20
    minqualityboardlengthmatching: float = 1.5
    minqualityboardlengthintervalsrpt: float = 0.005

    # ---- optimizer balancing params ----
    # NOTE: changing this will require code changes elsewhere, per the
    # original game_params.py comment -- carried over verbatim.
    effectiveboardlength: int = 120

    # ---- optimizer output params ----
    numbestpickstooutput: int = 5
    testtotraindataratio_bnds: Tuple = (0.2, 0.3, False)
    trainrandomstate_bnds: Tuple = (38, 44, True)
    trainlearningrate_bnds: Tuple = (0.01, 0.1, False)
    trainnumestimators_bnds: Tuple = (50, 300, True)

    # ---- evaluator params ----
    idealgamelength: int = 12
    opttwohitspct: float = 0.01
    optorthospct: float = 0.2
    optmultispct: float = 0.05

    # ---- iterative optimizer params ----
    changebaseincrperiter: float = 0.01
    iterscorecutoff: float = 2
    prescorecutoff: float = 2.5
    maxnumitermodeliters: int = 1000

    # ---- base dir for board-design file paths, see _default_data_root ----
    data_root: Path = field(default_factory=_default_data_root)

    def __post_init__(self):
        self._compute_derived()

    # The rest of this mirrors the "DO NOT MODIFY BELOW" derived section
    # of the old game_params.py -- same values, same formulas -- just
    # reading/writing self.* instead of module globals. The
    # numplayers/twodecks ruleset branch (probHandHist/probPegHist/
    # dealsize/etc.) was moved to the _RULESET_TABLES lookup above in
    # Phase 10 of the Mk II refactor; everything else here is unchanged
    # and easy to diff against the original if needed.
    def _compute_derived(self):
        self.flushmods = [[0.0] * 21 for _ in range(3)]

        if not self.twodecks:
            self.unknowncardsafterdeal = 46  # 52 minus 6 card deal
            self.numdecks = 1
            self.cardsperrank = 4
            self.flushmods[0][10] = 4 + (13.0 - 4.0) / 52.0
            self.flushmods[1][10] = 4 + (13.0 - 5.0) / 52.0
            self.flushmods[2][10] = 4 + (13.0 - 6.0) / 52.0
        else:
            self.unknowncardsafterdeal = 98  # 52*2 minus 6 card deal
            self.numdecks = 2
            self.cardsperrank = 8
            self.flushmods[0][10] = 4 + (13.0 - 4.0) / (52.0 * 2)
            self.flushmods[1][10] = 4 + (13.0 - 5.0) / (52.0 * 2)
            self.flushmods[2][10] = 4 + (13.0 - 6.0) / (52.0 * 2)

        for d in range(0, 3):
            for r in range(0, 21):
                if r < 10:
                    self.flushmods[d][r] = ((r + 1) / 10.0) * math.sqrt(self.flushmods[d][10]) + \
                        (1 - (r + 1) / 10.0) * self.flushmods[d][10]
                elif r > 10:
                    self.flushmods[d][r] = ((20 - r) / 10.0) * self.flushmods[d][10] + \
                        ((r - 10) / 10.0) * math.pow(self.flushmods[d][10], 2)

        ruleset = _RULESET_TABLES.get((self.numplayers, self.twodecks))
        if ruleset is None:
            raise ValueError(f"{self.numplayers} player play is not configured yet")

        self.dealsize = ruleset["dealsize"]
        self.handsize = ruleset["handsize"]
        self.discardsize = ruleset["discardsize"]
        self.cribstartsize = ruleset["cribstartsize"]
        self.probHandHist: List[dict] = [dict(move=m, prob=p) for m, p in ruleset["probHandHist"]]
        self.probPegHist: List[dict] = [dict(move=m, prob=p) for m, p in ruleset["probPegHist"]]
        self.avgMovesPerPegging = ruleset["avgMovesPerPegging"]
        self.ideallikelihoodholehit = ruleset["ideallikelihoodholehit"]

        # Discretized normal distribution of avg moves per pegging, for
        # the markov chain mini-model. Originally computed with
        # scipy.stats.norm.pdf; _normal_pdf above is the stdlib
        # equivalent (identical formula, so values match to float
        # precision).
        max_val = math.floor(2 * self.avgMovesPerPegging)
        list_pos = list(range(1, max_val + 1))
        mean = self.avgMovesPerPegging
        std_dev = 1.0
        probabilities = [_normal_pdf(x, mean, std_dev) for x in list_pos]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        self.probPegRounds = [dict(rounds=r, prob=p) for r, p in zip(list_pos, probabilities)]

    # ---- board-design file paths, derived from data_root. These used to
    # be hardcoded to a specific machine's user profile
    # ('C:\\Users\\Dell 5290\\...'); now they're computed from data_root,
    # which defaults to the repo root and can be overridden per-instance
    # or via CRIBS_AND_LADDERS_DATA_DIR. Note the Boards/MicroBoard1/CURVES
    # subpath doesn't currently exist in this repo -- that was already the
    # case before this refactor and is a Phase 3/4 concern, not fixed here. ----
    @property
    def eventenergyfile(self) -> str:
        return str(self.data_root / "Boards" / "MicroBoard1" / "CURVES" / "energy.svg")

    @property
    def eventsovertimecurvefile(self) -> str:
        return str(self.data_root / "Boards" / "MicroBoard1" / "CURVES" / "eventsovertime.svg")

    @property
    def eventlengthdisthistcurvefile(self) -> str:
        return str(self.data_root / "Boards" / "MicroBoard1" / "CURVES" / "event-length-dist-hist.svg")

    @property
    def eventlengthovertimeidealcurve1file(self) -> str:
        return str(self.data_root / "etc" / "eventlengthovertimeidealcurve1.svg")

    @property
    def eventspacingsdisthistcurvefile(self) -> str:
        return str(self.data_root / "Boards" / "MicroBoard1" / "CURVES" / "spacinghist.svg")

    @property
    def velocityovertimecurvefile(self) -> str:
        return str(self.data_root / "Boards" / "MicroBoard1" / "CURVES" / "velocity.svg")

    @property
    def db_path(self) -> str:
        return str(self.data_root / "Boards" / "AllBoards.db")

    # ---- Phase 4 (Optimizer/board-design subsystem) db paths, added
    # alongside the game_params -> GameConfig migration of
    # PossibleEvents/Stats/EventSetBuilder/Optimizer. Previously each of
    # these was a separate hardcoded literal ('etc/Temp.db',
    # 'etc/Optimizer.db') repeated at every call site instead of a single
    # source of truth, same issue db_path fixed for 'Boards/AllBoards.db'
    # in Phase 3. ----
    @property
    def temp_events_db_path(self) -> str:
        return str(self.data_root / "etc" / "Temp.db")

    @property
    def optimizer_db_path(self) -> str:
        return str(self.data_root / "etc" / "Optimizer.db")


DEFAULT_CONFIG = GameConfig()
