"""
Backward-compatible shim over cribsandladders.config.GameConfig.

This file used to hold ~300 lines of module-level constants (read
directly everywhere as `gp.numplayers`, `gp.flushmods`, etc.) and, as an
import-time side effect, opened a live sqlite3 connection against
'Boards/AllBoards.db' and ran a schema-introspection query against it.
That meant importing this module -- which nearly everything in the
package does -- required a working db at a path relative to the
process's CWD, plus scipy, before any game logic could even be
imported, let alone tested.

The actual config now lives in cribsandladders.config.GameConfig, a
plain dataclass with no import-time I/O that can be constructed
per-test with different values. This module re-exports the exact same
names as before, sourced from a single DEFAULT_CONFIG instance, so
every call site that still does `import game_params as gp; gp.X` keeps
working completely unchanged.

The one behavioral improvement: `sqliteConn`, `sqliteCursor`, and
`insertstatstub` are now created lazily on first access (see
__getattr__ below) instead of at import time, so importing this module
-- or anything that imports it -- no longer touches the filesystem or
a database. Only code that actually reads one of those three names
(currently: Stats.py) triggers the connection, exactly once, cached
after that.

Modules that need per-call/per-test configurability (ScoreHand,
CribbageGame, Player, CribSquad) have been updated to accept an
explicit `config: GameConfig` parameter instead of reading this module.
Everything else still reads these module-level names unchanged --
that's intentional; migrating those (Optimizer, EventSetBuilder,
PossibleEvents, Evaluator, Stats, Board, BaseLayout, BoardSetter) is
Phase 3/4 work, done alongside writing tests for them, not before.
"""
import sqlite3 as sql
from io import StringIO

from cribsandladders.config import GameConfig, DEFAULT_CONFIG

_cfg = DEFAULT_CONFIG

# ---- user-settable trial params ----
twodecks = _cfg.twodecks
numplayers = _cfg.numplayers
tracksused = _cfg.tracksused
numtrials = _cfg.numtrials
nummaxthreads = _cfg.nummaxthreads
batchnum = _cfg.batchnum
boardname = _cfg.boardname
findmode = _cfg.findmode

eventenergyfile = _cfg.eventenergyfile
eventsovertimecurvefile = _cfg.eventsovertimecurvefile
eventlengthdisthistcurvefile = _cfg.eventlengthdisthistcurvefile
eventlengthovertimeidealcurve1file = _cfg.eventlengthovertimeidealcurve1file
eventspacingsdisthistcurvefile = _cfg.eventspacingsdisthistcurvefile
velocityovertimecurvefile = _cfg.velocityovertimecurvefile

# ---- candidate set gen params ----
minanglefromtracktangent = _cfg.minanglefromtracktangent
maxloopyorthoeventdisplacementincrements = _cfg.maxloopyorthoeventdisplacementincrements
maxladderlength = _cfg.maxladderlength
eventminspacing = _cfg.eventminspacing
maxeventlineext = _cfg.maxeventlineext
mincrowvectordistcancel = _cfg.mincrowvectordistcancel
whenstartworryingaboutcancels = _cfg.whenstartworryingaboutcancels
probminimodeliters = _cfg.probminimodeliters
allowabletwohits = _cfg.allowabletwohits
onlysamedirtwohits = _cfg.onlysamedirtwohits
maxtwohitnetgainloss = _cfg.maxtwohitnetgainloss
randomfeatheringamount = _cfg.randomfeatheringamount
maxefflengthdisp = _cfg.maxefflengthdisp
goodscorecutoffperc = _cfg.goodscorecutoffperc
gamelengthtightness = _cfg.gamelengthtightness
idealcancelspct = _cfg.idealcancelspct
finishlinelength = _cfg.finishlinelength

# ---- optimizer bounding params ----
maxeventsetfinesseiters = _cfg.maxeventsetfinesseiters
maxeventsettrials = _cfg.maxeventsettrials
maxitersconvergeoneventtrialset = _cfg.maxitersconvergeoneventtrialset
maxitertrynewbuild = _cfg.maxitertrynewbuild
maxitertrackstalled = _cfg.maxitertrackstalled
minqualityboardlengthmatching = _cfg.minqualityboardlengthmatching
minqualityboardlengthintervalsrpt = _cfg.minqualityboardlengthintervalsrpt

# ---- optimizer balancing params ----
effectiveboardlength = _cfg.effectiveboardlength

# ---- optimizer output params ----
numbestpickstooutput = _cfg.numbestpickstooutput
testtotraindataratio_bnds = _cfg.testtotraindataratio_bnds
trainrandomstate_bnds = _cfg.trainrandomstate_bnds
trainlearningrate_bnds = _cfg.trainlearningrate_bnds
trainnumestimators_bnds = _cfg.trainnumestimators_bnds

# ---- evaluator params ----
idealgamelength = _cfg.idealgamelength
opttwohitspct = _cfg.opttwohitspct
optorthospct = _cfg.optorthospct
optmultispct = _cfg.optmultispct

# ---- iterative optimizer params ----
changebaseincrperiter = _cfg.changebaseincrperiter
iterscorecutoff = _cfg.iterscorecutoff
prescorecutoff = _cfg.prescorecutoff
maxnumitermodeliters = _cfg.maxnumitermodeliters

# ---- derived (previously the "DO NOT MODIFY BELOW" section) ----
flushmods = _cfg.flushmods
unknowncardsafterdeal = _cfg.unknowncardsafterdeal
numdecks = _cfg.numdecks
cardsperrank = _cfg.cardsperrank
dealsize = _cfg.dealsize
handsize = _cfg.handsize
discardsize = _cfg.discardsize
cribstartsize = _cfg.cribstartsize
probHandHist = _cfg.probHandHist
probPegHist = _cfg.probPegHist
avgMovesPerPegging = _cfg.avgMovesPerPegging
ideallikelihoodholehit = _cfg.ideallikelihoodholehit
probPegRounds = _cfg.probPegRounds

# ---- lazy DB seam ----
# Same public names as before (sqliteConn, sqliteCursor, insertstatstub),
# but nothing happens until one of them is actually accessed. See the
# module docstring above for why this matters.
_lazy_db = {}


def __getattr__(name):
    if name in ("sqliteConn", "sqliteCursor", "insertstatstub"):
        if not _lazy_db:
            conn = sql.connect(_cfg.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM pragma_table_info('Stat') as tblInfo")
            result = cursor.fetchall()
            result.remove(('Stat_ID',))
            sb = StringIO()
            sb.write("INSERT INTO Stat (")
            sb.write("".join([c[0] + "," for c in result])[:-1])
            sb.write(") Values (")
            _lazy_db['sqliteConn'] = conn
            _lazy_db['sqliteCursor'] = cursor
            _lazy_db['insertstatstub'] = sb.getvalue()
            sb.close()
        return _lazy_db[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
