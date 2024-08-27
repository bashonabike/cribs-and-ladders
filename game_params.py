#region Headers
from xml.dom import minidom
#DO NOT MODIFY ABOVE HERE
##################################################################################################

import numpy as np
import math
import sqlite3 as sql
from io import StringIO


#endregion
#region Trial_Params
##################################################################################################

#YOU CAN CHANGE THESE VALUES:

twodecks = False
#TODO: ensure 3 player mode works, scope out all the code
numplayers = 3
#Set to None if all tracks, otherwise tracks are 1 starting, enter list of ints
tracksused = None
#tracksused = [1,2]
#NOTE that realistically at least 100 trials are needed for any semblence of accuracy
numtrials = 1000
#NOTE: # logical cores minus 2 seems to be optimal
nummaxthreads = 6
#OPTIONAL run multiple boards in a batch
batchnum = 1
boardname = "Meg Bday Trial #2"

#Set if seeking optimal events layout
findmode = True
eventenergyfile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\Boards\\energy-curve-SONG.svg"
#TODO: nrg intensity curve?  going to be VERY track dependent, may struggle to converge
eventsovertimecurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\eventsovertimecurve1.svg"
eventlengthdisthistcurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\eventlengthdistcurve1.svg"
eventlengthovertimeidealcurve1file = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\eventlengthovertimeidealcurve1.svg"
eventspacingsdisthistcurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\eventspacingsdisthistcurve1.svg"
velocityovertimecurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\velocityovertimecurve1.svg"

####################################################################################
#CANDIDATE SET GEN PARAMS
minanglefromtracktangent = 30 #degrees
maxloopyorthoeventdisplacementincrements = 12
maxladderlength = 20
eventminspacing = 5 #mm
maxeventlineext = 100
mincrowvectordistcancel = 12 #mm, since straight line approx this equates to ~15mm curvy, need room to discernibly add lump on 1 side
whenstartworryingaboutcancels = 3 #After this many cancels on a track, start impeding

#OPTIMIZER BOUNDING PARAMS
maxeventsetfinesseiters = 10
maxeventsettrials = 100
maxitersconvergeoneventtrialset = 200
maxitertrynewbuild = 50
maxitertrackstalled = 20

#OPTIMIZER BALANCING PARAMS
#NOTE: changing these will require code changes!!!
effectiveboardlength = 120 #try to achieve parity with normal board

#OPTIMIZER OUTPUT PARAMS
numbestpickstooutput = 5
testtotraindataratio_bnds = (0.2, 0.3, False)
trainrandomstate_bnds = (38, 44, True)
trainlearningrate_bnds=(0.01, 0.1, False)
trainnumestimators_bnds=(50, 300, True)
####################################################################################


####################################################################################
#EVALUATOR PARAMS
idealgamelength = 12
opttwohitspct = 0.1
optorthospct = 0.2
optmultispct = 0.05


##################################################################################################

##################################################################################################
#Iterative optimizer params
changepctperiteration = 0.01
iterscorecutoff = 8



##################################################################################################
#endregion
#region xml_parser


##################################################################################################
#DO NOT MODIFY BELOW THIS LINE!!
#TODO: convert this to OOP
#Make BOARD file w/ subclasses, pass the board into all the child methods
#just have gameparams be the non-board related stuff
# board_xml_file_names = []
# board_xml_files = []
# boardnames = []
# lengths = []
# twodecklengths = []
#
# ladders = []
# chutes = []
# laddersbytrack = []
# chutesbytrack = []
#
# for batch in range(batchnum):
#     board_xml_file_names.append(input("Please enter board xml file path: "))
#     # parse an xml file by name
#     board_xml_files.append(minidom.parse(board_xml_file_names[batch]))
#
#     # use getElementsByTagName() to get tag
#     boardnames.append(str(board_xml_files[batch].getElementsByTagName('boardname')[0].firstChild.data).strip("\""))
#     batch_ladders = []
#     batch_chutes= []
#     batch_lengths = []
#     batch_twodecklengths = []
#
#     for track in board_xml_files[batch].getElementsByTagName('track'):
#         curtrack = int(track.getElementsByTagName('tracknum')[0].firstChild.data)
#         batch_lengths.append(int(track.getElementsByTagName('length')[0].firstChild.data))
#         try:
#             batch_twodecklengths.append(int(track.getElementsByTagName('twodecklength')[0].firstChild.data))
#         except:
#             batch_twodecklengths.append(int(track.getElementsByTagName('length')[0].firstChild.data))
#
#         for ladder in track.getElementsByTagName('ladder'):
#             start = int(ladder.getElementsByTagName('start')[0].firstChild.data)
#             end = int(ladder.getElementsByTagName('end')[0].firstChild.data)
#             batch_ladders.append((start, end, curtrack))
#         for chute in board_xml_files[batch].getElementsByTagName('chute'):
#             start = int(chute.getElementsByTagName('start')[0].firstChild.data)
#             end = int(chute.getElementsByTagName('end')[0].firstChild.data)
#             batch_chutes.append((start, end, curtrack))
#
#     ladders.append(batch_ladders)
#     chutes.append(batch_chutes)
#     lengths.append(batch_lengths)
#     twodecklengths.append(batch_twodecklengths)
#
#     batch_laddersbytrack = []
#     batch_chutesbytrack = []
#     for curtrack in range(numplayers):
#         (batch_laddersbytrack.append(sorted([(start, end) for (start, end, track) in batch_ladders if track in
#                                              {0, curtrack + 1}], key=lambda e: (e[0], e[1]))))
#         (batch_chutesbytrack.append(sorted([(start, end) for (start, end, track) in batch_chutes if track in
#                                             {0, curtrack + 1}], key=lambda e: (e[0], e[1]))))
#
#     laddersbytrack.append(batch_laddersbytrack)
#     chutesbytrack.append(batch_chutesbytrack)


flushmods = np.zeros((3, 21), dtype=float).tolist()

if not twodecks:
    unknowncardsafterdeal = 46 #52 minus 6 card deal
    numdecks = 1
    cardsperrank = 4
    flushmods[0][10] = 4+(13.0-4.0)/52.0
    flushmods[1][10] = 4+(13.0-5.0)/52.0
    flushmods[2][10] = 4+(13.0-6.0)/52.0
else:
    unknowncardsafterdeal = 98 #52*2 minus  6 card deal
    numdecks = 2
    cardsperrank = 8
    flushmods[0][10] = 4+(13.0-4.0)/(52.0*2)
    flushmods[1][10]= 4+(13.0-5.0)/(52.0*2)
    flushmods[2][10]= 4+(13.0-6.0)/(52.0*2)

for d in range(0,3):
    for r in range(0,21):
        if r < 10:
            flushmods[d][r] = ((r+1)/10.0)*math.sqrt(flushmods[d][10]) + (1 - (r+1)/10.0)*flushmods[d][10]
        elif r > 10:
            flushmods[d][r] = ((20-r) / 10.0) * flushmods[d][10] + ((r -10) / 10.0) * math.pow(flushmods[d][10],2)


if numplayers == 2:
    dealsize = 6
    handsize = 4
    discardsize = 2
    cribstartsize = 0
    if not twodecks:
        likelihoodoffourmove = 0.0729679583244131
        likelihoodoftwomove = 0.162153000784985
        likelihoodofonemove = 0.407104117605081
    else:
        likelihoodoffourmove = 0.070627226485653
        likelihoodoftwomove = 0.164968186732361
        likelihoodofonemove = 0.406004643496447
elif numplayers == 3:
    dealsize = 5
    handsize = 4
    discardsize = 1
    cribstartsize = 1
    if not twodecks:
        likelihoodoffourmove = 0.0864347659125092
        likelihoodoftwomove = 0.177552348341814
        likelihoodofonemove = 0.432365835706743
    else:
        likelihoodoffourmove = 0.0830682257856498
        likelihoodoftwomove = 0.177399939813422
        likelihoodofonemove = 0.43167748592064
else:
    raise Exception(str(numplayers) + " player play is not configured yet")

likelihoodofonemovemultiplier = likelihoodofonemove/likelihoodoftwomove

#Pre-create stub for inserting into stats
sqliteConn = sql.connect('Boards/AllBoards.db')
sqliteCursor = sqliteConn.cursor()

#Prepend columns to write to, all except auto-incrementing Stat_ID
retrievestatcolumnsquery = "SELECT name FROM pragma_table_info(\'Stat\') as tblInfo"
sqliteCursor.execute(retrievestatcolumnsquery)
result = sqliteCursor.fetchall()
result.remove(('Stat_ID',))
insertstatquery_sb = StringIO()
insertstatquery_sb.write("INSERT INTO Stat (")
insertstatquery_sb.write("".join([c[0]+"," for c in result])[:-1])
insertstatquery_sb.write(") Values (")
insertstatstub = insertstatquery_sb.getvalue()
insertstatquery_sb.close()

##################################################################################################
#endregion