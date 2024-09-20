#region Headers
from xml.dom import minidom
#DO NOT MODIFY ABOVE HERE
##################################################################################################

import numpy as np
import math
import sqlite3 as sql
from io import StringIO
from scipy.stats import norm


#endregion
#region Trial_Params
##################################################################################################

#YOU CAN CHANGE THESE VALUES:

twodecks = False
numplayers = 3
#Set to None if all tracks, otherwise tracks are 1 starting, enter list of ints
tracksused = None
#tracksused = [1,2]
#NOTE that realistically at least 100 trials are needed for any semblence of accuracy
numtrials = 1000
#NOTE: # logical cores minus 2 seems to be optimal
nummaxthreads = 2
#OPTIONAL run multiple boards in a batch
batchnum = 1
boardname = "Micro Board 2"

#Set if seeking optimal events layout
findmode = True
eventenergyfile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\Boards\\MicroBoard1\\CURVES\\energy.svg"
#TODO: nrg intensity curve?  going to be VERY track dependent, may struggle to converge
eventsovertimecurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\Boards\\MicroBoard1\\CURVES\\eventsovertime.svg"
eventlengthdisthistcurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\Boards\\MicroBoard1\\CURVES\\event-length-dist-hist.svg"
eventlengthovertimeidealcurve1file = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\eventlengthovertimeidealcurve1.svg"
eventspacingsdisthistcurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\Boards\\MicroBoard1\\CURVES\\spacinghist.svg"
velocityovertimecurvefile = "C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\Boards\\MicroBoard1\\CURVES\\velocity.svg"

####################################################################################
#CANDIDATE SET GEN PARAMS
minanglefromtracktangent = 30 #degrees
maxloopyorthoeventdisplacementincrements = 12
maxladderlength = 20
eventminspacing = 5 #mm
maxeventlineext = 100
mincrowvectordistcancel = 12 #mm, since straight line approx this equates to ~15mm curvy, need room to discernibly add lump on 1 side
whenstartworryingaboutcancels = 3 #After this many cancels on a track, start impeding
probminimodeliters = 200
allowabletwohits = 1
randomfeatheringamount = 15 #Nix holes randomly at nth interval to avoid endless opt loops
maxefflengthdisp = 14
goodscorecutoff = 230
gamelengthtightness = 4  #This is exponent mantissa, probably best keep below 6, should be int for perf

#OPTIMIZER BOUNDING PARAMS
maxeventsetfinesseiters = 10
maxeventsettrials = 100
maxitersconvergeoneventtrialset = 200
maxitertrynewbuild = 200
maxitertrackstalled = 20
minqualityboardlengthmatching = 2 #Try to get within this many holes of ideal
minqualityboardlengthintervalsrpt = 0.005

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
opttwohitspct = 0.01
optorthospct = 0.2
optmultispct = 0.05


##################################################################################################

##################################################################################################
#Iterative optimizer params
changepctperiteration = 0.02
iterscorecutoff = 6
prescorecutoff = 4



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


probHandHist, probPegHist = [], []
if numplayers == 2:
    dealsize = 6
    handsize = 4
    discardsize = 2
    cribstartsize = 0
    if not twodecks:
        probHandHist.append(dict(move=1, prob=0.00236094059873454))
        probHandHist.append(dict(move=2, prob=0.117287236325861))
        probHandHist.append(dict(move=3, prob=0.0236480395607792))
        probHandHist.append(dict(move=4, prob=0.150709569965402))
        probHandHist.append(dict(move=5, prob=0.0431622867641377))
        probHandHist.append(dict(move=6, prob=0.154135080143202))
        probHandHist.append(dict(move=7, prob=0.0618566436868448))
        probHandHist.append(dict(move=8, prob=0.155100919479048))
        probHandHist.append(dict(move=9, prob=0.0525760008241829))
        probHandHist.append(dict(move=10, prob=0.0703216888881258))
        probHandHist.append(dict(move=11, prob=0.0108302784192859))
        probHandHist.append(dict(move=12, prob=0.0828175035843371))
        probHandHist.append(dict(move=13, prob=0.00565337957915161))
        probHandHist.append(dict(move=14, prob=0.0259059572970235))
        probHandHist.append(dict(move=15, prob=0.00135217507018433))
        probHandHist.append(dict(move=16, prob=0.0242747619742615))
        probHandHist.append(dict(move=17, prob=0.00438276427510538))
        probHandHist.append(dict(move=18, prob=0.00387194258192464))
        probHandHist.append(dict(move=19, prob=0.000321946445281982))
        probPegHist.append(dict(move=1, prob=0.749741528628781))
        probPegHist.append(dict(move=2, prob=0.203735862399241))
        probPegHist.append(dict(move=3, prob=0.0304929082616849))
        probPegHist.append(dict(move=4, prob=0.00654570355683461))
        probPegHist.append(dict(move=5, prob=0.00252652968056967))
        probPegHist.append(dict(move=6, prob=0.00659717404634134))
        probPegHist.append(dict(move=7, prob=0.0000604218789861656))
        probPegHist.append(dict(move=8, prob=0.00000447569473971597))
        probPegHist.append(dict(move=9, prob=0.0000201406263287219))
        probPegHist.append(dict(move=10, prob=0))
        probPegHist.append(dict(move=11, prob=0))
        probPegHist.append(dict(move=12, prob=0.000228260431725515))
        probPegHist.append(dict(move=13, prob=0))
        probPegHist.append(dict(move=14, prob=0.0000469947947670177))

        avgMovesPerPegging = 2.335944235958582
        ideallikelihoodholehit = 0.28325666666666666
    else:
        probHandHist.append(dict(move=1, prob=0.002381436036184))
        probHandHist.append(dict(move=2, prob=0.116997229581671))
        probHandHist.append(dict(move=3, prob=0.0234297000946524))
        probHandHist.append(dict(move=4, prob=0.150510215107208))
        probHandHist.append(dict(move=5, prob=0.0434147458865123))
        probHandHist.append(dict(move=6, prob=0.15384681877315))
        probHandHist.append(dict(move=7, prob=0.0629805552073924))
        probHandHist.append(dict(move=8, prob=0.154326563600766))
        probHandHist.append(dict(move=9, prob=0.0524564231781582))
        probHandHist.append(dict(move=10, prob=0.0707602010606251))
        probHandHist.append(dict(move=11, prob=0.0112502323088692))
        probHandHist.append(dict(move=12, prob=0.0825333984518505))
        probHandHist.append(dict(move=13, prob=0.00588227667013869))
        probHandHist.append(dict(move=14, prob=0.0257506277742001))
        probHandHist.append(dict(move=15, prob=0.00132686182052357))
        probHandHist.append(dict(move=16, prob=0.0249078328067666))
        probHandHist.append(dict(move=17, prob=0.0037644841878698))
        probHandHist.append(dict(move=18, prob=0.00421829686264171))
        probHandHist.append(dict(move=19, prob=0.000280931655811179))
        probPegHist.append(dict(move=1, prob=0.806566806974083))
        probPegHist.append(dict(move=2, prob=0.146543923919091))
        probPegHist.append(dict(move=3, prob=0.0294897498609172))
        probPegHist.append(dict(move=4, prob=0.00809514312406185))
        probPegHist.append(dict(move=5, prob=0.00307976529123411))
        probPegHist.append(dict(move=6, prob=0.00593070003254012))
        probPegHist.append(dict(move=7, prob=0.0000356891683374096))
        probPegHist.append(dict(move=8, prob=0.00000839745137350814))
        probPegHist.append(dict(move=9, prob=0))
        probPegHist.append(dict(move=10, prob=0))
        probPegHist.append(dict(move=11, prob=0))
        probPegHist.append(dict(move=12, prob=0.000199439470120818))
        probPegHist.append(dict(move=13, prob=0))
        probPegHist.append(dict(move=14, prob=0.0000503847082410488))

        avgMovesPerPegging = 2.342926086134056

        ideallikelihoodholehit = 0.28256333333333333
elif numplayers == 3:
    dealsize = 5
    handsize = 4
    discardsize = 1
    cribstartsize = 1
    if not twodecks:
        probHandHist.append(dict(move=1, prob=0.00360317408125911))
        probHandHist.append(dict(move=2, prob=0.156059908591721))
        probHandHist.append(dict(move=3, prob=0.0302776225079188))
        probHandHist.append(dict(move=4, prob=0.188904963886058))
        probHandHist.append(dict(move=5, prob=0.0494251361807999))
        probHandHist.append(dict(move=6, prob=0.159432917940793))
        probHandHist.append(dict(move=7, prob=0.0671670009535396))
        probHandHist.append(dict(move=8, prob=0.140123740944114))
        probHandHist.append(dict(move=9, prob=0.0463754534793235))
        probHandHist.append(dict(move=10, prob=0.0537023641206063))
        probHandHist.append(dict(move=11, prob=0.00823660934468813))
        probHandHist.append(dict(move=12, prob=0.0539407490217999))
        probHandHist.append(dict(move=13, prob=0.00376483740505705))
        probHandHist.append(dict(move=14, prob=0.0161224914784248))
        probHandHist.append(dict(move=15, prob=0.000843937351351944))
        probHandHist.append(dict(move=16, prob=0.0131166496783174))
        probHandHist.append(dict(move=17, prob=0.0022386260261511))
        probHandHist.append(dict(move=18, prob=0.00197832067427306))
        probHandHist.append(dict(move=19, prob=0.000139742873113471))
        probPegHist.append(dict(move=1, prob=0.730293050282142))
        probPegHist.append(dict(move=2, prob=0.226196219924063))
        probPegHist.append(dict(move=3, prob=0.0274142910085097))
        probPegHist.append(dict(move=4, prob=0.0069140567640704))
        probPegHist.append(dict(move=5, prob=0.00363883003643025))
        probPegHist.append(dict(move=6, prob=0.00523309069804843))
        probPegHist.append(dict(move=7, prob=0.0000699237132288673))
        probPegHist.append(dict(move=8, prob=0.00000279694852915469))
        probPegHist.append(dict(move=9, prob=0.00000279694852915469))
        probPegHist.append(dict(move=10, prob=0))
        probPegHist.append(dict(move=11, prob=0))
        probPegHist.append(dict(move=12, prob=0.000173410808807591))
        probPegHist.append(dict(move=13, prob=0))
        probPegHist.append(dict(move=14, prob=0.0000615328676414032))

        avgMovesPerPegging = 2.115606979222163

        ideallikelihoodholehit = 0.30000583333333336
    else:
        probHandHist.append(dict(move=1, prob=0.00351686998571272))
        probHandHist.append(dict(move=2, prob=0.155440158259149))
        probHandHist.append(dict(move=3, prob=0.0310830860534125))
        probHandHist.append(dict(move=4, prob=0.187973953181668))
        probHandHist.append(dict(move=5, prob=0.0492718980107704))
        probHandHist.append(dict(move=6, prob=0.159152104626882))
        probHandHist.append(dict(move=7, prob=0.0673865259918672))
        probHandHist.append(dict(move=8, prob=0.140270908891087))
        probHandHist.append(dict(move=9, prob=0.0456533685020332))
        probHandHist.append(dict(move=10, prob=0.0546680953950984))
        probHandHist.append(dict(move=11, prob=0.00841850752829981))
        probHandHist.append(dict(move=12, prob=0.0542642048576767))
        probHandHist.append(dict(move=13, prob=0.00372843169579075))
        probHandHist.append(dict(move=14, prob=0.0157462358500934))
        probHandHist.append(dict(move=15, prob=0.000813276184196065))
        probHandHist.append(dict(move=16, prob=0.013353115727003))
        probHandHist.append(dict(move=17, prob=0.00224749972524453))
        probHandHist.append(dict(move=18, prob=0.00214858775689636))
        probHandHist.append(dict(move=19, prob=0.000173095944609298))
        probPegHist.append(dict(move=1, prob=0.727272982511308))
        probPegHist.append(dict(move=2, prob=0.22676339868547))
        probPegHist.append(dict(move=3, prob=0.0287304203294466))
        probPegHist.append(dict(move=4, prob=0.00777150429706912))
        probPegHist.append(dict(move=5, prob=0.00443043127918175))
        probPegHist.append(dict(move=6, prob=0.00472242421519879))
        probPegHist.append(dict(move=7, prob=0.0000926516046977171))
        probPegHist.append(dict(move=8, prob=0.00000421143657716896))
        probPegHist.append(dict(move=9, prob=0.00000140381219238965))
        probPegHist.append(dict(move=10, prob=0))
        probPegHist.append(dict(move=11, prob=0))
        probPegHist.append(dict(move=12, prob=0.000172668899663927))
        probPegHist.append(dict(move=13, prob=0))
        probPegHist.append(dict(move=14, prob=0.0000379029291945206))

        avgMovesPerPegging = 2.1183971225589944
        ideallikelihoodholehit = 0.2989738888888889
else:
    raise Exception(str(numplayers) + " player play is not configured yet")

#Det norm distr of avg moves per pegging for markov chain mini-model
maxVal = math.floor(2*avgMovesPerPegging)
list_pos = list(range(1, maxVal+1))
mean = avgMovesPerPegging
std_dev = 1.0
probabilities = norm.pdf(list_pos, loc=mean, scale=std_dev)
probabilities /= np.sum(probabilities)
prob_l = probabilities.tolist()
probPegRounds = [dict(rounds=r[0], prob=r[1]) for r in zip(list_pos, prob_l)]


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