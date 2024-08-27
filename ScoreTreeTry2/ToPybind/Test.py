import Card as cd
import ScoreTree_STANDALONE as st
import scoretree as stcpp
import time as tm

hand = [dict(rank=1, suit=2), dict(rank=3, suit=0), dict(rank=3, suit=3)]
sequence = [dict(rank=3, suit=2), dict(rank=3, suit=1)]
handmuxed = [201, 3, 303]
seqmuxed = [203, 103]
effLandingForHoles=[1,2,3,4,5,6,7,1,9,10,11,1,13,14,15,16,17,18]

start = tm.time()
test = st.getCardToPlay(hand, 1, sequence, effLandingForHoles,effLandingForHoles,
                    0, 2, 4, 1)
end = tm.time()
print(str(test['rank']) + "-" + str(test['suit']) + "-" + str(end-start))

start = tm.time()
testt = stcpp.getCardToPlay(handmuxed, 1, seqmuxed, effLandingForHoles, effLandingForHoles, 0, 2, 4, 1)
end = tm.time()

print(str(testt) + "-" + str(end-start))