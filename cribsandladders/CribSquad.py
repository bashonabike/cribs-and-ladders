import game_params as gp
import cribsandladders.Player as pl
import os
import random as r

class CribSquad:
    def __init__(self, rankLookupTable, tracks, tracksUsed = None, homoRisk = False):
        self.players = []
        self.tracksUsed = tracksUsed
        self.homoRisk = homoRisk
        if self.tracksUsed is None or len(self.tracksUsed) != gp.numplayers:
            self.tracksUsed = []
            for p in range(gp.numplayers):
                if len(tracks) in (0,1): self.tracksUsed.append(0)
                else: self.tracksUsed.append(tracks[p].num)
        for i in range(0,gp.numplayers):
            if homoRisk: risk = 11
            else: risk = r.randint(1,21)
            (self.players.append(pl.Player(risk, i + 1, rankLookupTable, self.tracksUsed[i])))

    def resetRisks(self):
        for player in self.players:
            if self.homoRisk: player.risk = 11
            else: player.risk = r.randint(1,21)

    def resetCanPlay(self):
        for player in self.players:
            player.canPlay = True

    def resetWins(self):
        for player in self.players:
            player.wins = 0

    def resetScores(self):
        for player in self.players:
            player.score = 0

    def getPlayerByNum (self, num):
        for player in self.players:
            if player.num == num:
                return player

    def getNextPeggingPlayer (self, num):
        for p in range((num -1) +1, num+len(self.players)):
            curPlayer = self.players[p%len(self.players)]
            if curPlayer.canPlay and curPlayer.num != num:
                return curPlayer
        return None


    def donePegging(self):
        return sum(len(p.pegginghand) for p in self.players) == 0

