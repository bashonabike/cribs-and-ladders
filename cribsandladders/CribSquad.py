import game_params as gp
import cribsandladders.Player as pl
import os
import random as r

class CribSquad:
    def __init__(self, rankLookupTable, tracksUsed = None):
        self.players = []
        self.tracksUsed = tracksUsed
        if self.tracksUsed is None or len(self.tracksUsed) != gp.numplayers:
            self.tracksUsed = [-1]*gp.numplayers
        for i in range(0,gp.numplayers):
            (self.players.append(pl.Player(r.randint(1,21), i + 1, rankLookupTable, self.tracksUsed[i])))

    def resetRisks(self):
        for player in self.players:
            player.risk = r.randint(1,21)

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

