class Node:
    def __init__(self, card, utility, hasEvent):
        self.children = []
        self.utility = utility
        self.myScore = 0
        self.opponentScore = 0
        self.likelyOpponentScoreLine=0
        self.likelyOpponentScore = 0
        self.sumFromPlay = 0
        self.card = card
        self.hasEvent = hasEvent

    def addChild(self, childNode):
        self.children.append(childNode)

    def debug(self):
        print("Debugging Info for Normal Node:")
        print("Children: ", self.children)
        print("Utility: ", self.utility)
        print("Card:", self.card)

    def getChildren(self):
        return self.children

    def getCard(self):
        return self.card

    def getSumFromPlay(self):
        return self.sumFromPlay

    def setSumFromPlay(self, sum):
        self.sumFromPlay = sum

    def getUtility(self):
        return self.utility