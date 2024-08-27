class ChanceNode:
    def __init__(self, card, probability):
        self.children = []
        self.utility = 0
        self.card = card
        self.probability = probability

    def debug(self):
        print("Debugging Info for Chance Node:")
        print("Children: ", self.children)
        print("Utility: ", self.utility)
        print("Card:", self.card)

    def addChild(self, childNode):
        self.children.append(childNode)

    def getChildren(self):
        return self.children

    def getCard(self):
        return self.card

    def getUtility(self):
        return self.utility