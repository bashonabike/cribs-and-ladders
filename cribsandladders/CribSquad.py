import cribsandladders.Player as pl
from cribsandladders.config import GameConfig, DEFAULT_CONFIG
import random as r


class CribSquad:
    """
    Manages a collection of players within a single game session.

    This class handles player initialization, track assignment, risk management,
    and turn-taking logic for the pegging phase.
    """

    def __init__(self, rankLookupTable, tracks, tracksUsed=None, homoRisk=False, config: GameConfig = DEFAULT_CONFIG,
                 rng=None, move_selector=None):
        """
        Initializes the squad and populates it with Player instances.

        Args:
            rankLookupTable (dict): Pre-calculated hand rankings for strategy.
            tracks (list): List of available Track objects from the Board.
            tracksUsed (list, optional): Specific track numbers to assign to players.
            homoRisk (bool): If True, all players receive the same baseline risk value (11).
                             If False, risks are randomized per player.
            config (GameConfig): game configuration (defaults to the module-level DEFAULT_CONFIG)
            rng: object exposing .randint(a, b), used for risk randomization.
                Defaults to the global random module. Inject a
                random.Random(seed) for deterministic tests.
            move_selector: passed through to each constructed Player (see
                Player.__init__) so a fake pegging move search can be
                injected without needing the compiled scoretree extension.
        """
        self.players = []
        self.tracksUsed = tracksUsed
        self.homoRisk = homoRisk
        self.config = config
        self.rng = rng or r

        # Ensure tracks are assigned correctly to each player slot
        if self.tracksUsed is None or len(self.tracksUsed) != self.config.numplayers:
            self.tracksUsed = []
            for p in range(self.config.numplayers):
                if len(tracks) in (0, 1):
                    self.tracksUsed.append(0)
                else:
                    self.tracksUsed.append(tracks[p].num)

        # Create the player instances
        for i in range(0, self.config.numplayers):
            if homoRisk:
                risk = 11
            else:
                risk = self.rng.randint(1, 21)
            self.players.append(pl.Player(risk, i + 1, rankLookupTable, self.tracksUsed[i], config=self.config,
                                           move_selector=move_selector))

    def resetRisks(self):
        """Re-randomizes risk levels for all players (unless homoRisk is enabled)."""
        for player in self.players:
            if self.homoRisk:
                player.risk = 11
            else:
                player.risk = self.rng.randint(1, 21)

    def resetCanPlay(self):
        """Sets the 'canPlay' status to True for all players at the start of a pegging 'Go'."""
        for player in self.players:
            player.canPlay = True

    def resetWins(self):
        """Resets the win counter for every player in the squad."""
        for player in self.players:
            player.wins = 0

    def resetScores(self):
        """Resets all player positions (scores) to 0 on the board."""
        for player in self.players:
            player.score = 0

    def getPlayerByNum(self, num):
        """
        Retrieves a player object based on their designated player number.

        Args:
            num (int): The player number (typically 1 through numplayers).

        Returns:
            Player: The matching player object, or None if not found.
        """
        for player in self.players:
            if player.num == num:
                return player

    def getNextPeggingPlayer(self, num):
        """
        Finds the next player in the rotation who still has valid moves.

        Args:
            num (int): The number of the current player.

        Returns:
            Player: The next available player in rotation, or None if no one can play.
        """
        for p in range((num - 1) + 1, num + len(self.players)):
            curPlayer = self.players[p % len(self.players)]
            if curPlayer.canPlay and curPlayer.num != num:
                return curPlayer
        return None

    def donePegging(self):
        """
        Checks if the pegging phase is complete.

        Returns:
            bool: True if all cards in all players' pegging hands have been played.
        """
        return sum(len(p.pegginghand) for p in self.players) == 0