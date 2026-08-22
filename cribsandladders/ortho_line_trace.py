"""
Split out of EventSetBuilder.py (Phase 4 decomposition follow-up).
OrthoLineTrace only ever needed a `possibleEvents`-shaped object (duck
typed -- it reads `.orthogonal_vector(...)` and `.config`) plus
`Enums.OrthoLineTraceType`, so it has no real dependency on
EventSetBuilder itself and moves verbatim into its own module.
EventSetBuilder.py re-imports it so existing internal call sites
(`OrthoLineTrace(...)` inside `updateVectorsTest`) keep working
unchanged.
"""
import Enums as en


class OrthoLineTrace:
    """
    Represents a trace line for orthogonal event placement.

    Used to calculate and validate the placement of orthogonal events
    such as ladders and chutes on the game board.

    Attributes:
        event: The event this trace is associated with.
        incr: Increment value for trace generation.
        rev: Boolean indicating if the trace is reversed.
        type: Type of the orthogonal line trace.
        vector: Tuple of coordinate pairs representing the trace vector.
    """

    def __init__(self, possibleEvents, event, incr, rev, type):
        """
        Initialize an OrthoLineTrace for an event.

        Args:
            possibleEvents: Collection of possible events.
            event: The event to create a trace for.
            incr: Increment value for trace generation.
            rev: Boolean indicating if the trace is reversed.
            type: Type of the orthogonal line trace (START or END).
        """
        self.event = event
        self.incr = incr
        self.rev = rev
        self.type = type
        self.vector = ((-1, -1), (-1, -1))

        p1, p2 = (-1, -1), (-1, -1)
        midpoint = tuple([sum(c) / 2 for c in zip(self.event.startHole.coords, self.event.endHole.coords)])
        orthogonal_vector = tuple([(-1 if rev else 1) * o for o in self.event.orthoVector])
        orthogonal_vector = possibleEvents.orthogonal_vector(self.event.startHole.coords, self.event.endHole.coords,
                                                             possibleEvents.config.maxloopyorthoeventdisplacementincrements
                                                             * possibleEvents.config.eventminspacing, rev)
        length_divider = incr / possibleEvents.config.maxloopyorthoeventdisplacementincrements
        match type:
            case en.OrthoLineTraceType.START:
                p1 = self.event.startHole.coords
            case en.OrthoLineTraceType.END:
                p1 = self.event.endHole.coords
            case _:
                raise Exception("No ortho line trace type specified!")

        p2 = (midpoint[0] + orthogonal_vector[0] * length_divider,
              midpoint[1] + orthogonal_vector[1] * length_divider)

        self.vector = (p1, p2)

    def __key(self):
        return (self.event, self.rev, self.incr)

    # NOTE: do not put objects of multiple types in a set!!!

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, OrthoLineTrace):
            return self.__key() == other.__key()
        return NotImplemented

    def __lt__(self, other):
        # Define the comparison order
        if self.event != other.event:
            return self.event < other.event
        if self.rev != other.rev:
            return self.rev < other.rev
        return self.incr < other.incr
