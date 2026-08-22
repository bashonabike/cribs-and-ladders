"""
Split out of EventSetBuilder.py (Phase 4 "decompose the 2600+ line god
class" follow-up). OrthoPath had no dependency on EventSetBuilder at all
-- it's a plain data holder -- so it moves verbatim into its own module
for independent import/testability. EventSetBuilder.py keeps
`from cribsandladders.ortho_path import OrthoPath` so existing call
sites (`from cribsandladders.EventSetBuilder import OrthoPath`, if any)
and this module's own re-export both keep working.
"""


class OrthoPath:
    """
    Represents an orthogonal path between points on the game board.

    Used for generating and managing orthogonal event paths such as ladders and chutes.

    Attributes:
        start: Starting coordinates of the path.
        mid: Midpoint coordinates of the path.
        end: Ending coordinates of the path.
        incr: Increment value for path generation.
        rev: Boolean indicating if the path is reversed.
        event: The event associated with this path.
    """

    def __init__(self, start, mid, end, incr, rev, event):
        """
        Initialize an OrthoPath with start, middle, and end points.

        Args:
            start: Starting coordinates (x, y) of the path.
            mid: Midpoint coordinates (x, y) of the path.
            end: Ending coordinates (x, y) of the path.
            incr: Increment value used in path generation.
            rev: Boolean indicating if the path is reversed.
            event: The event object associated with this path.
        """
        self.start = start
        self.mid = mid
        self.end = end

        self.incr = incr
        self.rev = rev
        self.event = event
