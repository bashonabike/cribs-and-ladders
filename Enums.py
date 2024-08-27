from enum import Enum

# class syntax

class Event(Enum):

    NONE = 0
    CHUTE = 1
    LADDER = 2

class OrthoLineTraceType(Enum):

    START = 0
    END = 2

class InstanceEventType(Enum):

    CHUTEONLY = 0
    LADDERONLY = 1
    CHUTEANDLADDER = 2

# functional syntax

# Color = Enum('Color', ['RED', 'GREEN', 'BLUE'])