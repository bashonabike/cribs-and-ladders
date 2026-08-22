"""
Phase 4 (Optimizer/board-design subsystem) tests for
cribsandladders.ortho_path.OrthoPath and
cribsandladders.ortho_line_trace.OrthoLineTrace -- split out of
EventSetBuilder.py verbatim in the "break EventSetBuilder's 2184-line
class into smaller, independently testable units" follow-up work.

Neither class ever depended on EventSetBuilder itself: OrthoPath is a
plain data holder, and OrthoLineTrace only needs a duck-typed
`possibleEvents` (`.orthogonal_vector(...)` + `.config`) plus a fake
event with `.startHole.coords`/`.endHole.coords`/`.orthoVector`. Both
are still re-exported from `cribsandladders.EventSetBuilder` for
backward compatibility -- see the identity-check tests at the bottom.
"""
import Enums as en
from cribsandladders.ortho_path import OrthoPath
from cribsandladders.ortho_line_trace import OrthoLineTrace


def test_ortho_path_stores_all_fields_verbatim():
    event = object()
    path = OrthoPath(start=(0, 0), mid=(5, 5), end=(10, 10), incr=2, rev=True, event=event)
    assert path.start == (0, 0)
    assert path.mid == (5, 5)
    assert path.end == (10, 10)
    assert path.incr == 2
    assert path.rev is True
    assert path.event is event


class _FakeHole:
    def __init__(self, coords):
        self.coords = coords


class _FakeEvent:
    def __init__(self, start_coords, end_coords, ortho_vector=(0, 1)):
        self.startHole = _FakeHole(start_coords)
        self.endHole = _FakeHole(end_coords)
        self.orthoVector = ortho_vector


class _FakeConfig:
    maxloopyorthoeventdisplacementincrements = 4
    eventminspacing = 1.0


class _FakePossibleEvents:
    """Records the args it's called with and returns a fixed ortho vector."""

    def __init__(self, ortho_vector=(0.0, 2.0)):
        self.config = _FakeConfig()
        self.ortho_vector = ortho_vector
        self.calls = []

    def orthogonal_vector(self, start, end, dist, rev):
        self.calls.append((start, end, dist, rev))
        return self.ortho_vector


def test_ortho_line_trace_start_uses_start_hole_as_p1():
    event = _FakeEvent((0, 0), (10, 0))
    pe = _FakePossibleEvents(ortho_vector=(0.0, 4.0))

    trace = OrthoLineTrace(pe, event, incr=2, rev=False, type=en.OrthoLineTraceType.START)

    p1, p2 = trace.vector
    assert p1 == (0, 0)  # startHole.coords
    midpoint = (5.0, 0.0)
    length_divider = 2 / pe.config.maxloopyorthoeventdisplacementincrements  # 2/4 = 0.5
    expected_p2 = (midpoint[0] + pe.ortho_vector[0] * length_divider,
                   midpoint[1] + pe.ortho_vector[1] * length_divider)
    assert p2 == expected_p2


def test_ortho_line_trace_end_uses_end_hole_as_p1():
    event = _FakeEvent((0, 0), (10, 0))
    pe = _FakePossibleEvents()

    trace = OrthoLineTrace(pe, event, incr=1, rev=True, type=en.OrthoLineTraceType.END)

    p1, _ = trace.vector
    assert p1 == (10, 0)  # endHole.coords


def test_ortho_line_trace_passes_rev_and_scaled_distance_to_orthogonal_vector():
    event = _FakeEvent((0, 0), (10, 0))
    pe = _FakePossibleEvents()

    OrthoLineTrace(pe, event, incr=3, rev=True, type=en.OrthoLineTraceType.START)

    (start, end, dist, rev) = pe.calls[-1]
    assert start == (0, 0)
    assert end == (10, 0)
    assert dist == pe.config.maxloopyorthoeventdisplacementincrements * pe.config.eventminspacing
    assert rev is True


def test_ortho_line_trace_raises_for_unrecognized_type():
    event = _FakeEvent((0, 0), (10, 0))
    pe = _FakePossibleEvents()
    try:
        OrthoLineTrace(pe, event, incr=1, rev=False, type=None)
        assert False, "expected an exception for an unrecognized type"
    except Exception as e:
        assert "No ortho line trace type specified" in str(e)


def test_ortho_line_trace_equality_and_hash_use_event_rev_incr():
    event = _FakeEvent((0, 0), (10, 0))
    pe = _FakePossibleEvents()
    t1 = OrthoLineTrace(pe, event, incr=1, rev=False, type=en.OrthoLineTraceType.START)
    t2 = OrthoLineTrace(pe, event, incr=1, rev=False, type=en.OrthoLineTraceType.START)
    t3 = OrthoLineTrace(pe, event, incr=2, rev=False, type=en.OrthoLineTraceType.START)

    assert t1 == t2
    assert hash(t1) == hash(t2)
    assert t1 != t3
    assert t1 < t3  # same event/rev, lower incr sorts first


def test_ortho_line_trace_not_equal_to_other_types():
    event = _FakeEvent((0, 0), (10, 0))
    pe = _FakePossibleEvents()
    trace = OrthoLineTrace(pe, event, incr=1, rev=False, type=en.OrthoLineTraceType.START)
    assert (trace == "not a trace") is False


# ---------------------------------------------------------------------
# Backward-compat re-export from EventSetBuilder.py
# ---------------------------------------------------------------------

def test_eventsetbuilder_reexports_same_ortho_path_class():
    import cribsandladders.EventSetBuilder as esb
    assert esb.OrthoPath is OrthoPath


def test_eventsetbuilder_reexports_same_ortho_line_trace_class():
    import cribsandladders.EventSetBuilder as esb
    assert esb.OrthoLineTrace is OrthoLineTrace
