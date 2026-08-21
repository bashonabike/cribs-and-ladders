"""
Phase 3 (Board/geometry) tests for cribsandladders.DXFWriter.

The pure coordinate/vector math (convert_mm_to_in, euclidean_distance,
midpoint, remove_close_coordinates, adjust_close_points,
searchOrderedListForVal, rotate_vector_2d, compute_offset_curve,
create_progress_marker_vectors) now lives in cribsandladders.dxf_geometry
-- extracted out of this module specifically because DXFWriter.py does
`import ezdxf` at module scope, which would otherwise force every one
of these numpy-only unit tests to require ezdxf installed just to
import the module under test. They're imported back into DXFWriter
unchanged, so testing them via cribsandladders.dxf_geometry directly
also covers what DXFWriter re-exports.

buildDXFFile itself is the I/O boundary (ezdxf document + file write) --
covered by one @pytest.mark.integration test below that writes to
temp_output_dir (via the new `output_dir` parameter, which replaces the
old hardcoded "Boards/" prefix) and then reads the DXF back with
ezdxf.readfile to assert on its structure: layer names and entity
counts. That's the "snapshot-test the DXF output at the I/O boundary"
called for in the Phase 3 plan -- a structural snapshot rather than a
byte-for-byte one, since the file name embeds a timestamp and layer
colors are randomized (see DXFWriter.buildDXFFile's use of
random.randint for layer color).
"""
import numpy as np
import pytest

import cribsandladders.dxf_geometry as dg


# ---------------------------------------------------------------------
# convert_mm_to_in / euclidean_distance / midpoint
# ---------------------------------------------------------------------

def test_convert_mm_to_in():
    assert dg.convert_mm_to_in([(25.4, 50.8), (0, 0)]) == [(1.0, 2.0), (0.0, 0.0)]


def test_euclidean_distance_3_4_5_triangle():
    assert dg.euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)


def test_midpoint():
    assert dg.midpoint((0, 0), (4, 6)) == (2.0, 3.0)


# ---------------------------------------------------------------------
# remove_close_coordinates / adjust_close_points
# ---------------------------------------------------------------------

def test_remove_close_coordinates_drops_points_within_threshold():
    result = dg.remove_close_coordinates([(0, 0), (0.05, 0.05), (5, 5)], [(0, 0)], threshold=0.1)
    assert result == [(5, 5)]


def test_remove_close_coordinates_keeps_points_outside_threshold():
    result = dg.remove_close_coordinates([(5, 5)], [(0, 0)], threshold=0.1)
    assert result == [(5, 5)]


def test_adjust_close_points_merges_nearby_points_to_midpoint():
    result = dg.adjust_close_points([(0, 0), (0.02, 0.02)], threshold=0.1)
    mid = dg.midpoint((0, 0), (0.02, 0.02))
    assert result[0] == mid
    assert result[1] == mid


def test_adjust_close_points_leaves_distant_points_unchanged():
    result = dg.adjust_close_points([(0, 0), (10, 10)], threshold=0.1)
    assert result == [(0, 0), (10, 10)]


# ---------------------------------------------------------------------
# searchOrderedListForVal
# ---------------------------------------------------------------------

def test_search_ordered_list_for_val_found():
    assert dg.searchOrderedListForVal([1, 3, 5, 7], 5) == 2


def test_search_ordered_list_for_val_not_found():
    assert dg.searchOrderedListForVal([1, 3, 5, 7], 4) == -1


def test_search_ordered_list_for_val_empty_list():
    assert dg.searchOrderedListForVal([], 1) == -1


# ---------------------------------------------------------------------
# rotate_vector_2d
# ---------------------------------------------------------------------

def test_rotate_vector_2d_ninety_degrees_ccw():
    rotated = dg.rotate_vector_2d(np.array([1, 0]), 90)
    assert rotated[0] == pytest.approx(0.0, abs=1e-9)
    assert rotated[1] == pytest.approx(1.0, abs=1e-9)


def test_rotate_vector_2d_zero_degrees_is_identity():
    rotated = dg.rotate_vector_2d(np.array([3, 4]), 0)
    assert rotated[0] == pytest.approx(3.0)
    assert rotated[1] == pytest.approx(4.0)


# ---------------------------------------------------------------------
# create_progress_marker_vectors / compute_offset_curve
# ---------------------------------------------------------------------

def _straight_line_holes(n=10, spacing=1.0):
    return np.array([[i * spacing, 0.0] for i in range(n)])


def test_create_progress_marker_vectors_returns_one_vector_per_five_holes():
    holes = _straight_line_holes(n=12)
    markers = dg.create_progress_marker_vectors(holes, length=0.5)
    # range(4, 12, 5) -> indices 4, 9 -> 2 markers
    assert len(markers) == 2
    for left, right in markers:
        assert len(left) == 2 and len(right) == 2


def test_create_progress_marker_vectors_empty_for_short_hole_list():
    holes = _straight_line_holes(n=3)
    assert dg.create_progress_marker_vectors(holes, length=0.5) == []


def test_compute_offset_curve_produces_parallel_offsets_for_straight_line():
    holes = _straight_line_holes(n=10)
    markers = dg.create_progress_marker_vectors(holes, length=0.24)
    left, right, arrows = dg.compute_offset_curve(holes, [], markers, offset_distance=0.12, proximityThresh=0.001)

    assert len(left) > 0
    assert len(right) > 0
    # For a straight horizontal line, the offset curves should sit at a
    # constant +/- y offset from the line (y=0).
    left_ys = {round(p[1], 6) for p in left}
    right_ys = {round(p[1], 6) for p in right}
    assert left_ys == {0.12}
    assert right_ys == {-0.12}


def test_compute_offset_curve_returns_empty_arrows_when_holes_have_events():
    # When every hole is flagged as having an event, the arrow-skip
    # check (searchOrderedListForVal) should suppress every arrow.
    holes = _straight_line_holes(n=10)
    markers = dg.create_progress_marker_vectors(holes, length=0.24)
    all_hole_positions = list(range(1, len(holes) + 1))
    _, _, arrows = dg.compute_offset_curve(holes, all_hole_positions, markers, 0.12, 0.001)
    assert arrows == []


# ---------------------------------------------------------------------
# buildDXFFile -- integration, I/O boundary
# ---------------------------------------------------------------------

class _FakeHole:
    def __init__(self, x, y, num):
        self.coords = (x, y)
        self.num = num


class _FakeTrack:
    def __init__(self, track_id, num, n_holes=12, spacing_mm=5.0):
        self.Track_ID = track_id
        self.num = num
        self.trackholes = [_FakeHole(i * spacing_mm, 0.0, i + 1) for i in range(n_holes)]
        self.eventSetBuild = []  # no chutes/ladders -- keeps the fixture minimal


class _FakeBoard:
    def __init__(self):
        self.boardName = "PhaseThreeTestBoard"
        self.width = 100.0
        self.height = 50.0
        self.tracks = [_FakeTrack(track_id=1, num=1)]


@pytest.mark.integration
def test_build_dxf_file_writes_expected_layers_and_entities(temp_output_dir):
    ezdxf = pytest.importorskip("ezdxf")

    board = _FakeBoard()
    from cribsandladders.DXFWriter import buildDXFFile

    buildDXFFile(board, output_dir=str(temp_output_dir))

    produced = list((temp_output_dir / board.boardName).glob("*.dxf"))
    assert len(produced) == 1

    doc = ezdxf.readfile(str(produced[0]))
    layer_names = {layer.dxf.name for layer in doc.layers}

    # One track (Track_ID=1) plus the shared finish-hole/arrow layers.
    assert "Holes_T1" in layer_names
    assert "NumMarks_T1" in layer_names
    assert "TrackPath_T1" in layer_names
    assert "Holes_Finish" in layer_names
    assert "TrackPath_ALL" in layer_names
    assert "NormEvents_T1" in layer_names
    assert "RampUpEvents_T1" in layer_names
    assert "RampDownEvents_T1" in layer_names

    msp = doc.modelspace()
    circles = list(msp.query("CIRCLE"))
    splines = list(msp.query("SPLINE"))

    # 12 track holes + 2 starter holes (per track) + 1 shared finish hole.
    assert len(circles) == 12 + 2 + 1
    # Left/right track-path splines + the closed starter-circle spline.
    assert len(splines) == 3
