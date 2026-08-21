"""
Phase 3 (Board/geometry) tests for cribsandladders.BaseLayout.

BaseLayout's intersection helpers (`_ccw`, `_intersect`,
`check_intersections`, `build_interception_test_vector_set`) were
already pure -- no extraction needed, just tests. The SVG parsers
(`svgParserHoles`, `svgParserVectors`) look like they need a real file
on disk, but `xml.dom.minidom.parse` accepts any file-like object, so
passing an `io.StringIO` of inline SVG markup exercises the exact same
coordinate-transform logic without touching the filesystem. That's the
seam this file relies on instead of adding fixture files under
Boards/.

`setTrackHolesets` is covered here too, now that it takes an injected
`GameConfig` (config.findmode) instead of reading the `game_params`
global -- this is the Phase 3 config-injection migration for
BaseLayout/Board/BoardSetter called out in game_params.py's docstring.
"""
import io

import cribsandladders.BaseLayout as bl
from cribsandladders.Board import Track
from cribsandladders.config import GameConfig


def _svg(*paths, height=100, width=200):
    path_tags = "\n".join('<path d="{}" />'.format(d) for d in paths)
    return io.StringIO(
        '<svg xmlns="http://www.w3.org/2000/svg" height="{}mm" width="{}mm">\n{}\n</svg>'.format(
            height, width, path_tags
        )
    )


# ---------------------------------------------------------------------
# _ccw / _intersect
# ---------------------------------------------------------------------

def test_ccw_true_for_counterclockwise_triangle():
    assert bl._ccw((0, 0), (1, 0), (1, 1)) is True


def test_ccw_false_for_clockwise_triangle():
    assert bl._ccw((0, 0), (1, 1), (1, 0)) is False


def test_intersect_true_for_crossing_segments():
    assert bl._intersect(((0, 0), (10, 10)), ((0, 10), (10, 0))) is True


def test_intersect_false_for_parallel_segments():
    assert bl._intersect(((0, 0), (10, 0)), ((0, 5), (10, 5))) is False


def test_intersect_false_for_non_crossing_collinear_segments():
    assert bl._intersect(((0, 0), (1, 0)), ((5, 0), (6, 0))) is False


# ---------------------------------------------------------------------
# check_intersections / build_interception_test_vector_set
# ---------------------------------------------------------------------

def test_check_intersections_returns_start_point_of_first_hit():
    test_set = [((0, 0), (10, 10))]
    candidates = [((0, 10), (10, 0))]
    assert bl.check_intersections(test_set, candidates) == (0, 0)


def test_check_intersections_returns_negative_one_when_no_hit():
    test_set = [((0, 0), (10, 0))]
    candidates = [((0, 5), (10, 5))]
    assert bl.check_intersections(test_set, candidates) == -1


def test_build_interception_test_vector_set_chains_consecutive_points():
    points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    vectors = bl.build_interception_test_vector_set(points)
    assert vectors == [
        ((0, 0), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (3, 3)),
    ]


def test_build_interception_test_vector_set_empty_for_single_point():
    assert bl.build_interception_test_vector_set([(0, 0)]) == []


# ---------------------------------------------------------------------
# svgParserHoles
# ---------------------------------------------------------------------

def test_svg_parser_holes_flips_y_axis_and_numbers_sequentially():
    # Three holes moving left-to-right, so no reversal needed.
    svg = _svg("m 10,20 5,0 5,0", "m 30,40 5,0 5,0", "m 50,60 5,0 5,0", height=100)
    holes = bl.svgParserHoles(svg, boardHeight=100, tracknum=1)

    assert [h.num for h in holes] == [1, 2, 3]
    assert [h.tracknum for h in holes] == [1, 1, 1]
    # y flips: boardHeight - y
    assert holes[0].coords == (10.0, 80.0)
    assert holes[1].coords == (30.0, 60.0)
    assert holes[2].coords == (50.0, 40.0)
    assert [h.lastHole for h in holes] == [False, False, True]


def test_svg_parser_holes_reverses_when_path_runs_right_to_left():
    # First point further right than the last -> parser reverses the order.
    svg = _svg("m 50,60 5,0 5,0", "m 30,40 5,0 5,0", "m 10,20 5,0 5,0", height=100)
    holes = bl.svgParserHoles(svg, boardHeight=100, tracknum=2)

    # After reversal, hole 1 should be the leftmost x (10.0), matching the
    # left-to-right case above.
    assert holes[0].coords == (10.0, 80.0)
    assert holes[-1].coords == (50.0, 40.0)


def test_svg_parser_holes_autodetects_board_height_from_svg_element():
    svg = _svg("m 10,20 5,0 5,0", "m 30,40 5,0 5,0", height=100)
    holes = bl.svgParserHoles(svg, boardHeight=-1, tracknum=1)
    # Same as passing boardHeight=100 explicitly.
    assert holes[0].coords == (10.0, 80.0)


def test_svg_parser_holes_return_raw_coords_skips_hole_construction():
    svg = _svg("m 10,20 5,0 5,0", "m 30,40 5,0 5,0", height=100)
    coords = bl.svgParserHoles(svg, boardHeight=100, returnRawCoords=True)
    assert coords == [(10.0, 80.0), (30.0, 60.0)]


def test_hole_hash_is_based_on_coords():
    from cribsandladders.BaseLayout import Hole
    h1 = Hole(1.0, 2.0, num=1, tracknum=1)
    h2 = Hole(1.0, 2.0, num=99, tracknum=5)  # different num/track, same coords
    assert hash(h1) == hash(h2)


# ---------------------------------------------------------------------
# svgParserVectors
# ---------------------------------------------------------------------

def test_svg_parser_vectors_computes_start_and_end_with_flipped_y():
    # Vector endpoint is p1 offset by the *last* relative pair only (not
    # the sum of all pairs) -- verified against the real implementation:
    # "m 10,20 5,0 5,0" -> start (10,20), end = start + last pair (5,0).
    svg = _svg("m 10,20 5,0 5,0", height=100)
    vectors = bl.svgParserVectors(svg, boardHeight=100)
    assert vectors == [((10.0, 80.0), (15.0, 80.0))]


# ---------------------------------------------------------------------
# setTrackHolesets (config-injected findmode)
# ---------------------------------------------------------------------

def _track_with_inline_svg(svg_text, num=1):
    t = Track()
    t.num = num
    t.holesetfilepath = io.StringIO(svg_text)
    return t


_SIMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" height="100mm" width="200mm">\n'
    '<path d="m 10,20 5,0 5,0" />\n'
    '<path d="m 30,40 5,0 5,0" />\n'
    '<path d="m 50,60 5,0 5,0" />\n'
    "</svg>"
)


def test_set_track_holesets_findmode_sets_lengths_from_parsed_holes():
    track = _track_with_inline_svg(_SIMPLE_SVG)
    bl.setTrackHolesets([track], boardHeight=100, config=GameConfig(findmode=True))

    assert len(track.trackholes) == 3
    assert track.length == 3
    assert track.efflength == 3
    assert track.twodeckslength == 3
    assert track.holesetIndexer == [1, 2, 3]


def test_set_track_holesets_non_findmode_does_not_override_length():
    track = _track_with_inline_svg(_SIMPLE_SVG)
    track.length = 999  # pre-set by BoardSetter in the non-findmode path
    bl.setTrackHolesets([track], boardHeight=100, config=GameConfig(findmode=False))

    # Holes are still parsed either way...
    assert len(track.trackholes) == 3
    # ...but findmode=False must not clobber the length BoardSetter set.
    assert track.length == 999
