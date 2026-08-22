"""
Phase 4 (Optimizer/board-design subsystem) tests for
cribsandladders.event_curve_math -- the pure curve/geometry helpers
split out of EventSetBuilder.py in the "break EventSetBuilder's
2184-line class into smaller, independently testable units" follow-up
work called for by the refactor plan.

Every function here is a plain function of its arguments: no
`self.board`/`self.possibleEvents` state, no sqlite, no matplotlib.
`get_normalized_ideal_curve` is the one function that still touches
the filesystem indirectly (via `BaseLayout.svgParserHoles`), so -- same
trick as test_base_layout.py -- it's exercised with an `io.StringIO`
SVG instead of a fixture file on disk, since `xml.dom.minidom.parse`
accepts any file-like object.

`EventSetBuilder` itself keeps thin wrapper methods
(`actualizeCurve`/`discretizeCurve`/etc.) delegating to these
functions -- covered indirectly by test_eventsetbuilder.py, which
doesn't re-test the math itself, just that the delegation happens.
"""
import io

import cribsandladders.event_curve_math as ecm


def _svg(*paths, height=100, width=200):
    path_tags = "\n".join('<path d="{}" />'.format(d) for d in paths)
    return io.StringIO(
        '<svg xmlns="http://www.w3.org/2000/svg" height="{}mm" width="{}mm">\n{}\n</svg>'.format(
            height, width, path_tags
        )
    )


# ---------------------------------------------------------------------
# get_normalized_ideal_curve
# ---------------------------------------------------------------------

def test_get_normalized_ideal_curve_scales_to_unit_range_and_sorts_by_x():
    # Raw coords (after BaseLayout's y-flip) will be (10, 80), (30, 60), (50, 40).
    svg = _svg("m 10,20 5,0 5,0", "m 30,40 5,0 5,0", "m 50,60 5,0 5,0", height=100)
    curve = ecm.get_normalized_ideal_curve(svg)

    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    assert xs == sorted(xs)
    assert min(xs) == 0.0 and max(xs) == 1.0
    assert min(ys) == 0.0 and max(ys) == 1.0


# ---------------------------------------------------------------------
# integrate_and_normalize_curve
# ---------------------------------------------------------------------

def test_integrate_and_normalize_curve_accumulates_and_divides_by_normalizer():
    curve = [(0, 2), (1, 2), (2, 4)]
    result = ecm.integrate_and_normalize_curve(curve, normalizer=8)
    assert result == [(0, 0.25), (1, 0.5), (2, 1.0)]


# ---------------------------------------------------------------------
# normalize_curve_magnitude
# ---------------------------------------------------------------------

def test_normalize_curve_magnitude_scales_by_largest_absolute_y():
    curve = [(0, -4), (1, 2), (2, 4)]
    result = ecm.normalize_curve_magnitude(curve)
    assert result == [(0, -1.0), (1, 0.5), (2, 1.0)]


def test_normalize_curve_magnitude_returns_curve_unchanged_when_all_zero():
    curve = [(0, 0), (1, 0)]
    assert ecm.normalize_curve_magnitude(curve) == curve


# ---------------------------------------------------------------------
# actualize_curve
# ---------------------------------------------------------------------

def test_actualize_curve_scales_x_and_y_independently():
    curve = [(1, 2), (2, 4)]
    result = ecm.actualize_curve(curve, x_actualizer=10, y_actualizer=0.5)
    assert result == [(10, 1.0), (20, 2.0)]


def test_actualize_curve_integrate_sums_adjacent_points():
    curve = [(0, 1), (1, 2), (2, 3)]
    result = ecm.actualize_curve(curve, x_actualizer=1, y_actualizer=1, integrate=True)
    # First point untouched by the integrate branch (i==0), rest are
    # (prev_y + cur_y) * y_actualizer.
    assert result == [(0, 1), (1, 3), (2, 5)]


# ---------------------------------------------------------------------
# discretize_curve
# ---------------------------------------------------------------------

def test_discretize_curve_non_accumulating_samples_one_point_per_bucket():
    curve = [(0, 1), (1, 2), (2, 3), (3, 4)]
    result = ecm.discretize_curve(curve, numBuckets=2)
    assert [b for b, _ in result] == [1, 2]


def test_discretize_curve_accumulating_is_lagged_by_one_bucket():
    # Characterization test: the accumulation loop condition
    # (`curveIdx < i * discFactor`) means bucket i actually accumulates
    # over the *previous* bucket's curve span -- bucket 0 always ends up
    # 0.0 regardless of curve contents, and bucket 1 picks up what you'd
    # naively expect bucket 0 to hold. Not "fixed" here since this is
    # pre-existing behavior unrelated to the Phase 4 decomposition --
    # just pinned down so it doesn't silently change.
    curve = [(0, 1), (1, 1), (2, 1), (3, 1)]
    result = ecm.discretize_curve(curve, numBuckets=2, accumulate=True)
    assert result[0][1] == 0.0
    assert result[1][1] == 1.4  # 0.7 * 2 + 0.3 * 0


# ---------------------------------------------------------------------
# get_points_in_proximity / try_get_disp_allowance
# ---------------------------------------------------------------------

def test_get_points_in_proximity_filters_by_signed_displacement_range():
    result = ecm.get_points_in_proximity((0, 5), [1, 3, 8, 12], inputPoint=10)
    # disp = inputPoint - p; only keep 0 <= disp <= 5 -- only point 8
    # (disp=2) qualifies; 12 gives disp=-2, outside [0, 5].
    assert result == [dict(point=8, disp=2)]


def test_get_points_in_proximity_excludes_out_of_range_points():
    # disp = inputPoint - p for each candidate: 9, 7, -10 -- none fall
    # in [0, 2].
    result = ecm.get_points_in_proximity((0, 2), [1, 3, 20], inputPoint=10)
    assert result == []


def test_try_get_disp_allowance_returns_matching_effect_and_mod():
    allowances = [dict(scalardisp=3, isallowed=True, mod=0.5)]
    result = ecm.try_get_disp_allowance(allowances, dict(disp=-3))
    assert result == dict(effect=True, mod=0.5)


def test_try_get_disp_allowance_defaults_when_no_match():
    result = ecm.try_get_disp_allowance([], dict(disp=5))
    assert result == dict(effect=False, mod=0)


# ---------------------------------------------------------------------
# search_ordered_list_for_val
# ---------------------------------------------------------------------

def test_search_ordered_list_for_val_finds_index():
    assert ecm.search_ordered_list_for_val([1, 3, 5, 7], 5) == 2


def test_search_ordered_list_for_val_returns_negative_one_when_absent():
    assert ecm.search_ordered_list_for_val([1, 3, 5, 7], 4) == -1


# ---------------------------------------------------------------------
# index_start_of_each_hole_in_cands
# ---------------------------------------------------------------------

class _Hole:
    def __init__(self, num):
        self.num = num


def test_index_start_of_each_hole_in_cands_builds_lookup_in_place():
    # NOTE: trackEventOverview is indexed both by int (candidate-event
    # position) and by the string key 'candeventstartlookup' that this
    # function adds -- so it has to be a dict (with int keys 0..n-1),
    # not a list, for the final assignment to even be legal. This
    # function has no call sites anywhere in the repo (grep confirms
    # it), so this dict-shaped contract is inferred from the body
    # rather than from any real caller.
    holes = [_Hole(1), _Hole(2), _Hole(3)]
    track_event_overview = {
        0: dict(eventtop=1),
        1: dict(eventtop=1),
        2: dict(eventtop=3),
    }
    ecm.index_start_of_each_hole_in_cands(holes, track_event_overview)
    lookup = track_event_overview['candeventstartlookup']
    assert lookup[0] == 0  # hole 1 -> first index with eventtop == 1
    assert lookup[1] == -1  # hole 2 has no matching eventtop
    assert lookup[2] == 2  # hole 3 -> index 2


# ---------------------------------------------------------------------
# recalc_track_completion_pcts
# ---------------------------------------------------------------------

class _FakeTrack:
    def __init__(self, num_holes):
        self.trackholes = [None] * num_holes


def test_recalc_track_completion_pcts_averages_over_viable_tracks():
    # Refactor Mk II Phase 8 step 2: trackEventsOverview elements are now
    # real TrackBuildState instances (attribute access), not plain dicts
    # -- this test used to build `dict(...)` fixtures directly; updated
    # to build TrackBuildState instances instead (recalc_track_completion_pcts
    # itself is unchanged in behavior, just how its input is shaped).
    from cribsandladders.EventSetBuilder import TrackBuildState

    overview = [
        TrackBuildState(track=_FakeTrack(10), trackidx=0, tracknum=1,
                        trackstalledcounter=0, curhole=5, eventsetbuild=[1, 2], optevents=4),
        TrackBuildState(track=_FakeTrack(10), trackidx=1, tracknum=2,
                        trackstalledcounter=100, curhole=1, eventsetbuild=[], optevents=4),
    ]
    avg_hole_pct, avg_chutes_pct = ecm.recalc_track_completion_pcts(overview, maxitertrackstalled=50)

    # Second track is stalled (100 > 50) and excluded from the average.
    assert overview[0].trackisstalled is False
    assert overview[1].trackisstalled is True
    assert avg_hole_pct == 0.5  # 5/10 for the one viable track
    assert avg_chutes_pct == 0.5  # 2/4 for the one viable track


# ---------------------------------------------------------------------
# ortho_bounding_box / bounding_box_plus_vector
# ---------------------------------------------------------------------

def test_ortho_bounding_box_offsets_both_endpoints_each_direction():
    vector = ((0, 0), (10, 0))
    ortho_dxdy = (0, 1)  # straight up
    corners = ecm.ortho_bounding_box(vector, ortho_dxdy)
    assert corners == ((0, 1), (10, 1), (10, -1), (0, -1))


def test_bounding_box_plus_vector_includes_original_vector_plus_four_sides():
    vector = ((0, 0), (10, 0))
    ortho_dxdy = (0, 1)
    result = ecm.bounding_box_plus_vector(vector, ortho_dxdy)
    assert len(result) == 5
    assert result[0] == vector


# ---------------------------------------------------------------------
# build_track_dict_from_benchmark_moves_df
# ---------------------------------------------------------------------

def test_build_track_dict_from_benchmark_moves_df_groups_by_track_and_orders_by_move_num():
    import pandas as pd

    df = pd.DataFrame.from_records([
        dict(Track_ID=1, Trial=0, MoveNum=1, MoveVal=5),
        dict(Track_ID=1, Trial=0, MoveNum=0, MoveVal=3),
        dict(Track_ID=1, Trial=1, MoveNum=0, MoveVal=7),
        dict(Track_ID=2, Trial=0, MoveNum=0, MoveVal=9),
    ])
    result = ecm.build_track_dict_from_benchmark_moves_df(df)

    assert result[1] == [[3, 5], [7]]
    assert result[2] == [[9]]
