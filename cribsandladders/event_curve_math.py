"""
Pure curve/geometry math split out of EventSetBuilder.py (Phase 4
decomposition follow-up -- "break EventSetBuilder's 2184-line class
into smaller, independently testable units" per the refactor plan).

Everything in this module is a plain function of its arguments: no
`self.board`/`self.possibleEvents` state, no sqlite, no matplotlib.
`get_normalized_ideal_curve` is the one exception that still touches
the filesystem (it parses an SVG curve file via
`cribsandladders.BaseLayout.svgParserHoles`), but it takes the
filename as a parameter rather than reading it off `self.config`, so
callers can point it at a temp fixture file.

`EventSetBuilder` keeps thin wrapper methods
(`actualizeCurve`/`discretizeCurve`/etc.) that delegate to these
functions, so existing call sites -- both production code and
test_eventsetbuilder.py/test_evaluator.py's `FakeEventSetBuilder`
stand-ins -- keep working unchanged.
"""
import bisect as bsc

import numpy as np

import cribsandladders.BaseLayout as bse


def get_normalized_ideal_curve(curve_file):
    """
    Load and normalize a curve from an SVG file.

    Args:
        curve_file: Path to the SVG file containing the curve.

    Returns:
        List of normalized (x, y) coordinates representing the curve.
    """
    rawCurve = np.array(bse.svgParserHoles(curve_file, returnRawCoords=True))
    # Extract all x and y values
    x_values = [coord[0] for coord in rawCurve]
    y_values = [coord[1] for coord in rawCurve]

    # Find the min & maximum x and y values
    min_x = min(x_values)
    min_y = min(y_values)
    max_x = max(x_values)
    max_y = max(y_values)
    scale_x = max_x - min_x
    scale_y = max_y - min_y

    # Normalize the coordinates
    normCurve = [((x - min_x) / scale_x, (y - min_y) / scale_y) for x, y in rawCurve]
    normCurve.sort(key=lambda e: e[0])

    return normCurve


def integrate_and_normalize_curve(curve, normalizer):
    """
    Integrate and normalize a curve.

    Args:
        curve: List of (x, y) points representing the curve.
        normalizer: Value to normalize the y-values by.

    Returns:
        Integrated and normalized curve as a list of (x, y) points.
    """
    integrated_curve = []
    for i in range(0, len(curve)):
        normx = curve[i][0]
        if i == 0:
            normy = curve[i][1] / normalizer
        else:
            normy = integrated_curve[i - 1][1] + curve[i][1] / normalizer
        integrated_curve.append((normx, normy))

    return integrated_curve


def normalize_curve_magnitude(curve):
    """
    Normalize the magnitude of a curve.

    Args:
        curve: List of (x, y) points representing the curve.

    Returns:
        Curve with y-values normalized to the range [-1, 1].
    """
    normalizer = max(abs(max([c[1] for c in curve])), abs(min([c[1] for c in curve])))
    if normalizer > 0:
        normalized_curve = []
        for i in range(0, len(curve)):
            normy = curve[i][1] / normalizer
            normalized_curve.append((curve[i][0], normy))

        return normalized_curve
    return curve


def actualize_curve(curve, x_actualizer, y_actualizer, integrate=False):
    """
    Scale a curve by given x and y factors, with optional integration.

    Args:
        curve: List of (x, y) points representing the curve.
        x_actualizer: Scaling factor for x-values.
        y_actualizer: Scaling factor for y-values.
        integrate: If True, accumulate y-values during scaling.

    Returns:
        Scaled curve as a list of (x, y) points.
    """
    actualized_curve = []
    for i in range(0, len(curve)):
        actx = curve[i][0] * x_actualizer
        if integrate and i > 0:
            acty = (curve[i - 1][1] + curve[i][1]) * y_actualizer
        else:
            acty = curve[i][1] * y_actualizer
        actualized_curve.append((actx, acty))

    return actualized_curve


def discretize_curve(curve, numBuckets, accumulate=False):
    """
    Convert a continuous curve into discrete buckets.

    Args:
        curve: List of (x, y) points representing the curve.
        numBuckets: Number of discrete buckets to create.
        accumulate: If True, accumulate y-values in each bucket.

    Returns:
        Discretized curve as a list of (bucket, value) pairs.
    """
    # If accumulating, NORMALIZE AFTER!!!
    # TODO(liam): PRE-EXISTING BUG, caught by
    # tests/test_event_curve_math.py::test_discretize_curve_accumulating_is_lagged_by_one_bucket.
    # The `while curveIdx < i * discFactor` boundary check below advances
    # curveIdx up to (but not including) the start of bucket i's own
    # span, so accum_y for bucket i only ever picks up whatever's left
    # over from the *previous* bucket's advancement -- the accumulation
    # is lagged by one bucket. Concretely: bucket 0 always comes out
    # 0.0 regardless of the curve's actual values, and every later
    # bucket reflects the curve one bucket behind where a straightforward
    # reading of this loop would expect. Not fixed yet; see the test for
    # the full characterization.
    discretized_curve = []
    curveIdx = 0
    discFactor = len(curve) / numBuckets
    for i in range(0, numBuckets):
        discx = i + 1
        accum_y = 0.0
        while curveIdx < len(curve) - 1 and curveIdx < i * discFactor:
            if accumulate:
                accum_y += curve[curveIdx][1]
            curveIdx += 1
        if accumulate:
            if i == 0:
                discy = accum_y
            else:
                discy = 0.7 * accum_y + 0.3 * discretized_curve[i - 1][1]
        else:
            discy = curve[curveIdx][1]
        discretized_curve.append((discx, discy))

    return discretized_curve


def get_points_in_proximity(searchRange, searchPoints, inputPoint):
    """
    Find points within a specified range of an input point.

    Args:
        searchRange: Tuple of (min, max) distances to search within.
        searchPoints: List of points to search through.
        inputPoint: The reference point for distance calculations.

    Returns:
        List of dictionaries containing points and their distances from the input point.
    """
    # NOTE: searchPoints MUST BE SORTED!!
    pointPairs = []
    for p in searchPoints:
        if searchRange[0] <= (inputPoint - p) <= searchRange[1]:
            pointPairs.append(dict(point=p, disp=inputPoint - p))

    return pointPairs


def try_get_disp_allowance(dispAllowances, proxPoint):
    """
    Try to get displacement allowance for a given proximity point.

    Args:
        dispAllowances: List of displacement allowances.
        proxPoint: Dictionary containing displacement information.

    Returns:
        Dictionary with effect and modification values if found, or default values.
    """
    allowance = next((allow for allow in dispAllowances
                      if allow['scalardisp'] == abs(proxPoint['disp'])), None)
    if allowance is not None:
        return dict(effect=allowance['isallowed'], mod=allowance['mod'])
    return dict(effect=False, mod=0)


def search_ordered_list_for_val(orderedList, val):
    """
    Search for a value in a sorted list using binary search.

    Args:
        orderedList: Sorted list to search in.
        val: Value to search for.

    Returns:
        Index of the value if found, -1 otherwise.
    """
    idx = bsc.bisect_left(orderedList, val)
    if idx < len(orderedList) and orderedList[idx] == val: return idx
    return -1


def index_start_of_each_hole_in_cands(holes, trackEventOverview):
    """
    Creates a lookup table mapping hole numbers to their starting indices in the candidate events list.

    This method builds an index that allows for efficient lookup of where events for each hole
    begin in the candidate events list, which is used to speed up event selection during board generation.

    Args:
        holes: List of hole objects on the track.
        trackEventOverview: List of all candidate events for the track, sorted by hole number.

    Note:
        Modifies the trackEventOverview dictionary in-place to add a 'candeventstartlookup' key
        containing the index mapping.

    TODO(liam): flagged by
    tests/test_event_curve_math.py::test_index_start_of_each_hole_in_cands_builds_lookup_in_place.
    This function has zero call sites anywhere in the repo (confirmed by
    grep) -- it's dead code as of this writing. Its own body also
    implies `trackEventOverview` must be a `dict` keyed by both
    sequential ints (0..n-1, read via `trackEventOverview[candEventCursor]`)
    *and* the string 'candeventstartlookup' this function adds at the
    end -- an inferred contract, not a documented one, since there's no
    real caller to confirm it against. Worth confirming intent (revive
    the call site, or delete this) before relying on it.
    """
    candEventCursor = 0
    candEventCursorStartLookups = [-1] * len(holes)

    for h in holes:
        while trackEventOverview[candEventCursor]['eventtop'] < h.num:
            candEventCursor += 1
            if candEventCursor >= len(trackEventOverview): break
        if candEventCursor >= len(trackEventOverview): break

        if trackEventOverview[candEventCursor]['eventtop'] == h.num:
            candEventCursorStartLookups[h.num - 1] = candEventCursor

    trackEventOverview['candeventstartlookup'] = candEventCursorStartLookups


def recalc_track_completion_pcts(trackEventsOverview, maxitertrackstalled):
    """
    Recalculates completion percentages and stall status for all tracks.

    This method updates the completion statistics for each track, including:
    - Whether the track is stalled (no progress made in recent iterations)
    - The percentage of holes processed
    - The percentage of target events placed

    Args:
        trackEventsOverview: list of `EventSetBuilder.TrackBuildState`
            instances (Refactor Mk II Phase 8 step 2 -- previously a
            list of plain dicts with the same keys as attributes).
        maxitertrackstalled: threshold (from `config.maxitertrackstalled`)
            above which a track's stalled-iteration counter marks it as stalled.

    Returns:
        tuple: A tuple containing:
            - Average hole completion percentage across all viable tracks
            - Average chute completion percentage across all viable tracks
    """
    for t in trackEventsOverview:
        t.trackisstalled = t.trackstalledcounter > maxitertrackstalled
        t.holescompletepct = t.curhole / len(t.track.trackholes)
        t.chutescompletepct = len(t.eventsetbuild) / t.optevents
    viableTracks = [e for e in trackEventsOverview if not e.trackisstalled]

    avgHolePct = sum([t.holescompletepct for t in viableTracks]) / len(viableTracks)
    avgChutesPct = sum([t.chutescompletepct for t in viableTracks]) / len(viableTracks)
    return avgHolePct, avgChutesPct


def ortho_bounding_box(vector, ortho_dxdy):
    """
    Create an orthogonal bounding box around a vector.

    Args:
        vector: A tuple of two points defining the vector.
        ortho_dxdy: The (dx, dy) orthogonal offset to apply at half the
            configured event spacing (i.e. what
            `possibleEvents.orthogonal_vector(vector[0], vector[1],
            config.eventminspacing / 2.0, False)` returns) -- computed
            by the caller since it needs `possibleEvents`/`config`, not
            just this pure function's own inputs.

    Returns:
        A tuple of four points defining the corners of the bounding box.
    """
    revOrtho_dxdy = [(-1) * d for d in ortho_dxdy]
    corners = []
    for o in [ortho_dxdy, revOrtho_dxdy]:
        for v in vector:
            corners.append(tuple([c + co for c, co in zip(v, o)]))
    # NOTE: we zigzagged in teh nested iterators, unzigging here
    corners = [corners[0], corners[1], corners[3], corners[2]]
    return tuple(corners)


def bounding_box_plus_vector(vector, ortho_dxdy):
    """
    Create a bounding box around a vector with additional space.

    Args:
        vector: A tuple of two points defining the vector.
        ortho_dxdy: see `ortho_bounding_box`.

    Returns:
        A tuple containing the original vector and the four sides of the bounding box.
    """
    intersects = [vector]
    corners = ortho_bounding_box(vector, ortho_dxdy)
    for i in range(0, 4):
        intersects.append((corners[i], corners[(i + 1) % 4]))
    return tuple(intersects)


def build_track_dict_from_benchmark_moves_df(benchmark_moves_df):
    """
    Turn a `BenchmarkMoves` DataFrame (columns Track_ID, Trial, MoveNum,
    MoveVal, plus Board_ID) into `{track_id: [[move_val, ...], ...]}` --
    one list of move values per trial, ordered by MoveNum.

    Split out of `EventSetBuilder.retrieveOrGenerateBenchmarkMoves` so
    this DataFrame-to-dict hydration is unit-testable with a small
    hand-built DataFrame instead of a real benchmark-moves db (mirrors
    the `hydrate_candidate_events_from_dataframe` /
    `hydrate_tracks_from_dataframes` precedent from earlier in Phase
    3/4).
    """
    from collections import defaultdict

    track_dict = defaultdict(list)
    grouped = benchmark_moves_df.groupby(['Track_ID', 'Trial'])
    for (track_id, trial), group in grouped:
        moves_list = [(row['MoveNum'], row['MoveVal']) for _, row in group.iterrows()]
        moves_list.sort(key=lambda x: x[0])
        trial_list = [move_val for _, move_val in moves_list]
        track_dict[track_id].append(trial_list)

    return dict(track_dict)
