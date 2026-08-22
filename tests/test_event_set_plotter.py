"""
Phase 4 (Optimizer/board-design subsystem) tests for
cribsandladders.event_set_plotter -- the plotting adapter split out of
EventSetBuilder.py's plotBoard/testPlotVectorsOnHoles/
plot_coordinates_and_vectors methods (the refactor plan's "a thin
plotting adapter tests can no-op").

IMPORTANT: `EventSetPlotter`'s real methods end with
`plt.waitforbuttonpress()`, which -- confirmed by hand in this sandbox
-- blocks forever even under matplotlib's non-interactive "Agg" backend
(there's no GUI event to wait for, but the call doesn't know that and
never times out). So the `EventSetPlotter` tests below patch
`cribsandladders.event_set_plotter.plt` with a MagicMock rather than
calling the real matplotlib functions, and just assert the right calls
happen (savefig gets the right filename, waitforbuttonpress is
invoked, etc). `NoOpEventSetPlotter` needs no such patching -- that's
the whole point of it existing for tests that don't care about
plotting at all.
"""
import unittest.mock as mock

from cribsandladders.event_set_plotter import EventSetPlotter, NoOpEventSetPlotter


class _FakeHole:
    def __init__(self, coords, num=1):
        self.coords = coords
        self.num = num


class _FakeTrack:
    def __init__(self, track_id, holes):
        self.Track_ID = track_id
        self.trackholes = holes
        self.chutes = []
        self.ladders = []


class _FakeBoard:
    def __init__(self, tracks):
        self.tracks = tracks


class _FakeBuilder:
    def __init__(self, tracks):
        self.board = _FakeBoard(tracks)


def _builder_with_one_track():
    holes = [_FakeHole((0, 0), num=1), _FakeHole((1, 1), num=5), _FakeHole((2, 2), num=10)]
    return _FakeBuilder([_FakeTrack(1, holes)])


# ---------------------------------------------------------------------
# NoOpEventSetPlotter -- no matplotlib patching needed, that's the point
# ---------------------------------------------------------------------

def test_noop_plotter_plot_board_does_nothing():
    plotter = NoOpEventSetPlotter()
    plotter.plot_board(_builder_with_one_track())  # should not raise


def test_noop_plotter_test_plot_vectors_on_holes_does_nothing():
    plotter = NoOpEventSetPlotter()
    plotter.test_plot_vectors_on_holes(_builder_with_one_track(), vectors=[((0, 0), (1, 1))])


def test_noop_plotter_plot_coordinates_and_vectors_does_nothing():
    plotter = NoOpEventSetPlotter()
    plotter.plot_coordinates_and_vectors(_builder_with_one_track())


# ---------------------------------------------------------------------
# EventSetPlotter -- real methods, but with matplotlib mocked out so
# nothing actually blocks on waitforbuttonpress or opens a window.
# ---------------------------------------------------------------------

@mock.patch('cribsandladders.event_set_plotter.plt')
def test_plot_board_delegates_to_plot_coordinates_and_vectors(mock_plt):
    plotter = EventSetPlotter()
    builder = _builder_with_one_track()
    plotter.plot_board(builder)
    mock_plt.savefig.assert_called_once()
    mock_plt.show.assert_called()
    mock_plt.waitforbuttonpress.assert_called()


@mock.patch('cribsandladders.event_set_plotter.plt')
def test_test_plot_vectors_on_holes_plots_each_vector_and_shows(mock_plt):
    plotter = EventSetPlotter()
    builder = _builder_with_one_track()
    vectors = [((0, 0), (1, 1)), ((2, 2), (3, 3))]

    plotter.test_plot_vectors_on_holes(builder, vectors)

    assert mock_plt.plot.call_count >= len(vectors)
    mock_plt.show.assert_called_once()
    mock_plt.waitforbuttonpress.assert_called_once()


@mock.patch('cribsandladders.event_set_plotter.plt')
def test_plot_coordinates_and_vectors_saves_to_given_bitmap_name(mock_plt):
    plotter = EventSetPlotter()
    builder = _builder_with_one_track()

    plotter.plot_coordinates_and_vectors(builder, bitmap_name='my_test_output.png')

    mock_plt.savefig.assert_called_once_with('my_test_output.png', format='png')
    mock_plt.waitforbuttonpress.assert_called_once()


@mock.patch('cribsandladders.event_set_plotter.plt')
def test_plot_coordinates_and_vectors_sets_axis_limits_from_hole_coords(mock_plt):
    plotter = EventSetPlotter()
    builder = _builder_with_one_track()  # holes at (0,0), (1,1), (2,2)

    plotter.plot_coordinates_and_vectors(builder)

    mock_plt.xlim.assert_called_once_with([-1, 3])
    mock_plt.ylim.assert_called_once_with([-1, 3])
