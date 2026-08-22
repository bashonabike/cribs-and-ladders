"""
Plotting adapter split out of EventSetBuilder.py (Phase 4 decomposition
follow-up -- the refactor plan calls for "a thin plotting adapter tests
can no-op"). `plotBoard`/`testPlotVectorsOnHoles`/
`plot_coordinates_and_vectors` all did `import matplotlib.pyplot as
plt` at EventSetBuilder module scope and called it directly inline
with board-reading logic. `matplotlib` itself is already installed in
this sandbox (unlike scipy/lightgbm/markovgame), so this isn't an
import-hygiene fix the way the other Phase 4 lazy-import changes were
-- it's specifically about giving tests (and any other caller that
doesn't want a blocking `plt.show()`/`plt.waitforbuttonpress()` popup)
something to substitute.

`EventSetBuilder` now takes an optional `plotter` in `__init__`
(defaults to `EventSetPlotter()`) and its `plotBoard`/
`testPlotVectorsOnHoles`/`plot_coordinates_and_vectors` methods
delegate to `self.plotter`, so existing call sites are unaffected, but
`test_eventsetbuilder.py` can inject a `NoOpEventSetPlotter()` instead
of a real one to exercise the calling code without ever touching
matplotlib.
"""
import matplotlib.pyplot as plt


class EventSetPlotter:
    """Real matplotlib-backed plotting for an EventSetBuilder's board."""

    def plot_board(self, builder):
        self.plot_coordinates_and_vectors(builder)

    def test_plot_vectors_on_holes(self, builder, vectors):
        """
        Creates a visualization of vectors overlaid on the board's hole positions.

        This method is primarily used for debugging and visualization purposes to see how
        vectors (representing potential events) interact with the board's hole positions.

        Args:
            builder: the EventSetBuilder whose board holes should be plotted.
            vectors: List of vectors to plot, where each vector is a tuple of two points (start, end).

        Note:
            Displays an interactive plot using matplotlib. The plot includes:
            - Hole positions as points
            - Track numbers and hole numbers as labels
            - The input vectors as lines
        """
        plt.figure(figsize=(15, 10))
        for vector in vectors:
            x_values = [vector[0][0], vector[1][0]]
            y_values = [vector[0][1], vector[1][1]]
            plt.plot(x_values, y_values)

        for t in builder.board.tracks:
            coordinates = [h.coords for h in t.trackholes]
            x_coords, y_coords = zip(*coordinates)
            plt.scatter(x_coords, y_coords, marker='o')

        # Add labels
        for t in builder.board.tracks:
            plt.annotate(str(t.Track_ID), t.trackholes[0].coords)
            for c in t.trackholes:
                if c.num % 5 == 0:
                    plt.annotate(str(c.num), c.coords)

        plt.show()
        plt.waitforbuttonpress()

    def plot_coordinates_and_vectors(self, builder, bitmap_name='output_bitmap.png'):
        """
        Plots multiple sets of coordinates and vectors, and saves the plot as a bitmap image.

        Args:
            builder: the EventSetBuilder whose board tracks/events should be plotted.
            bitmap_name (str): The name of the output bitmap file.
        """
        plt.figure(figsize=(20, 20))
        coordinate_sets = []
        path_dot_vectors = []
        vector_sets = []
        x_marks = []
        for t in builder.board.tracks:
            holes = [h.coords for h in t.trackholes]
            coordinate_sets.append(holes)
            for c_idx in range(len(holes) - 1):
                path_dot_vectors.append((holes[c_idx], holes[c_idx + 1]))
            trackVectorSet = []
            for l, ch in zip([True, True, False], [True, False, True]):
                trackVectorSubset = []
                trackVectorSubset.extend([c.crowVector for c in t.chutes if not c.eventDete.isOrtho
                                          and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                trackVectorSubset.extend([c.eventDete.instanceStartVector for c in t.chutes if c.eventDete.isOrtho
                                          and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                trackVectorSubset.extend([c.eventDete.instanceEndVector for c in t.chutes if c.eventDete.isOrtho
                                          and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                trackVectorSet.append(trackVectorSubset)

                if l and not ch:
                    x_marks.extend([c.crowVector[1] for c in t.chutes if not c.eventDete.isOrtho
                                    and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                    x_marks.extend([c.eventDete.instanceEndVector[0] for c in t.chutes if c.eventDete.isOrtho
                                    and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                elif ch and not l:
                    x_marks.extend([c.crowVector[0] for c in t.chutes if not c.eventDete.isOrtho
                                    and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                    x_marks.extend([c.eventDete.instanceStartVector[0] for c in t.chutes if c.eventDete.isOrtho
                                    and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])

            vector_sets.append(trackVectorSet)

        # Plot each set of coordinates
        for coordinates in coordinate_sets:
            x_coords, y_coords = zip(*coordinates)
            plt.scatter(x_coords, y_coords, marker='o')

        # Plot tracks in fine dots
        for vector in path_dot_vectors:
            x_values = [vector[0][0], vector[1][0]]
            y_values = [vector[0][1], vector[1][1]]
            plt.plot(x_values, y_values, linestyle=':', color='black', linewidth=1)

        # Plot lumps for cannot enter
        lumps = []
        lumps.extend([tuple(c.eventDete.instanceLump) for c in [c for t in builder.board.tracks for c in t.chutes]
                      if c.eventDete.instanceLump != (-1, -1)])
        lumps.extend([tuple(l.eventDete.instanceLump) for l in [l for t in builder.board.tracks for l in t.ladders]
                      if l.eventDete.instanceLump != (-1, -1)])
        for coordinates in lumps:
            x_coords, y_coords = coordinates[0], coordinates[1]
            plt.scatter(x_coords, y_coords, marker="s")

        # Add labels
        for t in builder.board.tracks:
            plt.annotate(str(t.Track_ID), t.trackholes[0].coords)
            for c in t.trackholes:
                if c.num % 5 == 0:
                    plt.annotate(str(c.num), c.coords)

        # Plot each set of vectors
        colourCounter = 0
        colours = [(.5, 0, 0), (1, 0.5, 0), (1, 0, 0.5),
                   (0, .5, 0), (0.5, 1, 0), (0, 1, 0.5),
                   (0, 0, .5), (0, 0.5, 1), (0.5, 0, 1)]
        for vector_subset in vector_sets:
            for vectors in vector_subset:
                for vector in vectors:
                    x_values = [vector[0][0], vector[1][0]]
                    y_values = [vector[0][1], vector[1][1]]
                    plt.plot(x_values, y_values, color=colours[colourCounter])  # 'r-' means red line
                colourCounter += 1

        # Ploy x's on vectors for ladders-only & chutes-onlly
        # for c in x_marks:
        #     plt.annotate("x", c, fontsize=18)

        # Set the axes' limits to fit all points and vectors nicely
        all_x = [coord[0] for coordinates in coordinate_sets for coord in coordinates]
        all_y = [coord[1] for coordinates in coordinate_sets for coord in coordinates]
        plt.xlim([min(all_x) - 1, max(all_x) + 1])
        plt.ylim([min(all_y) - 1, max(all_y) + 1])

        # Save the plot as a bitmap image
        plt.savefig(bitmap_name, format='png')
        plt.show()
        plt.waitforbuttonpress()
        # plt.close()


class NoOpEventSetPlotter:
    """
    Drop-in replacement for `EventSetPlotter` that does nothing --
    for tests (and any headless/batch run) that construct an
    `EventSetBuilder` but never want a blocking matplotlib window or a
    stray `output_bitmap.png` written to disk.
    """

    def plot_board(self, builder):
        pass

    def test_plot_vectors_on_holes(self, builder, vectors):
        pass

    def plot_coordinates_and_vectors(self, builder, bitmap_name='output_bitmap.png'):
        pass
