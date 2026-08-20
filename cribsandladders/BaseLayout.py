from xml.dom import minidom
import game_params as gp

#########################
#Public methods
#########################
def setTrackHolesets(tracks, boardHeight, twoDeckLineBoardPath = ""):
    """
    Sets the track holesets for each track in the given list of tracks.

    Args:
        tracks (list): List of Track objects.
        boardHeight (int): Height of the board.
        twoDeckLineBoardPath (str, optional): Path to the two deck line board file. Defaults to "".

    Returns:
        None
    """
    twoDeckLinePath = None
    if not twoDeckLineBoardPath in ("", None):
        twoDeckLinePath =svgParserVectors(twoDeckLineBoardPath, boardHeight)[0]

    for track in tracks:
        track.trackholes = svgParserHoles(track.holesetfilepath, boardHeight, track.num)
        track.setHolesetIndexer()

        #REMOVED two deck line since 2 decks don really speed up the game much
        if gp.findmode:
            track.length = len(track.trackholes)
            track.efflength = track.length
            track.twodeckslength = track.length


def svgParserHoles(svgFilePath, boardHeight = -1, tracknum = -1, returnRawCoords = False):
    """
    Parses an SVG file and extracts the coordinates of the holes.

    Args:
        svgFilePath (str): Path to the SVG file.
        boardHeight (int, optional): Height of the board. Defaults to -1.
        tracknum (int, optional): Number of the track. Defaults to -1.
        returnRawCoords (bool, optional): Whether to return raw coordinates or Hole objects. Defaults to False.

    Returns:
        list or tuple: List of Hole objects or tuple of raw coordinates.
    """
    board_xml_file=minidom.parse(svgFilePath)
    holes = []
    allcoords = []
    holenum = 0
    # omitFirst = False
    if boardHeight == -1:
        for svg in board_xml_file.getElementsByTagName('svg'):
            boardHeight = float(svg.getAttribute('height').replace("mm", ""))
            break
    for svg_path in board_xml_file.getElementsByTagName('path'):
        # if not omitFirst:
        #     omitFirst = True
        #     continue
        coords = [float(c) for c in svg_path.getAttribute('d').split()[1].split(",")]
        allcoords.append((coords[0],boardHeight-coords[1]))
        #Flip y axis so cartesian coords

    #Reverse if line is in reverse dir
    if allcoords[0][0] > allcoords[len(holes)-1][0]: allcoords =  list(reversed(allcoords))

    if not returnRawCoords:
        for coord_idx in range(0, len(allcoords)):
            holenum += 1
            coord = allcoords[coord_idx]
            holes.append(Hole(coord[0], coord[1], holenum, tracknum, lastHole=coord_idx == len(allcoords)-1))
        return holes
    else:
        return allcoords
def svgParserVectors(svgFilePath, boardHeight):
    """
    Parses an SVG file and extracts the coordinates of the vectors.

    Args:
        svgFilePath (str): Path to the SVG file.
        boardHeight (int): Height of the board.

    Returns:
        list: List of tuples representing the coordinates of the vectors.
    """
    board_xml_file=minidom.parse(svgFilePath)
    vectors = []
    for svg_path in board_xml_file.getElementsByTagName('path'):
        pairs = [t.split(",") for t in svg_path.getAttribute('d').split()]
        pairs.remove(['m'])
        p1 = tuple([float(c) for c in pairs[0]])
        p2_x = p1[0] + float(pairs[len(pairs)-1][0])
        p2_y = p1[1] + float(pairs[len(pairs)-1][1])
        #Flip y axis so cartesian coords
        vector = ((p1[0], boardHeight-p1[1]), (p2_x, boardHeight-p2_y))
        vectors.append(vector)

    return vectors


def check_intersections(test_path_set, possible_intercepts_set):
    """
    Check if any vector in test_path_set intersects with any vector in possible_intercepts_set.

    Parameters:
        test_path_set (list of tuples): List of vectors in the format ((x1, y1), (x2, y2)).
        possible_intercepts_set (list of tuples): List of vectors in the format ((x1, y1), (x2, y2)).

    Returns:
        True if intersects found, False if none found
    """

    for vector1 in test_path_set:
        for vector2 in possible_intercepts_set:
            if _intersect(vector1, vector2):
                return vector1[0]
    return -1

def build_interception_test_vector_set(main_set):
    """
    Given an ordered main set of points and a subset of points,
    draw vectors between each point in the subset and the next point in the main set.
    If the point immediately before the point in question in the main set is not in the subset,
    draw a vector between that previous point and the point in question as well.

    Args:
        main_set (list of tuples): An ordered list of points (x, y).

    Returns:
        List of tuples: Each tuple represents a vector from one point to another.
    """
    vectors = []

    # Ensure both sets are sorted as per the main set

    for holenum in range(0, len(main_set)):
        # Draw a vector to the next point in the main set, if it exists
        if holenum < len(main_set) - 1:
            vectors.append((main_set[holenum], main_set[holenum+1]))

    return vectors


#########################
#Private methods
#########################
def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Return true if line segments AB and CD intersect
def _intersect(vec1, vec2):
    return (_ccw(vec1[0], vec2[0], vec2[1]) != _ccw(vec1[1], vec2[0], vec2[1]) and
            _ccw(vec1[0], vec1[1], vec2[0]) != _ccw(vec1[0], vec1[1], vec2[1]))


class Hole:
    def __init__(self, x, y, num, tracknum, lastHole=False):
        self.coords = (x, y)
        self.num = num
        self.tracknum = tracknum
        self.lastHole = lastHole

    def __hash__(self):
        return hash(self.coords)