"""
Pure coordinate/vector math used to build the DXF track-path geometry.

Split out of DXFWriter.py (Phase 3 of the TDD refactor) because that
module does `import ezdxf` at the top -- a heavy dependency not
installed in every environment -- purely so `buildDXFFile` can build a
DXF document. None of the functions below touch ezdxf, the filesystem,
or a database; they're numpy-only coordinate geometry, so pulling them
into their own module means they (and DXFWriter's actual math) can be
unit tested without ezdxf installed at all. DXFWriter.py imports these
back in and uses them unchanged inside buildDXFFile.
"""
import bisect as bsc

import numpy as np


def convert_mm_to_in(coordinate_list):
    """
    Convert a list of (x, y) coordinate tuples from millimeters to inches.

    Args:
    coordinate_list (list of tuples): A list of (x, y) coordinate tuples in millimeters.

    Returns:
    list of tuples: A list of (x, y) coordinate tuples converted to inches.
    """
    return [(x / 25.4, y / 25.4) for (x, y) in coordinate_list]


# Function to calculate Euclidean distance between two points
def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


# Function to remove coordinates from list1 that are too close to any coordinate in list2
def remove_close_coordinates(list1, list2, threshold=0.125):
    result = []

    for coord1 in list1:
        too_close = False
        for coord2 in list2:
            if euclidean_distance(coord1, coord2) <= threshold:
                too_close = True
                break
        if not too_close:
            result.append(coord1)

    return result


# Function to calculate the midpoint between two points
def midpoint(coord1, coord2):
    return tuple((np.array(coord1) + np.array(coord2)) / 2)


# Function to adjust points that are too close by replacing them with their midpoint
def adjust_close_points(coords, threshold=0.125):
    modified = coords.copy()  # Copy the original list to avoid modifying it during iteration
    n = len(modified)
    i = 0

    while i < n:
        j = i + 1
        while j < n:
            if euclidean_distance(modified[i], modified[j]) <= threshold:
                mid = midpoint(modified[i], modified[j])
                modified[i] = mid
                modified[j] = mid
            j += 1
        i += 1

    return modified


def searchOrderedListForVal(orderedList, val):
    idx = bsc.bisect_left(orderedList, val)
    if idx < len(orderedList) and orderedList[idx] == val:
        return idx
    return -1


def rotate_vector_2d(direction_vector, angle_deg):
    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)

    # Define the 2D rotation matrix (counterclockwise rotation)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Apply the rotation
    new_direction_vector = np.dot(rotation_matrix, direction_vector)
    return new_direction_vector


def compute_offset_curve(points, holesWithEvents, progressMarkers, offset_distance, proximityThresh):
    # List to store points for the left and right offset curves
    left_curve, right_curve = [], []
    arrows = []
    direction_vector = np.array([(-1, -1), (-1, -1)])

    # Pad with pre-point and post-point so tracks extends past first and last hole
    if len(points) >= 2:
        p_pre = 1.6 * points[0] - 0.6 * points[1]
        p_post = 1.6 * points[len(points) - 1] - 0.6 * points[len(points) - 2]
        aug_points = np.vstack([p_pre, points, p_post])
    else:
        aug_points = points

    triggerArrow = False

    # Loop through each pair of consecutive aug_points on the curve
    progMarkerIdx = 0
    for i in range(len(aug_points) - 1):
        # Get two consecutive aug_points
        if i > 0:
            # use prev point ideally
            p1 = aug_points[i - 1]
        else:
            p1 = aug_points[i]
        p2 = aug_points[i + 1]

        # Compute the direction vector (from p1 to p2)
        direction_vector = p2 - p1

        # Normalize the direction vector
        direction_vector /= np.linalg.norm(direction_vector)

        # Compute the perpendicular vector (-dy, dx)
        norm_perp_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Scale the perpendicular vector to the offset distance
        perp_vector = norm_perp_vector * offset_distance

        # Compute the aug_points for the left and right offset curves
        left_curve.append((aug_points[i] + perp_vector).tolist())
        right_curve.append((aug_points[i] - perp_vector).tolist())

        # Add progress marker in if in line
        if i > 0 and i % 5 == 0 and progMarkerIdx < len(progressMarkers):
            left_curve.append((progressMarkers[progMarkerIdx][0]).tolist())
            right_curve.append((progressMarkers[progMarkerIdx][1]).tolist())
            progMarkerIdx += 1

        if i > 1 and i % 7 == 0:
            triggerArrow = True
        if triggerArrow and i % 5 != 0:
            # Check if event on next hole which would muddy up the arrow
            if searchOrderedListForVal(holesWithEvents, i + 1) == -1:
                # Det arrow directional vector
                arrow_dir_vector = aug_points[i + 1] - aug_points[i]
                arrow_dir_vector /= np.linalg.norm(arrow_dir_vector)

                # Det arrow start point
                arrow_base = aug_points[i] + (2 / 25.4) * arrow_dir_vector
                arrow_head = aug_points[i] + (4 / 25.4) * arrow_dir_vector

                # Build arrow unit vectors
                left_arrow_vect = rotate_vector_2d((-1) * arrow_dir_vector, -30) * 0.05
                right_arrow_vect = rotate_vector_2d((-1) * arrow_dir_vector, 30) * 0.05

                # Build vectors and append to lists
                arrows.append([arrow_base.tolist(), arrow_head.tolist()])
                arrows.append([arrow_head.tolist(), (arrow_head + left_arrow_vect).tolist()])
                arrows.append([arrow_head.tolist(), (arrow_head + right_arrow_vect).tolist()])

                triggerArrow = False

    # Add the last point offsets (for the endpoint of the curve)
    last_perp_vector = np.array([-direction_vector[1], direction_vector[0]]) * offset_distance
    left_curve.append((aug_points[-1] + last_perp_vector).tolist())
    right_curve.append((aug_points[-1] - last_perp_vector).tolist())

    # Check proximity to holes, if too close remove outright
    left_curve = remove_close_coordinates(left_curve, aug_points, threshold=proximityThresh)
    right_curve = remove_close_coordinates(right_curve, aug_points, threshold=proximityThresh)

    # Check proximity to neighbours, if too close set each to midpoint of each other
    left_curve = adjust_close_points(left_curve, threshold=proximityThresh - 0.050)
    right_curve = adjust_close_points(right_curve, threshold=proximityThresh - 0.05)

    # Run 2nd time to clean up
    left_curve = remove_close_coordinates(left_curve, aug_points, threshold=proximityThresh)
    right_curve = remove_close_coordinates(right_curve, aug_points, threshold=proximityThresh)
    left_curve = adjust_close_points(left_curve, threshold=proximityThresh - 0.05)
    right_curve = adjust_close_points(right_curve, threshold=proximityThresh - 0.05)

    return left_curve, right_curve, arrows


def create_progress_marker_vectors(hole_list, length):
    vectors = []  # List to store the orthogonal vectors

    # Loop through every 5th point (starting from index 4 for 0-based indexing)
    for i in range(4, len(hole_list), 5):
        # Get the current point and its neighbors
        p1 = hole_list[i - 1]  # Previous point
        p2 = hole_list[i]  # Current point
        last_hole = i + 1 >= len(hole_list)
        p3 = p2 if last_hole else hole_list[i + 1]  # Next point (handle last point)

        # Compute the direction vector from p1 to p3 (use p1 to p3 for smoother orthogonal vector)
        # direction_vector = p3 - p1

        # Compute literal dir vector since trying 5-slash between 5-hole and next hole
        if last_hole:
            direction_vector = p2 - p1
        else:
            direction_vector = p3 - p2

        # Normalize the direction vector
        direction_vector /= np.linalg.norm(direction_vector)

        # Compute the orthogonal (perpendicular) vector (-dy, dx)
        # Note this is 90º CCW rotation
        orthogonal_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Normalize and scale the orthogonal vector to half the desired length (0.5 in total length)
        orthogonal_vector *= (length / 2)

        # Find midpoint between 5 hole and next TRY IT OUT
        if last_hole:
            mid_point = (p2 - p1) / 2 + p2
        else:
            mid_point = (p2 + p3) / 2

        # Store the vector as a tuple of (left_point, right_point)
        linepoints = [mid_point + orthogonal_vector, mid_point - orthogonal_vector]
        vectors.append(tuple(linepoints))

    return vectors
