# Author: Porter Zach
# Python 3.9

import numpy as np

def cast(origin: np.ndarray, dir: np.ndarray, map: np.ndarray, want_endpoint = False) -> float:
    """Performs a raycast.

    Args:
        origin (np.ndarray): The cast origin.
        dir (np.ndarray): The direction of the cast.
        map (np.ndarray): The map in which the cast is performed.
        want_endpoint (bool): Whether the caller wants the intersection point of the ray to be returned.

    Returns:
        float: The distance from the origin to the first hit wall.
        --- OR ---
        tuple: (float) The distance from the origin to the first hit wall;
            (numpy.ndarray) The position of the ray intersection point.
    """

    # we'll try itercasting first. premature optimization is the root of all evil
    cur = np.array([origin[0], origin[1]])
    for i in range(10000):
        cur += dir
        if map[int(cur[1]), int(cur[0])]:
            dist = np.linalg.norm(cur - origin)
            return dist if not want_endpoint else (dist, cur)