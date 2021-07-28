# Author: Porter Zach
# Python 3.9

import numpy as np
import cv2
import pygame

def import_course(file_name: str) -> tuple:
    """Imports the desired course. Courses are images where black represents track,
    white represents wall, and a single (!) red pixel represents the spawn point of
    the cars (cars always spawn facing right).

    Args:
        fileName (str): The name of the course to import.

    Returns:
        tuple: (numpy.ndarray) The imported course as an array; 
            (pygame.Surface) The image representing the course; 
            (numpy.ndarray) The location of the spawn point.
    """

    # Read the course image
    img = cv2.imread(file_name)

    # Find the position of the single red pixel denoting the car spawn point
    # Essentially, find where there is a pixel where the red channel (cv2 uses
    # BGR, so i=2) is > 0 and the green channel (i=1) == 0. This works because
    # the course consists solely of black (0, 0, 0), white (255, 255, 255) and 
    # red (0, 0, 255).
    red = np.nonzero(np.bitwise_and(img[:, :, 2] > 0, img[:, :, 1] == 0))
    # Reformat the output for easier access
    red = np.array([red[1][0], red[0][0]])

    # Set image to grayscale to easily tell between white 
    # (wall) and black (track)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Set the single red pixel to match the black track
    img[red[1], red[0]] = 0

    surf = pygame.surfarray.make_surface(np.transpose(img))

    return img, surf, red