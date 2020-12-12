from math import cos, sin, radians
import numpy as np

def get_coords(L, theta):

    Lx = cos(radians(theta)) * L
    Ly = sin(radians(theta)) * L
    coords = np.array([[0., 0.], [Lx, Ly]])

    return coords
