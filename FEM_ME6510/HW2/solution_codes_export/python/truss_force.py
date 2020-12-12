from math import sqrt, pi, cos, sin, radians
import numpy as np
from get_coords import get_coords


def truss_force(coords, U, E, A):
    #
    # This function computes the axial force in a truss member
    # 
    # Inputs:
    #     coords: a (2 x 2) array of element global coordinates
    #     U: a (4 x 1) array of element global displacements
    #     E: Young's modulus
    #     A: cross-sectional area of the element
    #     
    # Outputs:
    #     axial stress: the scalar axial force in the truss (tension is positive)
    #     

    Lx = coords[1,0]-coords[0,0]
    Ly = coords[1,1]-coords[0,1]
    L = sqrt((Lx)**2. + (Ly)**2.)
    
    # defining direction cosines for element
    Cx = Lx / L
    Cy = Ly / L
    
    axial_stress = E / L * np.dot(np.array([-Cx, -Cy, Cx, Cy]), U)

    return axial_stress

def exercise_3():

    print("Exercise 3, part a")
    theta = 45.
    L = 60.
    coords = get_coords(L, theta)
    A = 2.
    E = 30.e6
    U = np.array([0., 0., 0.01, 0.02])
    stress = truss_force(coords, U.T, E, A)
    print(stress)

    print("Exercise 3, part b")
    theta = 30.
    L = 3.
    coords = get_coords(L, theta)
    A = 3.e-4
    E = 210.e9
    U = np.array([0.25e-3, 0., 1.e-3, 0.])
    stress = truss_force(coords, U.T, E, A)
    print(stress)
