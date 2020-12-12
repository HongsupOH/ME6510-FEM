from math import sqrt, pi, cos, sin, radians
import numpy as np
from get_coords import get_coords


def truss_stiffness(coords, E, A):
    # This function computes the stiffness matrix of a 2D linear 
    # elastic truss element
    # 
    # Inputs:
    #    coords: a (2 x 2) array of element global coordinates
    #    E: Young's Modulus
    #    A: Cross-sectional area of the element
    # 
    # Outputs:
    #    k: the element stiffness matrix in global coordinates
     
    # define length in terms of the x and y coordinates
    Lx = coords[1,0]-coords[0,0]
    Ly = coords[1,1]-coords[0,1]
    L = sqrt((Lx)**2. + (Ly)**2.)
    
    print(L)
    # define direction cosines for element
    C = Lx / L;
    S = Ly / L;
    
    # Global stiffness matrix of element (see lecture 4 notes)
    k = A * E / L * np.array([[ C**2., S*C,   -C**2., -S*C],
                              [ S*C,   S**2., -S*C,   -S**2.],
                              [-C**2, -S*C,    C**2,   S*C],
                              [-S*C,  -S**2,   S*C,    S**2]])
    
    return k

def exercise_2():

    print("Exercise 2, part a")
    theta = 45.
    L = 20.
    coords = get_coords(L, theta)
    A = 3.
    E = 30.e6
    k = truss_stiffness(coords, E, A)
    print(k)

    print("Exercise 2, part b")
    theta = 120.
    L = 15.
    coords = get_coords(L, theta)
    A = 1.
    E = 15.e6
    k = truss_stiffness(coords, E, A)
    print(k)

    print("Exercise 2, part c")
    theta = -30.
    L = 3.
    coords = get_coords(L, theta)
    A = 4.e-4
    E = 210.e9
    k = truss_stiffness(coords, E, A)
    print(k)

    print("Exercise 2, part d")
    theta = 20.
    L = 1.
    coords = get_coords(L, theta)
    A = 2.e-4
    E = 70.e9
    k = truss_stiffness(coords, E, A)
    print(k)
