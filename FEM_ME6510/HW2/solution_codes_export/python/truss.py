from math import sqrt, pi, cos, sin, tan, radians
import numpy as np
from truss_stiffness import truss_stiffness
from truss_force import truss_force


def truss():

    # define the truss
    # length of elem 2 given to be 100
    # 2 dimensional truss
    E = 10.e6
    A = 1.
    L2 = 100.
    num_dims = 2
    num_nodes = 4
    num_dofs = num_nodes * num_dims
    num_elems = 3
    
    # calculate the nodal coordinates
    # define node 1 as the origin of the global coordinate system
    node_coords = np.zeros([num_nodes, num_dims])
    node_coords[1,:] = [-L2, L2 * tan(radians(60.))]
    node_coords[2,:] = [-L2, 0.]
    node_coords[3,:] = [-L2, -L2 * tan(radians(30.))]
    
    # define connectivity of elems
    elem_conn = np.empty([num_elems, 2], dtype=int)
    elem_conn[0,:] = [0, 1]
    elem_conn[1,:] = [0, 2]
    elem_conn[2,:] = [0, 3]

    # number the dofs
    node_dofs = []
    for n in range(num_nodes):
        node_dofs.append([2 * n, 2 * n + 1])

    # assemble the global stiffness matrix by summing all
    # element stiffness matrices
    K = np.zeros([num_dofs, num_dofs])
    for ec in elem_conn:
        elem_k = truss_stiffness(node_coords[ec, :], E, A)
        elem_dofs = np.array(node_dofs[ec[0]] + node_dofs[ec[1]], dtype=int)
        K[np.ix_(elem_dofs, elem_dofs)] += elem_k
        print(elem_k)

    print(K)

    # apply bcs and solve using solve_unks code from class
    # define the 'free' DOFs, f
    # and corresponding known forces
    f = np.array([0, 1])
    F_f = np.array([1000., 1000.])
    
    # define the 'supported' DOFs, s
    # and corresponding known displacements
    s = np.array([2, 3, 4, 5, 6, 7])
    D_s = np.zeros(len(s))
    
    # solve for unknown displacements, D_f
    F_ext = F_f - np.dot(K[np.ix_(f, s)], D_s)
    D_f = np.linalg.solve(K[np.ix_(f, f)], F_ext)
    
    # solve for unknown forces
    F_s = np.dot(K[np.ix_(s, f)], D_f) + \
            np.dot(K[np.ix_(s, s)], D_s)

    print("unknown disps = ", D_f)
    print("unknown forces = ", F_s)
    
    # combine all disps into a single vector
    disps = np.empty([num_dofs])
    disps[s] = D_s
    disps[f] = D_f
    
    for e, ec in enumerate(elem_conn):
        elem_dofs = np.array(node_dofs[ec[0]] + node_dofs[ec[1]], dtype=int)
        elem_disps = disps[elem_dofs]
        elem_stress = truss_force(node_coords[ec, :], elem_disps, E, A)
        print("Element ", e + 1, "has stress = ", elem_stress, " psi")

if __name__ == '__main__':
    truss()
