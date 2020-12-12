import numpy as np
from shape import shape_function
from shape_grad import shape_gradient
from gauss_integration import gauss
from Jacobi import Jacobian

def quad_B_matrix(node_coors,xi):
    N_der = shape_gradient(xi)
    J,J_inv,detJ = Jacobian(node_coors,xi)

    B_vals = J_inv.dot(N_der)

    B = np.array([[B_vals[0,0],0,B_vals[0,1],0,B_vals[0,2],0,B_vals[0,3],0],
                  [0,B_vals[1,0],0,B_vals[1,1],0,B_vals[1,2],0,B_vals[1,3]],
                  [B_vals[1,0],B_vals[0,0],B_vals[1,1],B_vals[0,1],B_vals[1,2],B_vals[0,2],B_vals[1,3],B_vals[0,3]]])
    return B
