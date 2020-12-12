import numpy as np
from shape_grad import shape_gradient
from gauss_integration import gauss

def Jacobian(node_coors,xi):
    # 1. Call shape gradient function
    N_der = shape_gradient(xi)
    # 2. Get Jacobian matrix
    J = N_der.dot(node_coors)
    # 3. Get inverse of Jacobian
    J_inv = np.linalg.inv(J)
    # 4. Get determinant of Jacobian
    detJ = np.linalg.det(J)
    return J, J_inv, detJ


    
        
