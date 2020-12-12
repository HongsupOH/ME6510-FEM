import numpy as np

def shape_gradient(xi):
    N_der = np.zeros((2,4))

    N_der[0,0] = -(1/4)*(1-xi[1])
    N_der[0,1] = (1/4)*(1-xi[1])
    N_der[0,2] = (1/4)*(1+xi[1])
    N_der[0,3] = -(1/4)*(1+xi[1])

    N_der[1,0] = -(1/4)*(1-xi[0])
    N_der[1,1] = -(1/4)*(1+xi[0])
    N_der[1,2] = (1/4)*(1+xi[0])
    N_der[1,3] = (1/4)*(1-xi[0])
    return N_der
