import numpy as np

def shape_function(xi):
    N = np.zeros((1,4))
        
    N[0,0] = (1/4)*(1-xi[0])*(1-xi[1])
    N[0,1] = (1/4)*(1+xi[0])*(1-xi[1])
    N[0,2] = (1/4)*(1+xi[0])*(1+xi[1])
    N[0,3] = (1/4)*(1-xi[0])*(1+xi[1])
        
    Nq = np.array([[N[0,0],0,N[0,1],0,N[0,2],0,N[0,3],0],
                   [0,N[0,0],0,N[0,1],0,N[0,2],0,N[0,3]]])
    return N,Nq
