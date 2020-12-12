import numpy as np

def gauss(gauss_order):
    if gauss_order==1:
        gp = np.array([0])
        gw = np.array([2])
    elif gauss_order==2:
        gp = np.array([-np.sqrt(1/3),np.sqrt(1/3)])
        gw = np.array([1,1])
    elif gauss_order==3:
        gp = np.array([0,-np.sqrt(3/5),np.sqrt(3/5)])
        gw = np.array([8/9,5/9,5/9])
    return gp,gw
