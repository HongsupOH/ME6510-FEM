from Jacobi import Jacobian
from gauss_integration import gauss

def volume(node_coors,h,gauss_order):
    #Init volume
    vol = 0

    #Define gauss points (gp) and gauss weights (gw)
    gp,gw = gauss(gauss_order)
    
    #Get k matrix
    for i in range(len(gp)):
        for j in range(len(gp)):
            xi = [gp[i],gp[j]]
            J, J_inv, detJ = Jacobian(node_coors,xi)
            vol += gw[i]*gw[j]*h*detJ
            
    return vol
