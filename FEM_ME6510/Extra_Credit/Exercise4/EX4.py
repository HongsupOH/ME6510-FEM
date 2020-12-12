import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools


def elasticity_matrix(E,nu,elasticity_type):
    D = np.array([])
    if elasticity_type== "plane stress":
        D = (E/(1-nu**2))*np.array([[1,nu,0],
                                    [nu,1,0],
                                    [0,0,(1-nu)/2]])
    elif elasticity_type== "plane strain":
        D = (E/((1+nu)*(1-2*nu)))*np.array([[1-nu,nu,0],
                                          [nu,1-nu,0],
                                          [0,0,(1-2*nu)/2]])
    elif elasticity_type== "axisymmetric":
        D = (E/(1+nu)*(1-2*nu))*np.array([[1-nu,nu,0,nu],
                                          [nu,1-nu,0,nu],
                                          [0,0,(1-2*nu)/2,0],
                                          [nu,nu,0,1-nu]])
    
    return D

def quad_shape(xi):
    #shape function
    N = np.zeros((1,4))
    N_der = np.zeros((2,4))
    
    N[0,0] = (1/4)*(1-xi[0])*(1-xi[1])
    N[0,1] = (1/4)*(1+xi[0])*(1-xi[1])
    N[0,2] = (1/4)*(1+xi[0])*(1+xi[1])
    N[0,3] = (1/4)*(1-xi[0])*(1+xi[1])
    
    #shape der
    N_der[0,0] = -(1/4)*(1-xi[1])
    N_der[0,1] = (1/4)*(1-xi[1])
    N_der[0,2] = (1/4)*(1+xi[1])
    N_der[0,3] = -(1/4)*(1+xi[1])

    N_der[1,0] = -(1/4)*(1-xi[0])
    N_der[1,1] = -(1/4)*(1+xi[0])
    N_der[1,2] = (1/4)*(1+xi[0])
    N_der[1,3] = (1/4)*(1-xi[0])

    Nq = np.array([[N[0,0],0,N[0,1],0,N[0,2],0,N[0,3],0],
                       [0,N[0,0],0,N[0,1],0,N[0,2],0,N[0,3]]])
    return N,N_der,Nq

def quad_B_matrix(node_coors,xi):
    N,N_der,Nq = quad_shape(xi)
    J = N_der.dot(node_coors)
    J_inv = np.linalg.inv(J)
    detJ = np.linalg.det(J)

    B_vals = J_inv.dot(N_der)

    B = np.array([[B_vals[0,0],0,B_vals[0,1],0,B_vals[0,2],0,B_vals[0,3],0],
                  [0,B_vals[1,0],0,B_vals[1,1],0,B_vals[1,2],0,B_vals[1,3]],
                  [B_vals[1,0],B_vals[0,0],B_vals[1,1],B_vals[0,1],B_vals[1,2],B_vals[0,2],B_vals[1,3],B_vals[0,3]]])
    return B,detJ

def quad_stiffness(node_coors,h,E,nu,gauss_order,planeCode):
    k = np.zeros((8,8))
    if planeCode==1:
        D = elasticity_matrix(E,nu,"plane stress")
    elif planeCode==2:
        D = elasticity_matrix(E,nu,"plane strain")
    elif planeCode==3:
        D = elasticity_matrix(E,nu,"axisymmetric")

    if gauss_order==1:
        xi_s = np.array([0])
        wi = np.array([2])
    elif gauss_order==2:
        xi_s = np.array([-np.sqrt(1/3),np.sqrt(1/3)])
        wi = np.array([1,1])
    elif gauss_order==3:
        xi_s = np.array([0,-np.sqrt(3/5),np.sqrt(3/5)])
        wi = np.array([8/9,5/9,5/9])
        
    for i in range(len(wi)):
        for j in range(len(wi)):
            xi = [xi_s[i],xi_s[j]]
            B,detJ = quad_B_matrix(node_coors,xi)
            k += wi[i]*wi[j]*h*detJ*(B.transpose().dot(D.dot(B)))
    return k


def solve(K,f,s,f_f,d_s):
    f_ext = f_f - np.dot(K[np.ix_(f,s)],d_s)
    #d_f = np.linalg.solve(K[np.ix_(f,f)],f_ext)
    print("Condition number of Kff:")
    print(np.linalg.cond(K[np.ix_(f,f)]))
    d_f = np.dot(np.dot(np.linalg.inv(np.dot(K[np.ix_(f,f)].transpose(),K[np.ix_(f,f)])),K[np.ix_(f,f)].transpose()),f_ext)
    #d_f = np.linalg.lstsq(K[np.ix_(f,f)],f_ext,rcond=-1)[0]
    
    f_s = np.dot(K[np.ix_(s,f)],d_f) + np.dot(K[np.ix_(s,s)],d_s)
    return d_f,f_s

def main(case):
    glb_loc = {0:{0:0,5:1,4:2,3:3},
               1:{5:0,1:1,2:2,4:3}}

    node_coors = {0:[[0,0],[1,0],[1,1],[0,1]],
                  1:[[1,0],[2,0],[2,1],[1,1]]}

    L = 2
    H = 1
    E = 29e3
    nu = 0.3
    h = 0.1
    volume = 2*1*0.1
    planeCode = 1
    
    d = 0.2
    err = 0.2*(1/100)
    gauss_orders = np.array([1,2,3])
    for gauss_order in gauss_orders:
        K_glb = np.zeros((12,12))
        print("Gauss order: {}".format(gauss_order))
        vol = 0
        for ele in range(2):
            node_coor = node_coors[ele]
            link = glb_loc[ele]
            k_ele = quad_stiffness(node_coor,h,E,nu,gauss_order,planeCode)
            for glb1,loc1 in link.items():
                for glb2,loc2 in link.items():
                    for i in range(2):
                        for j in range(2):
                            K_glb[2*glb1+i,2*glb2+j] += k_ele[2*loc1+i,2*loc2+j]
        if case==1:
            s = np.array([0,1,2,3,4,6])
            f = np.array([5,7,8,9,10,11])
            d_s = np.array([0,0,d+err,0,d+err,0])
            f_f = np.array([0,0,0,0,0,0])
            d_f,f_s = solve(K_glb,f,s,f_f,d_s)
            
        elif case==2:
            s = np.array([0,1,2,4,6])
            f = np.array([3,5,7,8,9,10,11])
            
            d_s = np.array([0,0,d+err,d+err,0])
            f_f = np.array([0,0,0,0,0,0,0])
            
            d_f,f_s = solve(K_glb,f,s,f_f,d_s)

        #volume = vol
        eff_xA = volume/L
        force_an = E*(d/L) * eff_xA
        force_fea = sum([x for x in f_s if x>0])
        ext_err = err*100/d
        act_err = 100*abs(force_an - force_fea)/force_an
        print("Expected force is {}".format(force_an))
        print("Real force is {}".format(force_fea))
        print("Expected error is {}".format(ext_err))
        print("Real error is {}".format(act_err))
        print("")

if __name__=="__main__":
    print("Case 1:")
    main(1)
    print("")
    print("Case 2:")
    main(2)
            
            

        

        
            
    
    
    


    
