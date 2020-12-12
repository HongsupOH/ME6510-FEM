import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D


class Elastostatics(object):
    def __init__(self,dim,x,y,xNode,yNode,q, E,nu,h,pos,planeCode,gauss_order,r):
        self.dim = dim
        self.q = q
        self.x,self.y = x,y
        self.xNode,self.yNode = xNode,yNode
        self.dataLocal = defaultdict(dict)
        self.nodeLocation = defaultdict(list)
        self.boundary_values = defaultdict(int)
        self.NN_BC = defaultdict(int)
        self.dof = xNode*yNode*self.dim
        self.K = np.zeros((self.dof,self.dof))
        self.F = np.zeros(self.dof)
        self.D = np.zeros(self.dof)
        
        
        self.Stress = np.zeros(xNode*yNode*4)
        self.Strain = np.zeros(xNode*yNode*4)

        self.E = E
        self.nu = nu
        self.h = h
        self.pos = pos
        self.planeCode = planeCode 
        self.gauss_order = gauss_order

        self.r = r

    def elasticity_matrix(self,E,nu,elasticity_type):
        D = np.array([])
        if elasticity_type== "plane stress":
            D = (E/(1-nu**2))*np.array([[1,nu,0],
                                        [nu,1,0],
                                        [0,0,(1-nu)/2]])
        elif elasticity_type== "plane strain":
            D = (E/(1+nu)*(1-2*nu))*np.array([[1-nu,nu,0],
                                              [nu,1-nu,0],
                                              [0,0,(1-2*nu)/2]])
        elif elasticity_type== "axisymmetric":
            D = (E/((1+nu)*(1-2*nu)))*np.array([[1-nu,nu,0,nu],
                                              [nu,1-nu,0,nu],
                                              [0,0,(1-2*nu)/2,0],
                                              [nu,nu,0,1-nu]])
        return D

    def quad_shape(self,xi):
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

    def quad_r(self,xi):
        N,N_der,Nq = self.quad_shape(xi)
        rs = np.array([self.r,self.r+self.x,self.r+self.x,self.r])
        r_i = N.dot(rs.transpose())
        return r_i[0]

    def quad_B_matrix(self,node_coors,xi):
        N,N_der,Nq = self.quad_shape(xi)
        J = N_der.dot(node_coors)
        J_inv = np.linalg.inv(J)
        detJ = np.linalg.det(J)

        B_vals = J_inv.dot(N_der)
        
        
        r = self.quad_r(xi)
        B = np.array([[B_vals[0,0],0,B_vals[0,1],0,B_vals[0,2],0,B_vals[0,3],0],
                      [0,B_vals[1,0],0,B_vals[1,1],0,B_vals[1,2],0,B_vals[1,3]],
                      [B_vals[1,0],B_vals[0,0],B_vals[1,1],B_vals[0,1],B_vals[1,2],B_vals[0,2],B_vals[1,3],B_vals[0,3]],
                      [N[0,0]/r,0,N[0,1]/r,0,N[0,2]/r,0,N[0,3]/r,0]])
        return B,detJ


    def quad_stressC(self,planeCode,E,nu,node_coors):
        if planeCode==1:
            D = self.elasticity_matrix(E,nu,"plane stress")
        elif planeCode==2:
            D = self.elasticity_matrix(E,nu,"plane strain")
        elif planeCode==3:
            D = self.elasticity_matrix(E,nu,"axisymmetric")

        xi = [0,0]
        B,detJ = self.quad_B_matrix(node_coors,xi)
        strain = B.dot(self.D)
        stress = D.dot(strain)
        return stress,strain

    def quad_stiffness(self,node_coors,E,nu,gauss_order):
        k = np.zeros((8,8))
        if self.planeCode==1:
            D = self.elasticity_matrix(E,nu,"plane stress")
        elif self.planeCode==2:
            D = self.elasticity_matrix(E,nu,"plane strain")
        elif self.planeCode==3:
            D = self.elasticity_matrix(E,nu,"axisymmetric")

        if gauss_order==1:
            xi = [0]
            wi = [2]
            
            B,detJ = self.quad_B_matrix(node_coors,xi)
            k += wi*h*detJ*(B.transpose().dot(D.dot(B)))
                    
        elif gauss_order==2:
            xi_s = np.array([-np.sqrt(1/3),np.sqrt(1/3)])
            wi = np.array([1,1])
            
            for i in range(self.dim):
                for j in range(self.dim):
                    xi = [xi_s[i],xi_s[j]]
                    r = self.quad_r(xi)
                    B,detJ = self.quad_B_matrix(node_coors,xi)
                    k += wi[i]*wi[j]*(2*np.pi*r)*detJ*(B.transpose().dot(D.dot(B)))
        return k

    def quad_force(self,gauss_order,pos,node_coors):
        f = np.zeros(8)
        if gauss_order==1:
            xi = [0]
            wi = [2]
            
            B,detJ = self.quad_B_matrix(node_coors,xi)
            k += wi*h*detJ*(B.transpose().dot(D.dot(B)))
            
        elif gauss_order==2:
            xi_s = np.array([-np.sqrt(1/3),np.sqrt(1/3)])
            wi = np.array([1,1])
            
            for i in range(self.dim):
                xi = [pos,xi_s[i]]
                N,N_der,Nq = self.quad_shape(xi)
                r = self.quad_r(xi)
                B,detJ = self.quad_B_matrix(node_coors,xi)
                f+= np.pi*wi[i]*Nq.transpose().dot(self.q)*detJ
        return f

    def solve(self,K,f,s,f_f,d_s):
        f_ext = f_f - np.dot(K[np.ix_(f,s)],d_s)
        d_f = np.linalg.solve(K[np.ix_(f,f)],f_ext)
        f_s = np.dot(K[np.ix_(s,f)],d_f) + np.dot(K[np.ix_(s,s)],d_s)
        return d_f,f_s

    def main(self):
        node_coors = np.array([[3,0],[3.1,0],[3.1,1],[3,1]])
        E = 29e6
        nu = 0.3
        pos = -1
        planeCode = 3
        gauss_order = 2
        k = self.quad_stiffness(node_coors,E,nu,gauss_order)
        print("Stiffness Matrix:")
        print(k)
        print("")
        force = np.array([10e3*2*np.pi/2 ,0, 0,0, 0,0, 10e3*2*np.pi/2,0])
        f = np.array([0,2,3,4,5,6,7])
        s = np.array([1])
        d_s = np.array([0])
        f_f = force[np.ix_(f)]
        d_f,f_s = self.solve(k,f,s,f_f,d_s)
        print("Unknown deformation")
        print(d_f)
        print("")
        print("Unknown force")
        print(f_s)
        print("")
        self.D = np.zeros(8)
        for ind,ele in enumerate(s):
            self.D[ele]=d_s[ind]
            self.F[ele]=f_s[ind]
        for ind,ele in enumerate(f):
            self.D[ele]=d_f[ind]
            self.F[ele]=f_f[ind]
        """
        self,gauss_order,planeCode,E,nu,node_coors
        """
        print("Deformation vector")
        print(self.D)
        print("")
        print("Force vector")
        print(self.F)
        print("")
        stress,strain = self.quad_stressC(3,E,nu,node_coors)
        print("stress")
        print(stress)
        print("")
        print("strain")
        print(strain)
        print("")
        print("Check what happens in clockwise manner")
        node_inv = np.array([[3,0],[3,1],[3.1,1],[3.1,0]])
        print("Clock wise manner")
        print(node_inv)
        E = 29e6
        nu = 0.3
        h = 2*np.pi*self.r
        pos = -1
        planeCode = 3
        gauss_order = 2
        k = self.quad_stiffness(node_coors,E,nu,gauss_order)
        print("Stiffness Matrix:")
        print(k)
        print("")
        #force = self.quad_force(gauss_order,pos,node_coors)
        force = np.array([10e3*2*np.pi/2 ,0, 0,0, 0,0, 10e3*2*np.pi/2,0])
        print("check Force:", force)
        f = np.array([0,2,3,4,5,6,7])
        s = np.array([1])
        d_s = np.array([0])
        f_f = force[np.ix_(f)]
        d_f,f_s = self.solve(k,f,s,f_f,d_s)
        print("Unknown deformation")
        print(d_f)
        print("")
        print("Unknown force")
        print(f_s)
        print("")
        self.D = np.zeros(8)
        for ind,ele in enumerate(s):
            self.D[ele]=d_s[ind]
            self.F[ele]=f_s[ind]
        for ind,ele in enumerate(f):
            self.D[ele]=d_f[ind]
            self.F[ele]=f_f[ind]

    
    
    
if __name__=="__main__":
    """
    dim,x,y,xNode,yNode,q,r
    """
    q2 = np.array([10e3*2*np.pi*3/2 , 0])
    dim = 2
    E2 = 29e6
    nu2 = 0.3
    r = 3
    h2 = 2*np.pi*r
    pos2 = -1
    planeCode2 = 3
    gauss_order2 = 2

    x2,y2 = 0.1,1
    xNode2,yNode2 = 2,2
    p2 = Elastostatics(dim,x2,y2,xNode2,yNode2,q2,E2,nu2,h2,pos2,planeCode2,gauss_order2,r)
    p2.main()
    


    
    

    

    
    
    
