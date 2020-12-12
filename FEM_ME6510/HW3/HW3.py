import numpy as np
import matplotlib.pyplot as plt

class Beam(object):
    def __init__(self, num_ele, dim, L, E, I, force, deform):
        #Define variables
        self.num_ele = num_ele
        self.num_node = num_ele+1
        self.dof = dim*self.num_node
        self.L = L
        self.L_ele = int(L/num_ele)
        self.P = force[0] # At uniform distribution P is (w*L)/2
        self.E = E
        self.I = I
        #Define Matrixs and vectors
        self.K = np.zeros((self.dof,self.dof))
        self.d_ind = np.array([0,1])
        self.f_ind = np.array([x for x in range(self.dof) if x not in self.d_ind])
        
        self.force = force
        self.deform = deform

        self.P_f = np.array([0 for x in range(len(self.f_ind))])
        self.D_s = np.array([0 for x in range(len(self.d_ind))])

        self.xs = [x for x in range(0,L+1, self.L_ele)]

    def GlobK(self):
        E = self.E
        I = self.I
        L = self.L_ele
        #Define element K matrix
        K_ele = (E*I/(L**3))*np.array([[12,6*L,-12,6*L],
                                       [6*L,4*(L**2),-6*L,2*(L**2)],
                                       [-12,-6*L,12,-6*L],
                                       [6*L,2*(L**2),-6*L,4*(L**2)]])
        #Generate Global K matrix
        glbInd = [0,1,2,3]
        for ele in range(self.num_ele):
            for loc1,glb1 in enumerate(glbInd):
                for loc2,glb2 in enumerate(glbInd):
                    self.K[glb1,glb2]+= K_ele[loc1,loc2]
            #Remove first two elements in glbInd
            cnt = 0
            while cnt<2:
                glbInd.pop(0)
                cnt+=1
            #Insert new last two elements in glbInd
            cnt2 = 0
            newInd = glbInd[-1]+1
            while cnt2<2:
                glbInd.append(newInd)
                newInd+=1
                cnt2+=1
                
    def defineNeumannBC(self, ind, force):
        self.P_f[ind] = force

    def defineDirichletBC(self, ind, deform):
        self.D_s[ind] = Deform

    def solve(self, uniform = None):
        #Define free and supported indexes
        f = self.f_ind
        s = self.d_ind
        #Solve problem
        P_ext = self.P_f - np.dot(self.K[np.ix_(f,s)],self.D_s)
        D_f = np.linalg.solve(self.K[np.ix_(f,f)],P_ext)
        F_s = np.dot(self.K[np.ix_(s,f)],D_f)+np.dot(self.K[np.ix_(s,s)],self.D_s)
        if uniform == True:
            # P is equal to (w*L)/2
            F_s += np.array([-self.P, -self.P*self.L/6])
        #Convert local variable to global (public) variable
        self.D_f = D_f
        self.F_s = F_s

    def ExtractDisplacement(self):
        #Extract only displacement (remove rotation)
        Dv = []
        for ind,ele in enumerate(self.D_f):
            if ind%2==0:
                Dv.append(ele)
        #Convert local to global and combine free index displacement
        self.D = np.array([self.deform[0]] + Dv)

    def InterpolationFunction(self,uniform=None):
        #Define varialbe for analytical solution
        L = self.L
        v1 = self.deform[0]
        ph1 = self.deform[1]
        v2 = self.D_f[-2]
        ph2 = self.D_f[-1]
        #vs is displacement analytical answer
        #ms is moment analytical answer
        vs,ms = [],[]
        a1 = (2/(L**3))*(v1-v2) + (1/(L**2))*(ph1+ph2)
        a2 = (-3/(L**2))*(v1-v2) - (1/L)*(2*ph1+ph2)
        a3 = ph1
        a4 = v1
        #Generate analytical solution for both displacement and moment
        for x in self.xs:
            # v(displacement) = a1*x^3 + a2*x^2 + a3*x + a4
            #v = a1*(x**3)+a2*(x**2)+a3*x+a4
            v = self.P*x**2/(6*self.E*self.I)*(3*L - x)
            vs.append(v)
            # m(moment) = p*(L - x)
            if uniform:
                w = 2*self.P/L
                m = w*(L-x)**2/2
            else:
                m = self.P*(L-x)
            ms.append(m)
        #Convert local to global varible
        self.vs = vs
        self.ms = ms

    def solMoment(self,uniform=None):
        #Define variables
        E = self.E
        I = self.I
        L = self.L
        #Create v vector
        v1 = self.deform[0]
        ph1 = self.deform[1]
        v2 = self.D_f[-2]
        ph2 = self.D_f[-1]
        d = np.array([v1,ph1,v2,ph2])
        #Solve problem
        FE_M = []
        for x in range(0,L+1, self.L_ele):
            # m (moment) = EI*[B]{d}
            B = (1/L**3)*np.array([12*x-6*L, 6*x*L - 4*(L**2), -12*x+6*L,6*x*L - 2*(L**2)])
            m = E*I*np.dot(B,d)
            FE_M.append(m)
        #Convert local to global varible
        self.FE_M = FE_M

    def correctSol(self):
        L = self.L
        #Construct v vector
        v1 = self.deform[0]
        ph1 = self.deform[1]
        v2 = self.D_f[-2]
        ph2 = self.D_f[-1]
        d = np.array([v1,ph1,v2,ph2])
        cs = []
        #Solution
        for x in range(0,L+1, self.L_ele):
            # v(displacement) = [N]*{d}
            N = (1/L**3)*np.array([2*x**3 - 3*x**2*L+L**3,
                                   x**3*L-2*x**2*L**2+x*L**3,
                                   -2*x**3+3*x**2*L,
                                   x**3*L-x**2*L**2])
            c = np.dot(N,d)
            cs.append(c)    
        #Convert local to global varible
        self.cs = cs

    def error(self,v_a):
        errorD =[]
        for ind,v in enumerate(self.cs):
            eD = np.abs(v-v_a[ind])
            errorD.append(eD)
        self.errorD = errorD

if __name__== "__main__":
    #Problem1
    print("Problem 1: ")
    #Define variables
    num_eles1 = [1,10]
    for num_ele1 in num_eles1:
        print("The number of element: {}".format(num_ele1))
        force1 = np.array([-500,0])
        deform1 = np.array([0,0])
        dim1 = 2
        L1 = 100
        E1 = 30e6
        I1 = 100
        p1 = Beam(num_ele1,dim1,L1,E1,I1,force1,deform1)
        #Generate global K
        p1.GlobK()
        print("Stiffness Matrix")
        print(p1.K)
        #Define know force and deform
        p1.defineNeumannBC(len(p1.P_f)-2,-500) # P2 = -500
        p1.defineNeumannBC(len(p1.P_f)-1,0)    # M2 = 0
        #solve problem
        p1.solve()
        print("D_{f}: ")
        print(p1.D_f)
        print("F_{s}: ")
        print(p1.F_s)
        print("")
    #Extract displacement from D_{f} (remove rotation term)
    p1.ExtractDisplacement()
    #Analytical solution
    #p1.analyticalSol()
    x_a = np.linspace(0,L1,11)
    v = p1.P*(x_a)**2/(6*E1*I1)*(3*L1-x_a)
    print("Displacement(FEM):")
    print(p1.D)
    print("Analytical answer:")
    print(v)
    #Plot FEM vs analytical solution (displacement)
    plt.plot(x_a, v, label="Analytical solution")
    plt.plot(p1.xs,p1.D, label="FEM solution")
    plt.title("FEM vs Analytical solution of point load(displacement)")
    plt.xlabel("Location (inch)")
    plt.ylabel("Displacement (inch)")
    plt.legend()
    plt.grid()
    plt.show()
    #FEM moment solution
    p1.solMoment()
    m = p1.P*(L1-x_a)
    #Plot FEM vs analytical solution (moment)
    plt.plot(x_a,m, label="Analytical solution")
    plt.plot(p1.xs,p1.FE_M, label="FEM solution")
    plt.title("FEM vs Analytical solution of point load(moment)")
    plt.xlabel("Location (inch)")
    plt.ylabel("Moment (lb*in)")
    plt.legend()
    plt.grid()
    plt.show()
    #Correct solution
    p1.correctSol()
    #Error between FEM and correct solution
    p1.error(v)
    plt.plot(p1.xs,p1.errorD)
    plt.title("Error between FEM and correct solution of point load(displacement)")
    plt.xlabel("Location (inch)")
    plt.ylabel("Error (inch)")
    plt.grid()
    plt.show()
    #Deformation for different number of elements
    for numele in [2,5,10,20,50,100]:
        pD = Beam(numele ,dim1,L1,E1,I1,force1,deform1)
        pD.GlobK()
        pD.defineNeumannBC(len(pD.P_f)-2,-500) 
        pD.defineNeumannBC(len(pD.P_f)-1,0)
        pD.solve()
        pD.ExtractDisplacement()
        plt.plot(pD.xs, pD.D, label=numele)
    plt.title("FEM plots according to the number of node (point load)")
    plt.xlabel("Location (inch)")
    plt.ylabel("Displacement (inch)")
    plt.legend()
    plt.grid()
    plt.show()
    #Errors for different number of elements
    for numele in [2,5,10,20,50,100]:
        pL = Beam(numele ,dim1,L1,E1,I1,force1,deform1)
        pL.GlobK()
        pL.defineNeumannBC(len(pL.P_f)-2,-500) 
        pL.defineNeumannBC(len(pL.P_f)-1,0)
        pL.solve()
        pL.ExtractDisplacement()
        pL.correctSol()
        x_aL = np.linspace(0,L1,numele+1)
        vL = pL.P*(x_aL)**2/(6*E1*I1)*(3*L1-x_aL)
        pL.error(vL)
        plt.plot(pL.xs,pL.errorD, label=numele)
    plt.title("Error between FEM and correct solution (displacement) of point load")
    plt.xlabel("Location (inch)")
    plt.ylabel("Error (inch)")
    plt.legend()
    plt.grid()
    plt.show()
    print("")
    
    #Problem2
    print("Problem 2: ")
    #Define variables
    num_eles2 = [1,10] 
    for num_ele2 in num_eles2:
        force2 = np.array([-1000,50000/3])
        deform2 = np.array([0,0])
        dim2 = 2
        L2 = 100
        E2 = 30e6
        I2 = 100
        p2 = Beam(num_ele2,dim2,L2,E2,I2,force2,deform2)
        #Generate global K
        p2.GlobK()
        print("Stiffness Matrix")
        print(p2.K)
        #Define know force and deform
        p2.defineNeumannBC(len(p2.P_f)-2,-1000) # P2 = -wL/2
        p2.defineNeumannBC(len(p2.P_f)-1,50000/3)    # M2 = wL^2/12
        #solve problem
        p2.solve(uniform = True)
        print("D_{f}: ")
        print(p2.D_f)
        print("F_{s}: ")
        print(p2.F_s)
        print("")
    #Extract displacement from D_{f} (remove rotation term)
    p2.ExtractDisplacement()
    #Analytical solution
    #p2.analyticalSol(uniform=True)
    x_a2 = np.linspace(0,L1,11)
    w = 2*p2.P/L2
    v2 = w*(x_a2)**2/(24*E2*I2)*(x_a2**2-4*L2*x_a2+6*L2**2)
    print("Displacement(FEM):")
    print(p2.D)
    print("Analytical answer:")
    print(v2)
    #Plot FEM vs analytical solution (displacement)
    plt.plot(x_a2,v2, label="Analytical solution")
    plt.plot(p2.xs,p2.D, label="FEM solution")
    plt.title("FEM vs Analytical solution (displacement) of distribution load")
    plt.xlabel("Location (inch)")
    plt.ylabel("Displacement (inch)")
    plt.legend()
    plt.grid()
    plt.show()
    #FEM moment solution
    m = w*(L2-x_a2)**2/2
    p2.solMoment(uniform=True)
    #Plot FEM vs analytical solution (moment)
    plt.plot(x_a2,m, label="Analytical solution")
    plt.plot(p2.xs,p2.FE_M, label="FEM solution")
    plt.title("FEM vs Analytical solution (moment) of distribution load")
    plt.xlabel("Location (inch)")
    plt.ylabel("Moment (lb*in)")
    plt.legend()
    plt.grid()
    plt.show()
    #Correct solution
    p2.correctSol()
    #Error between FEM and correct solution
    p2.error(v2)
    plt.plot(p2.xs,p2.errorD)
    plt.title("Error between FEM and correct solution (displacement) of distribution load")
    plt.xlabel("Location (inch)")
    plt.ylabel("Error (inch)")
    plt.grid()
    plt.show()
    #Deformation for different number of elements
    for numele in [2,4,5,10,20,25,50,100]:
        pD2 = Beam(numele ,dim2,L2,E2,I2,force2,deform2)
        pD2.GlobK()
        pD2.defineNeumannBC(len(pD2.P_f)-2,-1000) # P2 = -wL/2
        pD2.defineNeumannBC(len(pD2.P_f)-1,50000/3)    # M2 = wL^2/12
        pD2.solve()
        pD2.ExtractDisplacement()
        plt.plot(pD2.xs, pD2.D, label=numele)
    plt.title("FEM plots according to the number of node (distribution load)")
    plt.xlabel("Location (inch)")
    plt.ylabel("Displacement (inch)")
    plt.legend()
    plt.grid()
    plt.show()
    #Errors for different number of elements
    for numele in [2,4,5,10,20,25,50,100]:
        pL2 = Beam(numele ,dim2,L2,E2,I2,force2,deform2)
        pL2.GlobK()
        pL2.defineNeumannBC(len(pL2.P_f)-2,-1000) 
        pL2.defineNeumannBC(len(pL2.P_f)-1,50000/3)    
        pL2.solve(uniform = True)
        pL2.ExtractDisplacement()
        pL2.correctSol()
        x_aL2 = np.linspace(0,L1,numele+1)
        w2 = 2*pL2.P/L2
        vL2 = w2*(x_aL2)**2/(24*E2*I2)*(x_aL2**2-4*L2*x_aL2+6*L2**2)
        pL2.error(vL2)
        plt.plot(pL2.xs,pL2.errorD, label=numele)
    plt.title("Error between FEM and correct solution (displacement) of distribution load")
    plt.xlabel("Location (inch)")
    plt.ylabel("Error (inch)")
    plt.legend()
    plt.grid()
    plt.show()

    
