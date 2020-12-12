import numpy as np
from collections import defaultdict

class HomeWork1(object):
    def __init__(self,Ks,dof, forces, deforms, d_ind, f_ind):
        self.eles = {}
        for k in Ks:
            localK = np.array([[Ks[k],-Ks[k]],[-Ks[k],Ks[k]]])
            self.eles[k] = localK

        self.forces = forces
        self.deforms = deforms

        self.d_ind = d_ind
        self.f_ind = f_ind
        
        self.dof = dof
        self.K = np.array([[0.0 for x in range(self.dof)] for x in range(self.dof)])
        self.loc_to_glb = defaultdict(dict)
        self.BC_Dirichlet = defaultdict(int)
        self.BC_Neumann = defaultdict(int)

    def local_to_global(self,ele,loc_1,loc_2,glob_1,glob_2):
        link = {loc_1:glob_1, loc_2:glob_2}
        self.loc_to_glb[ele] = link
        
    def Glob_K(self):
        for ele in self.loc_to_glb:
            locK = self.eles[ele]
            link = self.loc_to_glb[ele]
            locs = list(link)
            for y in locs:
                for x in locs:
                    gY,gX = link[y], link[x]
                    self.K[gY][gX] += locK[y][x]

    def boundaryCondition(self):
        for ele in range(self.dof):
            if ele in self.d_ind:
                self.BC_Dirichlet[ele] = self.deforms[ele]
            if ele in self.f_ind:
                self.BC_Neumann[ele] = self.forces[ele]

    def solve(self):
        s,f,D_s,F_f = [],[],[],[]
        for ele in range(self.dof):
            if ele in self.BC_Dirichlet:
                s.append(ele)
                D_s.append(self.BC_Dirichlet[ele])
            else:
                f.append(ele)
                if ele in self.BC_Neumann:
                    F_f.append(self.BC_Neumann[ele])
        f = np.array(f)
        F_f = np.array(F_f)
        s = np.array(s)
        D_s = np.array(D_s)
        F_ext = F_f - np.dot(self.K[np.ix_(f,s)],D_s)
        D_f = np.linalg.solve(self.K[np.ix_(f,f)],F_ext)
        F_s = np.dot(self.K[np.ix_(s,f)],D_f)+np.dot(self.K[np.ix_(s,s)],D_s)
        self.D_f = D_f
        self.F_s = F_s


if __name__=="__main__":
    #Question1
    #Input: Ks,dof, forces, deforms, d_ind, f_ind
    print("Question1")
    #Q1-1.Define Inputs
    Ks1 = {1:10,2:5,3:8,4:10,5:20}
    dof1 = 4
    forces1,deforms1 = {1:0,2:1000},{0:0,3:0}
    d_ind1,f_ind1 = np.array([0,3]),np.array([1,2])
    #Q1-2.Call class
    q1 = HomeWork1(Ks1,dof1,forces1,deforms1,d_ind1,f_ind1)
    #Q1-3.Link between local and global nodes
    q1.local_to_global(1,0,1,0,1)
    q1.local_to_global(2,0,1,1,2)
    q1.local_to_global(3,0,1,1,2)
    q1.local_to_global(4,0,1,2,3)
    q1.local_to_global(5,0,1,1,3)
    #Q1-4.Make global K
    q1.Glob_K()
    print("Global stiffness matrix K:")
    print(q1.K)
    #Q1-5.Define boundary values and solve problem
    q1.boundaryCondition()
    q1.solve()
    print("F_{s} = ", q1.F_s)
    print("D_{f} = ", q1.D_f)
    print("")
    #Question2
    print("Question2")
    #Q2-1.Define Inputs
    Ks2 = {1:1987.5*(10**3),2:331.25*(10**3),3:331.25*(10**3),4:1192.5*(10**3)}
    dof2 = 4
    forces2,deforms2 = {1:0,2:0,3:1500},{0:0}
    d_ind2,f_ind2 = np.array([0]),np.array([1,2,3])
    #Q2-2.Call class
    q2 = HomeWork1(Ks2,dof2,forces2,deforms2, d_ind2,f_ind2)
    #Q2-3.Link between local and global nodes
    q2.local_to_global(1,0,1,0,1)
    q2.local_to_global(2,0,1,1,2)
    q2.local_to_global(3,0,1,1,2)
    q2.local_to_global(4,0,1,2,3)
    #Q2-4.Make global K
    q2.Glob_K()
    print("Global stiffness matrix K:")
    print(q2.K)
    #Q2-5.Define boundary values and solve problem
    q2.boundaryCondition()
    q2.solve()
    print("F_{s} = ", q2.F_s)
    print("D_{f} = ", q2.D_f)
    print("")
    #Question3
    print("Question3")
    #Q3-1.Define Inputs
    Ks3 = {1:1/5,2:1/10,3:1/15}
    dof3 = 3
    I,V = {0:10,1:0},{2:0}
    V_ind,I_ind = np.array([2]),np.array([0,1])
    #Q3-2.Call class
    q3 = HomeWork1(Ks3,dof3,I,V,V_ind,I_ind)
    #Q3-3.Link between local and global nodes
    q3.local_to_global(1,0,1,0,1)
    q3.local_to_global(2,0,1,1,2)
    q3.local_to_global(3,0,1,1,2)
    #Q3-4.Make global K
    q3.Glob_K()
    print("Global stiffness matrix K:")
    print(q3.K)
    #Q3-5.Define boundary values and solve problem
    q3.boundaryCondition()
    q3.solve()
    print("I_{s} = ",q3.F_s)
    print("V_{f} = ",q3.D_f)
    print("")
    #HW2
    print("Exercise1")
    Ks4 = {1:1000,2:500,3:500}
    dof4 = 4
    forces4,deforms4 = {1:-4000},{0:0,2:0,3:0}
    d_ind4,f_ind4 = np.array([0,2,3]),np.array([1])
    HW2 = HomeWork1(Ks4,dof4,forces4,deforms4, d_ind4,f_ind4)
    HW2.local_to_global(1,0,1,0,1)
    HW2.local_to_global(2,0,1,1,2)
    HW2.local_to_global(3,0,1,1,3)
    HW2.Glob_K()
    print("Global stiffness matrix K:")
    print(HW2.K)
    #HW2.Define boundary values and solve problem
    HW2.boundaryCondition()
    HW2.solve()
    print("F_{s} = ",HW2.F_s)
    print("D_{f} = ",HW2.D_f)
        
