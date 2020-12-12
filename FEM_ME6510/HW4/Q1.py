import numpy as np

class HeatConduction(object):
    def __init__(self,num_ele,num_node,dim,Ls,As,Kxxs,glb_loc,f,s,f_f,d_s):
        self.num_ele = num_ele
        self.num_node = num_node
        self.glb_loc = glb_loc
        self.Ls = Ls
        self.As = As
        self.kxxs = kxxs
        self.dof = dim*num_node
        self.f = f
        self.s = s
        self.f_f = f_f
        self.d_s = d_s
        self.K = np.zeros((self.dof,self.dof))
        
    def K_ele(self,A,L,Kxx):
        return (A*Kxx/L)*np.array([[1,-1],[-1, 1]])

    def K_glob(self):
        for ele in range(self.num_ele):
            L = self.Ls[ele]
            A = self.As[ele]
            kxx = self.kxxs[ele]
            k_e = self.K_ele(A,L,kxx)
            link = self.glb_loc[ele]
            for glb1,loc1 in link.items():
                for glb2,loc2 in link.items():
                    self.K[glb1,glb2]+=k_e[loc1,loc2]
        
    def solve(self):
        f,s = self.f,self.s
        f_f,d_s = self.f_f,self.d_s
        
        f_ext = f_f - np.dot(self.K[np.ix_(f,s)],d_s)
        d_f = np.linalg.solve(self.K[np.ix_(f,f)],f_ext)
        f_s = np.dot(self.K[np.ix_(s,f)],d_f) + np.dot(self.K[np.ix_(s,s)],d_s)
        self.d_f = d_f
        self.f_s = f_s

    def answer(self):
        print("Global stiffness matrix K:")
        print(self.K)
        print("")
        print("Temperature at free index:")
        for ind,val in enumerate(self.f):
            print("At node {}, temperature is {}".format(val, self.d_f[ind]))
        print("")
        print("Heatflux at supported index:")
        for ind,val in enumerate(self.s):
            print("At node {}, heatflux is {}".format(val, self.f_s[ind]))

if __name__=="__main__":
    """
    num_ele,num_node,dim,Ls,As,Kxxs,glb_loc,f,s,f_f,d_s
    """
    num_ele = 3
    num_node = 4
    dim = 1
    Ls = np.array([0.1,0.1,0.1])
    As = np.array([0.1,0.1,0.1])
    kxxs = np.array([5,10,15])
    glb_loc = {0:{0:0,1:1},1:{1:0,2:1},2:{2:0,3:1}}
    f = [1,2]
    s = [0,3]
    f_f = [0,0]
    d_s = [200,600]
    hc = HeatConduction(num_ele,num_node,dim,Ls,As,kxxs,glb_loc,f,s,f_f,d_s)
    hc.K_glob()
    hc.solve()
    hc.answer()

