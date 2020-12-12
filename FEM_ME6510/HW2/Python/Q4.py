import numpy as np

class Truss(object):
    def __init__(self,num_node,num_elem,dims,E,A,f,s,force,deform):
        self.num_node = num_node
        self.num_elem = num_elem
        self.dims = dims
        self.num_dofs = num_node*dims
        
        self.node_coors = np.zeros((self.num_node,2))
        self.elem_conn = np.zeros((self.num_elem,2),dtype=int)
        self.node_dofs = np.zeros((self.num_node,2),dtype=int)
        self.K = np.zeros((self.num_dofs,self.num_dofs))
        self.E,self.A = E,A
        self.f,self.s = f,s
        self.force = force
        self.deform = deform

        self.thetas = []
        
    def add_node(self,node,x,y):
        self.node_coors[node,0] = x
        self.node_coors[node,1] = y

    def add_elem(self,ele,n1,n2):
        self.elem_conn[ele,0] = n1
        self.elem_conn[ele,1] = n2

    def add_dofs(self):
        for i in range(self.num_node):
            self.node_dofs[i] = [2*i,2*i+1]

    def add_theta(self,theta):
        self.thetas.append(theta)

    def truss_stiffness(self,theta,E,A,L):
        theta_r = theta*np.pi/180
        k = E*A/L
        k_locd = np.array([[k,0,-k,0],
                           [0,0,0,0],
                           [-k,0,k,0],
                           [0,0,0,0]])
        C = np.cos(theta_r)
        S = np.sin(theta_r)
        Tr = np.array([[C,S,0,0],
                       [-S,C,0,0],
                       [0,0,C,S],
                       [0,0,-S,C]])
        self.Tr = Tr
        k_globd = np.dot(Tr.T,np.dot(k_locd,Tr))
        return k_globd

    def globK(self):
        for e in range(self.num_elem):
            elem_nodes = self.elem_conn[e]
            elem_node_coors = self.node_coors[np.ix_(elem_nodes)]
            L_ele = np.sqrt((elem_node_coors[0][0]-elem_node_coors[1][0])**2+\
                            (elem_node_coors[0][1]-elem_node_coors[1][1])**2)
            theta = self.thetas[e]
            k_ele = self.truss_stiffness(theta,self.E,self.A, L_ele)
            n1,n2 = elem_nodes
            dof1,dof2 = self.node_dofs[n1],self.node_dofs[n2]
            cur_dofs = np.concatenate((dof1,dof2))
            for loc1,glb1 in enumerate(cur_dofs):
                for loc2,glb2 in enumerate(cur_dofs):
                    self.K[glb1][glb2] += k_ele[loc1][loc2]

    def solve(self):
        F_f = self.force
        D_s = self.deform
        F_ext = F_f - np.dot(self.K[np.ix_(f,s)],D_s)
        D_f = np.linalg.solve(self.K[np.ix_(f,f)],F_ext)
        F_s = np.dot(self.K[np.ix_(s,f)],D_f)+np.dot(self.K[np.ix_(s,s)],D_s)
        self.D_f = D_f
        self.F_s = F_s
            

if __name__=="__main__":
    """
    num_node,num_elem,dims,E,A,f,s,force,deform
    """
    print("Exercise4")
    f = np.array([0,1])
    s = np.array([x for x in range(2,8)])
    force = np.array([1000,-1000])
    deform = np.array([0.0 for x in range(6)])
    tr = Truss(4,3,2,10e6,1,f,s,force,deform)
    #Add node coordinate information
    tr.add_node(0,0,0)
    tr.add_node(1,100*np.sqrt(3),100)
    tr.add_node(2,0,100)
    tr.add_node(3,100/np.sqrt(3),100)
    #Add elem connection
    tr.add_elem(0,0,1)
    tr.add_elem(1,0,2)
    tr.add_elem(2,0,3)
    #Add dof information
    tr.add_dofs()
    #Add theta
    tr.add_theta(60)
    tr.add_theta(0)
    tr.add_theta(-30)
    #GlobalStiffness
    tr.globK()
    print("Global K: ")
    print(tr.K)
    #Solve
    tr.solve()
    print("D_{f} = ",tr.D_f)
    print("F_{s} = ",tr.F_s)
    print("")
    print("Lab2:")
    s = np.array([0,1,4,5])
    f = np.array([2,3,6,7,8,9])
    force = np.array([0,0,-500,0,-500,0])
    deform = np.array([0,0,0,0])
    lab = Truss(5,6,2,1.9e6,1,f,s,force,deform)
    #Add node coordinate information
    lab.add_node(0,0,0)
    lab.add_node(1,0,3)
    lab.add_node(2,3,0)
    lab.add_node(3,3,3)
    lab.add_node(4,3,6)
    #Add elem connection
    lab.add_elem(0,0,1)
    lab.add_elem(1,1,2)
    lab.add_elem(2,2,3)
    lab.add_elem(3,1,3)
    lab.add_elem(4,1,4)
    lab.add_elem(5,3,4)
    #Add dof information
    lab.add_dofs()
    #Add theta
    lab.add_theta(0)
    lab.add_theta(-45)
    lab.add_theta(0)
    lab.add_theta(-90)
    lab.add_theta(45)
    lab.add_theta(0)
    #GlobalStiffness
    lab.globK()
    #print("Global K: ")
    #print(lab.K)
    #Solve
    lab.solve()
    print("D_{f} = ",lab.D_f)
    print("F_{s} = ",lab.F_s)
