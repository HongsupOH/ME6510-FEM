import numpy as np

def k_e(theta,E,A,L):
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
    k_globd = np.dot(Tr.T,np.dot(k_locd,Tr))
    return k_globd

def gK(Es,As,Ls,thetas,num_ele,num_node,Nodes):
    dofs = 2*num_node
    K = np.zeros((dofs,dofs))
    for e in range(num_ele):
        E = Es[e]
        A = As[e]
        L = Ls[e]
        theta = thetas[e]
        n1,n2 = Nodes[e]
        tdofs = []
        tdofs += [2*n1,2*n1+1]
        tdofs += [2*n2,2*n2+1]
        for loc1,glb1 in enumerate(tdofs):
            for loc2,glb2 in enumerate(tdofs):
                k_ele = k_e(theta,E,A,L)
                K[glb1][glb2] += k_ele[loc1][loc2]
    return K

def solve(f,s,f_f,d_s,K):
    f_ext = f_f - np.dot(K[np.ix_(f,s)],d_s)
    d_f = np.linalg.solve(K[np.ix_(f,f)],f_ext)
    f_s = np.dot(K[np.ix_(s,f)],d_f) + np.dot(K[np.ix_(s,s)],d_s)
    return d_f,f_s

def main(Es,As,Ls,thetas,num_ele,num_node,f,s,f_f,d_s,Nodes):
    print("Glob K:")
    K = gK(Es,As,Ls,thetas,num_ele,num_node,Nodes)
    print(K)
    print("")
    print("Answer for unknow factors:")
    d_f,f_s = solve(f,s,f_f,d_s,K)
    print("Unknown deformation:")
    print("d_{f}: ",d_f)
    print("Unknown force:")
    print("f_{s}: ",f_s)
    
if __name__=="__main__":
    Nodes = {0:[0,2],1:[0,1],2:[1,2],3:[2,3],4:[1,3]}
    """
    gK(Es,As,Ls,thetas,num_ele,num_node,Nodes)
    """
    Es = [30e6,30e6,30e6,30e6,30e6]
    As = [10,10,20,10,10]
    Ls = [120,96,72,120,96]
    thetas = [36.87,0,90,143,13,0]
    num_ele = 5
    num_node = 4
    f = np.array([2,3,4,5])
    s = np.array([0,1,6,7])
    f_f = np.array([0,-20000,0,0])
    d_s = np.array([0,0,0,0])
    main(Es,As,Ls,thetas,num_ele,num_node,f,s,f_f,d_s,Nodes)
    print("")
    print("Test1")
    Nodes = {0:[0,1],1:[0,2],2:[0,3]}
    Es = [30e6,30e6,30e6]
    As = [2,2,2]
    Ls = [120,120*np.sqrt(2),120]
    thetas = [90,45,0]
    num_ele = 3
    num_node = 4
    f = np.array([0,1])
    s = np.array([2,3,4,5,6,7])
    f_f = np.array([0,-10000])
    d_s = np.array([0,0,0,0,0,0])
    main(Es,As,Ls,thetas,num_ele,num_node,f,s,f_f,d_s,Nodes)
    print("")
    print("Test1")
    Nodes = {0:[0,1],1:[0,2],2:[0,3]}
    Es = [10e6,10e6,10e6]
    As = [1,1,1]
    Ls = [200,100,200/np.sqrt(3)]
    thetas = [-60,0,30]
    num_ele = 3
    num_node = 4
    f = np.array([0,1])
    s = np.array([2,3,4,5,6,7])
    f_f = np.array([1000,-1000])
    d_s = np.array([0,0,0,0,0,0])
    main(Es,As,Ls,thetas,num_ele,num_node,f,s,f_f,d_s,Nodes)
