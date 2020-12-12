import numpy as np

def k_e(k):
    k_ele = np.array([[k,-k],
                       [-k,k]])
    return k_ele

def k_glob(Ks,Nodes):
    K = np.zeros((6,6))
    num_ele = list(Nodes)
    for e in range(len(num_ele)):
        k = Ks[e]
        locs = [0,1]
        glbs = Nodes[e]
        k_ele = k_e(k)
        for i in range(len(locs)):
            for j in range(len(locs)):
                K[glbs[i],glbs[j]]+=k_ele[locs[i],locs[j]]
    return K
        
def solve(f,s,f_f,d_s,K):
    f_ext = f_f - np.dot(K[np.ix_(f,s)],d_s)
    d_f = np.linalg.solve(K[np.ix_(f,f)],f_ext)
    f_s = np.dot(K[np.ix_(s,f)],d_f) + np.dot(K[np.ix_(s,s)],d_s)
    return d_f,f_s

def stress(A,p):
    return p/A


if __name__=="__main__":
    Nodes = {0:[0,1],1:[1,2],2:[1,2],3:[2,3],4:[3,4],5:[3,4],6:[4,5]}
    Ks = [110200,34800,34800,87000,23200,23200,63800]
    K = k_glob(Ks,Nodes)
    f = np.array([1,2,3,4,5])
    s = np.array([0])
    f_f = np.array([0,0,0,0,-2])
    d_s = np.array([0])
    d_f,f_s = solve(f,s,f_f,d_s,K)
    print("stiffness K:")
    print(K)
    print("Unknown deformation")
    print(d_f)
    print("Unknown force")
    print(f_s)
    print("")
    As = [3.8,2.4,3,1.6,2.2]
    print("stress")
    for A in As:
        print(stress(A,2000))
    
   
